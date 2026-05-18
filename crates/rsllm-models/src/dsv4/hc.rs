//! Hyper-Connection (HC) pre/post merge layer.
//!
//! DS V4 Flash routes each token through `N_HC = 4` parallel residual
//! streams instead of a single one. Around every sublayer (attention or
//! MoE FFN) we run two HC operations:
//!
//! - [`hc_pre`] reduces the four streams into a single merged residual
//!   that the sublayer reads from. The mix coefficients come from a
//!   per-token Sinkhorn-Knopp doubly-stochastic matrix produced by a
//!   learned projection of the (flattened, RMS-normed) residual.
//! - [`hc_post`] takes the sublayer's output and scatters it back into
//!   the four streams using the same Sinkhorn output's `post[h]` lane
//!   gate plus the doubly-stochastic combine matrix `comb[dst, src]`.
//!   **hc_post owns no weights of its own** — it consumes the split
//!   tensor produced by the most recent `hc_pre` for the same sublayer.
//!
//! Layout of the per-token Sinkhorn buffer (`HC_MIX_DIM = 24`):
//!
//! ```text
//! [ pre[0..4]  | post[0..4]  | comb[0..16] (row-major dst + src*N_HC) ]
//! ```
//!
//! Ported by reference from `ds4.c:4186-4385` (`hc_split_sinkhorn_one`,
//! `hc_pre_from_state_one`, `hc_post_one`, MIT, The ds4.c authors).
//! Line numbers pinned to ds4 commit `ef0a490` (2026-05-17).
//!
//! ## Weight layout (per sublayer, matches ds4)
//!
//! Each transformer block has **two** sublayers (attention + FFN). Each
//! sublayer carries one [`HcSublayerWeights`] bundle with three tensors:
//!
//! | Tensor | Shape | dtype | Role |
//! |---|---|---|---|
//! | `mix_fn` | `[HC_DIM × HC_MIX_DIM]` = `[16384 × 24]` | F16 | Project flattened+normed residual to mix logits |
//! | `scale` | `[3]` | F32 | Channel scales `[pre, post, comb]` for the Sinkhorn split |
//! | `base`  | `[HC_MIX_DIM]` = `[24]` | F32 | Per-logit bias added before sigmoid/softmax |
//!
//! The upstream GGUF tensor names are
//! `blk.{il}.hc_attn_fn.weight`, `hc_attn_scale.weight`,
//! `hc_attn_base.weight` (and the `hc_ffn_*` analogues for the FFN
//! sublayer). v0.1.0's loader maps to those names — see
//! [`crate::dsv4::loader::load_hc_sublayer_weights`].

use rsllm_backend_cpu::SimdTier;
use rsllm_backend_cpu::ops::sinkhorn::{N_HC, N_HC_SINKHORN_ITER, hc_split_sinkhorn};

use super::shape::DSV4_N_EMBD;
use super::weight::{WeightBlob, matmul_weight_f32};
use crate::Error;

/// Length of the per-token Sinkhorn mix buffer:
/// `2 * N_HC + N_HC * N_HC = 8 + 16 = 24`.
pub const HC_MIX_DIM: usize = 2 * N_HC + N_HC * N_HC;

/// Flattened residual length consumed by the `mix_fn` projection:
/// `N_HC × DSV4_N_EMBD = 4 × 4096 = 16384`.
pub const HC_DIM: usize = N_HC * DSV4_N_EMBD;

/// Backwards-compatible alias retained for callers (e.g. F008.B test
/// builders) that referenced the old name. New code should prefer
/// [`HC_MIX_DIM`].
pub const HC_SINKHORN_BUF_LEN: usize = HC_MIX_DIM;

/// Numerical floor passed to [`hc_split_sinkhorn`] and the
/// no-weight RMSNorm prelude (ds4.c:4301 uses `DS4_RMS_EPS = 1e-6`).
pub const HC_SINKHORN_EPS: f32 = 1e-6;

/// Per-sublayer HC weights (attention or FFN). Each transformer block
/// holds two of these — `hc_attn` and `hc_ffn`. The post step has no
/// weights of its own; it reuses the [`HcScratch::split`] tensor that
/// the matching pre step produced.
///
/// All three fields come from ds4-format GGUF tensors:
/// `blk.{il}.hc_{attn,ffn}_fn.weight` / `_scale.weight` / `_base.weight`.
#[derive(Debug, Clone, Copy)]
pub struct HcSublayerWeights<'a> {
    /// `[HC_DIM × HC_MIX_DIM]` = `[16384 × 24]` F16 (or F32 in tests):
    /// projection from the flattened-normed residual to mix logits.
    pub mix_fn: WeightBlob<'a>,
    /// `[3]` F32 channel scales `[pre, post, comb]` for the Sinkhorn
    /// split's three sub-regions.
    pub scale: &'a [f32],
    /// `[HC_MIX_DIM]` F32 per-logit bias added before
    /// sigmoid/softmax inside [`hc_split_sinkhorn`].
    pub base: &'a [f32],
}

/// Reusable scratch for one HC pre + matching HC post on a batch.
///
/// `split` is sized to hold one full Sinkhorn output per token across
/// the pre→[sublayer]→post boundary. The post step reads it; the next
/// pre call overwrites it.
#[derive(Debug, Default)]
pub struct HcScratch {
    /// `[n_tok × HC_DIM]` flattened + RMS-normed-no-weight residual.
    pub flat: Vec<f32>,
    /// `[n_tok × HC_MIX_DIM]` projected mix logits.
    pub mix_logits: Vec<f32>,
    /// `[n_tok × HC_MIX_DIM]` Sinkhorn output split per token, persisted
    /// from the [`hc_pre`] call to the matching [`hc_post`] call.
    pub split: Vec<f32>,
    /// `[N_HC × DSV4_N_EMBD]` per-token residual snapshot used as
    /// read-side state during the in-place [`hc_post`] update.
    pub prev_token: Vec<f32>,
}

impl HcScratch {
    /// Allocate scratch sized for `n_tok` tokens.
    #[must_use]
    pub fn new(n_tok: usize) -> Self {
        Self {
            flat: vec![0.0_f32; n_tok * HC_DIM],
            mix_logits: vec![0.0_f32; n_tok * HC_MIX_DIM],
            split: vec![0.0_f32; n_tok * HC_MIX_DIM],
            prev_token: vec![0.0_f32; N_HC * DSV4_N_EMBD],
        }
    }

    /// Resize scratch in place for a new `n_tok`.
    pub fn resize(&mut self, n_tok: usize) {
        self.flat.resize(n_tok * HC_DIM, 0.0);
        self.mix_logits.resize(n_tok * HC_MIX_DIM, 0.0);
        self.split.resize(n_tok * HC_MIX_DIM, 0.0);
        self.prev_token.resize(N_HC * DSV4_N_EMBD, 0.0);
    }
}

/// HC pre-merge: flatten + RMS-norm the residual, project to mix
/// logits via `mix_fn`, run Sinkhorn split, and produce the merged
/// residual via a per-stream weighted sum.
///
/// The `[n_tok × HC_MIX_DIM]` Sinkhorn split is stashed in
/// [`HcScratch::split`] for the matching [`hc_post`] call.
///
/// `streams` is `[n_tok × N_HC × DSV4_N_EMBD]`. `merged` is
/// `[n_tok × DSV4_N_EMBD]`. Both are row-major.
///
/// Mirrors `ds4.c:4284-4315` `hc_pre_from_state_one_scratch`.
///
/// # Errors
/// - [`Error::ShapeMismatch`] if input slice lengths disagree with
///   the documented shape.
pub fn hc_pre(
    streams: &[f32],
    merged: &mut [f32],
    weights: &HcSublayerWeights<'_>,
    scratch: &mut HcScratch,
    n_tok: usize,
    tier: SimdTier,
) -> Result<(), Error> {
    if streams.len() != n_tok * HC_DIM {
        return Err(Error::ShapeMismatch {
            key: "hc_pre.streams",
            expected: format!("{}", n_tok * HC_DIM),
            actual: format!("{}", streams.len()),
        });
    }
    if merged.len() != n_tok * DSV4_N_EMBD {
        return Err(Error::ShapeMismatch {
            key: "hc_pre.merged",
            expected: format!("{}", n_tok * DSV4_N_EMBD),
            actual: format!("{}", merged.len()),
        });
    }
    if weights.scale.len() != 3 {
        return Err(Error::ShapeMismatch {
            key: "hc_pre.scale",
            expected: "3".to_string(),
            actual: format!("{}", weights.scale.len()),
        });
    }
    if weights.base.len() != HC_MIX_DIM {
        return Err(Error::ShapeMismatch {
            key: "hc_pre.base",
            expected: format!("{HC_MIX_DIM}"),
            actual: format!("{}", weights.base.len()),
        });
    }
    scratch.resize(n_tok);

    // 1. Per-token flatten + RMS-norm-no-weight (ds4.c:4301
    //    `rms_norm_no_weight(flat, residual_hc, hc_dim, DS4_RMS_EPS)`).
    //    The four streams are already contiguous in memory; flattening
    //    is a copy + in-place normalization on the copy.
    for t in 0..n_tok {
        let src = &streams[t * HC_DIM..(t + 1) * HC_DIM];
        let dst = &mut scratch.flat[t * HC_DIM..(t + 1) * HC_DIM];
        rms_norm_no_weight_into(src, dst, HC_SINKHORN_EPS);
    }

    // 2. Project flat → mix logits via mix_fn: shape
    //    `[n_tok × HC_DIM] × [HC_DIM × HC_MIX_DIM] → [n_tok × HC_MIX_DIM]`.
    let scale: [f32; 3] = [weights.scale[0], weights.scale[1], weights.scale[2]];
    matmul_weight_f32(
        &mut scratch.mix_logits,
        &weights.mix_fn,
        &scratch.flat,
        n_tok,
        HC_DIM,
        HC_MIX_DIM,
        tier,
    )?;

    // 3. Per-token Sinkhorn split + weighted sum.
    for t in 0..n_tok {
        let mix_t = &scratch.mix_logits[t * HC_MIX_DIM..(t + 1) * HC_MIX_DIM];
        let split_t = &mut scratch.split[t * HC_MIX_DIM..(t + 1) * HC_MIX_DIM];
        hc_split_sinkhorn(
            split_t,
            mix_t,
            &scale,
            weights.base,
            N_HC,
            N_HC_SINKHORN_ITER,
            HC_SINKHORN_EPS,
            tier,
        )
        .map_err(map_cpu_err("hc_pre.sinkhorn"))?;

        // `split[0..N_HC]` is the per-stream weight `pre[h]` consumed by
        // hc_weighted_sum_one (ds4.c:4267-4280). Each output dim d is a
        // convex combination of streams[t, h, d] weighted by pre[h].
        let pre_w = &split_t[0..N_HC];
        let merged_t = &mut merged[t * DSV4_N_EMBD..(t + 1) * DSV4_N_EMBD];
        merged_t.fill(0.0);
        for (h, &w) in pre_w.iter().enumerate().take(N_HC) {
            let stream_h_off = (t * N_HC + h) * DSV4_N_EMBD;
            let stream_h = &streams[stream_h_off..stream_h_off + DSV4_N_EMBD];
            for i in 0..DSV4_N_EMBD {
                merged_t[i] += w * stream_h[i];
            }
        }
    }
    Ok(())
}

/// HC post-scatter: replace each residual stream with a learned
/// combination of (a) the previous stream state weighted by the
/// doubly-stochastic combine matrix and (b) the sublayer output
/// scaled by `post[h]`.
///
/// Per `ds4.c:4366-4385` (`hc_post_one`):
///
/// ```text
/// new_streams[t, dst, d] = sublayer_out[t, d] * post[t, dst]
///                        + Σ_src comb[t, dst + src*N_HC] * old_streams[t, src, d]
/// ```
///
/// `streams` is `[n_tok × N_HC × DSV4_N_EMBD]` and is updated in
/// place. `sublayer_out` is `[n_tok × DSV4_N_EMBD]`. `scratch.split`
/// must have been filled by the matching [`hc_pre`] call on the same
/// `n_tok`; this function reads the `post` + `comb` slices out of it.
///
/// # Errors
/// - [`Error::ShapeMismatch`] if input slice lengths disagree.
pub fn hc_post(
    streams: &mut [f32],
    sublayer_out: &[f32],
    scratch: &mut HcScratch,
    n_tok: usize,
) -> Result<(), Error> {
    if streams.len() != n_tok * HC_DIM {
        return Err(Error::ShapeMismatch {
            key: "hc_post.streams",
            expected: format!("{}", n_tok * HC_DIM),
            actual: format!("{}", streams.len()),
        });
    }
    if sublayer_out.len() != n_tok * DSV4_N_EMBD {
        return Err(Error::ShapeMismatch {
            key: "hc_post.sublayer_out",
            expected: format!("{}", n_tok * DSV4_N_EMBD),
            actual: format!("{}", sublayer_out.len()),
        });
    }
    if scratch.split.len() != n_tok * HC_MIX_DIM {
        return Err(Error::ShapeMismatch {
            key: "hc_post.scratch.split",
            expected: format!("{} (n_tok × HC_MIX_DIM)", n_tok * HC_MIX_DIM),
            actual: format!("{}", scratch.split.len()),
        });
    }
    if scratch.prev_token.len() != N_HC * DSV4_N_EMBD {
        // The matching hc_pre would have sized this to N_HC * N_EMBD;
        // if we got here with a smaller buffer, the caller paired
        // hc_post with a mismatched scratch.
        scratch.prev_token.resize(N_HC * DSV4_N_EMBD, 0.0);
    }

    for t in 0..n_tok {
        let split_t = &scratch.split[t * HC_MIX_DIM..(t + 1) * HC_MIX_DIM];
        // Layout from hc_split_sinkhorn: [pre | post | comb (dst-major)].
        let post = &split_t[N_HC..2 * N_HC];
        let comb = &split_t[2 * N_HC..2 * N_HC + N_HC * N_HC];

        // Snapshot old streams[t, :, :] before in-place overwrite.
        let stream_t_off = t * HC_DIM;
        scratch
            .prev_token
            .copy_from_slice(&streams[stream_t_off..stream_t_off + HC_DIM]);

        let sub_t = &sublayer_out[t * DSV4_N_EMBD..(t + 1) * DSV4_N_EMBD];
        for dst in 0..N_HC {
            let p_dst = post[dst];
            let stream_dst_off = stream_t_off + dst * DSV4_N_EMBD;
            let stream_dst = &mut streams[stream_dst_off..stream_dst_off + DSV4_N_EMBD];
            for d in 0..DSV4_N_EMBD {
                let mut acc = sub_t[d] * p_dst;
                for src in 0..N_HC {
                    // comb is row-major with `dst + src * N_HC` —
                    // matches ds4.c:4380 exactly.
                    acc += comb[dst + src * N_HC]
                        * scratch.prev_token[src * DSV4_N_EMBD + d];
                }
                stream_dst[d] = acc;
            }
        }
    }
    Ok(())
}

/// Output-side HC head collapse: reduce one token's `[N_HC × N_EMBD]`
/// residual streams to a single `[N_EMBD]` vector, ready for the
/// final RMSNorm + LM head projection.
///
/// Ported from `ds4.c:7916-7944` (`output_hc_head_one`):
///
/// ```text
/// flat = rms_norm_no_weight(inp_hc)             # [HC_DIM]
/// pre  = matvec_f16(output_hc_fn, flat)         # [N_HC]
/// w[i] = sigmoid(pre[i] * scale[0] + base[i]) + eps   # [N_HC]
/// out  = Σ_h w[h] * inp_hc[h, :]                 # [N_EMBD]
/// ```
///
/// `inp_hc` is `[N_HC × N_EMBD]` for one token; `out` is `[N_EMBD]`.
/// `mix_fn` must be `[HC_DIM × N_HC]` (caller-validated).
///
/// # Errors
/// [`Error::ShapeMismatch`] if any slice length disagrees.
pub fn output_hc_collapse(
    inp_hc: &[f32],
    out: &mut [f32],
    mix_fn: &WeightBlob<'_>,
    scale: &[f32],
    base: &[f32],
    tier: SimdTier,
) -> Result<(), Error> {
    if inp_hc.len() != HC_DIM {
        return Err(Error::ShapeMismatch {
            key: "output_hc_collapse.inp_hc",
            expected: format!("{HC_DIM}"),
            actual: format!("{}", inp_hc.len()),
        });
    }
    if out.len() != DSV4_N_EMBD {
        return Err(Error::ShapeMismatch {
            key: "output_hc_collapse.out",
            expected: format!("{DSV4_N_EMBD}"),
            actual: format!("{}", out.len()),
        });
    }
    if scale.len() != 1 {
        return Err(Error::ShapeMismatch {
            key: "output_hc_collapse.scale",
            expected: "1".to_string(),
            actual: format!("{}", scale.len()),
        });
    }
    if base.len() != N_HC {
        return Err(Error::ShapeMismatch {
            key: "output_hc_collapse.base",
            expected: format!("{N_HC}"),
            actual: format!("{}", base.len()),
        });
    }

    // Flatten + RMS-norm-no-weight (matches ds4.c:7930).
    let mut flat = vec![0.0_f32; HC_DIM];
    rms_norm_no_weight_into(inp_hc, &mut flat, HC_SINKHORN_EPS);

    // Project: pre = mix_fn @ flat, shape [HC_DIM × N_HC] @ [HC_DIM] → [N_HC].
    let mut pre = vec![0.0_f32; N_HC];
    matmul_weight_f32(&mut pre, mix_fn, &flat, 1, HC_DIM, N_HC, tier)?;

    // Per-stream sigmoid gate.
    let s = scale[0];
    let mut w = [0.0_f32; N_HC];
    for (i, w_i) in w.iter_mut().enumerate() {
        let z = pre[i] * s + base[i];
        *w_i = sigmoid(z) + HC_SINKHORN_EPS;
    }

    // Weighted sum across streams. ds4's hc_weighted_sum_one
    // (ds4.c:4267-4280) takes per-stream weights and the per-stream
    // residual; same shape as our hc_pre's pre-region weighted_sum.
    for d in 0..DSV4_N_EMBD {
        let mut acc = 0.0_f32;
        for (h, &wh) in w.iter().enumerate() {
            acc += wh * inp_hc[h * DSV4_N_EMBD + d];
        }
        out[d] = acc;
    }
    Ok(())
}

/// Numerically stable sigmoid (mirrors `ds4.c:4885` `sigmoid_stable`).
#[inline]
fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        let e = (-x).exp();
        1.0 / (1.0 + e)
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// In-place RMS-norm-no-weight: `dst[i] = src[i] / sqrt(mean(src²) +
/// eps)`. Mirrors ds4's `rms_norm_no_weight` (used at `ds4.c:4301`).
///
/// Sum-of-squares accumulates in f64 to match ds4's `double ss`
/// precision pattern (`ds4.c:rmsnorm_kernel`).
///
/// Uses `assert_eq!` (not `debug_assert_eq!`) so a mismatched
/// `src`/`dst` length is loud in release builds too — a silent
/// `zip().take(min)` would produce wrong-but-non-panicking output
/// for any future caller that wires up the wrong scratch buffer.
fn rms_norm_no_weight_into(src: &[f32], dst: &mut [f32], eps: f32) {
    assert_eq!(
        src.len(),
        dst.len(),
        "rms_norm_no_weight_into: src.len ({}) != dst.len ({})",
        src.len(),
        dst.len()
    );
    let n = src.len() as f64;
    let mut sumsq = 0.0_f64;
    for &v in src {
        let vd = v as f64;
        sumsq += vd * vd;
    }
    let rms_recip = (1.0_f64 / (sumsq / n + eps as f64).sqrt()) as f32;
    for (d, &s) in dst.iter_mut().zip(src.iter()) {
        *d = s * rms_recip;
    }
}

fn map_cpu_err(stage: &'static str) -> impl FnOnce(rsllm_backend_cpu::Error) -> Error {
    move |e| Error::ShapeMismatch {
        key: stage,
        expected: "valid kernel shape".to_string(),
        actual: format!("{e}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a zero-valued `HcSublayerWeights` for a given `mix_fn`
    /// matrix + base bias slice. `scale` is `[1.0, 1.0, 1.0]` —
    /// neutral mix scaling.
    fn zero_op_weights<'a>(
        mix_fn: WeightBlob<'a>,
        base: &'a [f32],
        scale: &'a [f32; 3],
    ) -> HcSublayerWeights<'a> {
        HcSublayerWeights {
            mix_fn,
            scale: scale.as_slice(),
            base,
        }
    }

    /// Build a `[HC_DIM × HC_MIX_DIM]` F32 matrix initialized to zeros.
    /// Use this when you want hc_pre's projection to feed all-zero
    /// logits into Sinkhorn → uniform `pre[h] ≈ 1/N_HC` mix.
    fn zero_mix_fn_storage() -> Vec<f32> {
        vec![0.0_f32; HC_DIM * HC_MIX_DIM]
    }

    #[test]
    fn hc_pre_merged_is_finite_and_shape_correct() {
        let n_tok = 2;
        let mix_storage = zero_mix_fn_storage();
        let base = vec![0.0_f32; HC_MIX_DIM];
        let scale = [1.0_f32, 1.0, 1.0];
        let mut streams = vec![0.1_f32; n_tok * HC_DIM];
        let mut merged = vec![0.0_f32; n_tok * DSV4_N_EMBD];
        let mut scratch = HcScratch::new(n_tok);

        let w = zero_op_weights(WeightBlob::F32(&mix_storage), &base, &scale);
        hc_pre(
            &streams,
            &mut merged,
            &w,
            &mut scratch,
            n_tok,
            SimdTier::Scalar,
        )
        .unwrap();
        for &v in &merged {
            assert!(v.is_finite(), "merged contains non-finite value");
        }
        // Mark `streams` as used to avoid a clippy warning.
        streams[0] = 0.1;
    }

    #[test]
    fn hc_post_preserves_finite_invariant() {
        let n_tok = 2;
        let mix_storage = zero_mix_fn_storage();
        let base = vec![0.0_f32; HC_MIX_DIM];
        let scale = [1.0_f32, 1.0, 1.0];
        let mut streams = vec![0.1_f32; n_tok * HC_DIM];
        let mut merged = vec![0.0_f32; n_tok * DSV4_N_EMBD];
        let sublayer_out = vec![0.05_f32; n_tok * DSV4_N_EMBD];
        let mut scratch = HcScratch::new(n_tok);
        let w = zero_op_weights(WeightBlob::F32(&mix_storage), &base, &scale);
        // hc_pre populates scratch.split; hc_post then reads it.
        hc_pre(
            &streams,
            &mut merged,
            &w,
            &mut scratch,
            n_tok,
            SimdTier::Scalar,
        )
        .unwrap();
        hc_post(&mut streams, &sublayer_out, &mut scratch, n_tok).unwrap();
        for &v in &streams {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn hc_pre_rejects_wrong_streams_shape() {
        let n_tok = 1;
        let mix_storage = zero_mix_fn_storage();
        let base = vec![0.0_f32; HC_MIX_DIM];
        let scale = [1.0_f32, 1.0, 1.0];
        let streams = vec![0.0_f32; HC_DIM - 1]; // wrong size
        let mut merged = vec![0.0_f32; n_tok * DSV4_N_EMBD];
        let mut scratch = HcScratch::new(n_tok);
        let w = zero_op_weights(WeightBlob::F32(&mix_storage), &base, &scale);
        let err = hc_pre(
            &streams,
            &mut merged,
            &w,
            &mut scratch,
            n_tok,
            SimdTier::Scalar,
        )
        .unwrap_err();
        assert!(matches!(err, Error::ShapeMismatch { .. }));
    }

    #[test]
    fn hc_pre_uniform_mix_yields_average_of_streams() {
        // All-zero mix_fn + all-zero base → mix=0 → sigmoid(0)=0.5 →
        // pre[h] = 0.5 + eps for all h, sum ≈ 4·0.5 = 2. With
        // streams identical across h, merged ≈ sum_h pre[h] * stream[h]
        // = 2 * stream (single stream value). So if we set stream[h, :]
        // = h+1 for all d, merged[d] should be 0.5 * (1+2+3+4) = 5
        // (plus small eps drift).
        let n_tok = 1;
        let mix_storage = zero_mix_fn_storage();
        let base = vec![0.0_f32; HC_MIX_DIM];
        let scale = [1.0_f32, 1.0, 1.0];
        let mut streams = vec![0.0_f32; HC_DIM];
        for h in 0..N_HC {
            let off = h * DSV4_N_EMBD;
            for d in 0..DSV4_N_EMBD {
                streams[off + d] = (h + 1) as f32;
            }
        }
        let mut merged = vec![0.0_f32; n_tok * DSV4_N_EMBD];
        let mut scratch = HcScratch::new(n_tok);
        let w = zero_op_weights(WeightBlob::F32(&mix_storage), &base, &scale);
        hc_pre(
            &streams,
            &mut merged,
            &w,
            &mut scratch,
            n_tok,
            SimdTier::Scalar,
        )
        .unwrap();
        // Expected: 0.5 * (1+2+3+4) = 5.0; allow some slack for
        // sigmoid(0)+eps vs 0.5 exact.
        for &v in merged.iter().take(8) {
            assert!(
                (v - 5.0).abs() < 0.01,
                "expected merged ≈ 5.0, got {v}"
            );
        }
    }

    #[test]
    fn output_hc_collapse_zero_weights_yields_half_sigmoid_mix() {
        // With mix_fn = 0, scale[0] = 0, base = 0: pre = 0, w[h] =
        // sigmoid(0) + eps ≈ 0.5 for every stream. So
        //   out[d] = Σ_h 0.5 * inp_hc[h, d] = 0.5 * Σ_h inp_hc[h, d].
        //
        // Set inp_hc[h, :] = (h + 1) * 1.0; then Σ_h(h+1) = 1+2+3+4 = 10.
        // out[d] should be ≈ 0.5 * 10 = 5.0 for every d.
        let mut inp_hc = vec![0.0_f32; HC_DIM];
        for h in 0..N_HC {
            for d in 0..DSV4_N_EMBD {
                inp_hc[h * DSV4_N_EMBD + d] = (h + 1) as f32;
            }
        }
        let mix_storage = vec![0.0_f32; HC_DIM * N_HC];
        let scale = [0.0_f32];
        let base = vec![0.0_f32; N_HC];
        let mut out = vec![0.0_f32; DSV4_N_EMBD];
        output_hc_collapse(
            &inp_hc,
            &mut out,
            &WeightBlob::F32(&mix_storage),
            &scale,
            &base,
            SimdTier::Scalar,
        )
        .unwrap();
        for &v in out.iter().take(16) {
            assert!(
                (v - 5.0).abs() < 0.05,
                "expected ≈ 5.0, got {v}"
            );
        }
    }

    #[test]
    fn output_hc_collapse_rejects_wrong_shapes() {
        let bad_inp = vec![0.0_f32; HC_DIM - 1];
        let mut out = vec![0.0_f32; DSV4_N_EMBD];
        let mix = vec![0.0_f32; HC_DIM * N_HC];
        let err = output_hc_collapse(
            &bad_inp,
            &mut out,
            &WeightBlob::F32(&mix),
            &[0.0_f32],
            &[0.0_f32; N_HC],
            SimdTier::Scalar,
        )
        .unwrap_err();
        assert!(matches!(err, Error::ShapeMismatch { .. }));
    }

    #[test]
    fn hc_pre_post_multi_token_offsets_are_independent() {
        // Multi-token regression: confirm that `t * HC_DIM` offset
        // arithmetic in hc_pre / hc_post stays consistent across the
        // batch. Different tokens get different stream values, so any
        // off-by-one in the per-token indexing would surface as one
        // token contaminating another's state.
        let n_tok = 4;
        let mix_storage = zero_mix_fn_storage();
        let base = vec![0.0_f32; HC_MIX_DIM];
        let scale = [1.0_f32, 1.0, 1.0];

        // Per-token sentinel: token t's streams hold (t+1)*10, so
        // expected merged ≈ 0.5 * 4 * (t+1)*10 = 20*(t+1).
        let mut streams = vec![0.0_f32; n_tok * HC_DIM];
        for t in 0..n_tok {
            let v = (t + 1) as f32 * 10.0;
            for h in 0..N_HC {
                let off = (t * N_HC + h) * DSV4_N_EMBD;
                for d in 0..DSV4_N_EMBD {
                    streams[off + d] = v;
                }
            }
        }
        let mut merged = vec![0.0_f32; n_tok * DSV4_N_EMBD];
        let mut scratch = HcScratch::new(n_tok);
        let w = zero_op_weights(WeightBlob::F32(&mix_storage), &base, &scale);
        hc_pre(
            &streams,
            &mut merged,
            &w,
            &mut scratch,
            n_tok,
            SimdTier::Scalar,
        )
        .unwrap();
        // Each token's merged value should reflect ONLY that token's
        // streams. With sigmoid(0)+eps ≈ 0.5, pre weights sum to ≈ 2,
        // and a uniform stream of v gives merged ≈ 2v.
        for t in 0..n_tok {
            let expected = 2.0 * (t + 1) as f32 * 10.0;
            let actual = merged[t * DSV4_N_EMBD];
            assert!(
                (actual - expected).abs() < 0.5,
                "token {t}: expected merged ≈ {expected}, got {actual}"
            );
            // Spot-check end of row too — catches stride errors where
            // only the first few elements of each token are correct.
            let actual_end = merged[(t + 1) * DSV4_N_EMBD - 1];
            assert!(
                (actual_end - expected).abs() < 0.5,
                "token {t} end: expected merged ≈ {expected}, got {actual_end}"
            );
        }

        // hc_post then writes new streams. With a uniform sublayer_out,
        // each token's new state should depend ONLY on that token's
        // previous state.
        let mut sublayer_out = vec![0.0_f32; n_tok * DSV4_N_EMBD];
        for t in 0..n_tok {
            let v = (t + 1) as f32;
            for d in 0..DSV4_N_EMBD {
                sublayer_out[t * DSV4_N_EMBD + d] = v;
            }
        }
        hc_post(&mut streams, &sublayer_out, &mut scratch, n_tok).unwrap();
        // Per token t: new_stream ≈ sublayer_out (= t+1) * post (≈ 1.0)
        // + sum_over_src comb[*, src] * old_streams (≈ (t+1)*10).
        // Since comb is doubly stochastic, row-sum ≈ 1, so new ≈
        // (t+1) + (t+1)*10 = (t+1)*11. Spot-check first + last of
        // first stream of each token.
        for t in 0..n_tok {
            let expected = (t + 1) as f32 * 11.0;
            let actual = streams[t * HC_DIM];
            assert!(
                (actual - expected).abs() < 1.0,
                "post token {t}: expected ≈ {expected}, got {actual}"
            );
        }
    }

    #[test]
    fn hc_post_in_place_uses_pre_split() {
        // Confirm that hc_post reads scratch.split (filled by hc_pre)
        // rather than reaching for any external state. With zero mix_fn,
        // post[h] = 2 * sigmoid(0) = 1.0 and comb is doubly stochastic
        // with row-sum 1, so each new stream[dst, d] ≈ sublayer_out[d]
        // + avg(old_streams[*, d]).
        let n_tok = 1;
        let mix_storage = zero_mix_fn_storage();
        let base = vec![0.0_f32; HC_MIX_DIM];
        let scale = [1.0_f32, 1.0, 1.0];
        let mut streams = vec![2.0_f32; HC_DIM];
        let mut merged = vec![0.0_f32; n_tok * DSV4_N_EMBD];
        let sublayer_out = vec![3.0_f32; n_tok * DSV4_N_EMBD];
        let mut scratch = HcScratch::new(n_tok);
        let w = zero_op_weights(WeightBlob::F32(&mix_storage), &base, &scale);

        hc_pre(
            &streams,
            &mut merged,
            &w,
            &mut scratch,
            n_tok,
            SimdTier::Scalar,
        )
        .unwrap();
        hc_post(&mut streams, &sublayer_out, &mut scratch, n_tok).unwrap();

        // Each stream lane should now hold ≈ 3.0 (sub_out * post) +
        // 2.0 (old residual * row-sum of comb ≈ 1) = 5.0.
        for &v in streams.iter().take(8) {
            assert!(
                (v - 5.0).abs() < 0.1,
                "expected stream ≈ 5.0, got {v}"
            );
        }
    }
}
