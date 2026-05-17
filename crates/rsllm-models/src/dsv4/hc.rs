//! Hyper-Connection (HC) pre/post merge layer.
//!
//! DS V4 Flash routes each token through `N_HC = 4` parallel residual
//! streams instead of a single one. Around every sublayer (attention or
//! MoE FFN) we run two HC operations:
//!
//! - `hc_pre` reduces the four streams into a single merged residual
//!   that the sublayer reads from. The reduction weights come from a
//!   Sinkhorn-Knopp doubly-stochastic mix matrix learned per-token.
//! - `hc_post` takes the sublayer's output and scatters it back into
//!   the four streams using a per-stream sigmoid gate `g[h]` and a
//!   `2*sigmoid`-shaped post-gate `p[h]` from the same Sinkhorn call.
//!
//! The Sinkhorn kernel itself lives in
//! [`rsllm_backend_cpu::ops::sinkhorn`]; HC just supplies the per-token
//! mix logits (from a learned projection of the streams' sum) and
//! applies the resulting `(g, p, c)` triple.
//!
//! Layout of the per-token Sinkhorn buffer (`SINKHORN_BUF_LEN = 24`):
//!
//! ```text
//! [ g[0..4]  | p[0..4]  | c[0..16] (row-major dst*4 + src) ]
//! ```
//!
//! Ported by reference from `ds4.c:4186-4310` (`hc_split_sinkhorn_one`,
//! MIT, The ds4.c authors). Line numbers pinned to ds4 commit
//! `ef0a490` (2026-05-17).

use rsllm_backend_cpu::SimdTier;
use rsllm_backend_cpu::ops::sinkhorn::{N_HC, N_HC_SINKHORN_ITER, hc_split_sinkhorn};

use super::shape::DSV4_N_EMBD;
use super::weight::{WeightBlob, matmul_weight_f32};
use crate::Error;

/// Length of the per-token Sinkhorn mix buffer:
/// `2 * N_HC + N_HC * N_HC = 8 + 16 = 24`.
pub const HC_SINKHORN_BUF_LEN: usize = 2 * N_HC + N_HC * N_HC;

/// Numerical floor passed to [`hc_split_sinkhorn`].
pub const HC_SINKHORN_EPS: f32 = 1e-6;

/// Weights for one HC pre **or** post operation. Each sublayer has its
/// own copy: a transformer block stores four sets per layer
/// (`pre_attn`, `post_attn`, `pre_ffn`, `post_ffn`).
///
/// The `mix_w` projection consumes the per-token stream sum
/// (`[N_EMBD]`) and produces `HC_SINKHORN_BUF_LEN` logits which feed
/// directly into [`hc_split_sinkhorn`]. `mix_base` is the per-logit
/// bias (added after the projection, before Sinkhorn).
#[derive(Debug, Clone, Copy)]
pub struct HcOpWeights<'a> {
    /// `[HC_SINKHORN_BUF_LEN × N_EMBD]` = `[24 × 4096]`.
    pub mix_w: WeightBlob<'a>,
    /// `[HC_SINKHORN_BUF_LEN]` = `[24]` bias.
    pub mix_base: &'a [f32],
    /// Three channel scales `[pre_scale, post_scale, comb_scale]` from
    /// ds4's HC config. Pre / post / comb correspond to the three
    /// sub-regions of the Sinkhorn buffer.
    pub scale: [f32; 3],
}

/// Reusable scratch for HC pre or post over a batch of tokens.
#[derive(Debug, Default)]
pub struct HcScratch {
    /// `[n_tok × N_EMBD]` — per-token sum of the four streams (input to the mix projection).
    pub stream_sum: Vec<f32>,
    /// `[n_tok × HC_SINKHORN_BUF_LEN]` — projected mix logits per token.
    pub mix_logits: Vec<f32>,
    /// `[HC_SINKHORN_BUF_LEN]` — Sinkhorn output for the current token.
    pub sinkhorn_out: Vec<f32>,
}

impl HcScratch {
    /// Allocate scratch sized for `n_tok` tokens.
    #[must_use]
    pub fn new(n_tok: usize) -> Self {
        Self {
            stream_sum: vec![0.0_f32; n_tok * DSV4_N_EMBD],
            mix_logits: vec![0.0_f32; n_tok * HC_SINKHORN_BUF_LEN],
            sinkhorn_out: vec![0.0_f32; HC_SINKHORN_BUF_LEN],
        }
    }

    /// Resize scratch in place for a new `n_tok`.
    pub fn resize(&mut self, n_tok: usize) {
        self.stream_sum.resize(n_tok * DSV4_N_EMBD, 0.0);
        self.mix_logits.resize(n_tok * HC_SINKHORN_BUF_LEN, 0.0);
        self.sinkhorn_out.resize(HC_SINKHORN_BUF_LEN, 0.0);
    }
}

/// Pre-merge: reduce four residual streams to one merged token using
/// the Sinkhorn mix matrix's first row.
///
/// `streams` is `[n_tok × N_HC × N_EMBD]`. `merged` is `[n_tok × N_EMBD]`.
///
/// Reasoning for using row 0 of `c`: the Sinkhorn output is doubly
/// stochastic, so every row sums to 1 — picking any fixed row gives a
/// valid convex combination of streams. Row 0 matches our reading of
/// `ds4.c:4186+` (`hc_split_sinkhorn_one`).
///
/// **TODO (numerical-parity gate)**: this assumption — that ds4 reads
/// row 0 unconditionally rather than a per-token learned destination
/// row — should be verified against the ds4.c reference when running
/// the 50-token greedy-decode parity test. If ds4 actually picks the
/// destination row based on some other signal, the merge weights will
/// differ systematically and the parity gate will fail; that's the
/// signal to revisit this choice.
pub fn hc_pre(
    streams: &[f32],
    merged: &mut [f32],
    weights: &HcOpWeights<'_>,
    scratch: &mut HcScratch,
    n_tok: usize,
    tier: SimdTier,
) -> Result<(), Error> {
    let n_embd = DSV4_N_EMBD;
    if streams.len() != n_tok * N_HC * n_embd {
        return Err(Error::ShapeMismatch {
            key: "hc_pre.streams",
            expected: format!("{}", n_tok * N_HC * n_embd),
            actual: format!("{}", streams.len()),
        });
    }
    if merged.len() != n_tok * n_embd {
        return Err(Error::ShapeMismatch {
            key: "hc_pre.merged",
            expected: format!("{}", n_tok * n_embd),
            actual: format!("{}", merged.len()),
        });
    }
    if weights.mix_base.len() != HC_SINKHORN_BUF_LEN {
        return Err(Error::ShapeMismatch {
            key: "hc_pre.mix_base",
            expected: format!("{HC_SINKHORN_BUF_LEN}"),
            actual: format!("{}", weights.mix_base.len()),
        });
    }
    scratch.resize(n_tok);

    // 1. Stream sum per token: input to the mix projection.
    stream_sum(streams, &mut scratch.stream_sum, n_tok);

    // 2. Project sum → 24 logits per token: mix_logits = mix_w @ stream_sum.
    matmul_weight_f32(
        &mut scratch.mix_logits,
        &weights.mix_w,
        &scratch.stream_sum,
        n_tok,
        n_embd,
        HC_SINKHORN_BUF_LEN,
        tier,
    )?;

    // 3. For each token: Sinkhorn → merge stream lanes with c[0, *].
    for t in 0..n_tok {
        let mix_t =
            &scratch.mix_logits[t * HC_SINKHORN_BUF_LEN..(t + 1) * HC_SINKHORN_BUF_LEN];
        hc_split_sinkhorn(
            &mut scratch.sinkhorn_out,
            mix_t,
            &weights.scale,
            weights.mix_base,
            N_HC,
            N_HC_SINKHORN_ITER,
            HC_SINKHORN_EPS,
            tier,
        )
        .map_err(map_cpu_err("hc_pre.sinkhorn"))?;

        // c is laid out at offset 2*N_HC, row-major (dst * N_HC + src).
        let c_off = 2 * N_HC;
        // Row 0 of c: weights c[0,0..N_HC].
        let merge_w = &scratch.sinkhorn_out[c_off..c_off + N_HC];
        let merged_t = &mut merged[t * n_embd..(t + 1) * n_embd];
        for col in merged_t.iter_mut() {
            *col = 0.0;
        }
        for (h, &w) in merge_w.iter().enumerate().take(N_HC) {
            let stream_h_off = (t * N_HC + h) * n_embd;
            for i in 0..n_embd {
                merged_t[i] += w * streams[stream_h_off + i];
            }
        }
    }
    Ok(())
}

/// Post-scatter: add the sublayer output back to each stream weighted
/// by a learned per-stream gate.
///
/// `streams` carries the `[n_tok × N_HC × N_EMBD]` residual state and
/// is updated **in place**. `sublayer_out` is the `[n_tok × N_EMBD]`
/// activation that the wrapped sublayer (attention or MoE FFN)
/// produced for the merged token.
///
/// Update rule per stream `h` and token `t`:
///   `streams[t, h] = p[h] * streams[t, h] + g[h] * sublayer_out[t]`
///
/// Both `g[h]` and `p[h]` come from the Sinkhorn output buffer; ds4's
/// `hc_split_post` (`ds4.c:4186+` HC family) uses the same combination.
pub fn hc_post(
    streams: &mut [f32],
    sublayer_out: &[f32],
    weights: &HcOpWeights<'_>,
    scratch: &mut HcScratch,
    n_tok: usize,
    tier: SimdTier,
) -> Result<(), Error> {
    let n_embd = DSV4_N_EMBD;
    if streams.len() != n_tok * N_HC * n_embd {
        return Err(Error::ShapeMismatch {
            key: "hc_post.streams",
            expected: format!("{}", n_tok * N_HC * n_embd),
            actual: format!("{}", streams.len()),
        });
    }
    if sublayer_out.len() != n_tok * n_embd {
        return Err(Error::ShapeMismatch {
            key: "hc_post.sublayer_out",
            expected: format!("{}", n_tok * n_embd),
            actual: format!("{}", sublayer_out.len()),
        });
    }
    if weights.mix_base.len() != HC_SINKHORN_BUF_LEN {
        return Err(Error::ShapeMismatch {
            key: "hc_post.mix_base",
            expected: format!("{HC_SINKHORN_BUF_LEN}"),
            actual: format!("{}", weights.mix_base.len()),
        });
    }
    scratch.resize(n_tok);

    stream_sum(streams, &mut scratch.stream_sum, n_tok);

    matmul_weight_f32(
        &mut scratch.mix_logits,
        &weights.mix_w,
        &scratch.stream_sum,
        n_tok,
        n_embd,
        HC_SINKHORN_BUF_LEN,
        tier,
    )?;

    for t in 0..n_tok {
        let mix_t =
            &scratch.mix_logits[t * HC_SINKHORN_BUF_LEN..(t + 1) * HC_SINKHORN_BUF_LEN];
        hc_split_sinkhorn(
            &mut scratch.sinkhorn_out,
            mix_t,
            &weights.scale,
            weights.mix_base,
            N_HC,
            N_HC_SINKHORN_ITER,
            HC_SINKHORN_EPS,
            tier,
        )
        .map_err(map_cpu_err("hc_post.sinkhorn"))?;

        let g = &scratch.sinkhorn_out[0..N_HC];
        let p = &scratch.sinkhorn_out[N_HC..2 * N_HC];
        let sub_t = &sublayer_out[t * n_embd..(t + 1) * n_embd];

        for h in 0..N_HC {
            let gh = g[h];
            let ph = p[h];
            let stream_h_off = (t * N_HC + h) * n_embd;
            let stream_h = &mut streams[stream_h_off..stream_h_off + n_embd];
            for i in 0..n_embd {
                stream_h[i] = ph * stream_h[i] + gh * sub_t[i];
            }
        }
    }
    Ok(())
}

/// Compute `out[t, *] = sum_h streams[t, h, *]` per token.
fn stream_sum(streams: &[f32], out: &mut [f32], n_tok: usize) {
    let n_embd = DSV4_N_EMBD;
    out.fill(0.0);
    for t in 0..n_tok {
        let dst = &mut out[t * n_embd..(t + 1) * n_embd];
        for h in 0..N_HC {
            let src_off = (t * N_HC + h) * n_embd;
            for i in 0..n_embd {
                dst[i] += streams[src_off + i];
            }
        }
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

    fn zero_op_weights<'a>(base: &'a [f32], w: &'a [f32]) -> HcOpWeights<'a> {
        HcOpWeights {
            mix_w: WeightBlob::F32(w),
            mix_base: base,
            scale: [1.0, 1.0, 1.0],
        }
    }

    #[test]
    fn hc_pre_merged_is_finite_and_shape_correct() {
        let n_tok = 2;
        // Mix projection that always outputs ones; this exercises every
        // dim of the projection and forces a uniform Sinkhorn input —
        // expected result: c is doubly stochastic ≈ uniform, p, g positive.
        let mix_w = vec![1.0_f32 / DSV4_N_EMBD as f32; HC_SINKHORN_BUF_LEN * DSV4_N_EMBD];
        let mix_base = vec![0.0_f32; HC_SINKHORN_BUF_LEN];
        let weights = zero_op_weights(&mix_base, &mix_w);

        let streams: Vec<f32> = (0..n_tok * N_HC * DSV4_N_EMBD)
            .map(|i| ((i as f32) * 0.001).sin())
            .collect();
        let mut merged = vec![0.0_f32; n_tok * DSV4_N_EMBD];
        let mut scratch = HcScratch::new(n_tok);

        hc_pre(
            &streams,
            &mut merged,
            &weights,
            &mut scratch,
            n_tok,
            SimdTier::Scalar,
        )
        .unwrap();

        assert!(merged.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn hc_post_preserves_finite_invariant() {
        let n_tok = 1;
        let mix_w = vec![0.01_f32; HC_SINKHORN_BUF_LEN * DSV4_N_EMBD];
        let mix_base = vec![0.0_f32; HC_SINKHORN_BUF_LEN];
        let weights = zero_op_weights(&mix_base, &mix_w);

        let mut streams = vec![0.5_f32; n_tok * N_HC * DSV4_N_EMBD];
        let sublayer_out = vec![1.0_f32; n_tok * DSV4_N_EMBD];
        let mut scratch = HcScratch::new(n_tok);

        hc_post(
            &mut streams,
            &sublayer_out,
            &weights,
            &mut scratch,
            n_tok,
            SimdTier::Scalar,
        )
        .unwrap();

        assert!(streams.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn hc_pre_rejects_wrong_streams_shape() {
        let mix_w = vec![0.0_f32; HC_SINKHORN_BUF_LEN * DSV4_N_EMBD];
        let mix_base = vec![0.0_f32; HC_SINKHORN_BUF_LEN];
        let weights = zero_op_weights(&mix_base, &mix_w);
        let streams = vec![0.0_f32; 17]; // wrong
        let mut merged = vec![0.0_f32; DSV4_N_EMBD];
        let mut scratch = HcScratch::new(1);
        let err = hc_pre(
            &streams,
            &mut merged,
            &weights,
            &mut scratch,
            1,
            SimdTier::Scalar,
        )
        .unwrap_err();
        assert!(matches!(err, Error::ShapeMismatch { .. }));
    }

    #[test]
    fn hc_pre_uniform_mix_yields_average_of_streams() {
        // If every stream carries identical values, the merged result
        // must equal one of them (a convex combination of equal numbers
        // is the number itself), regardless of the Sinkhorn weights.
        let n_tok = 1;
        let mix_w = vec![0.0_f32; HC_SINKHORN_BUF_LEN * DSV4_N_EMBD];
        let mix_base = vec![0.0_f32; HC_SINKHORN_BUF_LEN];
        let weights = zero_op_weights(&mix_base, &mix_w);

        // All four streams carry value 0.25.
        let streams = vec![0.25_f32; n_tok * N_HC * DSV4_N_EMBD];
        let mut merged = vec![0.0_f32; n_tok * DSV4_N_EMBD];
        let mut scratch = HcScratch::new(n_tok);
        hc_pre(
            &streams,
            &mut merged,
            &weights,
            &mut scratch,
            n_tok,
            SimdTier::Scalar,
        )
        .unwrap();
        // Convex combo of 0.25, 0.25, 0.25, 0.25 = 0.25 exactly.
        for &v in &merged[..16] {
            assert!((v - 0.25).abs() < 1e-5, "merged != 0.25: {v}");
        }
    }
}
