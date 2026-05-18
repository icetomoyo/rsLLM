//! Per-layer LoRA weights + projection functions for the
//! compressed-KV scoring (`attn_compressor`) and the ratio-4 indexer
//! (`attn_indexer_*`).
//!
//! These produce the per-token *score* and *latent* inputs that
//! [`crate::dsv4::attention::ThreeTierAttention`] currently
//! placeholders with zeros (the `TODO(F008)` in `attention.rs`). The
//! actual replacement of those zeros lands in F008.C alongside the
//! [`crate::AttentionFn`] signature extension needed to thread the
//! residual `x` through to the adapter.
//!
//! ds4 anchors:
//! - `attn_compressor` family — see the layer-weight struct at
//!   `ds4.c:2306+` (search anchor `attn_compressor`).
//! - `attn_indexer` family — same struct, search anchor
//!   `attn_indexer`.
//!
//! Ported by reference from `ds4.c` (MIT, The ds4.c authors).
//! Line numbers pinned to ds4 commit `ef0a490` (2026-05-17).

use rsllm_backend_cpu::SimdTier;

use super::shape::{DSV4_HEAD_DIM, DSV4_N_EMBD, DSV4_N_INDEXER_HEAD, DSV4_N_INDEXER_HEAD_DIM};
use super::weight::{WeightBlob, matmul_weight_f32};
use crate::Error;

/// Multiply `a * b` and surface a `ShapeMismatch` on overflow.
/// Pattern established in F007 review fixes; applied here so a
/// caller-supplied `n_tok` cannot wrap and bypass the shape checks.
fn checked_mul_or_err(a: usize, b: usize, tag: &'static str) -> Result<usize, Error> {
    a.checked_mul(b).ok_or(Error::ShapeMismatch {
        key: tag,
        expected: format!("{a} * {b} (overflow)"),
        actual: "n/a".to_string(),
    })
}

/// Per-layer compressor weights — the 4-tensor bundle that ds4's
/// `compressor_decode_one` (`ds4.c:6431+`) consumes to produce one
/// `[HEAD_DIM]` compressed row every `compress_ratio` tokens.
///
/// Present on every layer with `compress_ratio > 0` (all but the
/// first two dense layers).
///
/// Tensor shapes (`ds4.c:2316-2321`) depend on the layer's regime:
///
/// | Tensor | Shape | dtype | Role |
/// |---|---|---|---|
/// | `kv` | `[N_EMBD × comp_width]` | F16 | KV latent projection |
/// | `gate` | `[N_EMBD × comp_width]` | F16 | Gate-side score projection |
/// | `ape` | `[comp_width × compress_ratio]` | F16 | Absolute position embed for compression |
/// | `norm` | `[N_HEAD_DIM]` = `[512]` | F32 | Post-pool RMSNorm scale |
///
/// where `comp_width = (compress_ratio == 4 ? 2 : 1) * N_HEAD_DIM` —
/// ratio-4 layers carry `comp_width = 1024`, ratio-128 layers carry
/// `comp_width = 512`.
///
/// **Algorithmic gap (F011 follow-up).** Loading these tensors lets
/// the GGUF parse succeed against a real model. The downstream
/// [`project_compressor_score`] helper, the F006
/// [`rsllm_kvcache::dsv4::compressed::CompressedKvPool`], and the
/// F008.C.2 attention compressor path STILL implement the old
/// per-token single-matmul shortcut — they do not yet model the
/// stateful per-position pooling, APE bias, gate sigmoid, or
/// post-pool RMSNorm. dsv4-vectors top-1 cannot pass until that
/// algorithmic rewrite (F011) lands. Until then, this struct is
/// loaded with the correct tensors but only `kv` is used.
#[derive(Debug, Clone, Copy)]
pub struct CompressorWeights<'a> {
    /// `[N_EMBD × comp_width]` F16. KV latent projection. Used by
    /// the legacy [`project_compressor_score`] path as the single
    /// "compressor matrix" until F011 lands.
    pub kv: WeightBlob<'a>,
    /// `[N_EMBD × comp_width]` F16. Gate-side score projection
    /// (combined with `kv` + APE bias in ds4's `compressor_decode_one`).
    /// Unused until F011.
    pub gate: WeightBlob<'a>,
    /// `[comp_width × compress_ratio]` F16. Absolute position embed
    /// added to `gate(x)` per ds4.c:6473-6475. Unused until F011.
    pub ape: WeightBlob<'a>,
    /// `[N_HEAD_DIM = 512]` F32. RMSNorm scale applied after the
    /// per-ratio pool reduction. Unused until F011.
    pub norm: &'a [f32],
}

/// Per-layer **write-side** indexer weights (present only on ratio-4
/// layers — even layers `il >= 2`). These produce the per-token KV
/// row and per-dim score that populate the indexer's internal
/// compressed pool ([`rsllm_kvcache::dsv4::indexer::IndexerPool`]).
///
/// Per token:
/// ```text
/// indexer_kv    = attn_indexer_kv · x         # [N_INDEXER_HEAD_DIM] = [128]
/// indexer_score = attn_indexer_kv_score · x   # [N_INDEXER_HEAD_DIM]
/// ```
#[derive(Debug, Clone, Copy)]
pub struct IndexerWriteWeights<'a> {
    /// `[N_INDEXER_HEAD_DIM × N_EMBD]` = `[128 × 4096]`.
    pub attn_indexer_kv: WeightBlob<'a>,
    /// `[N_INDEXER_HEAD_DIM × N_EMBD]` = `[128 × 4096]`.
    pub attn_indexer_kv_score: WeightBlob<'a>,
}

/// Per-layer **read-side** indexer weights. Used at attention time
/// to project the residual stream into the per-head indexer query
/// (`[N_INDEXER_HEAD × N_INDEXER_HEAD_DIM]` per token) and to scale
/// the per-head dot-product contribution.
#[derive(Debug, Clone, Copy)]
pub struct IndexerReadWeights<'a> {
    /// `[(N_INDEXER_HEAD * N_INDEXER_HEAD_DIM) × N_EMBD]`
    /// = `[(64 * 128) × 4096]` = `[8192 × 4096]`.
    pub attn_indexer_q: WeightBlob<'a>,
    /// Per-head scoring weights `[N_INDEXER_HEAD]` = `[64]`. ds4
    /// learns these alongside the indexer Q; they multiply each
    /// head's ReLU(Q·K_c) before the sum.
    pub attn_indexer_head_weight: &'a [f32],
}

/// Project the residual stream through the compressor LoRA, writing
/// one `HEAD_DIM`-wide score row per token.
///
/// Output buffer layout: `[n_tok × HEAD_DIM]` row-major.
///
/// # Errors
/// [`Error::ShapeMismatch`] if any of the input/output buffer lengths
/// disagree with the documented dimensions.
pub fn project_compressor_score(
    weights: &CompressorWeights<'_>,
    x: &[f32],
    out: &mut [f32],
    n_tok: usize,
    tier: SimdTier,
) -> Result<(), Error> {
    let in_total = checked_mul_or_err(n_tok, DSV4_N_EMBD, "compressor.x")?;
    let out_total = checked_mul_or_err(n_tok, DSV4_HEAD_DIM, "compressor.out")?;
    if x.len() != in_total {
        return Err(Error::ShapeMismatch {
            key: "compressor.x",
            expected: format!("{in_total}"),
            actual: format!("{}", x.len()),
        });
    }
    if out.len() != out_total {
        return Err(Error::ShapeMismatch {
            key: "compressor.out",
            expected: format!("{out_total}"),
            actual: format!("{}", out.len()),
        });
    }
    // F010.B: route through `kv` as the single-matrix proxy until
    // F011 lands the full ds4 stateful compressor (gate sigmoid + APE
    // bias + per-ratio pool + RMSNorm). The output shape is right
    // (`[head_dim]` per token) but it's emitted EVERY token rather
    // than every `compress_ratio` tokens, and without the gate/ape/
    // norm composition. dsv4-vectors top-1 cannot pass on this path.
    //
    // Note that for ratio-4 layers `comp_width = 1024`, so this
    // matmul actually produces a `[1024]` row rather than `[512]`.
    // The caller currently passes a `[HEAD_DIM = 512]` slice; for
    // ratio-4 layers we'd overrun. Until F011 wires the correct
    // per-regime width, callers must only pass this on ratio-128
    // layers (`coff = 1`, `comp_width = HEAD_DIM`). The attention
    // path's compressor branch needs the F011 rewrite to handle
    // ratio-4 correctly.
    matmul_weight_f32(
        out,
        &weights.kv,
        x,
        n_tok,
        DSV4_N_EMBD,
        DSV4_HEAD_DIM,
        tier,
    )
}

/// Project the residual stream through the write-side indexer LoRA
/// pair, writing one indexer-KV row and one indexer-score row per
/// token. Both outputs are `[n_tok × N_INDEXER_HEAD_DIM]` row-major.
///
/// # Errors
/// [`Error::ShapeMismatch`] on any buffer-length disagreement.
pub fn project_indexer_write(
    weights: &IndexerWriteWeights<'_>,
    x: &[f32],
    kv_out: &mut [f32],
    score_out: &mut [f32],
    n_tok: usize,
    tier: SimdTier,
) -> Result<(), Error> {
    let expected_in = checked_mul_or_err(n_tok, DSV4_N_EMBD, "indexer_write.x")?;
    let expected_out =
        checked_mul_or_err(n_tok, DSV4_N_INDEXER_HEAD_DIM, "indexer_write.out")?;
    if x.len() != expected_in {
        return Err(Error::ShapeMismatch {
            key: "indexer_write.x",
            expected: format!("{expected_in}"),
            actual: format!("{}", x.len()),
        });
    }
    if kv_out.len() != expected_out {
        return Err(Error::ShapeMismatch {
            key: "indexer_write.kv_out",
            expected: format!("{expected_out}"),
            actual: format!("{}", kv_out.len()),
        });
    }
    if score_out.len() != expected_out {
        return Err(Error::ShapeMismatch {
            key: "indexer_write.score_out",
            expected: format!("{expected_out}"),
            actual: format!("{}", score_out.len()),
        });
    }
    matmul_weight_f32(
        kv_out,
        &weights.attn_indexer_kv,
        x,
        n_tok,
        DSV4_N_EMBD,
        DSV4_N_INDEXER_HEAD_DIM,
        tier,
    )?;
    matmul_weight_f32(
        score_out,
        &weights.attn_indexer_kv_score,
        x,
        n_tok,
        DSV4_N_EMBD,
        DSV4_N_INDEXER_HEAD_DIM,
        tier,
    )
}

/// Project the residual stream through the read-side indexer LoRA
/// to produce the per-token, per-head indexer query.
///
/// Output layout: `[n_tok × (N_INDEXER_HEAD * N_INDEXER_HEAD_DIM)]`
/// row-major — same convention as `out_dim = N_INDEXER_HEAD * N_INDEXER_HEAD_DIM = 8192`.
///
/// # Errors
/// [`Error::ShapeMismatch`] on any buffer-length disagreement.
pub fn project_indexer_query(
    weights: &IndexerReadWeights<'_>,
    x: &[f32],
    q_out: &mut [f32],
    n_tok: usize,
    tier: SimdTier,
) -> Result<(), Error> {
    let q_lanes = DSV4_N_INDEXER_HEAD * DSV4_N_INDEXER_HEAD_DIM;
    let expected_in = checked_mul_or_err(n_tok, DSV4_N_EMBD, "indexer_query.x")?;
    let expected_q = checked_mul_or_err(n_tok, q_lanes, "indexer_query.q_out")?;
    if x.len() != expected_in {
        return Err(Error::ShapeMismatch {
            key: "indexer_query.x",
            expected: format!("{expected_in}"),
            actual: format!("{}", x.len()),
        });
    }
    if q_out.len() != expected_q {
        return Err(Error::ShapeMismatch {
            key: "indexer_query.q_out",
            expected: format!("{expected_q}"),
            actual: format!("{}", q_out.len()),
        });
    }
    if weights.attn_indexer_head_weight.len() != DSV4_N_INDEXER_HEAD {
        return Err(Error::ShapeMismatch {
            key: "indexer_query.head_weight",
            expected: format!("{DSV4_N_INDEXER_HEAD}"),
            actual: format!("{}", weights.attn_indexer_head_weight.len()),
        });
    }
    matmul_weight_f32(
        q_out,
        &weights.attn_indexer_q,
        x,
        n_tok,
        DSV4_N_EMBD,
        q_lanes,
        tier,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a weight that copies a deterministic slice of `x` into
    /// each output lane. Row `o` has a single 1.0 at column
    /// `o % in_dim` — i.e. `out[t, o] == x[t, o % in_dim]`. For
    /// `out_dim <= in_dim` this is identity-by-truncation; for
    /// `out_dim > in_dim` it wraps, so the test asserts on lanes that
    /// don't overlap the wrap (`o < in_dim`).
    fn truncating_weight(out_dim: usize, in_dim: usize) -> Vec<f32> {
        let mut w = vec![0.0_f32; out_dim * in_dim];
        for o in 0..out_dim {
            w[o * in_dim + (o % in_dim)] = 1.0;
        }
        w
    }

    #[test]
    fn compressor_rejects_wrong_x() {
        let w = vec![0.0_f32; DSV4_HEAD_DIM * DSV4_N_EMBD];
        // F010.B: the test exercises `project_compressor_score`, which
        // currently routes through `kv` as a single-matrix proxy. The
        // other three tensors are inert; supply zero/one placeholders.
        let zero_gate = vec![0.0_f32; w.len()];
        let zero_ape = vec![0.0_f32; DSV4_HEAD_DIM * 128];
        let ones_norm = vec![1.0_f32; DSV4_HEAD_DIM];
        let weights = CompressorWeights {
            kv: WeightBlob::F32(&w),
            gate: WeightBlob::F32(&zero_gate),
            ape: WeightBlob::F32(&zero_ape),
            norm: &ones_norm,
        };
        let x = vec![0.0_f32; DSV4_N_EMBD - 1]; // wrong
        let mut out = vec![0.0_f32; DSV4_HEAD_DIM];
        let err = project_compressor_score(&weights, &x, &mut out, 1, SimdTier::Scalar)
            .unwrap_err();
        assert!(matches!(err, Error::ShapeMismatch { .. }));
    }

    #[test]
    fn compressor_passes_through_truncating_weight() {
        let w = truncating_weight(DSV4_HEAD_DIM, DSV4_N_EMBD);
        // F010.B: the test exercises `project_compressor_score`, which
        // currently routes through `kv` as a single-matrix proxy. The
        // other three tensors are inert; supply zero/one placeholders.
        let zero_gate = vec![0.0_f32; w.len()];
        let zero_ape = vec![0.0_f32; DSV4_HEAD_DIM * 128];
        let ones_norm = vec![1.0_f32; DSV4_HEAD_DIM];
        let weights = CompressorWeights {
            kv: WeightBlob::F32(&w),
            gate: WeightBlob::F32(&zero_gate),
            ape: WeightBlob::F32(&zero_ape),
            norm: &ones_norm,
        };
        let mut x = vec![0.0_f32; DSV4_N_EMBD];
        for (i, v) in x.iter_mut().enumerate().take(DSV4_HEAD_DIM) {
            *v = (i as f32) + 1.0;
        }
        let mut out = vec![0.0_f32; DSV4_HEAD_DIM];
        project_compressor_score(&weights, &x, &mut out, 1, SimdTier::Scalar).unwrap();
        for (i, &v) in out.iter().enumerate() {
            assert!((v - ((i as f32) + 1.0)).abs() < 1e-5, "mismatch at {i}");
        }
    }

    #[test]
    fn indexer_write_produces_kv_and_score() {
        let w_kv = truncating_weight(DSV4_N_INDEXER_HEAD_DIM, DSV4_N_EMBD);
        let w_score = truncating_weight(DSV4_N_INDEXER_HEAD_DIM, DSV4_N_EMBD);
        let weights = IndexerWriteWeights {
            attn_indexer_kv: WeightBlob::F32(&w_kv),
            attn_indexer_kv_score: WeightBlob::F32(&w_score),
        };
        let mut x = vec![0.0_f32; 2 * DSV4_N_EMBD];
        // Token 0 has lane 0 = 7.0; token 1 has lane 1 = 9.0.
        x[0] = 7.0;
        x[DSV4_N_EMBD + 1] = 9.0;
        let mut kv_out = vec![0.0_f32; 2 * DSV4_N_INDEXER_HEAD_DIM];
        let mut score_out = vec![0.0_f32; 2 * DSV4_N_INDEXER_HEAD_DIM];
        project_indexer_write(
            &weights,
            &x,
            &mut kv_out,
            &mut score_out,
            2,
            SimdTier::Scalar,
        )
        .unwrap();
        assert!((kv_out[0] - 7.0).abs() < 1e-5);
        assert!((score_out[0] - 7.0).abs() < 1e-5);
        assert!((kv_out[DSV4_N_INDEXER_HEAD_DIM + 1] - 9.0).abs() < 1e-5);
        assert!((score_out[DSV4_N_INDEXER_HEAD_DIM + 1] - 9.0).abs() < 1e-5);
    }

    #[test]
    fn indexer_write_rejects_mismatched_out() {
        let w = vec![0.0_f32; DSV4_N_INDEXER_HEAD_DIM * DSV4_N_EMBD];
        let weights = IndexerWriteWeights {
            attn_indexer_kv: WeightBlob::F32(&w),
            attn_indexer_kv_score: WeightBlob::F32(&w),
        };
        let x = vec![0.0_f32; DSV4_N_EMBD];
        let mut kv_out = vec![0.0_f32; DSV4_N_INDEXER_HEAD_DIM - 1];
        let mut score_out = vec![0.0_f32; DSV4_N_INDEXER_HEAD_DIM];
        let err = project_indexer_write(
            &weights,
            &x,
            &mut kv_out,
            &mut score_out,
            1,
            SimdTier::Scalar,
        )
        .unwrap_err();
        assert!(matches!(err, Error::ShapeMismatch { .. }));
    }

    #[test]
    fn indexer_query_projects_per_head_q() {
        let q_lanes = DSV4_N_INDEXER_HEAD * DSV4_N_INDEXER_HEAD_DIM;
        let w_q = truncating_weight(q_lanes, DSV4_N_EMBD);
        let head_weights = vec![1.0_f32; DSV4_N_INDEXER_HEAD];
        let weights = IndexerReadWeights {
            attn_indexer_q: WeightBlob::F32(&w_q),
            attn_indexer_head_weight: &head_weights,
        };
        let mut x = vec![0.0_f32; DSV4_N_EMBD];
        x[5] = 3.5;
        let mut q_out = vec![0.0_f32; q_lanes];
        project_indexer_query(&weights, &x, &mut q_out, 1, SimdTier::Scalar).unwrap();
        // Lane 5 of the output should carry the input value because
        // truncating_weight is identity on the first `out_dim` rows.
        assert!((q_out[5] - 3.5).abs() < 1e-5);
    }

    #[test]
    fn compressor_threads_multiple_tokens() {
        // Exercises the per-token row stride — a 1-token test would
        // pass even if the matmul forgot to advance the output row.
        let w = truncating_weight(DSV4_HEAD_DIM, DSV4_N_EMBD);
        // F010.B: the test exercises `project_compressor_score`, which
        // currently routes through `kv` as a single-matrix proxy. The
        // other three tensors are inert; supply zero/one placeholders.
        let zero_gate = vec![0.0_f32; w.len()];
        let zero_ape = vec![0.0_f32; DSV4_HEAD_DIM * 128];
        let ones_norm = vec![1.0_f32; DSV4_HEAD_DIM];
        let weights = CompressorWeights {
            kv: WeightBlob::F32(&w),
            gate: WeightBlob::F32(&zero_gate),
            ape: WeightBlob::F32(&zero_ape),
            norm: &ones_norm,
        };
        let mut x = vec![0.0_f32; 3 * DSV4_N_EMBD];
        // Token t puts (t+1)*10 at lane t.
        for t in 0..3 {
            x[t * DSV4_N_EMBD + t] = ((t as f32) + 1.0) * 10.0;
        }
        let mut out = vec![0.0_f32; 3 * DSV4_HEAD_DIM];
        project_compressor_score(&weights, &x, &mut out, 3, SimdTier::Scalar).unwrap();
        for t in 0..3 {
            let v = out[t * DSV4_HEAD_DIM + t];
            assert!(
                (v - ((t as f32) + 1.0) * 10.0).abs() < 1e-5,
                "token {t} lane {t} = {v}"
            );
        }
    }

    #[test]
    fn indexer_query_per_head_stride_is_correct() {
        // Sanity-pin the per-head stride convention F008.C relies on:
        // head h's Q vector lives at `q_out[h * N_INDEXER_HEAD_DIM ..]`.
        // truncating_weight wraps after `in_dim = N_EMBD`, so rows
        // 0..4096 are identity-on-x and rows >= 4096 wrap. We set
        // lane 0 of x; lanes 0 of every head row should pick it up.
        let q_lanes = DSV4_N_INDEXER_HEAD * DSV4_N_INDEXER_HEAD_DIM;
        let w_q = truncating_weight(q_lanes, DSV4_N_EMBD);
        let head_weights = vec![1.0_f32; DSV4_N_INDEXER_HEAD];
        let weights = IndexerReadWeights {
            attn_indexer_q: WeightBlob::F32(&w_q),
            attn_indexer_head_weight: &head_weights,
        };
        let mut x = vec![0.0_f32; DSV4_N_EMBD];
        x[0] = 11.0;
        let mut q_out = vec![0.0_f32; q_lanes];
        project_indexer_query(&weights, &x, &mut q_out, 1, SimdTier::Scalar).unwrap();
        // Head 0 lane 0 = lane 0 of x = 11.0.
        assert!((q_out[0] - 11.0).abs() < 1e-5);
        // Head 1 lane 0 lives at q_out[128]; truncating_weight has a
        // 1.0 at row=128, col=128 — but x[128] = 0.0. So q_out[128] = 0.
        assert!(q_out[DSV4_N_INDEXER_HEAD_DIM].abs() < 1e-5);
    }

    #[test]
    fn indexer_write_rejects_mismatched_score_out() {
        // Symmetric of indexer_write_rejects_mismatched_out — verify
        // the second guard fires even when kv_out is correct.
        let w = vec![0.0_f32; DSV4_N_INDEXER_HEAD_DIM * DSV4_N_EMBD];
        let weights = IndexerWriteWeights {
            attn_indexer_kv: WeightBlob::F32(&w),
            attn_indexer_kv_score: WeightBlob::F32(&w),
        };
        let x = vec![0.0_f32; DSV4_N_EMBD];
        let mut kv_out = vec![0.0_f32; DSV4_N_INDEXER_HEAD_DIM];
        let mut score_out = vec![0.0_f32; DSV4_N_INDEXER_HEAD_DIM - 1];
        let err = project_indexer_write(
            &weights,
            &x,
            &mut kv_out,
            &mut score_out,
            1,
            SimdTier::Scalar,
        )
        .unwrap_err();
        assert!(matches!(err, Error::ShapeMismatch { key, .. } if key == "indexer_write.score_out"));
    }

    #[test]
    fn check_shape_rejects_wrong_byte_len() {
        // Exercises WeightBlob::check_shape directly — undersized
        // F32 storage must error out at load time.
        let w_short = vec![0.0_f32; DSV4_HEAD_DIM * DSV4_N_EMBD - 1];
        let blob = WeightBlob::F32(&w_short);
        let err = blob
            .check_shape(DSV4_HEAD_DIM, DSV4_N_EMBD, "test.compressor")
            .unwrap_err();
        assert!(matches!(err, Error::ShapeMismatch { .. }));
    }

    #[test]
    fn indexer_query_rejects_wrong_head_weight_len() {
        let q_lanes = DSV4_N_INDEXER_HEAD * DSV4_N_INDEXER_HEAD_DIM;
        let w_q = vec![0.0_f32; q_lanes * DSV4_N_EMBD];
        let head_weights = vec![1.0_f32; DSV4_N_INDEXER_HEAD - 1]; // wrong
        let weights = IndexerReadWeights {
            attn_indexer_q: WeightBlob::F32(&w_q),
            attn_indexer_head_weight: &head_weights,
        };
        let x = vec![0.0_f32; DSV4_N_EMBD];
        let mut q_out = vec![0.0_f32; q_lanes];
        let err = project_indexer_query(&weights, &x, &mut q_out, 1, SimdTier::Scalar)
            .unwrap_err();
        assert!(matches!(err, Error::ShapeMismatch { .. }));
    }
}
