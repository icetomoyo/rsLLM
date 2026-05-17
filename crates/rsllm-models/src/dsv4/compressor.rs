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

/// Per-layer weights for the compressed-KV scoring path. Present on
/// every layer with `compress_ratio > 0` (i.e. all but the first
/// two dense layers, per
/// [`rsllm_kvcache::dsv4::shape::layer_compress_ratio`]).
///
/// Logical operation per token:
/// ```text
/// compress_score = attn_compressor · x   # [HEAD_DIM] = [HEAD_DIM × N_EMBD] · [N_EMBD]
/// ```
/// The result feeds [`rsllm_kvcache::dsv4::compressed::CompressedKvPool::accumulate`]
/// as the per-dim softmax score that aggregates KV latents at each
/// `compress_ratio` boundary.
#[derive(Debug, Clone, Copy)]
pub struct CompressorWeights<'a> {
    /// `[HEAD_DIM × N_EMBD]` = `[512 × 4096]`.
    pub attn_compressor: WeightBlob<'a>,
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
    if x.len() != n_tok * DSV4_N_EMBD {
        return Err(Error::ShapeMismatch {
            key: "compressor.x",
            expected: format!("{}", n_tok * DSV4_N_EMBD),
            actual: format!("{}", x.len()),
        });
    }
    if out.len() != n_tok * DSV4_HEAD_DIM {
        return Err(Error::ShapeMismatch {
            key: "compressor.out",
            expected: format!("{}", n_tok * DSV4_HEAD_DIM),
            actual: format!("{}", out.len()),
        });
    }
    matmul_weight_f32(
        out,
        &weights.attn_compressor,
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
    let expected_in = n_tok * DSV4_N_EMBD;
    let expected_out = n_tok * DSV4_N_INDEXER_HEAD_DIM;
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
    if x.len() != n_tok * DSV4_N_EMBD {
        return Err(Error::ShapeMismatch {
            key: "indexer_query.x",
            expected: format!("{}", n_tok * DSV4_N_EMBD),
            actual: format!("{}", x.len()),
        });
    }
    if q_out.len() != n_tok * q_lanes {
        return Err(Error::ShapeMismatch {
            key: "indexer_query.q_out",
            expected: format!("{}", n_tok * q_lanes),
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
        let weights = CompressorWeights {
            attn_compressor: WeightBlob::F32(&w),
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
        let weights = CompressorWeights {
            attn_compressor: WeightBlob::F32(&w),
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
