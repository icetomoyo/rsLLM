//! Per-layer LoRA weights + projection functions for the
//! compressed-KV scoring (`attn_compressor`) and the ratio-4 indexer
//! (`indexer.*` / `indexer_compressor_*`).
//!
//! `CompressorWeights` produces the per-token *score* and *latent*
//! inputs used by the compressor pool.  `IndexerWeights` holds the
//! six-tensor bundle for ratio-4 layers aligned to ds4 upstream
//! (`ds4.c:2326-2331` for shapes, `ds4.c:2610-2615` for load).
//!
//! **Algorithmic gap (F011 follow-up).** The indexer algorithm —
//! `project_compressor_score` equivalent for `IndexerWeights` — is
//! intentionally absent here. F011 will add it once the stateful
//! per-position pooling, gate sigmoid, APE bias, and RMSNorm are
//! properly modelled. Until then `IndexerWeights` is loaded with the
//! correct tensors but the downstream adapter in `attention.rs` keeps
//! the F006 zero-placeholder path.
//!
//! ds4 anchors:
//! - `attn_compressor` family — layer-weight struct at `ds4.c:2306+`.
//! - `indexer` family — shapes at `ds4.c:2326-2331`, load at
//!   `ds4.c:2610-2615`.
//!
//! Ported by reference from `ds4.c` (MIT, The ds4.c authors).
//! Line numbers pinned to ds4 commit `ef0a490` (2026-05-17).

use rsllm_backend_cpu::SimdTier;

use super::shape::{DSV4_HEAD_DIM, DSV4_N_EMBD};
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

/// Per-layer indexer weights — the six-tensor bundle present on every
/// ratio-4 layer (even `il >= 2`, per
/// [`rsllm_kvcache::dsv4::shape::layer_has_indexer`]).
///
/// Shape table (ds4.c:2326-2331, load at ds4.c:2610-2615):
///
/// | Field | GGUF key suffix | Shape | dtype |
/// |---|---|---|---|
/// | `attn_q_b` | `indexer.attn_q_b.weight` | `[N_LORA_Q × (N_INDEXER_HEAD × N_INDEXER_HEAD_DIM)]` = `[1024 × 8192]` | F16 |
/// | `proj` | `indexer.proj.weight` | `[N_EMBD × N_INDEXER_HEAD]` = `[4096 × 64]` | F16 |
/// | `comp_ape` | `indexer_compressor_ape.weight` | `[index_width × 4]` where `index_width = 2 × N_INDEXER_HEAD_DIM = 256` | F16 |
/// | `comp_kv` | `indexer_compressor_kv.weight` | `[N_EMBD × index_width]` = `[4096 × 256]` | F16 |
/// | `comp_gate` | `indexer_compressor_gate.weight` | `[N_EMBD × index_width]` = `[4096 × 256]` | F16 |
/// | `comp_norm` | `indexer_compressor_norm.weight` | `[N_INDEXER_HEAD_DIM]` = `[128]` | F32 |
///
/// The last four tensors (`comp_*`) are a parallel structure to the
/// main `CompressorWeights` family but with `index_width = 256`
/// (`2 × N_INDEXER_HEAD_DIM`) instead of `comp_width = 1024`
/// (`2 × N_HEAD_DIM` for ratio-4).
///
/// **Algorithm deferred to F011.** The projection function (analogous
/// to `project_compressor_score` for the compressor path) is not
/// defined here — F011 will add it once the stateful per-position
/// pooling, gate sigmoid, APE bias, and RMSNorm are modelled. Until
/// then `attention.rs` keeps the zero-placeholder path for the indexer
/// tier.
#[derive(Debug, Clone, Copy)]
pub struct IndexerWeights<'a> {
    /// `[N_LORA_Q × (N_INDEXER_HEAD × N_INDEXER_HEAD_DIM)]`
    /// = `[1024 × 8192]` F16. Indexer attention Q_b projection
    /// (analogous to MLA `attn_q_b`). ds4.c:2326.
    pub attn_q_b: WeightBlob<'a>,
    /// `[N_EMBD × N_INDEXER_HEAD]` = `[4096 × 64]` F16. Indexer
    /// output projection. ds4.c:2327.
    pub proj: WeightBlob<'a>,
    /// `[index_width × 4]` where `index_width = 2 × N_INDEXER_HEAD_DIM
    /// = 256`. F16. APE bias for the indexer compressor sub-pool.
    /// ds4.c:2328.
    pub comp_ape: WeightBlob<'a>,
    /// `[N_EMBD × index_width]` = `[4096 × 256]` F16. KV-side
    /// latent projection for the indexer compressor. ds4.c:2329.
    pub comp_kv: WeightBlob<'a>,
    /// `[N_EMBD × index_width]` = `[4096 × 256]` F16. Gate-side
    /// projection for the indexer compressor. ds4.c:2330.
    pub comp_gate: WeightBlob<'a>,
    /// `[N_INDEXER_HEAD_DIM]` = `[128]` F32. RMSNorm scale applied
    /// after the indexer compressor pool reduction. ds4.c:2331.
    pub comp_norm: &'a [f32],
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

// NOTE(F011): project_indexer_* functions are intentionally absent.
// F011 will add the proper stateful indexer pipeline once the full
// algorithm (per-position APE bias, gate sigmoid, pool reduction,
// RMSNorm) is modelled. `attention.rs` keeps zero-placeholder scores
// for the indexer tier until then.

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::shape::{DSV4_N_INDEXER_HEAD, DSV4_N_INDEXER_HEAD_DIM, DSV4_N_LORA_Q};

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

    /// Verify `IndexerWeights` is constructible with the correct
    /// upstream shapes (ds4.c:2326-2331). This is a struct
    /// construction test — no algorithm runs (that is F011's job).
    #[test]
    fn indexer_weights_struct_accepts_correct_shapes() {
        // index_width = 2 * N_INDEXER_HEAD_DIM = 256
        let index_width = 2 * DSV4_N_INDEXER_HEAD_DIM;
        let ratio = 4usize;

        let attn_q_b_w =
            vec![0.0_f32; DSV4_N_LORA_Q * DSV4_N_INDEXER_HEAD * DSV4_N_INDEXER_HEAD_DIM];
        let proj_w = vec![0.0_f32; DSV4_N_EMBD * DSV4_N_INDEXER_HEAD];
        let comp_ape_w = vec![0.0_f32; index_width * ratio];
        let comp_kv_w = vec![0.0_f32; DSV4_N_EMBD * index_width];
        let comp_gate_w = vec![0.0_f32; DSV4_N_EMBD * index_width];
        let comp_norm_w = vec![1.0_f32; DSV4_N_INDEXER_HEAD_DIM];

        let iw = IndexerWeights {
            attn_q_b: WeightBlob::F32(&attn_q_b_w),
            proj: WeightBlob::F32(&proj_w),
            comp_ape: WeightBlob::F32(&comp_ape_w),
            comp_kv: WeightBlob::F32(&comp_kv_w),
            comp_gate: WeightBlob::F32(&comp_gate_w),
            comp_norm: &comp_norm_w,
        };
        // Verify expected byte counts match the shape specification.
        assert_eq!(
            iw.attn_q_b.byte_len(),
            DSV4_N_LORA_Q * DSV4_N_INDEXER_HEAD * DSV4_N_INDEXER_HEAD_DIM * 4
        );
        assert_eq!(iw.proj.byte_len(), DSV4_N_EMBD * DSV4_N_INDEXER_HEAD * 4);
        assert_eq!(iw.comp_ape.byte_len(), index_width * ratio * 4);
        assert_eq!(iw.comp_kv.byte_len(), DSV4_N_EMBD * index_width * 4);
        assert_eq!(iw.comp_gate.byte_len(), DSV4_N_EMBD * index_width * 4);
        assert_eq!(iw.comp_norm.len(), DSV4_N_INDEXER_HEAD_DIM);
    }
}
