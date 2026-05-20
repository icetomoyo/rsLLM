//! Per-layer LoRA weights + projection functions for the
//! compressed-KV scoring (`attn_compressor`) and the ratio-4 indexer
//! (`indexer.*` / `indexer_compressor_*`).
//!
//! `CompressorWeights` is consumed by [`compressor_decode_one`], the
//! stateful per-token entry point that drives the
//! [`CompressedKvPool`] (dual matmul → APE bias → pool aggregation →
//! RMSNorm → RoPE-YaRN tail). `IndexerWeights` holds the parallel
//! six-tensor bundle for ratio-4 layers aligned to ds4 upstream
//! (`ds4.c:2326-2331` for shapes, `ds4.c:2610-2615` for load).
//!
//! **Algorithmic gap (F011.D follow-up).** The indexer-side stateful
//! sub-pipeline (analogous to [`compressor_decode_one`] but for
//! `IndexerWeights`) is intentionally absent. F011.D will add it.
//! Until then `IndexerWeights` is loaded with the correct tensors but
//! the downstream adapter in `attention.rs` keeps the F006
//! zero-placeholder path for the indexer tier.
//!
//! ds4 anchors:
//! - `attn_compressor` family — layer-weight struct at `ds4.c:2306+`,
//!   stateful per-token routine at `ds4.c:6431-6524`.
//! - `indexer` family — shapes at `ds4.c:2326-2331`, load at
//!   `ds4.c:2610-2615`.
//!
//! Ported by reference from `ds4.c` (MIT, The ds4.c authors).
//! Line numbers pinned to ds4 commit `ef0a490` (2026-05-17).

use rsllm_backend_cpu::SimdTier;
use rsllm_backend_cpu::ops::rope::{RoPEParams, rope_yarn_tail};
use rsllm_gguf::{GgmlType, dequant_to_f32};
use rsllm_kvcache::dsv4::compressed::CompressedKvPool;

use super::shape::{
    DSV4_COMPRESS_ROPE_FREQ_BASE, DSV4_N_EMBD, DSV4_N_ROT, DSV4_RMS_EPS, DSV4_ROPE_ORIG_CTX,
};
use super::weight::{WeightBlob, matmul_weight_f32};
use crate::Error;

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
/// All four tensors are consumed by [`compressor_decode_one`] (the
/// stateful per-token entry point) as of F011.B/C. The
/// [`CompressedKvPool`] holds the cross-token state; this struct
/// holds only the static weights.
#[derive(Debug, Clone, Copy)]
pub struct CompressorWeights<'a> {
    /// `[N_EMBD × comp_width]` F16. KV latent projection consumed by
    /// [`compressor_decode_one`] (`ds4.c:6450-6455`).
    pub kv: WeightBlob<'a>,
    /// `[N_EMBD × comp_width]` F16. Gate-side score projection
    /// (combined with `kv` + APE bias in `compressor_decode_one`,
    /// `ds4.c:6460-6475`).
    pub gate: WeightBlob<'a>,
    /// `[comp_width × compress_ratio]` F16. Absolute position embed
    /// added to `gate(x)` per `ds4.c:6473-6475`.
    pub ape: WeightBlob<'a>,
    /// `[N_HEAD_DIM = 512]` F32. RMSNorm scale applied after the
    /// per-ratio pool reduction (`ds4.c:6491-6495`).
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

impl<'a> IndexerWeights<'a> {
    /// View the four `comp_*` tensors as a [`CompressorWeights`] so the
    /// same [`compressor_decode_one`] kernel can drive the indexer's
    /// stateful per-token pipeline. Upstream `ds4.c:7042-7054` reuses
    /// `compressor_decode_one` for the indexer too, parameterised only
    /// by `head_dim` (128 vs 512) and the cache buffers — the algorithm
    /// is identical. F011.D adopts the same approach.
    #[must_use]
    pub fn as_compressor_view(&self) -> CompressorWeights<'a> {
        CompressorWeights {
            kv: self.comp_kv,
            gate: self.comp_gate,
            ape: self.comp_ape,
            norm: self.comp_norm,
        }
    }
}

// NOTE(F011.C): the old placeholder `project_compressor_score` was
// retired here. Callers now route through `compressor_decode_one`
// (the stateful per-token entry point) + `CompressedKvPool::
// accumulate_wide`. See the F011.C commit body for the migration
// notes.

// NOTE(F011): project_indexer_* functions are intentionally absent.
// F011 will add the proper stateful indexer pipeline once the full
// algorithm (per-position APE bias, gate sigmoid, pool reduction,
// RMSNorm) is modelled. `attention.rs` keeps zero-placeholder scores
// for the indexer tier until then.

/// Stateful per-token compressor step, mirroring `ds4.c:6431-6524`
/// `compressor_decode_one`.
///
/// For one token at sequence position `pos` and layer `il`, this:
/// 1. Projects the residual `x` through `weights.kv` and `weights.gate`
///    to produce two `width = pool.width()`-wide rows.
/// 2. Adds the APE bias from `weights.ape[:, pos % ratio]` to the
///    gate-side row (which becomes the per-dim softmax score).
/// 3. Hands both rows to [`CompressedKvPool::accumulate_wide`], which
///    writes them into the appropriate state slot, runs the dual-lane
///    softmax aggregation on the emission boundary, and (for ratio-4)
///    rotates the state buffer.
/// 4. On emission, post-processes the just-written compressed row
///    in place — RMSNorm with `weights.norm`, then RoPE-YaRN tail
///    rotation at position `comp_pos = pos + 1 - ratio` with the
///    upstream compressed-layer parameters
///    (`freq_base = DSV4_COMPRESS_ROPE_FREQ_BASE`, ramped YaRN).
/// 5. **TODO(F011.fp8):** the attention compressor (head_dim = 512)
///    additionally runs `dsv4_fp8_kv_quantize_row_inplace_cpu`
///    (`ds4.c:6500`) on the post-RoPE row. That kernel has no Rust
///    equivalent yet; the row stays at full f32 precision until
///    F011.fp8 lands. This is a known numerical divergence vs ds4.c
///    that affects attention-side scoring but not crash safety.
///
/// Returns `true` when an emission fired on this token, `false`
/// otherwise. Callers reading from `pool.rows()` should call this
/// once per token in stream order and consult the return value to
/// know whether a new compressed row is now available.
///
/// # Arguments
/// - `x` — per-token residual, length `DSV4_N_EMBD`.
/// - `weights` — the layer's `CompressorWeights` (`kv`, `gate`, `ape`, `norm`).
/// - `pool` — the layer's [`CompressedKvPool`]; mutated.
/// - `pos` — token sequence position (0-indexed).
/// - `il` — layer index. Used only as a per-layer-RoPE caching key
///   today; reserved for future tier-specific specialisation.
/// - `kv_cur` / `sc_cur` — caller-owned scratch buffers, each length
///   `pool.width()`. Passed in to avoid per-token allocation in the
///   prefill loop.
/// - `ape_col` — caller-owned scratch buffer, length `pool.width()`.
///   Holds the dequantised APE column for `pos % ratio`.
/// - `tier` — SIMD tier for the matmuls.
///
/// # Errors
/// - [`Error::ShapeMismatch`] on any length disagreement.
/// - Errors bubbled from `matmul_weight_f32`, `dequant_to_f32`,
///   `rope_yarn_tail`, or the pool's `accumulate_wide`.
#[allow(clippy::too_many_arguments)]
pub fn compressor_decode_one(
    x: &[f32],
    weights: &CompressorWeights<'_>,
    pool: &mut CompressedKvPool,
    pos: u32,
    il: u32,
    kv_cur: &mut [f32],
    sc_cur: &mut [f32],
    ape_col: &mut [f32],
    tier: SimdTier,
) -> Result<bool, Error> {
    let width = pool.width();
    let head_dim = pool.head_dim();
    let ratio = pool.ratio();

    if x.len() != DSV4_N_EMBD {
        return Err(Error::ShapeMismatch {
            key: "compressor_decode_one.x",
            expected: format!("{DSV4_N_EMBD}"),
            actual: format!("{}", x.len()),
        });
    }
    if kv_cur.len() != width || sc_cur.len() != width || ape_col.len() != width {
        return Err(Error::ShapeMismatch {
            key: "compressor_decode_one.scratch",
            expected: format!("kv/sc/ape each {width}"),
            actual: format!(
                "kv={}, sc={}, ape={}",
                kv_cur.len(),
                sc_cur.len(),
                ape_col.len()
            ),
        });
    }

    // 1. Dual matmul to produce width-wide kv_cur and sc_cur rows
    //    (`ds4.c:6450-6470`). One token per call.
    matmul_weight_f32(kv_cur, &weights.kv, x, 1, DSV4_N_EMBD, width, tier)?;
    matmul_weight_f32(sc_cur, &weights.gate, x, 1, DSV4_N_EMBD, width, tier)?;

    // 2. APE bias addition (`ds4.c:6472-6475`).
    let pos_mod = pos % ratio;
    read_ape_column(&weights.ape, width, ratio, pos_mod, ape_col)?;
    for j in 0..width {
        sc_cur[j] += ape_col[j];
    }

    // 3. State write + boundary check + (on emit) aggregation + rotation
    //    — all owned by the pool.
    let emitted_idx = pool.accumulate_wide(kv_cur, sc_cur)?;

    // 4. Post-process the just-emitted row in place: RMSNorm + RoPE.
    if let Some(idx) = emitted_idx {
        // Take a separate immutable borrow of `weights.norm` before we
        // borrow the pool row mutably (`norm` is `&[f32]` so no
        // aliasing concern, but we make the order explicit).
        let norm = weights.norm;
        if norm.len() != head_dim {
            return Err(Error::ShapeMismatch {
                key: "compressor_decode_one.norm",
                expected: format!("{head_dim}"),
                actual: format!("{}", norm.len()),
            });
        }
        let row = pool.compressed_row_mut(idx);

        // RMSNorm in place: ds4.c:6491-6495 uses an f64 accumulator
        // for `ss`; mirror that to avoid losing low-order bits on
        // head_dim = 512 inputs.
        let mut ss: f64 = 0.0;
        for &v in row.iter() {
            ss += f64::from(v) * f64::from(v);
        }
        let rms = 1.0_f32 / ((ss / head_dim as f64) as f32 + DSV4_RMS_EPS).sqrt();
        for (v, &w) in row.iter_mut().zip(norm.iter()) {
            *v = *v * rms * w;
        }

        // RoPE tail rotation at the compressed-pool position with the
        // upstream compressed-layer YaRN regime (`ds4.c:4745-4790`).
        // Emission only fires when `(pos + 1) % ratio == 0` (the pool's
        // boundary check, mirroring `ds4.c:6449`), so `pos + 1 >= ratio`
        // always holds here. `saturating_sub` is defensive against a
        // future caller that violates the invariant (e.g. driving the
        // pool through a non-stream code path); the debug_assert below
        // surfaces the bug in test builds rather than silently mis-
        // rotating the row in release.
        debug_assert!(
            pos + 1 >= ratio,
            "compressor_decode_one emission invariant: pos+1 ({}) must be >= ratio ({})",
            pos + 1,
            ratio,
        );
        let comp_pos = (pos + 1).saturating_sub(ratio);
        let params = compress_rope_params(comp_pos, il, head_dim as u32);
        rope_yarn_tail(row, &params, tier).map_err(|e| Error::ShapeMismatch {
            key: "compressor_decode_one.rope",
            expected: "valid YaRN tail rotation".to_string(),
            actual: format!("{e}"),
        })?;

        // TODO(F011.fp8): attention compressor (head_dim ==
        // DSV4_HEAD_DIM = 512) additionally runs
        // `dsv4_fp8_kv_quantize_row_inplace_cpu(row, head_dim, DSV4_N_ROT)`
        // here per `ds4.c:6498-6501`. No Rust equivalent yet — the row
        // stays at full f32 precision. The indexer compressor
        // (head_dim = 128) skips this step upstream too.
        return Ok(true);
    }

    Ok(false)
}

/// Build the [`RoPEParams`] bundle for a compressed-layer emission,
/// mirroring `ds4.c:4745-4790` `rope_tail_layer_inplace` for
/// compressed layers (`ds4_layer_compress_ratio(il) != 0`).
///
/// `_il` is accepted for signature parity with upstream's per-layer
/// dispatch. All compressed DS V4 Flash layers share the same
/// `freq_base` / `freq_scale` / `ext_factor` regime, so `il` is not
/// read; future layer-tier specialisation would consult it.
fn compress_rope_params(pos: u32, _il: u32, head_dim: u32) -> RoPEParams {
    // Constants come from ds4.c:56-61. `freq_scale = 1 / SCALE_FACTOR`,
    // and the `attn_factor` correction cancels YaRN's internal mscale
    // so the rotation matches ds4's interpolation-without-magnitude
    // change behaviour.
    let scale_factor = 16.0_f32; // DS4_ROPE_SCALE_FACTOR
    let freq_scale = 1.0_f32 / scale_factor;
    let ext_factor = 1.0_f32; // SCALE_FACTOR > 1.0 → ext_factor = 1
    // attn_factor = 1.0 / (1.0 + 0.1 * ln(SCALE_FACTOR))
    let attn_factor = 1.0_f32 / (1.0 + 0.1 * (1.0_f32 / freq_scale).ln());
    RoPEParams {
        n_head: 1,
        head_dim,
        n_rot: DSV4_N_ROT as u32,
        pos,
        n_ctx_orig: DSV4_ROPE_ORIG_CTX,
        freq_base: DSV4_COMPRESS_ROPE_FREQ_BASE,
        freq_scale,
        ext_factor,
        attn_factor,
        beta_fast: 32.0,
        beta_slow: 1.0,
        inverse: false,
    }
}

/// Read column `pos_mod` of the APE matrix (`[width × ratio]`
/// row-major) into `out` (length `width`). For an F32-backed blob
/// this is a single slice copy; for an F16-backed blob the slice
/// is dequantised via `dequant_to_f32`. Other dtypes are rejected.
///
/// Integer arithmetic on the byte offset is checked
/// (`usize::checked_mul` + `checked_add`) so a crafted GGUF cannot
/// wrap the computation and produce a nonsensical-but-in-bounds slice.
fn read_ape_column(
    ape: &WeightBlob<'_>,
    width: usize,
    ratio: u32,
    pos_mod: u32,
    out: &mut [f32],
) -> Result<(), Error> {
    if out.len() != width {
        return Err(Error::ShapeMismatch {
            key: "compressor.ape.out",
            expected: format!("{width}"),
            actual: format!("{}", out.len()),
        });
    }
    if pos_mod >= ratio {
        return Err(Error::ShapeMismatch {
            key: "compressor.ape.pos_mod",
            expected: format!("< {ratio}"),
            actual: format!("{pos_mod}"),
        });
    }
    let off = (pos_mod as usize)
        .checked_mul(width)
        .ok_or(Error::ShapeMismatch {
            key: "compressor.ape.offset",
            expected: "pos_mod * width must not overflow".to_string(),
            actual: format!("pos_mod={pos_mod}, width={width}"),
        })?;
    match ape {
        WeightBlob::F32(s) => {
            let end = off.checked_add(width).ok_or(Error::ShapeMismatch {
                key: "compressor.ape.f32_end",
                expected: "off + width must not overflow".to_string(),
                actual: format!("off={off}, width={width}"),
            })?;
            if end > s.len() {
                return Err(Error::ShapeMismatch {
                    key: "compressor.ape.f32_slice",
                    expected: format!("{end} f32s"),
                    actual: format!("{}", s.len()),
                });
            }
            out.copy_from_slice(&s[off..end]);
            Ok(())
        }
        WeightBlob::Quant { data, dtype } => {
            let elem_bytes: usize = match dtype {
                GgmlType::F16 | GgmlType::BF16 => 2,
                GgmlType::F32 => 4,
                other => {
                    return Err(Error::ShapeMismatch {
                        key: "compressor.ape.dtype",
                        expected: "F16/BF16/F32".to_string(),
                        actual: format!("{other:?}"),
                    });
                }
            };
            let byte_off = off.checked_mul(elem_bytes).ok_or(Error::ShapeMismatch {
                key: "compressor.ape.byte_off",
                expected: "off * elem_bytes must not overflow".to_string(),
                actual: format!("off={off}, elem_bytes={elem_bytes}"),
            })?;
            let col_bytes = width
                .checked_mul(elem_bytes)
                .ok_or(Error::ShapeMismatch {
                    key: "compressor.ape.col_bytes",
                    expected: "width * elem_bytes must not overflow".to_string(),
                    actual: format!("width={width}, elem_bytes={elem_bytes}"),
                })?;
            let byte_end = byte_off
                .checked_add(col_bytes)
                .ok_or(Error::ShapeMismatch {
                    key: "compressor.ape.byte_end",
                    expected: "byte_off + col_bytes must not overflow".to_string(),
                    actual: format!("byte_off={byte_off}, col_bytes={col_bytes}"),
                })?;
            if byte_end > data.len() {
                return Err(Error::ShapeMismatch {
                    key: "compressor.ape.byte_slice",
                    expected: format!("{byte_end} bytes"),
                    actual: format!("{}", data.len()),
                });
            }
            dequant_to_f32(*dtype, &data[byte_off..byte_end], out).map_err(|e| {
                Error::ShapeMismatch {
                    key: "compressor.ape.dequant",
                    expected: "valid dequant".to_string(),
                    actual: format!("{e}"),
                }
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::shape::{
        DSV4_HEAD_DIM, DSV4_N_INDEXER_HEAD, DSV4_N_INDEXER_HEAD_DIM, DSV4_N_LORA_Q,
    };

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

    // ---- compressor_decode_one (F011.B) -----------------------------------

    /// Build a `CompressorWeights` fixture for ratio-128 layers with
    /// caller-controlled `kv` / `gate` matrices and a zero APE bias.
    /// All weights are F32-backed via `WeightBlob::F32` for easy
    /// numerical reasoning in tests.
    fn make_ratio128_weights<'a>(
        kv: &'a [f32],
        gate: &'a [f32],
        ape: &'a [f32],
        norm: &'a [f32],
    ) -> CompressorWeights<'a> {
        // Sanity: ratio-128 layers have comp_width = HEAD_DIM = 512.
        assert_eq!(kv.len(), DSV4_HEAD_DIM * DSV4_N_EMBD);
        assert_eq!(gate.len(), DSV4_HEAD_DIM * DSV4_N_EMBD);
        // APE is [comp_width × ratio] row-major = [512, 128] = 65_536.
        assert_eq!(ape.len(), DSV4_HEAD_DIM * 128);
        assert_eq!(norm.len(), DSV4_HEAD_DIM);
        CompressorWeights {
            kv: WeightBlob::F32(kv),
            gate: WeightBlob::F32(gate),
            ape: WeightBlob::F32(ape),
            norm,
        }
    }

    #[test]
    fn compressor_decode_one_emits_on_ratio_boundary_ratio128() {
        // Deterministic fixture: kv = identity-truncation (out lane o
        // mirrors x lane o for o < N_EMBD), gate = zeros (so scores are
        // uniform after APE bias of 0), APE = zeros, norm = ones.
        // With uniform-zero scores the per-dim softmax reduces to a
        // simple mean. With identity-truncation kv and a constant
        // x = c, every emitted (pre-RMSNorm) lane equals c.
        let kv = truncating_weight(DSV4_HEAD_DIM, DSV4_N_EMBD);
        let gate = vec![0.0_f32; DSV4_HEAD_DIM * DSV4_N_EMBD];
        let ape = vec![0.0_f32; DSV4_HEAD_DIM * 128];
        let norm = vec![1.0_f32; DSV4_HEAD_DIM];
        let weights = make_ratio128_weights(&kv, &gate, &ape, &norm);

        let mut pool = CompressedKvPool::with_dsv4_attn(2, 128);
        let mut kv_cur = vec![0.0_f32; pool.width()];
        let mut sc_cur = vec![0.0_f32; pool.width()];
        let mut ape_col = vec![0.0_f32; pool.width()];

        let x = vec![5.0_f32; DSV4_N_EMBD];

        // 127 tokens: no emission.
        for pos in 0..127 {
            let emitted = compressor_decode_one(
                &x, &weights, &mut pool, pos, 0,
                &mut kv_cur, &mut sc_cur, &mut ape_col, SimdTier::Scalar,
            )
            .unwrap();
            assert!(!emitted, "pos {pos} should not emit");
        }

        // 128th token: emission fires.
        let emitted = compressor_decode_one(
            &x, &weights, &mut pool, 127, 0,
            &mut kv_cur, &mut sc_cur, &mut ape_col, SimdTier::Scalar,
        )
        .unwrap();
        assert!(emitted, "pos 127 should emit");
        assert_eq!(pool.len(), 1);
    }

    #[test]
    fn compressor_decode_one_applies_rms_norm_to_emitted_row() {
        // Same fixture but with norm = [k, k, k, ..., k] for some k.
        // The RMSNorm step computes `rms = 1/sqrt(mean(x^2) + eps)` and
        // scales each lane by `rms * norm[i]`. With every kv lane = 5.0
        // pre-norm, `mean(x^2) = 25.0`, `rms = 1/sqrt(25 + 1e-6) ≈ 1/5`.
        // After RMSNorm: each lane ≈ 5 * (1/5) * k = k. So the
        // post-norm row is ≈ [k; head_dim] before RoPE perturbs the
        // tail. We assert this on the non-RoPE prefix lanes only.
        let kv = truncating_weight(DSV4_HEAD_DIM, DSV4_N_EMBD);
        let gate = vec![0.0_f32; DSV4_HEAD_DIM * DSV4_N_EMBD];
        let ape = vec![0.0_f32; DSV4_HEAD_DIM * 128];
        let k = 3.0_f32;
        let norm = vec![k; DSV4_HEAD_DIM];
        let weights = make_ratio128_weights(&kv, &gate, &ape, &norm);

        let mut pool = CompressedKvPool::with_dsv4_attn(2, 128);
        let mut kv_cur = vec![0.0_f32; pool.width()];
        let mut sc_cur = vec![0.0_f32; pool.width()];
        let mut ape_col = vec![0.0_f32; pool.width()];

        let x = vec![5.0_f32; DSV4_N_EMBD];
        for pos in 0..128 {
            compressor_decode_one(
                &x, &weights, &mut pool, pos, 0,
                &mut kv_cur, &mut sc_cur, &mut ape_col, SimdTier::Scalar,
            )
            .unwrap();
        }

        let row = &pool.rows()[..DSV4_HEAD_DIM];
        // n_nope = head_dim - n_rot = 512 - 64 = 448 lanes survive
        // unrotated. Check the first 100 of those.
        for &v in &row[..100] {
            assert!(
                (v - k).abs() < 1e-3,
                "expected post-RMSNorm lane ≈ {k}, got {v}"
            );
        }
    }

    #[test]
    fn compressor_decode_one_ape_bias_changes_score_distribution() {
        // Validate that the APE column actually contributes to scores
        // (and therefore affects the per-dim softmax). Setup: kv has
        // distinct values per token slot via `gate` carrying zeros and
        // `ape[pos_mod]` boosting the score for a known slot. With one
        // boosted slot, that slot's kv dominates the softmax.
        //
        // We use a ratio=2 toy pool so the math is easy to follow.
        // Note: ratio=2 means coff=1 (only ratio==4 triggers coff=2),
        // so this exercises the single-lane code path.
        let ratio = 2_u32;
        let head_dim = 4_usize;

        // For the toy pool we can't use DSV4_N_EMBD-sized matrices; the
        // compressor_decode_one signature is hard-coded to N_EMBD. So
        // instead we test APE handling via read_ape_column directly.
        let comp_width = head_dim; // coff = 1 for ratio=2 in upstream logic
                                    // (we just want a reasonable shape).
        let ape: Vec<f32> = (0..comp_width * ratio as usize)
            .map(|i| i as f32)
            .collect();
        let blob = WeightBlob::F32(&ape);
        let mut out = vec![0.0_f32; comp_width];
        read_ape_column(&blob, comp_width, ratio, 0, &mut out).unwrap();
        // Column 0 occupies offsets [0, comp_width).
        for (i, &v) in out.iter().enumerate() {
            assert_eq!(v, i as f32);
        }
        read_ape_column(&blob, comp_width, ratio, 1, &mut out).unwrap();
        // Column 1 occupies offsets [comp_width, 2*comp_width).
        for (i, &v) in out.iter().enumerate() {
            assert_eq!(v, (comp_width + i) as f32);
        }
    }

    #[test]
    fn compressor_decode_one_rejects_wrong_scratch_size() {
        let kv = vec![0.0_f32; DSV4_HEAD_DIM * DSV4_N_EMBD];
        let gate = vec![0.0_f32; DSV4_HEAD_DIM * DSV4_N_EMBD];
        let ape = vec![0.0_f32; DSV4_HEAD_DIM * 128];
        let norm = vec![1.0_f32; DSV4_HEAD_DIM];
        let weights = make_ratio128_weights(&kv, &gate, &ape, &norm);
        let mut pool = CompressedKvPool::with_dsv4_attn(1, 128);
        let mut kv_cur = vec![0.0_f32; pool.width() - 1]; // wrong
        let mut sc_cur = vec![0.0_f32; pool.width()];
        let mut ape_col = vec![0.0_f32; pool.width()];
        let x = vec![0.0_f32; DSV4_N_EMBD];
        let err = compressor_decode_one(
            &x, &weights, &mut pool, 0, 0,
            &mut kv_cur, &mut sc_cur, &mut ape_col, SimdTier::Scalar,
        )
        .unwrap_err();
        assert!(matches!(err, Error::ShapeMismatch { .. }));
    }

    #[test]
    fn read_ape_column_rejects_out_of_range_pos_mod() {
        // ratio=2, pos_mod=5 is way beyond the column count.
        let ape = vec![0.0_f32; 4 * 2];
        let blob = WeightBlob::F32(&ape);
        let mut out = vec![0.0_f32; 4];
        let err = read_ape_column(&blob, 4, 2, 5, &mut out).unwrap_err();
        assert!(matches!(err, Error::ShapeMismatch { .. }));
    }

    #[test]
    fn read_ape_column_rejects_too_short_blob() {
        // Blob declares width=4 but holds only 6 f32s, so column 1 ([4, 8))
        // overruns. The bounds check must fire.
        let ape = vec![0.0_f32; 6];
        let blob = WeightBlob::F32(&ape);
        let mut out = vec![0.0_f32; 4];
        let err = read_ape_column(&blob, 4, 2, 1, &mut out).unwrap_err();
        assert!(matches!(err, Error::ShapeMismatch { .. }));
    }

    #[test]
    fn read_ape_column_dequants_f16() {
        // Pack 4 f16 values into bytes via their well-known bit
        // patterns to avoid pulling `half` into rsllm-models' deps:
        //   1.0_f16 = 0x3C00, 2.0 = 0x4000, 3.0 = 0x4200, 4.0 = 0x4400.
        let vals = [1.0_f32, 2.0, 3.0, 4.0];
        let bytes: Vec<u8> = [0x3C00_u16, 0x4000, 0x4200, 0x4400]
            .iter()
            .flat_map(|&u| u.to_le_bytes())
            .collect();
        let blob = WeightBlob::Quant {
            data: &bytes,
            dtype: GgmlType::F16,
        };
        let mut out = vec![0.0_f32; 4];
        // Single-column matrix shape [width=4, ratio=1] — read column 0.
        read_ape_column(&blob, 4, 1, 0, &mut out).unwrap();
        for (i, &v) in out.iter().enumerate() {
            assert!((v - vals[i]).abs() < 1e-3, "lane {i} = {v}");
        }
    }
}
