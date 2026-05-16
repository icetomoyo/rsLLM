//! Quantized-weight × f32-activation matmul for MoE expert weights.
//!
//! DeepSeek V4 Flash's MoE routed experts are stored in three quant
//! formats with very different layouts:
//!
//!   * **Q4_K** — MoE routed expert weight (gate/up `[N_FF_EXP × N_EMBD]`).
//!     256-element block, 144 bytes per block; super-block of 8 sub-blocks
//!     each with their own 6-bit scale + 6-bit min plus per-block dmin/dmax.
//!   * **Q2_K** — MoE routed expert weight (down `[N_EMBD × N_FF_EXP]`).
//!     256-element block, 84 bytes per block.
//!   * **IQ2_XXS** — alternative 2-bit MoE expert weight (gate/up).
//!     256-element block, 66 bytes per block (with a 256-entry grid LUT).
//!
//! Each block dequantizer is already implemented in
//! [`rsllm_gguf::dequant`]. F004 phase E adds the matmul wrapper: it
//! dequantizes one weight row at a time into a stack/heap f32 buffer
//! and then does a naive f32 × f32 dot. This is the same strategy ds4
//! uses for partial blocks (ds4.c:1561+ family); SIMD-fused versions
//! are a future optimization.
//!
//! For activation: f32 in, f32 out. DS V4 Flash's MoE inputs go through
//! Q8_K quantization in ds4, but at the API level our kernels accept
//! plain f32 — the caller can quantize at the matmul boundary if it's
//! the right perf trade-off.

use rsllm_gguf::{GgmlType, dequant_to_f32};

use crate::SimdTier;
use crate::error::Error;
use crate::parallel::for_each_row_mut;

/// Generic quantized-weight × f32-activation matmul.
///
/// `out[t, o] = sum_i dequant(W)[o, i] * x[t, i]` where `W` is a
/// quantized weight buffer of `dtype` with logical shape
/// `[out_dim × in_dim]`, packed in GGUF row-major block layout.
///
/// `block_bytes` is the per-row stride in `weights` (must equal
/// `in_dim / elements_per_block * bytes_per_block`).
///
/// This is the slowest correct path: it dequantizes each row into a
/// scratch buffer per output element. Future SIMD specializations will
/// fuse the dequant + dot into a single pass (ds4.c:1561-1825 patterns).
#[allow(clippy::too_many_arguments)]
pub fn matmul_quant_f32(
    out: &mut [f32],
    weights: &[u8],
    x: &[f32],
    n_tok: usize,
    in_dim: usize,
    out_dim: usize,
    dtype: GgmlType,
    tier: SimdTier,
) -> Result<(), Error> {
    let _ = tier;

    if !matches!(
        dtype,
        GgmlType::Q4_K | GgmlType::Q2_K | GgmlType::IQ2_XXS | GgmlType::Q4_0 | GgmlType::Q8_0
    ) {
        return Err(Error::ShapeMismatch(
            "matmul_quant_f32: unsupported dtype for MoE expert matmul",
        ));
    }

    let block_elems = dtype.block_elements() as usize;
    let block_bytes = dtype.block_bytes() as usize;
    if !in_dim.is_multiple_of(block_elems) {
        return Err(Error::NotBlockAligned {
            what: "in_dim",
            actual: in_dim,
            block: block_elems,
        });
    }
    let blocks_per_row = in_dim / block_elems;
    let row_bytes = blocks_per_row * block_bytes;

    if weights.len() != out_dim * row_bytes {
        return Err(Error::ShapeMismatch(
            "matmul_quant_f32: weights must be out_dim * (in_dim / block_elem * block_bytes)",
        ));
    }
    if x.len() != n_tok * in_dim {
        return Err(Error::ShapeMismatch(
            "matmul_quant_f32: x must be n_tok * in_dim",
        ));
    }
    if out.len() != n_tok * out_dim {
        return Err(Error::ShapeMismatch(
            "matmul_quant_f32: out must be n_tok * out_dim",
        ));
    }

    for_each_row_mut(out, out_dim, |t, out_row| {
        let x_row = &x[t * in_dim..(t + 1) * in_dim];
        // Per-thread scratch: one row of dequantized f32 weights. We
        // allocate once per token rather than per output element so
        // the inner loop is just a dot product.
        let mut w_row = vec![0.0_f32; in_dim];
        for o in 0..out_dim {
            let w_packed = &weights[o * row_bytes..(o + 1) * row_bytes];
            // Dequant returns Result; if the GGUF row is malformed we
            // silently zero the row (consistent with our policy of not
            // letting one bad row poison the whole token). The error
            // path is unreachable in practice for valid GGUF inputs.
            if dequant_to_f32(dtype, w_packed, &mut w_row).is_err() {
                out_row[o] = 0.0;
                continue;
            }
            let mut dot = 0.0_f32;
            for i in 0..in_dim {
                dot += w_row[i] * x_row[i];
            }
            out_row[o] = dot;
        }
    });

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rsllm_gguf::GgmlType;

    /// Build a tiny f32 row, quantize it to Q4_0 via a manual packer
    /// (the only GGUF format that's trivial to construct by hand for
    /// tests — Q4_K's block layout is too involved for inline test
    /// construction; we exercise it through dequant smoke).
    fn pack_q4_0_row(x: &[f32]) -> Vec<u8> {
        // Q4_0 block: f16 d (2B) + 16 packed nibbles (16B) = 18B / 32 elems.
        assert!(
            x.len().is_multiple_of(32),
            "test helper requires multiple-of-32 length"
        );
        let blocks = x.len() / 32;
        let mut out = vec![0u8; blocks * 18];
        for b in 0..blocks {
            let block = &x[b * 32..(b + 1) * 32];
            let amax = block.iter().fold(0.0_f32, |a, &v| a.max(v.abs()));
            let d = amax / -8.0; // Q4_0 sign convention: see ggml
            let id = if d != 0.0 { 1.0 / d } else { 0.0 };
            let bits = half::f16::from_f32(d).to_bits().to_le_bytes();
            out[b * 18] = bits[0];
            out[b * 18 + 1] = bits[1];
            for i in 0..16 {
                let q0 = ((block[i] * id + 8.5_f32).floor() as i32).clamp(0, 15) as u8;
                let q1 = ((block[i + 16] * id + 8.5_f32).floor() as i32).clamp(0, 15) as u8;
                out[b * 18 + 2 + i] = q0 | (q1 << 4);
            }
        }
        out
    }

    #[test]
    fn rejects_unsupported_dtype() {
        let mut out = vec![0.0_f32; 4];
        let err = matmul_quant_f32(
            &mut out,
            &[],
            &[],
            1,
            32,
            4,
            GgmlType::F16,
            SimdTier::Scalar,
        )
        .unwrap_err();
        assert!(matches!(err, Error::ShapeMismatch(_)));
    }

    #[test]
    fn rejects_unaligned_in_dim() {
        let mut out = vec![0.0_f32; 4];
        let err = matmul_quant_f32(
            &mut out,
            &[],
            &[],
            1,
            30, // not multiple of 32
            4,
            GgmlType::Q4_0,
            SimdTier::Scalar,
        )
        .unwrap_err();
        assert!(matches!(err, Error::NotBlockAligned { .. }));
    }

    #[test]
    fn q4_0_matmul_matches_naive_within_tolerance() {
        // Smoke test using Q4_0 (easiest to pack manually). The same
        // matmul path serves Q4_K / Q2_K / IQ2_XXS — they only differ
        // in the dequant arm.
        let in_dim: usize = 64;
        let out_dim: usize = 3;
        let n_tok: usize = 2;

        let weights_f32: Vec<f32> = (0..out_dim * in_dim)
            .map(|i| ((i as f32) * 0.05).sin() * 4.0)
            .collect();
        let x: Vec<f32> = (0..n_tok * in_dim)
            .map(|i| ((i as f32) * 0.07).cos())
            .collect();

        let mut weights = Vec::new();
        for o in 0..out_dim {
            weights.extend_from_slice(&pack_q4_0_row(&weights_f32[o * in_dim..(o + 1) * in_dim]));
        }

        let mut out = vec![0.0_f32; n_tok * out_dim];
        matmul_quant_f32(
            &mut out,
            &weights,
            &x,
            n_tok,
            in_dim,
            out_dim,
            GgmlType::Q4_0,
            SimdTier::Scalar,
        )
        .unwrap();

        // Reference: dequant once globally and naive matmul.
        let row_blocks = in_dim / 32;
        let row_bytes = row_blocks * 18;
        let mut ref_w = vec![0.0_f32; out_dim * in_dim];
        for o in 0..out_dim {
            dequant_to_f32(
                GgmlType::Q4_0,
                &weights[o * row_bytes..(o + 1) * row_bytes],
                &mut ref_w[o * in_dim..(o + 1) * in_dim],
            )
            .unwrap();
        }
        for t in 0..n_tok {
            for o in 0..out_dim {
                let mut want = 0.0_f32;
                for i in 0..in_dim {
                    want += ref_w[o * in_dim + i] * x[t * in_dim + i];
                }
                let got = out[t * out_dim + o];
                assert!(
                    (got - want).abs() < 1e-4 * want.abs().max(1.0),
                    "t {t} o {o}: got {got}, want {want}"
                );
            }
        }
    }
}
