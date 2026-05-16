//! Weight blob dispatch for DeepSeek V4 Flash.
//!
//! GGUF tensors in a DS V4 Flash file ship in a mix of f32, f16, and
//! several quantized formats (Q8_0 for the dense attention path, Q4_K /
//! Q2_K / IQ2_XXS for MoE experts). The forward path needs a single
//! matmul entry point that hides this variety.
//!
//! Strategy: we dequantize one weight row at a time and run a naive
//! f32 dot product, leaning on [`rsllm_backend_cpu::ops::matmul_quant_f32`]
//! for quantized formats and a hand-rolled loop for f32. The dequant
//! kernel itself stays in [`rsllm_gguf::dequant`].
//!
//! SIMD-fused dequant + dot kernels (avoiding the per-row buffer) are a
//! future optimization; v0.1.0 deliberately keeps the model path simple
//! and correct, optimizing only the matmul that dominates wall-clock
//! cost (Q8_0 attention, which has its own batched path).

use rsllm_backend_cpu::SimdTier;
use rsllm_backend_cpu::ops::quant_matmul::matmul_quant_f32;
use rsllm_gguf::GgmlType;

use crate::Error;

/// A weight tensor in any of the formats DS V4 Flash can ship.
///
/// All variants are immutable views into memory that the caller owns
/// (typically an mmap'd GGUF file). The lifetime `'a` ties the blob to
/// the backing storage.
#[derive(Debug, Clone, Copy)]
pub enum WeightBlob<'a> {
    /// Plain `f32` row-major. Used for norm scale vectors and a few
    /// small embedding-like tensors.
    F32(&'a [f32]),
    /// GGUF-encoded quantized tensor with its declared dtype.
    Quant {
        /// Raw byte storage in GGUF row-major block layout.
        data: &'a [u8],
        /// Element type — one of the supported quant formats.
        dtype: GgmlType,
    },
}

impl WeightBlob<'_> {
    /// Number of bytes occupied by this blob (for debug / shape checks).
    pub fn byte_len(&self) -> usize {
        match self {
            Self::F32(s) => s.len() * 4,
            Self::Quant { data, .. } => data.len(),
        }
    }
}

/// Compute `out[t, o] = sum_i W[o, i] * x[t, i]` for any supported
/// weight format. Shapes are logical: `W` is `[out_dim × in_dim]`,
/// `x` is `[n_tok × in_dim]`, `out` is `[n_tok × out_dim]`.
///
/// Dispatch:
/// - `F32` → naive triple loop (used by tests and small weight rows).
/// - `Quant` → [`matmul_quant_f32`].
///
/// # Errors
/// Bubbles up the underlying kernel's shape errors or returns
/// [`Error::ShapeMismatch`] if the f32 path's lengths disagree.
pub fn matmul_weight_f32(
    out: &mut [f32],
    weight: &WeightBlob<'_>,
    x: &[f32],
    n_tok: usize,
    in_dim: usize,
    out_dim: usize,
    tier: SimdTier,
) -> Result<(), Error> {
    if out.len() != n_tok * out_dim {
        return Err(Error::ShapeMismatch {
            key: "matmul_weight_f32.out",
            expected: format!("{}", n_tok * out_dim),
            actual: format!("{}", out.len()),
        });
    }
    if x.len() != n_tok * in_dim {
        return Err(Error::ShapeMismatch {
            key: "matmul_weight_f32.x",
            expected: format!("{}", n_tok * in_dim),
            actual: format!("{}", x.len()),
        });
    }

    match weight {
        WeightBlob::F32(w) => {
            if w.len() != out_dim * in_dim {
                return Err(Error::ShapeMismatch {
                    key: "matmul_weight_f32.weight_f32",
                    expected: format!("{}", out_dim * in_dim),
                    actual: format!("{}", w.len()),
                });
            }
            let _ = tier;
            for t in 0..n_tok {
                let x_row = &x[t * in_dim..(t + 1) * in_dim];
                let out_row = &mut out[t * out_dim..(t + 1) * out_dim];
                for o in 0..out_dim {
                    let w_row = &w[o * in_dim..(o + 1) * in_dim];
                    let mut dot = 0.0_f32;
                    for i in 0..in_dim {
                        dot += w_row[i] * x_row[i];
                    }
                    out_row[o] = dot;
                }
            }
            Ok(())
        }
        WeightBlob::Quant { data, dtype } => matmul_quant_f32(
            out, data, x, n_tok, in_dim, out_dim, *dtype, tier,
        )
        .map_err(|e| Error::ShapeMismatch {
            key: "matmul_weight_f32.quant",
            expected: "valid matmul shape".to_string(),
            actual: format!("{e}"),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f32_matmul_matches_hand_dot() {
        // 2-token, 4-in, 3-out problem with hand-checked numbers.
        let w = vec![
            1.0_f32, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, //
            0.0, 0.0, 1.0, 1.0,
        ];
        let x = vec![
            1.0_f32, 2.0, 3.0, 4.0, //
            5.0, 6.0, 7.0, 8.0,
        ];
        let mut out = vec![0.0_f32; 2 * 3];
        matmul_weight_f32(
            &mut out,
            &WeightBlob::F32(&w),
            &x,
            2,
            4,
            3,
            SimdTier::Scalar,
        )
        .unwrap();
        // Row 0: [1*1, 1*2, 1*3 + 1*4] = [1, 2, 7]
        // Row 1: [1*5, 1*6, 1*7 + 1*8] = [5, 6, 15]
        assert_eq!(out, vec![1.0, 2.0, 7.0, 5.0, 6.0, 15.0]);
    }

    #[test]
    fn rejects_wrong_out_shape() {
        let w = vec![0.0_f32; 12];
        let x = vec![0.0_f32; 8];
        let mut out = vec![0.0_f32; 10]; // expected 6
        let err = matmul_weight_f32(
            &mut out,
            &WeightBlob::F32(&w),
            &x,
            2,
            4,
            3,
            SimdTier::Scalar,
        )
        .unwrap_err();
        assert!(matches!(err, Error::ShapeMismatch { .. }));
    }
}
