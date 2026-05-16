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

/// Grouped matmul for the attention output LoRA down-projection.
///
/// DS V4 Flash's `attn_output_a` is a `[N_HEAD * HEAD_DIM × out_low_dim]`
/// weight matrix whose rows are partitioned into `n_groups` blocks of
/// `group_dim = HEAD_DIM * (N_HEAD / N_OUT_GROUP)` consecutive input
/// rows. Each group projects to its own `rank` output slot, and the
/// `n_groups` outputs are concatenated into `[n_groups × rank]`.
///
/// In other words: input row `i ∈ [g*group_dim, (g+1)*group_dim)`
/// only contributes to output column `[g*rank, (g+1)*rank)`. Cross-
/// group contributions are zero, so a dense matmul would compute the
/// wrong result.
///
/// `weight_per_group` carries the per-group `[group_dim × rank]`
/// matrix. The total `weight_per_group.byte_len()` must equal
/// `n_groups * group_dim * rank * (bytes_per_element)`.
///
/// Shape contracts:
/// - `x`     : `[n_tok × (n_groups * group_dim)]`
/// - `out`   : `[n_tok × (n_groups * rank)]`
///
/// # Errors
/// Bubbles up shape errors from the underlying matmul kernel.
#[allow(clippy::too_many_arguments)]
pub fn matmul_grouped_lora_down(
    out: &mut [f32],
    weight_per_group: &WeightBlob<'_>,
    x: &[f32],
    n_tok: usize,
    n_groups: usize,
    group_dim: usize,
    rank: usize,
    tier: SimdTier,
) -> Result<(), Error> {
    let total_in = n_groups * group_dim;
    let total_out = n_groups * rank;
    if x.len() != n_tok * total_in {
        return Err(Error::ShapeMismatch {
            key: "matmul_grouped_lora_down.x",
            expected: format!("{}", n_tok * total_in),
            actual: format!("{}", x.len()),
        });
    }
    if out.len() != n_tok * total_out {
        return Err(Error::ShapeMismatch {
            key: "matmul_grouped_lora_down.out",
            expected: format!("{}", n_tok * total_out),
            actual: format!("{}", out.len()),
        });
    }

    // Strategy: extract each group's input column-slice into a contiguous
    // [n_tok × group_dim] buffer, run a normal matmul against the group's
    // [group_dim × rank] weight slice, and write into the appropriate
    // output slot. This is `n_groups` standard matmuls; for DS V4 Flash
    // (n_groups = 8) the overhead vs. a single fused kernel is small.
    let mut x_group = vec![0.0_f32; n_tok * group_dim];
    let mut out_group = vec![0.0_f32; n_tok * rank];
    for g in 0..n_groups {
        // Gather per-group input rows.
        for t in 0..n_tok {
            let src = &x[t * total_in + g * group_dim..t * total_in + (g + 1) * group_dim];
            x_group[t * group_dim..(t + 1) * group_dim].copy_from_slice(src);
        }
        let w_g = grouped_weight_slice(weight_per_group, g, group_dim * rank)?;
        matmul_weight_f32(
            &mut out_group,
            &w_g,
            &x_group,
            n_tok,
            group_dim,
            rank,
            tier,
        )?;
        // Scatter per-group outputs.
        for t in 0..n_tok {
            let dst = &mut out[t * total_out + g * rank..t * total_out + (g + 1) * rank];
            dst.copy_from_slice(&out_group[t * rank..(t + 1) * rank]);
        }
    }
    Ok(())
}

/// Borrow the `g`-th group's weight slice from a stacked group blob.
/// Mirrors the row-slicing in [`super::moe::StackedExperts::expert`]
/// but with a configurable `elements_per_group`.
fn grouped_weight_slice<'a>(
    weight: &WeightBlob<'a>,
    g: usize,
    elements_per_group: usize,
) -> Result<WeightBlob<'a>, Error> {
    match *weight {
        WeightBlob::F32(s) => {
            let start = g * elements_per_group;
            let end = start + elements_per_group;
            if end > s.len() {
                return Err(Error::ShapeMismatch {
                    key: "grouped_weight_slice.f32",
                    expected: format!("at least {end} f32 elements"),
                    actual: format!("{}", s.len()),
                });
            }
            Ok(WeightBlob::F32(&s[start..end]))
        }
        WeightBlob::Quant { data, dtype } => {
            let block_elems = dtype.block_elements() as usize;
            let block_bytes = dtype.block_bytes() as usize;
            let blocks = elements_per_group / block_elems;
            let stride = blocks * block_bytes;
            let start = g * stride;
            let end = start + stride;
            if end > data.len() {
                return Err(Error::ShapeMismatch {
                    key: "grouped_weight_slice.quant",
                    expected: format!("at least {end} bytes"),
                    actual: format!("{}", data.len()),
                });
            }
            Ok(WeightBlob::Quant {
                data: &data[start..end],
                dtype,
            })
        }
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
    fn grouped_matmul_isolates_groups() {
        // 2 groups, group_dim=3, rank=2 → input [n_tok, 6] → output [n_tok, 4].
        // Per-group weight stored as [rank × group_dim] row-major (matches
        // the [out × in] convention of `matmul_weight_f32`).
        let n_tok = 2;
        let n_groups = 2;
        let group_dim = 3;
        let rank = 2;
        // Group 0 weight [rank=2 × group_dim=3]:
        //   row 0 (output 0): [1, 0, 1]
        //   row 1 (output 1): [0, 1, 1]
        // Group 1 weight [2 × 3]:
        //   row 0: [2, 0, 0]
        //   row 1: [0, 2, 0]
        #[rustfmt::skip]
        let w: Vec<f32> = vec![
            // Group 0 (6 elements)
            1.0, 0.0, 1.0,
            0.0, 1.0, 1.0,
            // Group 1 (6 elements)
            2.0, 0.0, 0.0,
            0.0, 2.0, 0.0,
        ];
        let x: Vec<f32> = vec![
            // tok 0: [g0 input | g1 input]
            1.0, 2.0, 3.0,  4.0, 5.0, 6.0,
            // tok 1
            10.0, 20.0, 30.0,  40.0, 50.0, 60.0,
        ];
        let mut out = vec![0.0_f32; n_tok * n_groups * rank];
        matmul_grouped_lora_down(
            &mut out,
            &WeightBlob::F32(&w),
            &x,
            n_tok,
            n_groups,
            group_dim,
            rank,
            SimdTier::Scalar,
        )
        .unwrap();
        // tok 0 g0: [1*1+0*2+1*3, 0*1+1*2+1*3]     = [4, 5]
        // tok 0 g1: [2*4+0*5+0*6, 0*4+2*5+0*6]     = [8, 10]
        // tok 1 g0: [1*10+0*20+1*30, 0*10+1*20+1*30] = [40, 50]
        // tok 1 g1: [2*40+0*50+0*60, 0*40+2*50+0*60] = [80, 100]
        assert_eq!(out, vec![4.0, 5.0, 8.0, 10.0, 40.0, 50.0, 80.0, 100.0]);
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
