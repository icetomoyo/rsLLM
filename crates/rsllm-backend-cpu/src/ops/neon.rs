//! ARM NEON dotprod kernel specializations (aarch64-only).
//!
//! Inspiration: `ds4.c:2740-2801` (MIT, The ds4.c authors). The
//! `dot_q8_0_row_neon` function below mirrors ds4's NEON Q8_0 dot
//! product, including its **pairwise block accumulation** with two
//! independent `float32x4_t` accumulators (`accv0`, `accv1`). The
//! pairwise form matters because floating-point addition is not
//! associative — the scalar reference in [`super::q8_0::dot_q8_0_row_scalar`]
//! accumulates blocks sequentially, while the NEON path accumulates
//! in pairs, so the two paths can differ in the last few ULPs even
//! on identical input. Cross-path correctness tests must allow for
//! this divergence (F004 spec: 1e-4 relative tolerance).
//!
//! Dispatch is via [`crate::SimdTier::Neon`], decided at runtime in
//! [`crate::detect`]. When the host lacks the `dotprod` extension or
//! the target arch isn't aarch64, callers fall back to the scalar
//! implementation in [`super::q8_0`].

#![cfg(target_arch = "aarch64")]

use core::arch::aarch64::{
    int32x4_t, vaddq_f32, vaddvq_f32, vaddvq_s32, vcvtq_f32_s32, vdotq_s32, vdupq_n_f32,
    vdupq_n_s32, vfmaq_n_f32, vld1q_s8,
};

use bytemuck::cast_slice;
use half::f16;

use super::q8_0::{Q8_0_BLOCK, Q8_0_BLOCK_BYTES};

/// NEON-accelerated Q8_0 row dot product. Caller must hold a runtime
/// guarantee that `dotprod` is supported (e.g. via
/// `std::arch::is_aarch64_feature_detected!("dotprod")` checked once
/// at construction time and stashed in [`crate::SimdTier::Neon`]).
///
/// Safety: this function uses unsafe NEON intrinsics. The bounds on
/// `row.len()`, `xq.len()`, `xscale.len()` are checked by the public
/// dispatcher in [`super::q8_0::matmul_q8_0_batch`]. We additionally
/// `debug_assert_eq!` them here for catch-in-test.
///
/// Mirrors `ds4.c:2746-2800` (the NEON+DOTPROD branch).
///
/// # Safety
/// Caller must guarantee that the host CPU supports the NEON
/// `dotprod` extension (e.g. checked once via
/// `std::arch::is_aarch64_feature_detected!("dotprod")`). Calling
/// this on a CPU without `dotprod` produces an illegal instruction.
#[target_feature(enable = "dotprod")]
pub unsafe fn dot_q8_0_row_neon(row: &[u8], xq: &[i8], xscale: &[f32], in_dim: usize) -> f32 {
    let blocks = in_dim.div_ceil(Q8_0_BLOCK);
    debug_assert_eq!(row.len(), blocks * Q8_0_BLOCK_BYTES);
    debug_assert_eq!(xq.len(), blocks * Q8_0_BLOCK);
    debug_assert_eq!(xscale.len(), blocks);

    // Fast path: in_dim divisible by 32 (no tail). Same condition as
    // ds4.c:2747 `(in_dim & 31u) == 0`.
    if in_dim.is_multiple_of(Q8_0_BLOCK) {
        let mut accv0 = unsafe { vdupq_n_f32(0.0) };
        let mut accv1 = unsafe { vdupq_n_f32(0.0) };

        let mut b = 0usize;
        while b + 1 < blocks {
            let block0 = &row[b * Q8_0_BLOCK_BYTES..];
            let block1 = &row[(b + 1) * Q8_0_BLOCK_BYTES..];
            let scale0 = block_scale_finite(block0);
            let scale1 = block_scale_finite(block1);

            let qs0: &[i8] = cast_slice(&block0[2..2 + Q8_0_BLOCK]);
            let qs1: &[i8] = cast_slice(&block1[2..2 + Q8_0_BLOCK]);
            let xq0 = &xq[b * Q8_0_BLOCK..];
            let xq1 = &xq[(b + 1) * Q8_0_BLOCK..];

            // SAFETY: the slice indexing above guarantees 32 i8 bytes
            // are addressable at each pointer; ds4.c:2764-2769 does
            // the same two `vld1q_s8` + two `vdotq_s32` per block.
            unsafe {
                let dot0 = accumulate_block(qs0.as_ptr(), xq0.as_ptr());
                let dot1 = accumulate_block(qs1.as_ptr(), xq1.as_ptr());
                accv0 = vfmaq_n_f32(accv0, vcvtq_f32_s32(dot0), scale0 * xscale[b]);
                accv1 = vfmaq_n_f32(accv1, vcvtq_f32_s32(dot1), scale1 * xscale[b + 1]);
            }
            b += 2;
        }

        // Odd-block tail.
        if b < blocks {
            let block = &row[b * Q8_0_BLOCK_BYTES..];
            let scale = block_scale_finite(block);
            let qs: &[i8] = cast_slice(&block[2..2 + Q8_0_BLOCK]);
            let xqb = &xq[b * Q8_0_BLOCK..];
            unsafe {
                let dot = accumulate_block(qs.as_ptr(), xqb.as_ptr());
                accv0 = vfmaq_n_f32(accv0, vcvtq_f32_s32(dot), scale * xscale[b]);
            }
        }

        return unsafe { vaddvq_f32(vaddq_f32(accv0, accv1)) };
    }

    // Tail-handling fallback for in_dim not divisible by 32: do per-block
    // dotprod over whole 32-lane segments, and scalar over the final
    // partial block. Matches ds4.c:2790-2799.
    let mut acc = 0.0_f32;
    for b in 0..blocks {
        let block = &row[b * Q8_0_BLOCK_BYTES..];
        let scale = block_scale_finite(block);
        let qs: &[i8] = cast_slice(&block[2..2 + Q8_0_BLOCK]);
        let xqb = &xq[b * Q8_0_BLOCK..];

        let i0 = b * Q8_0_BLOCK;
        let bn = (in_dim - i0).min(Q8_0_BLOCK);

        let dot = if bn == Q8_0_BLOCK {
            unsafe { vaddvq_s32(accumulate_block(qs.as_ptr(), xqb.as_ptr())) }
        } else {
            // Partial block — must do scalar to avoid reading past the
            // valid `bn` lanes (xq is zero-padded so reading would not
            // panic, but stays defensive and bit-identical to ds4 scalar).
            let mut s: i32 = 0;
            for i in 0..bn {
                s += i32::from(qs[i]) * i32::from(xqb[i]);
            }
            s
        };
        acc += scale * xscale[b] * dot as f32;
    }
    acc
}

/// SAFETY: caller guarantees `a` and `b` each address ≥ 32 i8 bytes.
#[inline]
#[target_feature(enable = "dotprod")]
unsafe fn accumulate_block(a: *const i8, b: *const i8) -> int32x4_t {
    unsafe {
        let mut acc = vdupq_n_s32(0);
        acc = vdotq_s32(acc, vld1q_s8(a), vld1q_s8(b));
        acc = vdotq_s32(acc, vld1q_s8(a.add(16)), vld1q_s8(b.add(16)));
        acc
    }
}

/// Read the f16 scale at the start of a Q8_0 block and coerce
/// non-finite values to 0.0. Matches the scalar path's sanitization
/// (see [`super::q8_0::dot_q8_0_row_scalar`]).
#[inline]
fn block_scale_finite(block: &[u8]) -> f32 {
    let bits = u16::from_le_bytes([block[0], block[1]]);
    let s = f16::from_bits(bits).to_f32();
    if s.is_finite() { s } else { 0.0 }
}

#[cfg(test)]
mod tests {
    use super::super::q8_0::{Q8_0_BLOCK, Q8_0_BLOCK_BYTES, dot_q8_0_row_scalar};
    use super::*;

    /// Pack one row of f32 weights into Q8_0 blocks (mirrors
    /// [`super::super::q8_0::quantize_q8_0_activation`]).
    fn pack_row(x: &[f32]) -> Vec<u8> {
        use super::super::q8_0::quantize_q8_0_activation;
        let blocks = x.len().div_ceil(Q8_0_BLOCK);
        let mut out = vec![0u8; blocks * Q8_0_BLOCK_BYTES];
        let mut tmp_q = vec![0i8; blocks * Q8_0_BLOCK];
        let mut tmp_s = vec![0.0_f32; blocks];
        quantize_q8_0_activation(x, &mut tmp_q, &mut tmp_s).unwrap();
        for b in 0..blocks {
            let dst = &mut out[b * Q8_0_BLOCK_BYTES..(b + 1) * Q8_0_BLOCK_BYTES];
            let bits = half::f16::from_f32(tmp_s[b]).to_bits().to_le_bytes();
            dst[0] = bits[0];
            dst[1] = bits[1];
            for i in 0..Q8_0_BLOCK {
                dst[2 + i] = tmp_q[b * Q8_0_BLOCK + i] as u8;
            }
        }
        out
    }

    #[test]
    fn neon_matches_scalar_when_supported() {
        if !std::arch::is_aarch64_feature_detected!("dotprod") {
            eprintln!("dotprod not detected; skipping");
            return;
        }
        let in_dim = 128;
        let w: Vec<f32> = (0..in_dim)
            .map(|i| ((i as f32) * 0.07 - 3.0).sin())
            .collect();
        let x: Vec<f32> = (0..in_dim)
            .map(|i| ((i as f32) * 0.11 + 1.0).cos())
            .collect();
        let row = pack_row(&w);

        use super::super::q8_0::quantize_q8_0_activation;
        let blocks = in_dim / Q8_0_BLOCK;
        let mut xq = vec![0i8; blocks * Q8_0_BLOCK];
        let mut xs = vec![0.0_f32; blocks];
        quantize_q8_0_activation(&x, &mut xq, &mut xs).unwrap();

        let scalar = dot_q8_0_row_scalar(&row, &xq, &xs, in_dim);
        let neon = unsafe { dot_q8_0_row_neon(&row, &xq, &xs, in_dim) };
        let denom = scalar.abs().max(1.0);
        assert!(
            (scalar - neon).abs() / denom < 1e-4,
            "neon {neon} vs scalar {scalar}"
        );
    }
}
