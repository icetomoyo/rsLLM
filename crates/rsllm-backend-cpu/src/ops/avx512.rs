//! x86_64 AVX-512 kernel specializations (x86_64-only).
//!
//! Inspiration: ggml's `ggml-cpu-x86.c` AVX-512 idioms (MIT, the ggml
//! authors). The Q8_0 dot product uses `_mm512_cvtepi8_epi16` to
//! sign-extend both operands then `_mm512_madd_epi16` to do 16-way
//! pairwise multiply-add — requires only AVX-512F + AVX-512BW, so
//! works on Skylake-X / Cascade Lake and every later AVX-512 part.
//!
//! Future optimization: AVX-512 **VNNI** (`_mm512_dpbusd_epi32`) is
//! significantly faster on Zen 5 / Sapphire Rapids+, but requires
//! either signed/unsigned bias-trick fixup or `dpbssd` (AVX-VNNI-INT8).
//! Phase D ships the portable madd path; benchmarks in phase E drive
//! the decision on whether to add a VNNI specialization.

#![cfg(target_arch = "x86_64")]

use core::arch::x86_64::{
    __m256i, __m512i, _mm256_loadu_si256, _mm512_cvtepi8_epi16, _mm512_madd_epi16,
    _mm512_reduce_add_epi32,
};

use bytemuck::cast_slice;

use super::q8_0::{Q8_0_BLOCK, Q8_0_BLOCK_BYTES, block_scale_finite};

/// AVX-512-accelerated Q8_0 row dot product. Caller must hold a
/// runtime guarantee that `avx512f` + `avx512bw` are supported.
///
/// Mirrors the shape of [`super::neon::dot_q8_0_row_neon`] but with a
/// single zmm accumulator — AVX-512 has 16 i32 lanes per register and
/// `madd_epi16` already does pairwise reduction, so two-register
/// pairwise unrolling buys less than on NEON's 4-lane vectors.
///
/// # Safety
/// Caller must guarantee that the host CPU supports both AVX-512F and
/// AVX-512BW (e.g. checked via
/// `is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw")`).
/// Calling this on a host without those extensions produces an illegal
/// instruction.
#[target_feature(enable = "avx512f,avx512bw")]
pub unsafe fn dot_q8_0_row_avx512(row: &[u8], xq: &[i8], xscale: &[f32], in_dim: usize) -> f32 {
    let blocks = in_dim.div_ceil(Q8_0_BLOCK);
    debug_assert_eq!(row.len(), blocks * Q8_0_BLOCK_BYTES);
    debug_assert_eq!(xq.len(), blocks * Q8_0_BLOCK);
    debug_assert_eq!(xscale.len(), blocks);

    let mut acc = 0.0_f32;
    for b in 0..blocks {
        let block = &row[b * Q8_0_BLOCK_BYTES..];
        let scale = block_scale_finite(block);
        let qs: &[i8] = cast_slice(&block[2..2 + Q8_0_BLOCK]);
        let xqb = &xq[b * Q8_0_BLOCK..];

        let i0 = b * Q8_0_BLOCK;
        let bn = (in_dim - i0).min(Q8_0_BLOCK);

        let dot = if bn == Q8_0_BLOCK {
            unsafe { madd_block(qs.as_ptr(), xqb.as_ptr()) }
        } else {
            // Partial-block tail (in_dim % 32 != 0): scalar.
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

/// Compute `sum(a[0..32] * b[0..32])` for 32 i8 lanes via AVX-512
/// sign-extend + madd + horizontal reduce. Returns a single i32.
///
/// # Safety
/// `a` and `b` must each address at least 32 readable i8 bytes.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn madd_block(a: *const i8, b: *const i8) -> i32 {
    unsafe {
        // Load 32 i8 (256 bits) from each pointer.
        let a256: __m256i = _mm256_loadu_si256(a as *const __m256i);
        let b256: __m256i = _mm256_loadu_si256(b as *const __m256i);
        // Sign-extend i8 → i16 across 512 bits (32 i16 lanes per register).
        let a512: __m512i = _mm512_cvtepi8_epi16(a256);
        let b512: __m512i = _mm512_cvtepi8_epi16(b256);
        // `madd_epi16`: pairs adjacent i16 lanes, multiplies, sums to
        // i32 → 16 i32 lanes. Sum-of-products of all 32 i16 pairs.
        let prod: __m512i = _mm512_madd_epi16(a512, b512);
        _mm512_reduce_add_epi32(prod)
    }
}

#[cfg(test)]
mod tests {
    use super::super::q8_0::{Q8_0_BLOCK, Q8_0_BLOCK_BYTES, dot_q8_0_row_scalar};
    use super::*;

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
    fn avx512_matches_scalar_when_supported() {
        if !is_x86_feature_detected!("avx512f") || !is_x86_feature_detected!("avx512bw") {
            eprintln!("avx512f+avx512bw not detected; skipping");
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
        let avx = unsafe { dot_q8_0_row_avx512(&row, &xq, &xs, in_dim) };
        let denom = scalar.abs().max(1.0);
        assert!(
            (scalar - avx).abs() / denom < 1e-4,
            "avx512 {avx} vs scalar {scalar}"
        );
    }
}
