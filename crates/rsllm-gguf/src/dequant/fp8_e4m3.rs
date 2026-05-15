//! FP8 E4M3 (1 sign / 4 exponent / 3 mantissa) element-level conversion.
//!
//! ## Why this lives here
//!
//! FP8 E4M3 is **not a GGUF on-disk weight type** — it has no `GgmlType`
//! variant and no on-disk block format. DeepSeek V4 Flash uses E4M3 as the
//! **runtime KV cache quantization format on Metal** (see `ds4_metal.h:241`
//! `ds4_metal_dsv4_fp8_kv_quantize_tensor`). The KV cache module
//! (`rsllm-kvcache`, FEATURE_006) will use these conversions to round-trip
//! KV rows in and out of f32; this module just provides the per-element
//! bitwise conversion that both the CPU reference path and the Metal kernel
//! agree on numerically.
//!
//! ## Format
//!
//! ```text
//! bit:  7  6 5 4 3  2 1 0
//!       │  └───┬─┘  └─┬─┘
//!       sign   E4     M3
//! ```
//!
//! - **Normal**: `(-1)^S × (1 + M/8) × 2^(E - 7)`, exponent bias = 7
//! - **Subnormal** (E = 0): `(-1)^S × (M/8) × 2^(-6)`
//! - **NaN**: `S.1111.111` (all-ones exp + all-ones mantissa); E4M3 has no
//!   infinity by design — values saturate to ±448 (max normal)
//! - **Zero**: `S.0000.000`
//!
//! Range: ±0 to ±448. Smallest positive normal = `2^-6` ≈ 0.0156. Smallest
//! positive subnormal = `2^-9` ≈ 0.00195.
//!
//! Reference: the OFP8 spec (IEEE working group), as implemented by NVIDIA
//! Transformer Engine and ggml's `e4m3_to_fp32_value`. We do not link any
//! library — the conversion is a 4-line bit fiddle.

/// Decode one E4M3 byte to f32. Lossless within E4M3's representable range.
#[inline]
pub fn fp8_e4m3_to_f32(byte: u8) -> f32 {
    let sign_bit = byte & 0x80;
    let exp = (byte >> 3) & 0x0F;
    let mantissa = byte & 0x07;

    let abs = if exp == 0 {
        // Subnormal: (mantissa / 8) * 2^-6 = mantissa * 2^-9
        f32::from(mantissa) / 512.0
    } else if exp == 0x0F && mantissa == 0x07 {
        // NaN
        return f32::NAN;
    } else {
        // Normal: (1 + mantissa/8) * 2^(exp - 7)
        let mant_f = 1.0 + f32::from(mantissa) / 8.0;
        let exp_i = i32::from(exp) - 7;
        let scale = if exp_i >= 0 {
            (1u64 << exp_i) as f32
        } else {
            1.0 / (1u64 << -exp_i) as f32
        };
        mant_f * scale
    };

    if sign_bit != 0 { -abs } else { abs }
}

/// Encode one f32 to E4M3. Saturates to ±448 on overflow. Round-to-nearest-even
/// (banker's rounding) on the mantissa, matching the OFP8 spec.
///
/// Note: this is a reference implementation, not optimized for throughput.
/// Hot paths (KV cache writes) should use SIMD kernels (Metal / AVX-512 /
/// NEON) instead of looping through this function.
#[inline]
pub fn f32_to_fp8_e4m3(value: f32) -> u8 {
    if value.is_nan() {
        // Canonical E4M3 NaN per the OFP8 spec and ggml convention: positive
        // NaN with all-ones exponent and mantissa = `0x7F` (S=0, E=15, M=7).
        // Both NaN bit patterns `0x7F` and `0xFF` decode as NaN, but writers
        // canonicalize to `0x7F` so byte-level comparisons across producers
        // (NVIDIA Transformer Engine, Metal kernels, ggml) agree.
        return 0x7F;
    }
    let sign_bit: u8 = if value.is_sign_negative() { 0x80 } else { 0x00 };
    let abs = value.abs();

    // Saturate to E4M3 max representable: max normal magnitude is
    // (1 + 6/8) × 2^8 = 1.75 × 256 = 448. The next code (`S.1111.111`) is
    // reserved for NaN per the OFP8 spec, so the largest finite magnitude
    // value is encoded as `S.1111.110`.
    const MAX_NORMAL: f32 = 448.0;
    if abs >= MAX_NORMAL {
        return sign_bit | 0x7E; // S.1111.110 = max normal magnitude
    }
    if abs == 0.0 {
        return sign_bit;
    }

    // Determine exponent. For normals, exp_e4m3 = floor(log2(abs)) + 7,
    // clamped to 1..=15. For subnormals, exp = 0.
    //
    // `round_ties_even` (banker's rounding) was stabilized in Rust 1.77;
    // our workspace MSRV is 1.87.
    let unbiased = abs.log2().floor() as i32;
    if unbiased < -6 {
        // Subnormal: mantissa = round_ties_even(abs / 2^-9)
        let scaled = abs * 512.0; // = abs / 2^-9
        let m = scaled.round_ties_even() as u32;
        if m >= 8 {
            // Round-up promoted to smallest normal.
            return sign_bit | 0x08; // S.0001.000
        }
        return sign_bit | (m as u8 & 0x07);
    }

    let exp = (unbiased + 7).clamp(1, 15) as u8;
    let exp_scale = if unbiased >= 0 {
        (1u64 << unbiased) as f32
    } else {
        1.0 / (1u64 << -unbiased) as f32
    };
    // Round mantissa: m = round_ties_even((abs / 2^exp_unbiased - 1) * 8)
    let mant_f = (abs / exp_scale - 1.0) * 8.0;
    let mut m = mant_f.round_ties_even() as i32;
    let mut e = exp;

    if m == 8 {
        // Rounded up — bump exponent.
        m = 0;
        e = e.saturating_add(1);
        if e >= 0x0F {
            // Watch for NaN encoding (E=15, M=7).
            return sign_bit | 0x7E;
        }
    }
    if !(0..=7).contains(&m) {
        // Shouldn't happen with the clamps above, but be defensive.
        m = m.clamp(0, 7);
    }
    sign_bit | (e << 3) | (m as u8 & 0x07)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_round_trip() {
        assert_eq!(fp8_e4m3_to_f32(0x00), 0.0);
        assert_eq!(fp8_e4m3_to_f32(0x80), -0.0);
        assert_eq!(f32_to_fp8_e4m3(0.0), 0x00);
        assert_eq!(f32_to_fp8_e4m3(-0.0), 0x80);
    }

    #[test]
    fn small_normal_values() {
        // 1.0 = (1 + 0/8) * 2^0 → exp = 7, mantissa = 0 → S=0, E=7, M=0 → 0b0_0111_000 = 0x38
        assert_eq!(f32_to_fp8_e4m3(1.0), 0x38);
        assert_eq!(fp8_e4m3_to_f32(0x38), 1.0);

        // 2.0 = (1 + 0/8) * 2^1 → E = 8 → 0b0_1000_000 = 0x40
        assert_eq!(f32_to_fp8_e4m3(2.0), 0x40);
        assert_eq!(fp8_e4m3_to_f32(0x40), 2.0);

        // -1.0
        assert_eq!(f32_to_fp8_e4m3(-1.0), 0xB8);
        assert_eq!(fp8_e4m3_to_f32(0xB8), -1.0);

        // 1.5 = (1 + 4/8) * 2^0 → E = 7, M = 4 → 0b0_0111_100 = 0x3C
        assert_eq!(f32_to_fp8_e4m3(1.5), 0x3C);
        assert_eq!(fp8_e4m3_to_f32(0x3C), 1.5);
    }

    #[test]
    fn max_normal_saturates() {
        // 448 = (1 + 6/8) * 2^8 → S=0, E=15, M=6 → 0b0_1111_110 = 0x7E
        assert_eq!(f32_to_fp8_e4m3(448.0), 0x7E);
        assert_eq!(fp8_e4m3_to_f32(0x7E), 448.0);

        // Anything above saturates.
        assert_eq!(f32_to_fp8_e4m3(1000.0), 0x7E);
        assert_eq!(f32_to_fp8_e4m3(-1000.0), 0xFE);
    }

    #[test]
    fn nan_handling() {
        // Both NaN bit patterns decode as NaN.
        assert!(fp8_e4m3_to_f32(0x7F).is_nan()); // S.1111.111 = canonical positive NaN
        assert!(fp8_e4m3_to_f32(0xFF).is_nan()); // S.1111.111 with sign bit also valid
        // Writers canonicalize NaN to 0x7F per OFP8 spec / ggml convention.
        assert_eq!(f32_to_fp8_e4m3(f32::NAN), 0x7F);
    }

    #[test]
    fn round_ties_to_even_normals() {
        // E4M3 codes 0x38 = 1.0 (M=0), 0x39 = 1.125 (M=1), 0x3A = 1.25 (M=2).
        // The exact midpoint 1.0625 is halfway between 1.0 and 1.125.
        // Round-to-nearest-even should choose the even mantissa (M=0 → 1.0,
        // bit pattern 0x38). Round-half-away-from-zero would choose 0x39.
        assert_eq!(f32_to_fp8_e4m3(1.0625), 0x38);
        // Midpoint between 1.125 (M=1, odd) and 1.25 (M=2, even) is 1.1875.
        // RtNE picks M=2 (even) → 0x3A. Round-half-away-from-zero also gives 0x3A.
        assert_eq!(f32_to_fp8_e4m3(1.1875), 0x3A);
        // Midpoint between 1.25 (M=2, even) and 1.375 (M=3, odd) is 1.3125.
        // RtNE picks M=2 (even) → 0x3A. Round-half-away-from-zero gives 0x3B (1.375).
        assert_eq!(f32_to_fp8_e4m3(1.3125), 0x3A);
    }

    #[test]
    fn round_ties_to_even_subnormals() {
        // Subnormal codes are M × 2^-9 for M in 0..=7.
        // Midpoint between M=2 (2/512 ≈ 0.00390625) and M=3 (3/512 ≈ 0.00585):
        // (2.5/512) = 0.0048828125. RtNE picks M=2 (even) → 0x02.
        assert_eq!(f32_to_fp8_e4m3(2.5 / 512.0), 0x02);
        // Midpoint between M=3 and M=4 (= 3.5/512): RtNE picks M=4 (even) → 0x04.
        assert_eq!(f32_to_fp8_e4m3(3.5 / 512.0), 0x04);
    }

    #[test]
    fn subnormal_values() {
        // Smallest positive subnormal: M=1, E=0 → 2^-9 = 1/512 ≈ 0.001953125
        let v = fp8_e4m3_to_f32(0x01);
        assert!((v - (1.0 / 512.0)).abs() < 1e-8);
        // Round-trip
        assert_eq!(f32_to_fp8_e4m3(1.0 / 512.0), 0x01);

        // Largest subnormal: M=7, E=0 → 7/512 ≈ 0.013671875
        let v = fp8_e4m3_to_f32(0x07);
        assert!((v - (7.0 / 512.0)).abs() < 1e-8);
    }

    #[test]
    fn round_trip_all_256_codes() {
        // For every E4M3 bit pattern, decode-then-encode must reproduce the
        // original (except for NaN encodings, which canonicalize to 0x7F per
        // the OFP8 spec).
        for byte in 0..=255u8 {
            let v = fp8_e4m3_to_f32(byte);
            if v.is_nan() {
                let re = f32_to_fp8_e4m3(v);
                assert!(re == 0x7F, "NaN re-encode: got 0x{re:02x}, want 0x7F");
            } else {
                let re = f32_to_fp8_e4m3(v);
                assert_eq!(
                    re, byte,
                    "byte 0x{byte:02x} -> f32 {v} -> 0x{re:02x}, want 0x{byte:02x}"
                );
            }
        }
    }

    #[test]
    fn out_of_range_clamps() {
        // Tiny positive: smaller than smallest subnormal → 0
        assert_eq!(f32_to_fp8_e4m3(1e-10), 0x00);
        // Tiny negative
        assert_eq!(f32_to_fp8_e4m3(-1e-10), 0x80);
    }
}
