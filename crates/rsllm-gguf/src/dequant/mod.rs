//! Dequantization: convert on-disk GGUF tensor bytes to `f32` arrays.
//!
//! Public entry point is [`dequant_to_f32`]. Per-type implementations live in
//! sibling modules and are not exposed directly to avoid an unstable surface.
//!
//! ## v0.1.0 supported types
//!
//! | Type   | Module       | Reference        |
//! |--------|--------------|------------------|
//! | F32    | `float`      | trivial          |
//! | F16    | `float`      | `half::f16`      |
//! | BF16   | `float`      | `half::bf16`     |
//! | Q8_0   | `q8_0`       | ggml `ggml-quants.c` (MIT, ggml authors) |
//! | Q4_0   | `q4_0`       | same             |
//! | Q4_1   | `q4_0`       | same             |
//! | Q4_K   | `q4_k`       | same             |
//! | Q5_K   | `q5_k`       | same             |
//! | Q6_K   | `q6_k`       | same             |
//!
//! Phase 4 of FEATURE_002 lands these one module at a time. Modules that
//! have not yet been implemented are listed in [`GgmlType::is_decodable_v0_1_0`]
//! as `false`.

use crate::error::Error;
use crate::tensor::GgmlType;

mod float;
mod q4_0;
mod q4_k;
mod q5_k;
mod q6_k;
mod q8_0;

/// Extract the 6-bit scale and 6-bit min for sub-block `j` (0..8) from a
/// K-quant style packed `scales` array (12 bytes).
///
/// Direct port of `get_scale_min_k4` in `ggml-quants.c` (MIT, ggml authors).
/// Shared by Q4_K and Q5_K — both use the identical scale-packing scheme.
fn q4_k_scale_min(j: usize, q: &[u8; 12]) -> (u8, u8) {
    if j < 4 {
        let sc = q[j] & 63;
        let m = q[j + 4] & 63;
        (sc, m)
    } else {
        let sc = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
        let m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
        (sc, m)
    }
}

/// Dequantize `src` bytes (the on-disk representation of a tensor) into the
/// `dst` `f32` slice. `dst.len()` is interpreted as the logical element count.
///
/// Returns:
/// - `Ok(())` on success.
/// - [`Error::DequantSizeMismatch`] if `src.len()` does not match the expected
///   byte count for `dtype` and `dst.len()`.
/// - [`Error::UnsupportedDequant`] if `dtype` is not yet implemented in this
///   build (see [`GgmlType::is_decodable_v0_1_0`]).
pub fn dequant_to_f32(dtype: GgmlType, src: &[u8], dst: &mut [f32]) -> Result<(), Error> {
    match dtype {
        GgmlType::F32 => float::dequant_f32(src, dst),
        GgmlType::F16 => float::dequant_f16(src, dst),
        GgmlType::BF16 => float::dequant_bf16(src, dst),
        GgmlType::Q4_0 => q4_0::dequant_q4_0(src, dst),
        GgmlType::Q4_1 => q4_0::dequant_q4_1(src, dst),
        GgmlType::Q4_K => q4_k::dequant_q4_k(src, dst),
        GgmlType::Q5_K => q5_k::dequant_q5_k(src, dst),
        GgmlType::Q6_K => q6_k::dequant_q6_k(src, dst),
        GgmlType::Q8_0 => q8_0::dequant_q8_0(src, dst),
        _ => Err(Error::UnsupportedDequant(dtype.name())),
    }
}

#[cfg(test)]
mod tests {
    //! Dispatch-level wiring tests. Each test feeds the smallest valid
    //! input for one supported dtype through `dequant_to_f32(dtype, …)` and
    //! checks that the result is non-trivial. The goal is to catch arm
    //! swaps in the `match` dispatch (e.g. Q4_K / Q5_K accidentally bound
    //! to each other) — *not* to re-verify the per-format unit tests.

    use super::*;
    use half::{bf16, f16};

    #[test]
    fn unsupported_dtype_reports_error() {
        // Q2_K is recognized but not decodable in v0.1.0.
        let src = vec![0u8; 84]; // 1 block of Q2_K
        let mut dst = vec![0.0f32; 256];
        match dequant_to_f32(GgmlType::Q2_K, &src, &mut dst) {
            Err(Error::UnsupportedDequant(name)) => assert_eq!(name, "q2_k"),
            other => panic!("expected UnsupportedDequant, got {other:?}"),
        }
    }

    #[test]
    fn f32_dispatch_roundtrips() {
        let src: Vec<u8> = [1.0f32, -2.5, 1.5, 0.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let mut dst = vec![0.0f32; 4];
        dequant_to_f32(GgmlType::F32, &src, &mut dst).unwrap();
        assert_eq!(dst, vec![1.0, -2.5, 1.5, 0.0]);
    }

    #[test]
    fn f16_dispatch_roundtrips() {
        let src: Vec<u8> = [1.0f32, -1.0, 2.0, 0.0]
            .iter()
            .flat_map(|v| f16::from_f32(*v).to_le_bytes())
            .collect();
        let mut dst = vec![0.0f32; 4];
        dequant_to_f32(GgmlType::F16, &src, &mut dst).unwrap();
        assert_eq!(dst, vec![1.0, -1.0, 2.0, 0.0]);
    }

    #[test]
    fn bf16_dispatch_roundtrips() {
        let src: Vec<u8> = [1.0f32, -1.0, 2.0, 0.0]
            .iter()
            .flat_map(|v| bf16::from_f32(*v).to_le_bytes())
            .collect();
        let mut dst = vec![0.0f32; 4];
        dequant_to_f32(GgmlType::BF16, &src, &mut dst).unwrap();
        assert_eq!(dst, vec![1.0, -1.0, 2.0, 0.0]);
    }

    /// Build one Q4_0 block: f16(d) + 16 bytes packed nibbles.
    fn q4_0_block(d: f32, nibbles: [u8; 32]) -> Vec<u8> {
        let mut out = f16::from_f32(d).to_le_bytes().to_vec();
        for j in 0..16 {
            out.push(((nibbles[j + 16] & 0x0F) << 4) | (nibbles[j] & 0x0F));
        }
        out
    }

    #[test]
    fn q4_0_dispatch_routes_to_q4_0_arm() {
        // All nibbles = 8 → dst = d * (8 - 8) = 0. With d = 2.0 every output
        // must be zero. If the dispatch accidentally routed to Q4_1 (which
        // has no -8 bias), every output would be d*8 = 16.0 instead.
        let block = q4_0_block(2.0, [8u8; 32]);
        let mut dst = vec![0.0f32; 32];
        dequant_to_f32(GgmlType::Q4_0, &block, &mut dst).unwrap();
        for v in &dst {
            assert!(v.abs() < 1e-3, "Q4_0 arm: got {v}, want 0");
        }
    }

    #[test]
    fn q4_1_dispatch_routes_to_q4_1_arm() {
        // Q4_1 layout: f16(d) + f16(m) + 16 bytes nibbles. dst = d*n + m.
        // Pick d=0, m=7 so every output equals m (impossible for Q4_0 which
        // would interpret the m bytes as nibble data).
        let mut block = f16::from_f32(0.0).to_le_bytes().to_vec();
        block.extend_from_slice(&f16::from_f32(7.0).to_le_bytes());
        block.extend(std::iter::repeat_n(0u8, 16));
        let mut dst = vec![0.0f32; 32];
        dequant_to_f32(GgmlType::Q4_1, &block, &mut dst).unwrap();
        for v in &dst {
            assert!((v - 7.0).abs() < 1e-3, "Q4_1 arm: got {v}, want 7");
        }
    }

    #[test]
    fn q8_0_dispatch_routes_to_q8_0_arm() {
        // f16(d=1) + 32 i8 quants all equal to 5. dst = d*5 = 5 everywhere.
        let mut block = f16::from_f32(1.0).to_le_bytes().to_vec();
        block.extend(std::iter::repeat_n(5u8, 32));
        let mut dst = vec![0.0f32; 32];
        dequant_to_f32(GgmlType::Q8_0, &block, &mut dst).unwrap();
        for v in &dst {
            assert!((v - 5.0).abs() < 1e-3, "Q8_0 arm: got {v}, want 5");
        }
    }

    /// Build one Q4_K block (144 bytes) with the given super-block scales
    /// and uniform nibbles. Tests use this to validate dispatch routing —
    /// the per-format file has its own pack_block. This stays minimal.
    fn q4_k_block(d: f32, dmin: f32, sc_all: u8, m_all: u8, nibble: u8) -> Vec<u8> {
        let mut out = f16::from_f32(d).to_le_bytes().to_vec();
        out.extend_from_slice(&f16::from_f32(dmin).to_le_bytes());
        // Pack scales: sc[0..4] in low 6 bits of q[0..4], m[0..4] in q[4..8],
        // and the j>=4 pairs split across two bytes (see q4_k.rs::pack_scales).
        // For uniform sc/m we can just fill in the simple branch and replicate
        // the high-pair packing.
        let mut q = [0u8; 12];
        for j in 0..4 {
            q[j] = sc_all & 0x3F;
            q[j + 4] = m_all & 0x3F;
        }
        for j in 4..8 {
            let sc_low = sc_all & 0x0F;
            let sc_high = (sc_all >> 4) & 0x03;
            let m_low = m_all & 0x0F;
            let m_high = (m_all >> 4) & 0x03;
            q[j + 4] = (m_low << 4) | sc_low;
            q[j - 4] |= sc_high << 6;
            q[j] |= m_high << 6;
        }
        out.extend_from_slice(&q);
        // 128 bytes of qs: low nibble + high nibble both equal to `nibble`.
        let packed = ((nibble & 0x0F) << 4) | (nibble & 0x0F);
        out.extend(std::iter::repeat_n(packed, 128));
        out
    }

    #[test]
    fn q4_k_dispatch_routes_to_q4_k_arm() {
        // d=1, dmin=0, sc=2, m=0, nibble=3 → every output = 1*2*3 - 0 = 6.
        // Q5_K and Q6_K have different block sizes / payloads, so even an
        // arm-swap would produce a size-mismatch error rather than 6.
        let block = q4_k_block(1.0, 0.0, 2, 0, 3);
        let mut dst = vec![0.0f32; 256];
        dequant_to_f32(GgmlType::Q4_K, &block, &mut dst).unwrap();
        for v in &dst {
            assert!((v - 6.0).abs() < 1e-2, "Q4_K arm: got {v}, want 6");
        }
    }

    #[test]
    fn q5_k_dispatch_routes_to_q5_k_arm() {
        // Q5_K block = 176 bytes. Build by hand:
        // header (4) + scales (12) + qh (32) + qs (128) = 176.
        let mut block = f16::from_f32(1.0).to_le_bytes().to_vec(); // d=1
        block.extend_from_slice(&f16::from_f32(0.0).to_le_bytes()); // dmin=0
        // Reuse Q4_K scales packing: sc=2, m=0 for all sub-blocks.
        block.extend_from_slice(&q4_k_block(1.0, 0.0, 2, 0, 0)[4..16]);
        // qh = 0 → no high bit set, all 5-bit values stay in low 4 bits.
        block.extend(std::iter::repeat_n(0u8, 32));
        // qs: low+high nibbles = 1 → 5-bit value 1.
        block.extend(std::iter::repeat_n(0b0001_0001u8, 128));
        let mut dst = vec![0.0f32; 256];
        dequant_to_f32(GgmlType::Q5_K, &block, &mut dst).unwrap();
        // dst = d * sc * q5 = 1 * 2 * 1 = 2.
        for v in &dst {
            assert!((v - 2.0).abs() < 1e-2, "Q5_K arm: got {v}, want 2");
        }
    }

    #[test]
    fn q6_k_dispatch_routes_to_q6_k_arm() {
        // Q6_K block = 210 bytes. ql[128] + qh[64] + scales[16] (i8) + d (f16).
        // All ql/qh = 0 → 6-bit value 0; after the -32 centering, q = -32.
        // Scales = 1, d = 1 → dst = 1 * 1 * (-32) = -32.
        let mut block = vec![0u8; 128]; // ql
        block.extend_from_slice(&[0u8; 64]); // qh
        block.extend(std::iter::repeat_n(1u8, 16)); // scales (signed i8 = 1)
        block.extend_from_slice(&f16::from_f32(1.0).to_le_bytes());
        let mut dst = vec![0.0f32; 256];
        dequant_to_f32(GgmlType::Q6_K, &block, &mut dst).unwrap();
        for v in &dst {
            assert!((v - (-32.0)).abs() < 1e-2, "Q6_K arm: got {v}, want -32");
        }
    }
}
