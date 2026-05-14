//! Q6_K dequantization (6-bit K-quants).
//!
//! Ported by reference from `ggml-quants.c` (`block_q6_K` /
//! `dequantize_row_q6_K`, MIT, ggml authors).
//!
//! ## Block layout (256 elements / 210 bytes)
//!
//! ```text
//! struct block_q6_K {
//!     uint8_t ql[128];       // 128 bytes — low 4 bits per element
//!     uint8_t qh[64];        // 64  bytes — high 2 bits per element
//!     int8_t  scales[16];    // 16  bytes — 16 sub-block 8-bit signed scales
//!     fp16    d;             // 2   bytes — super-block scale
//! }
//! ```
//!
//! Note the differences from Q4_K / Q5_K:
//! - 16 sub-blocks of 16 elements (not 8 × 32).
//! - Each scale is a signed `int8_t`, not a packed 6-bit value.
//! - There is no `dmin` — Q6_K is a pure scale-only quantization centered
//!   around zero (every 6-bit nibble is biased by `-32`).
//!
//! Each block is processed in two 128-element halves. Within each half, four
//! 32-element strides interleave through `ql[64]` + `qh[32]` to extract 128
//! 6-bit quants using the 4-bit low and 2-bit high pieces.

use half::f16;

use crate::error::Error;

const ELEMENTS_PER_BLOCK: usize = 256;
const BYTES_PER_BLOCK: usize = 210;

pub(crate) fn dequant_q6_k(src: &[u8], dst: &mut [f32]) -> Result<(), Error> {
    if !dst.len().is_multiple_of(ELEMENTS_PER_BLOCK) {
        return Err(Error::DequantSizeMismatch {
            src_bytes: src.len(),
            expected_bytes: dst.len().div_ceil(ELEMENTS_PER_BLOCK) * BYTES_PER_BLOCK,
        });
    }
    let n_blocks = dst.len() / ELEMENTS_PER_BLOCK;
    let expected = n_blocks * BYTES_PER_BLOCK;
    if src.len() != expected {
        return Err(Error::DequantSizeMismatch {
            src_bytes: src.len(),
            expected_bytes: expected,
        });
    }

    for b in 0..n_blocks {
        let block_off = b * BYTES_PER_BLOCK;
        let ql_off = block_off; // 128 bytes
        let qh_off = block_off + 128; // 64 bytes
        let sc_off = block_off + 128 + 64; // 16 bytes
        let d_off = block_off + 128 + 64 + 16; // 2 bytes

        let d = f16::from_le_bytes([src[d_off], src[d_off + 1]]).to_f32();

        // Each block is decoded as two 128-element halves.
        let mut out_off = b * ELEMENTS_PER_BLOCK;
        let mut ql_local = ql_off;
        let mut qh_local = qh_off;
        let mut sc_local = sc_off;

        for _half in 0..2 {
            // Process 128 elements per half (32 elements × 4 strides).
            for l in 0..32 {
                let is = l / 16;
                let q1 = (((src[ql_local + l] & 0x0F) as i32)
                    | ((src[qh_local + l] as i32 & 0x03) << 4))
                    - 32;
                let q2 = (((src[ql_local + l + 32] & 0x0F) as i32)
                    | (((src[qh_local + l] >> 2) as i32 & 0x03) << 4))
                    - 32;
                let q3 = (((src[ql_local + l] >> 4) as i32)
                    | (((src[qh_local + l] >> 4) as i32 & 0x03) << 4))
                    - 32;
                let q4 = (((src[ql_local + l + 32] >> 4) as i32)
                    | (((src[qh_local + l] >> 6) as i32 & 0x03) << 4))
                    - 32;

                // `scales` are signed bytes; reinterpret each u8 as i8.
                let sc0 = src[sc_local + is] as i8;
                let sc2 = src[sc_local + is + 2] as i8;
                let sc4 = src[sc_local + is + 4] as i8;
                let sc6 = src[sc_local + is + 6] as i8;

                dst[out_off + l] = d * f32::from(sc0) * (q1 as f32);
                dst[out_off + l + 32] = d * f32::from(sc2) * (q2 as f32);
                dst[out_off + l + 64] = d * f32::from(sc4) * (q3 as f32);
                dst[out_off + l + 96] = d * f32::from(sc6) * (q4 as f32);
            }

            out_off += 128;
            ql_local += 64;
            qh_local += 32;
            sc_local += 8;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pack one Q6_K block from raw inputs.
    ///
    /// `q6[i]` is the 6-bit value for element i (range 0..64).
    /// `scales[i]` is the i-th sub-block's 8-bit signed scale.
    fn pack_block(q6: [u8; 256], scales: [i8; 16], d: f32) -> Vec<u8> {
        let mut out = vec![0u8; BYTES_PER_BLOCK];

        // Compute ql/qh from q6 using the inverse of the dequant indexing.
        // The dequant loop reads, per (half h, stride l in 0..32):
        //   q1 = ql[h*64 + l]      low nibble + qh[h*32 + l] bits 0-1 → out[h*128 + l]
        //   q2 = ql[h*64 + l + 32] low nibble + qh[h*32 + l] bits 2-3 → out[h*128 + l + 32]
        //   q3 = ql[h*64 + l]      high nibble + qh[h*32 + l] bits 4-5 → out[h*128 + l + 64]
        //   q4 = ql[h*64 + l + 32] high nibble + qh[h*32 + l] bits 6-7 → out[h*128 + l + 96]
        // We invert that mapping below.
        for h in 0..2 {
            for l in 0..32 {
                let v1 = q6[h * 128 + l];
                let v2 = q6[h * 128 + l + 32];
                let v3 = q6[h * 128 + l + 64];
                let v4 = q6[h * 128 + l + 96];

                // Low nibbles in ql[h*64 + l] and ql[h*64 + l + 32].
                out[h * 64 + l] = v1 & 0x0F;
                out[h * 64 + l + 32] = v2 & 0x0F;
                // High nibbles: OR into the same bytes.
                out[h * 64 + l] |= (v3 & 0x0F) << 4;
                out[h * 64 + l + 32] |= (v4 & 0x0F) << 4;
                // qh: bits 0-1, 2-3, 4-5, 6-7 for v1, v2, v3, v4.
                let qh_byte = ((v1 >> 4) & 0x03)
                    | (((v2 >> 4) & 0x03) << 2)
                    | (((v3 >> 4) & 0x03) << 4)
                    | (((v4 >> 4) & 0x03) << 6);
                out[128 + h * 32 + l] = qh_byte;
            }
        }

        // 16 signed scales at offset 192.
        for (i, &s) in scales.iter().enumerate() {
            out[192 + i] = s as u8;
        }

        // 2-byte LE f16 d at offset 208.
        let d_bytes = f16::from_f32(d).to_le_bytes();
        out[208] = d_bytes[0];
        out[209] = d_bytes[1];

        out
    }

    #[test]
    fn block_size_matches_ds4_table() {
        assert_eq!(BYTES_PER_BLOCK, 210);
        assert_eq!(ELEMENTS_PER_BLOCK, 256);
    }

    #[test]
    fn zero_quants_yields_zero() {
        // All quants = 32 (the "zero" point in Q6_K); any scale; dst should be 0.
        let block = pack_block([32u8; 256], [5i8; 16], 1.0);
        let mut dst = vec![0.0f32; 256];
        dequant_q6_k(&block, &mut dst).unwrap();
        for v in &dst {
            assert!(v.abs() < 1e-3, "got {v}, want 0");
        }
    }

    #[test]
    fn unit_scale_centered_dequant() {
        // d = 1, all scales = 1; dst[i] = q6[i] - 32.
        let mut q6 = [0u8; 256];
        for (i, v) in q6.iter_mut().enumerate() {
            *v = (i % 64) as u8;
        }
        let block = pack_block(q6, [1i8; 16], 1.0);
        let mut dst = vec![0.0f32; 256];
        dequant_q6_k(&block, &mut dst).unwrap();
        for i in 0..256 {
            let want = f32::from(q6[i]) - 32.0;
            assert!(
                (dst[i] - want).abs() < 1e-3,
                "i={i}: got {}, want {want}",
                dst[i]
            );
        }
    }

    #[test]
    fn negative_scales_invert_sign() {
        // All quants = 33 (one above zero); scale = -1; dst = -1.
        let block = pack_block([33u8; 256], [-1i8; 16], 1.0);
        let mut dst = vec![0.0f32; 256];
        dequant_q6_k(&block, &mut dst).unwrap();
        for v in &dst {
            assert!((v - (-1.0)).abs() < 1e-3, "got {v}, want -1");
        }
    }

    #[test]
    fn distinct_per_sub_block_scales() {
        // Q6_K has 16 sub-blocks of 16 elements each. Use distinct scale
        // per sub-block to detect any sub-block-ordering bug: scale[i] = i+1.
        // d = 1, quants all = 33 (one above zero = +1 after centering).
        // → for element e in sub-block i: dst[e] = 1 * (i+1) * 1 = i+1.
        let mut scales = [0i8; 16];
        for (i, s) in scales.iter_mut().enumerate() {
            *s = (i + 1) as i8;
        }
        let block = pack_block([33u8; 256], scales, 1.0);
        let mut dst = vec![0.0f32; 256];
        dequant_q6_k(&block, &mut dst).unwrap();
        for sub in 0..16 {
            let want = (sub + 1) as f32;
            for l in 0..16 {
                let got = dst[sub * 16 + l];
                assert!(
                    (got - want).abs() < 1e-3,
                    "sub={sub} l={l} got {got} want {want}"
                );
            }
        }
    }

    #[test]
    fn distinct_per_sub_block_quants() {
        // Inverse coverage: hold scales uniform (=1), but vary the quant
        // value per sub-block. quant[sub_i, l] = sub_i * 2 (still in 0..64).
        // After centering: q - 32 → 2*sub_i - 32. So dst = d * 1 * (2*sub_i - 32).
        let mut q6 = [0u8; 256];
        for sub in 0..16 {
            for l in 0..16 {
                q6[sub * 16 + l] = (sub * 2) as u8;
            }
        }
        let block = pack_block(q6, [1i8; 16], 1.0);
        let mut dst = vec![0.0f32; 256];
        dequant_q6_k(&block, &mut dst).unwrap();
        for sub in 0..16 {
            let want = (2 * sub as i32 - 32) as f32;
            for l in 0..16 {
                let got = dst[sub * 16 + l];
                assert!(
                    (got - want).abs() < 1e-3,
                    "sub={sub} l={l} got {got} want {want}"
                );
            }
        }
    }

    #[test]
    fn wrong_src_size_errors() {
        let src = vec![0u8; 209];
        let mut dst = vec![0.0f32; 256];
        assert!(matches!(
            dequant_q6_k(&src, &mut dst),
            Err(Error::DequantSizeMismatch { .. })
        ));
    }
}
