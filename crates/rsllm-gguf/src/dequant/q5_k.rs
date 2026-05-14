//! Q5_K dequantization (5-bit K-quants).
//!
//! Ported by reference from `ggml-quants.c` (`block_q5_K` /
//! `dequantize_row_q5_K`, MIT, ggml authors).
//!
//! ## Block layout (256 elements / 176 bytes)
//!
//! ```text
//! struct block_q5_K {
//!     fp16    d;           // 2 bytes
//!     fp16    dmin;        // 2 bytes
//!     uint8_t scales[12];  // 12 bytes — same packing as Q4_K
//!     uint8_t qh[32];      // 32 bytes — high bit per element (256 bits)
//!     uint8_t qs[128];     // 128 bytes — low 4 bits per element
//! }
//! ```
//!
//! Each logical 5-bit value `q[i]` is the concatenation of:
//! - the 4 low bits from `qs` (same packing as Q4_K)
//! - 1 high bit from `qh`
//!
//! The bit assignment in `qh[l]` advances by 2 bits per outer-loop iteration:
//! sub-block pair `(0,1)` uses `u1 = bit 0`, `u2 = bit 1`; pair `(2,3)` uses
//! `bit 2 / bit 3`; etc. After 4 iterations all 8 bits of every `qh[l]` are
//! consumed (32 bytes × 8 = 256 bits = one bit per element).

use half::f16;

use crate::error::Error;

const ELEMENTS_PER_BLOCK: usize = 256;
const BYTES_PER_BLOCK: usize = 176;

pub(crate) fn dequant_q5_k(src: &[u8], dst: &mut [f32]) -> Result<(), Error> {
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
        let d = f16::from_le_bytes([src[block_off], src[block_off + 1]]).to_f32();
        let dmin = f16::from_le_bytes([src[block_off + 2], src[block_off + 3]]).to_f32();

        let mut scales = [0u8; 12];
        scales.copy_from_slice(&src[block_off + 4..block_off + 16]);

        let qh_off = block_off + 16;
        let qs_off = block_off + 16 + 32;

        let mut out_off = b * ELEMENTS_PER_BLOCK;
        let mut q_off = 0;
        let mut is = 0;
        let mut u1: u8 = 1;
        let mut u2: u8 = 2;

        for _ in 0..4 {
            let (sc1, m1) = super::q4_k_scale_min(is, &scales);
            let (sc2, m2) = super::q4_k_scale_min(is + 1, &scales);
            let d1 = d * f32::from(sc1);
            let m1f = dmin * f32::from(m1);
            let d2 = d * f32::from(sc2);
            let m2f = dmin * f32::from(m2);

            // Low nibble + (u1 high bit).
            for l in 0..32 {
                let lo = src[qs_off + q_off + l] & 0x0F;
                let hi_bit = if src[qh_off + l] & u1 != 0 { 16 } else { 0 };
                dst[out_off + l] = d1 * f32::from(lo + hi_bit) - m1f;
            }
            // High nibble + (u2 high bit).
            for l in 0..32 {
                let lo = src[qs_off + q_off + l] >> 4;
                let hi_bit = if src[qh_off + l] & u2 != 0 { 16 } else { 0 };
                dst[out_off + 32 + l] = d2 * f32::from(lo + hi_bit) - m2f;
            }

            out_off += 64;
            q_off += 32;
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pack_scales(sc: [u8; 8], m: [u8; 8]) -> [u8; 12] {
        // Reuse Q4_K's pack logic (inverse of `q4_k_scale_min`).
        let mut q = [0u8; 12];
        q[..4].copy_from_slice(&sc[..4]);
        q[4..8].copy_from_slice(&m[..4]);
        for j in 4..8 {
            let sc_low = sc[j] & 0x0F;
            let sc_high = (sc[j] >> 4) & 0x03;
            let m_low = m[j] & 0x0F;
            let m_high = (m[j] >> 4) & 0x03;
            q[j + 4] = (m_low << 4) | sc_low;
            q[j - 4] |= sc_high << 6;
            q[j] |= m_high << 6;
        }
        q
    }

    /// Pack one Q5_K block.
    /// `q5[i]` is the 5-bit value for element i (range 0..32).
    fn pack_block(d: f32, dmin: f32, sc: [u8; 8], m: [u8; 8], q5: [u8; 256]) -> Vec<u8> {
        let mut out = f16::from_f32(d).to_le_bytes().to_vec();
        out.extend_from_slice(&f16::from_f32(dmin).to_le_bytes());
        out.extend_from_slice(&pack_scales(sc, m));

        // qh[32]: bit b of qh[l] is the high bit for some element. The
        // mapping follows the dequant loop: sub-block pair (2k, 2k+1) uses
        //   u1 = 1 << (2k)  → bit 2k for elements (2k)*32 + l
        //   u2 = 1 << (2k+1)→ bit 2k+1 for elements (2k+1)*32 + l
        // i.e. `l` is the byte index, the sub-block determines which bit.
        let mut qh = [0u8; 32];
        for k in 0..4 {
            for (l, slot) in qh.iter_mut().enumerate() {
                let lo_elem = (2 * k) * 32 + l;
                let hi_elem = (2 * k + 1) * 32 + l;
                if q5[lo_elem] >= 16 {
                    *slot |= 1 << (2 * k);
                }
                if q5[hi_elem] >= 16 {
                    *slot |= 1 << (2 * k + 1);
                }
            }
        }
        out.extend_from_slice(&qh);

        // qs[128]: low 4 bits of each element (same packing as Q4_K).
        let mut qs = [0u8; 128];
        for k in 0..4 {
            for l in 0..32 {
                let lo = q5[(2 * k) * 32 + l] & 0x0F;
                let hi = q5[(2 * k + 1) * 32 + l] & 0x0F;
                qs[k * 32 + l] = (hi << 4) | lo;
            }
        }
        out.extend_from_slice(&qs);
        out
    }

    #[test]
    fn block_size_matches_ds4_table() {
        assert_eq!(BYTES_PER_BLOCK, 176);
        assert_eq!(ELEMENTS_PER_BLOCK, 256);
    }

    #[test]
    fn five_bit_range_zero_min() {
        // d = 1, dmin = 0; sc[j] = 1, m = 0; q5 covers 0..32.
        let sc = [1u8; 8];
        let m = [0u8; 8];
        let mut q5 = [0u8; 256];
        for (i, v) in q5.iter_mut().enumerate() {
            *v = (i % 32) as u8;
        }
        let block = pack_block(1.0, 0.0, sc, m, q5);
        let mut dst = vec![0.0f32; 256];
        dequant_q5_k(&block, &mut dst).unwrap();
        for i in 0..256 {
            let want = f32::from(q5[i]);
            let got = dst[i];
            assert!((got - want).abs() < 1e-2, "i={i}: got {got}, want {want}");
        }
    }

    #[test]
    fn full_range_with_offset() {
        // d = 0.5, dmin = 0.5; sc[j] = 2, m[j] = 1.
        // real_scale = 1.0, real_min = 0.5; dst = q5 - 0.5.
        let block = pack_block(0.5, 0.5, [2u8; 8], [1u8; 8], [31u8; 256]);
        let mut dst = vec![0.0f32; 256];
        dequant_q5_k(&block, &mut dst).unwrap();
        for v in &dst {
            assert!((v - 30.5).abs() < 1e-2, "got {v}, want 30.5");
        }
    }

    #[test]
    fn wrong_src_size_errors() {
        let src = vec![0u8; 175];
        let mut dst = vec![0.0f32; 256];
        assert!(matches!(
            dequant_q5_k(&src, &mut dst),
            Err(Error::DequantSizeMismatch { .. })
        ));
    }
}
