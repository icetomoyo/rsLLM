//! Q4_K dequantization (4-bit K-quants).
//!
//! Ported by reference from `ggml-quants.c` (`block_q4_K` /
//! `dequantize_row_q4_K` / `get_scale_min_k4`, MIT, ggml authors).
//!
//! ## Block layout (256 elements / 144 bytes)
//!
//! ```text
//! struct block_q4_K {
//!     fp16    d;           // 2 bytes — super-block scale for quantized scales
//!     fp16    dmin;        // 2 bytes — super-block scale for quantized mins
//!     uint8_t scales[12];  // 12 bytes — packed 6-bit scales (8) + mins (8)
//!     uint8_t qs[128];     // 128 bytes — 256 4-bit quants (2 per byte)
//! }
//! ```
//!
//! Each super-block holds 8 sub-blocks of 32 elements. For sub-block `j`:
//!
//! - `sc_j` = 6-bit quantized scale (range 0..63), extracted from `scales[12]`
//! - `m_j`  = 6-bit quantized min   (range 0..63), extracted from `scales[12]`
//! - real scale  = `d * sc_j`
//! - real min    = `dmin * m_j`
//! - `dst[i]` = real_scale * nibble[i] − real_min
//!
//! `get_scale_min_k4` performs the bit-fiddling that unpacks 8 × (6+6) =
//! 96 bits out of `scales[12]` = 96 bits exactly.

use half::f16;

use crate::error::Error;

const ELEMENTS_PER_BLOCK: usize = 256;
const BYTES_PER_BLOCK: usize = 144;

pub(crate) fn dequant_q4_k(src: &[u8], dst: &mut [f32]) -> Result<(), Error> {
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

        let qs_off = block_off + 16;
        let mut out_off = b * ELEMENTS_PER_BLOCK;
        let mut q_off = 0;
        let mut is = 0;

        // 4 iterations × 64 elements = 256 total. Each iteration handles two
        // 32-element sub-blocks (low-nibble pass + high-nibble pass).
        for _ in 0..4 {
            let (sc1, m1) = super::q4_k_scale_min(is, &scales);
            let (sc2, m2) = super::q4_k_scale_min(is + 1, &scales);
            let d1 = d * f32::from(sc1);
            let m1f = dmin * f32::from(m1);
            let d2 = d * f32::from(sc2);
            let m2f = dmin * f32::from(m2);

            // Low nibbles → first 32 outputs of this 64-element chunk.
            for l in 0..32 {
                let q = src[qs_off + q_off + l] & 0x0F;
                dst[out_off + l] = d1 * f32::from(q) - m1f;
            }
            // High nibbles → next 32 outputs.
            for l in 0..32 {
                let q = src[qs_off + q_off + l] >> 4;
                dst[out_off + 32 + l] = d2 * f32::from(q) - m2f;
            }

            out_off += 64;
            q_off += 32;
            is += 2;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Inverse of `get_scale_min_k4`: pack 8 (sc, m) pairs of 6-bit values
    /// into 12 bytes. Useful for synthesizing test blocks with known values.
    fn pack_scales(sc: [u8; 8], m: [u8; 8]) -> [u8; 12] {
        // Verify inputs are within 6-bit range.
        for &v in sc.iter().chain(m.iter()) {
            assert!(v <= 63, "scale/min must fit in 6 bits, got {v}");
        }

        let mut q = [0u8; 12];
        // j < 4: low 6 bits of q[j] / q[j+4].
        q[..4].copy_from_slice(&sc[..4]);
        q[4..8].copy_from_slice(&m[..4]);
        // j >= 4: split each 6-bit value across two bytes.
        for j in 4..8 {
            let sc_low = sc[j] & 0x0F;
            let sc_high = (sc[j] >> 4) & 0x03;
            let m_low = m[j] & 0x0F;
            let m_high = (m[j] >> 4) & 0x03;

            q[j + 4] = (m_low << 4) | sc_low;
            // sc_high → q[j-4] bits 6-7
            q[j - 4] |= sc_high << 6;
            // m_high → q[j] bits 6-7
            q[j] |= m_high << 6;
        }
        q
    }

    /// Pack one Q4_K block: header (d, dmin) + scales + 256 nibbles.
    fn pack_block(d: f32, dmin: f32, sc: [u8; 8], m: [u8; 8], nibbles: [u8; 256]) -> Vec<u8> {
        let mut out = f16::from_f32(d).to_le_bytes().to_vec();
        out.extend_from_slice(&f16::from_f32(dmin).to_le_bytes());
        out.extend_from_slice(&pack_scales(sc, m));

        // qs[128]: low nibble → element l; high nibble → element l + 32.
        // The on-disk layout pairs sub-block 2k (low) with sub-block 2k+1 (high).
        // Concretely: for each pair (k=0..3), bytes [k*32..k*32+32] hold
        //   low nibble = nibble[(2k)*32 + l],  high = nibble[(2k+1)*32 + l].
        let mut qs = [0u8; 128];
        for k in 0..4 {
            for l in 0..32 {
                let lo = nibbles[(2 * k) * 32 + l] & 0x0F;
                let hi = nibbles[(2 * k + 1) * 32 + l] & 0x0F;
                qs[k * 32 + l] = (hi << 4) | lo;
            }
        }
        out.extend_from_slice(&qs);
        out
    }

    #[test]
    fn block_size_matches_ds4_table() {
        assert_eq!(BYTES_PER_BLOCK, 144);
        assert_eq!(ELEMENTS_PER_BLOCK, 256);
    }

    #[test]
    fn pack_unpack_scales_roundtrip() {
        // Cover both j<4 and j>=4 branches, plus boundary values.
        let sc = [0u8, 1, 31, 63, 0, 1, 31, 63];
        let m = [63u8, 31, 1, 0, 63, 31, 1, 0];
        let packed = pack_scales(sc, m);
        for j in 0..8 {
            let (g_sc, g_m) = super::super::q4_k_scale_min(j, &packed);
            assert_eq!(g_sc, sc[j], "sc[{j}]");
            assert_eq!(g_m, m[j], "m[{j}]");
        }
    }

    #[test]
    fn zero_quants_yields_minus_min_per_subblock() {
        // d = 1, dmin = 1, sc all 0, m varying.
        // With nibble=0: dst = d*sc*0 - dmin*m = -m for each sub-block.
        let mut m = [0u8; 8];
        for (i, slot) in m.iter_mut().enumerate() {
            *slot = (i + 1) as u8;
        }
        let block = pack_block(1.0, 1.0, [0u8; 8], m, [0u8; 256]);
        let mut dst = vec![0.0f32; 256];
        dequant_q4_k(&block, &mut dst).unwrap();
        for sub in 0..8 {
            for l in 0..32 {
                let want = -f32::from(m[sub]);
                let got = dst[sub * 32 + l];
                assert!(
                    (got - want).abs() < 1e-3,
                    "sub={sub} l={l}: got {got}, want {want}"
                );
            }
        }
    }

    #[test]
    fn distinct_sub_block_scales() {
        // d = 1, dmin = 0; sc[j] = j+1, m = 0; nibble = 1 everywhere.
        // → dst = (j+1) * 1 = j+1 for every element in sub-block j.
        let sc: [u8; 8] = std::array::from_fn(|j| (j + 1) as u8);
        let m = [0u8; 8];
        let block = pack_block(1.0, 0.0, sc, m, [1u8; 256]);
        let mut dst = vec![0.0f32; 256];
        dequant_q4_k(&block, &mut dst).unwrap();
        for sub in 0..8 {
            let want = f32::from(sc[sub]);
            for l in 0..32 {
                let got = dst[sub * 32 + l];
                assert!(
                    (got - want).abs() < 1e-2,
                    "sub={sub} got {got}, want {want}"
                );
            }
        }
    }

    #[test]
    fn linear_with_offset_combined() {
        // d = 0.5, dmin = 0.25; sc[j] = 4, m[j] = 2;
        // real_scale = 0.5 * 4 = 2; real_min = 0.25 * 2 = 0.5
        // dst = 2 * nibble - 0.5
        let sc = [4u8; 8];
        let m = [2u8; 8];
        let mut nibbles = [0u8; 256];
        for (i, n) in nibbles.iter_mut().enumerate() {
            *n = (i % 16) as u8;
        }
        let block = pack_block(0.5, 0.25, sc, m, nibbles);
        let mut dst = vec![0.0f32; 256];
        dequant_q4_k(&block, &mut dst).unwrap();
        for i in 0..256 {
            let want = 2.0 * f32::from(nibbles[i]) - 0.5;
            let got = dst[i];
            assert!((got - want).abs() < 1e-2, "i={i}: got {got}, want {want}");
        }
    }

    #[test]
    fn wrong_src_size_errors() {
        let src = vec![0u8; 143];
        let mut dst = vec![0.0f32; 256];
        assert!(matches!(
            dequant_q4_k(&src, &mut dst),
            Err(Error::DequantSizeMismatch { .. })
        ));
    }

    #[test]
    fn partial_block_dst_errors() {
        let src = vec![0u8; 144];
        let mut dst = vec![0.0f32; 128];
        assert!(matches!(
            dequant_q4_k(&src, &mut dst),
            Err(Error::DequantSizeMismatch { .. })
        ));
    }
}
