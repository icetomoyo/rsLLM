//! Q2_K dequantization (2-bit K-quants).
//!
//! Used by DeepSeek V4 Flash for the **routed MoE down expert** weights
//! (see [v0.1.0 design](../../../../docs/features/v0.1.0.md#feature_002-gguf-文件解析器)).
//!
//! Ported by reference from `ggml-quants.c` (`block_q2_K` /
//! `dequantize_row_q2_K`, MIT, ggml authors).
//!
//! ## Block layout (256 elements / 84 bytes)
//!
//! ```text
//! struct block_q2_K {
//!     uint8_t scales[16];  // 16 bytes — packed 4-bit scale (low) + 4-bit min (high) per sub-block
//!     uint8_t qs[64];      // 64 bytes — 256 × 2-bit quants (4 per byte)
//!     fp16    d;           // 2 bytes — super-block scale for quantized scales
//!     fp16    dmin;        // 2 bytes — super-block scale for quantized mins
//! }
//! ```
//!
//! Each super-block holds 16 sub-blocks of 16 elements. For sub-block `j`:
//!
//! - `sc_j` = `scales[j] & 0xF` (4-bit scale, range 0..15)
//! - `m_j`  = `scales[j] >> 4`  (4-bit min,   range 0..15)
//! - real scale = `d * sc_j`
//! - real min   = `dmin * m_j`
//! - `dst[i]`   = real_scale × q_2bit_i − real_min
//!
//! The 2-bit packing in `qs[64]` interleaves 4 sub-blocks across the same
//! byte indices via different bit shifts (0 / 2 / 4 / 6), and the dequant
//! loop walks two halves of 128 elements each — see `pack_block` in tests
//! for the inverse.

use half::f16;

use crate::error::Error;

const ELEMENTS_PER_BLOCK: usize = 256;
const BYTES_PER_BLOCK: usize = 84;

pub(crate) fn dequant_q2_k(src: &[u8], dst: &mut [f32]) -> Result<(), Error> {
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
        let scales_off = block_off; // 16 bytes
        let qs_off = block_off + 16; // 64 bytes
        let d_off = block_off + 16 + 64; // 2 bytes f16 d
        let dmin_off = block_off + 16 + 64 + 2; // 2 bytes f16 dmin

        let d = f16::from_le_bytes([src[d_off], src[d_off + 1]]).to_f32();
        let dmin = f16::from_le_bytes([src[dmin_off], src[dmin_off + 1]]).to_f32();

        let mut out_off = b * ELEMENTS_PER_BLOCK;
        let mut q_off = qs_off;
        let mut is = 0usize;

        // 2 halves × 128 elements each = 256 total.
        // Each half consumes 32 bytes of qs and produces 128 outputs via
        // 4 shifts (0, 2, 4, 6), with each shift handling 32 elements
        // split into two 16-element sub-blocks.
        for _ in 0..2 {
            for shift in (0..8).step_by(2) {
                let sc1 = src[scales_off + is];
                let dl1 = d * f32::from(sc1 & 0x0F);
                let ml1 = dmin * f32::from(sc1 >> 4);
                for l in 0..16 {
                    let q = i32::from((src[q_off + l] >> shift) & 0x03);
                    dst[out_off + l] = dl1 * (q as f32) - ml1;
                }

                let sc2 = src[scales_off + is + 1];
                let dl2 = d * f32::from(sc2 & 0x0F);
                let ml2 = dmin * f32::from(sc2 >> 4);
                for l in 0..16 {
                    let q = i32::from((src[q_off + l + 16] >> shift) & 0x03);
                    dst[out_off + l + 16] = dl2 * (q as f32) - ml2;
                }

                out_off += 32;
                is += 2;
            }
            q_off += 32;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pack one Q2_K block from raw inputs.
    ///
    /// `sc[16]` = 4-bit scale per sub-block (0..15)
    /// `m[16]`  = 4-bit min   per sub-block (0..15)
    /// `q2[256]` = 2-bit quant per element (0..3)
    fn pack_block(d: f32, dmin: f32, sc: [u8; 16], m: [u8; 16], q2: [u8; 256]) -> Vec<u8> {
        // Verify inputs.
        for &v in sc.iter().chain(m.iter()) {
            assert!(v <= 15, "sc/m must fit in 4 bits, got {v}");
        }
        for &v in q2.iter() {
            assert!(v <= 3, "q2 must fit in 2 bits, got {v}");
        }

        let mut out = vec![0u8; BYTES_PER_BLOCK];

        // Scales: low 4 bits = scale, high 4 bits = min.
        for j in 0..16 {
            out[j] = (m[j] << 4) | (sc[j] & 0x0F);
        }

        // qs: invert the dequant's interleaved 2-bit packing.
        // Dequant uses:
        //   out element `out_off + l` (l in 0..16) reads `qs[q_off + l] >> shift`
        //   out element `out_off + l + 16` reads `qs[q_off + l + 16] >> shift`
        // out_off advances by 32 per shift, q_off advances by 32 per half.
        //
        // So for half h (0..2), shift index k (0..4), l (0..16):
        //   sub-block index in scales: is = h*8 + k*2  -> first 16 outs
        //   sub-block index in scales: is = h*8 + k*2 + 1  -> second 16 outs
        //   global output index = h*128 + k*32 + l (first 16) or +16 (second 16)
        //   qs byte index = h*32 + l (first 16) or +16 (second 16)
        //   bit position in byte: shift = k*2
        for h in 0..2 {
            for k in 0..4 {
                let shift = (k * 2) as u8;
                for l in 0..16 {
                    let elem1 = h * 128 + k * 32 + l;
                    let elem2 = h * 128 + k * 32 + l + 16;
                    let byte1 = h * 32 + l;
                    let byte2 = h * 32 + l + 16;
                    out[16 + byte1] |= (q2[elem1] & 0x03) << shift;
                    out[16 + byte2] |= (q2[elem2] & 0x03) << shift;
                }
            }
        }

        // d and dmin at end.
        let d_bytes = f16::from_f32(d).to_le_bytes();
        let dmin_bytes = f16::from_f32(dmin).to_le_bytes();
        out[80] = d_bytes[0];
        out[81] = d_bytes[1];
        out[82] = dmin_bytes[0];
        out[83] = dmin_bytes[1];

        out
    }

    #[test]
    fn block_size_matches_spec() {
        assert_eq!(BYTES_PER_BLOCK, 84);
        assert_eq!(ELEMENTS_PER_BLOCK, 256);
    }

    #[test]
    fn zero_quants_yields_minus_min_per_sub_block() {
        // d = 1, dmin = 1, sc = 0; mins varying; q2 = 0 everywhere.
        // dst = d * sc * 0 - dmin * m = -m for each sub-block.
        // m is 4-bit (0..15), so use values 0..15 (one per sub-block).
        let m: [u8; 16] = std::array::from_fn(|i| i as u8);
        let block = pack_block(1.0, 1.0, [0u8; 16], m, [0u8; 256]);
        let mut dst = vec![0.0f32; 256];
        dequant_q2_k(&block, &mut dst).unwrap();
        for sub in 0..16 {
            for l in 0..16 {
                let want = -f32::from(m[sub]);
                let got = dst[sub * 16 + l];
                assert!(
                    (got - want).abs() < 1e-3,
                    "sub={sub} l={l}: got {got}, want {want}"
                );
            }
        }
    }

    #[test]
    fn distinct_per_sub_block_scales() {
        // d = 1, dmin = 0, sc[j] = j, m = 0, q2 = 1 everywhere.
        // dst = 1 * j * 1 - 0 = j for every element in sub-block j.
        // sc is 4-bit (0..15), so we use j directly.
        let sc: [u8; 16] = std::array::from_fn(|j| j as u8);
        let m = [0u8; 16];
        let block = pack_block(1.0, 0.0, sc, m, [1u8; 256]);
        let mut dst = vec![0.0f32; 256];
        dequant_q2_k(&block, &mut dst).unwrap();
        for sub in 0..16 {
            let want = f32::from(sc[sub]);
            for l in 0..16 {
                let got = dst[sub * 16 + l];
                assert!(
                    (got - want).abs() < 1e-2,
                    "sub={sub} got {got}, want {want}"
                );
            }
        }
    }

    #[test]
    fn full_2bit_range_with_offset() {
        // d = 0.5, dmin = 0.25; sc = 4, m = 2.
        // real_scale = 0.5*4 = 2, real_min = 0.25*2 = 0.5
        // dst = 2 * q2 - 0.5; for q2 in {0,1,2,3} -> {-0.5, 1.5, 3.5, 5.5}
        let sc = [4u8; 16];
        let m = [2u8; 16];
        let mut q2 = [0u8; 256];
        for (i, slot) in q2.iter_mut().enumerate() {
            *slot = (i % 4) as u8;
        }
        let block = pack_block(0.5, 0.25, sc, m, q2);
        let mut dst = vec![0.0f32; 256];
        dequant_q2_k(&block, &mut dst).unwrap();
        for i in 0..256 {
            let want = 2.0 * f32::from(q2[i]) - 0.5;
            let got = dst[i];
            assert!((got - want).abs() < 1e-2, "i={i}: got {got}, want {want}");
        }
    }

    #[test]
    fn distinct_sc_and_m_combined() {
        // The two existing distinct-* tests each fix one of sc/m to zero so
        // they cannot catch a bug that mixes up the scale-nibble and the
        // min-nibble (e.g. swapping the `& 0x0F` and `>> 4` masks). This test
        // varies both simultaneously across all 16 sub-blocks.
        //
        // d = 1, dmin = 1; q2 = 2 everywhere.
        // sc[j] = (j % 4) + 1 -> values cycle through {1,2,3,4}
        // m[j]  = ((15 - j) % 4) + 1 -> values cycle inverse
        // dst[sub j] = 1 * sc[j] * 2 - 1 * m[j] = 2*sc[j] - m[j]
        let sc: [u8; 16] = std::array::from_fn(|j| ((j % 4) + 1) as u8);
        let m: [u8; 16] = std::array::from_fn(|j| (((15 - j) % 4) + 1) as u8);
        let block = pack_block(1.0, 1.0, sc, m, [2u8; 256]);
        let mut dst = vec![0.0f32; 256];
        dequant_q2_k(&block, &mut dst).unwrap();
        for sub in 0..16 {
            let want = 2.0 * f32::from(sc[sub]) - f32::from(m[sub]);
            for l in 0..16 {
                let got = dst[sub * 16 + l];
                assert!(
                    (got - want).abs() < 1e-3,
                    "sub={sub} l={l} sc={} m={}: got {got}, want {want}",
                    sc[sub],
                    m[sub]
                );
            }
        }
    }

    #[test]
    fn wrong_src_size_errors() {
        let src = vec![0u8; 83];
        let mut dst = vec![0.0f32; 256];
        assert!(matches!(
            dequant_q2_k(&src, &mut dst),
            Err(Error::DequantSizeMismatch { .. })
        ));
    }
}
