//! IQ2_XXS dequantization (~2.06 bits per weight, "Important Quants 2-bit XXS").
//!
//! Used by DeepSeek V4 Flash for the **routed MoE gate/up expert** weights
//! (see [v0.1.0 design](../../../../docs/features/v0.1.0.md#feature_002-gguf-文件解析器)).
//!
//! Ported by reference from `ds4.c:217-297` (MIT, The ds4.c authors), which
//! in turn borrows the grid + signs lookup tables and the decode algorithm
//! from `ggml-quants.c` (MIT, ggml authors). Both ds4 and rsLLM are
//! "format-compatible but code-independent" — we do not link ggml; we
//! reproduce the constants and algorithm under MIT terms.
//!
//! ## Block layout (256 elements / 66 bytes)
//!
//! ```text
//! struct block_iq2_xxs {
//!     fp16     d;       // 2 bytes — super-block scale
//!     uint16_t qs[32];  // 64 bytes — 8 sub-groups × 4 u16 each
//! }
//! ```
//!
//! Each block decodes **8 sub-groups of 32 elements**. Per sub-group `ib32`
//! the parser reads `qs[ib32*4 .. ib32*4+4]` (4 little-endian `u16`s) as two
//! `u32`s `aux32[0]` and `aux32[1]`:
//!
//! - **`aux32[0]`** — the 4 little-endian bytes are 4 grid indices into
//!   `IQ2XXS_GRID`. Each grid entry decodes to 8 elements, so 4 × 8 = 32
//!   elements per sub-group.
//! - **`aux32[1]` bits 0-27** — packed as 4 × 7-bit sign indices into
//!   `KSIGNS_IQ2XS` (sign index `l` lives in bits `7*l .. 7*l+6`).
//! - **`aux32[1]` bits 28-31** — per-sub-group extra exponent.
//!
//! Decoded value: `dst[i] = d × (0.5 + extra) × 0.25 × grid[i] × sign[i]`.

use half::f16;

use crate::error::Error;

mod tables;

use tables::{IQ2XXS_GRID, KMASK_IQ2XS, KSIGNS_IQ2XS};

const ELEMENTS_PER_BLOCK: usize = 256;
const BYTES_PER_BLOCK: usize = 66;
const SUBGROUPS_PER_BLOCK: usize = 8; // 256 / 32
const ELEMENTS_PER_SUBGROUP: usize = 32;

pub(crate) fn dequant_iq2_xxs(src: &[u8], dst: &mut [f32]) -> Result<(), Error> {
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
        let qs_off = block_off + 2;
        let mut out_off = b * ELEMENTS_PER_BLOCK;

        for ib32 in 0..SUBGROUPS_PER_BLOCK {
            // Read 4 u16 (= 8 bytes = 2 u32) for this sub-group.
            let qs_ib32 = qs_off + ib32 * 8;
            let aux32_0 = u32::from_le_bytes([
                src[qs_ib32],
                src[qs_ib32 + 1],
                src[qs_ib32 + 2],
                src[qs_ib32 + 3],
            ]);
            let aux32_1 = u32::from_le_bytes([
                src[qs_ib32 + 4],
                src[qs_ib32 + 5],
                src[qs_ib32 + 6],
                src[qs_ib32 + 7],
            ]);

            // aux8[0..4] = 4 grid indices into IQ2XXS_GRID.
            let aux8 = aux32_0.to_le_bytes();

            // Sub-group scale: db = d × (0.5 + extra) × 0.25
            // where `extra` = top 4 bits of aux32_1.
            let extra = (aux32_1 >> 28) as f32;
            let db = d * (0.5 + extra) * 0.25;

            // 4 inner pattern loads, each emits 8 elements.
            for l in 0..4 {
                let grid_idx = aux8[l] as usize;
                let signs_idx = ((aux32_1 >> (7 * l)) & 0x7F) as usize;
                let grid_pattern = IQ2XXS_GRID[grid_idx];
                let signs = KSIGNS_IQ2XS[signs_idx];

                // The grid entry is a u64 holding 8 i8 values (LE order).
                let grid_bytes = grid_pattern.to_le_bytes();
                for j in 0..8 {
                    // Interpret the grid byte as signed i8 (range -128..127).
                    let g = grid_bytes[j] as i8;
                    let sign = if signs & KMASK_IQ2XS[j] != 0 {
                        -1.0
                    } else {
                        1.0
                    };
                    dst[out_off + l * 8 + j] = db * f32::from(g) * sign;
                }
            }

            out_off += ELEMENTS_PER_SUBGROUP;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pack one IQ2_XXS block from raw inputs.
    ///
    /// `d` = super-block scale.
    /// `grids[8][4]` = 4 grid indices (0..256) per sub-group.
    /// `signs[8][4]` = 4 sign indices (0..128) per sub-group.
    /// `extras[8]` = per-sub-group extra exponent (0..16).
    fn pack_block(d: f32, grids: [[u8; 4]; 8], signs: [[u8; 4]; 8], extras: [u8; 8]) -> Vec<u8> {
        for s in signs.iter() {
            for &v in s {
                assert!(v < 128, "sign index must be 7-bit");
            }
        }
        for &e in &extras {
            assert!(e < 16, "extra must be 4-bit");
        }

        let mut out = f16::from_f32(d).to_le_bytes().to_vec();

        for ib32 in 0..8 {
            let aux32_0 = u32::from_le_bytes(grids[ib32]);
            let mut aux32_1 = 0u32;
            for (l, sign) in signs[ib32].iter().enumerate() {
                aux32_1 |= u32::from(*sign) << (7 * l);
            }
            aux32_1 |= u32::from(extras[ib32]) << 28;
            out.extend_from_slice(&aux32_0.to_le_bytes());
            out.extend_from_slice(&aux32_1.to_le_bytes());
        }

        assert_eq!(out.len(), BYTES_PER_BLOCK);
        out
    }

    #[test]
    fn block_size_matches_spec() {
        assert_eq!(BYTES_PER_BLOCK, 66);
        assert_eq!(ELEMENTS_PER_BLOCK, 256);
    }

    #[test]
    fn grid_table_has_256_entries() {
        assert_eq!(IQ2XXS_GRID.len(), 256);
        // First entry is the canonical all-eights pattern (8 × 0x08).
        assert_eq!(IQ2XXS_GRID[0], 0x0808_0808_0808_0808);
        // Last entry from ds4.c:297.
        assert_eq!(IQ2XXS_GRID[255], 0x2b2b_2b19_0808_1908);
    }

    #[test]
    fn ksigns_table_has_128_entries() {
        assert_eq!(KSIGNS_IQ2XS.len(), 128);
        // First entry is 0 (no signs flipped) per ds4.c:222.
        assert_eq!(KSIGNS_IQ2XS[0], 0);
        // Last entry per ds4.c:229: 255 (all signs flipped).
        assert_eq!(KSIGNS_IQ2XS[127], 255);
    }

    #[test]
    fn kmask_table_is_bit_powers() {
        assert_eq!(KMASK_IQ2XS, [1u8, 2, 4, 8, 16, 32, 64, 128]);
    }

    #[test]
    fn zero_grid_yields_zero() {
        // Grid index 0 happens to be all-positive values, but if we pair it
        // with sign index 0 (no flips), and d=0, every output is zero.
        let block = pack_block(0.0, [[0u8; 4]; 8], [[0u8; 4]; 8], [0u8; 8]);
        let mut dst = vec![0.0f32; 256];
        dequant_iq2_xxs(&block, &mut dst).unwrap();
        for v in &dst {
            assert!(v.abs() < 1e-3, "got {v}");
        }
    }

    #[test]
    fn uniform_pattern_uniform_output() {
        // Use IQ2XXS_GRID[0] = 0x0808080808080808 → 8 bytes of 0x08 = +8.
        // d = 1.0, extra = 0, signs = 0 (no flips).
        // db = 1 × (0.5 + 0) × 0.25 = 0.125
        // dst[i] = 0.125 × 8 × 1 = 1.0
        let block = pack_block(1.0, [[0u8; 4]; 8], [[0u8; 4]; 8], [0u8; 8]);
        let mut dst = vec![0.0f32; 256];
        dequant_iq2_xxs(&block, &mut dst).unwrap();
        for (i, v) in dst.iter().enumerate() {
            assert!((v - 1.0).abs() < 1e-3, "i={i}: got {v}, want 1.0");
        }
    }

    #[test]
    fn extras_scale_subgroups_independently() {
        // d = 1.0, all grids = 0 (pattern = +8 ×8), no sign flips.
        // extra[ib32] = ib32 → db = 1 × (0.5 + ib32) × 0.25
        // dst in sub-group ib32 = db × 8 = (0.5 + ib32) × 2 = 1 + 2*ib32
        let extras: [u8; 8] = std::array::from_fn(|i| i as u8);
        let block = pack_block(1.0, [[0u8; 4]; 8], [[0u8; 4]; 8], extras);
        let mut dst = vec![0.0f32; 256];
        dequant_iq2_xxs(&block, &mut dst).unwrap();
        for ib32 in 0..8 {
            let want = 1.0 + 2.0 * (ib32 as f32);
            for j in 0..32 {
                let got = dst[ib32 * 32 + j];
                assert!(
                    (got - want).abs() < 1e-3,
                    "ib32={ib32} j={j}: got {got}, want {want}"
                );
            }
        }
    }

    #[test]
    fn signs_flip_negate_elements() {
        // Sign index 127 = ksigns[127] = 255 → all 8 bits set → all flips.
        // With grid=0 (+8 ×8), d=1, extra=0:
        //   db = 0.125, value = 0.125 × 8 = 1.0, with sign flip = -1.0
        let block = pack_block(1.0, [[0u8; 4]; 8], [[127u8; 4]; 8], [0u8; 8]);
        let mut dst = vec![0.0f32; 256];
        dequant_iq2_xxs(&block, &mut dst).unwrap();
        for v in &dst {
            assert!((v - (-1.0)).abs() < 1e-3, "got {v}, want -1.0");
        }
    }

    #[test]
    fn distinct_l_slots_within_one_subgroup() {
        // Probe the 4 inner `l` loads inside a single sub-group with 4
        // distinct grid indices + 4 distinct sign indices, so an off-by-one
        // between aux8[l] indexing and the `>> (7*l)` sign extraction would
        // produce visible cross-talk.
        //
        // ib32 = 0 specifically:
        //   grids[0]  = [0, 1, 2, 3]   (4 distinct grid indices)
        //   signs[0]  = [0, 1, 2, 3]   (4 distinct sign indices)
        //
        // From ds4.c:222 KSIGNS_IQ2XS:
        //   ksigns[0] = 0   -> all 8 positions positive
        //   ksigns[1] = 129 = 0b10000001 -> bits 0 and 7 set
        //                                    (positions 0 and 7 negated)
        //   ksigns[2] = 130 = 0b10000010 -> bits 1 and 7 set
        //                                    (positions 1 and 7 negated)
        //   ksigns[3] = 3   = 0b00000011 -> bits 0 and 1 set
        //                                    (positions 0 and 1 negated)
        //
        // IQ2XXS_GRID[0..4] from ds4.c:233, decoded as little-endian bytes
        // (byte 0 = lowest byte of the u64):
        //   [0] = 0x0808080808080808 -> [+8, +8, +8, +8, +8, +8, +8, +8]
        //   [1] = 0x080808080808082b -> [+43, +8, +8, +8, +8, +8, +8, +8]
        //   [2] = 0x0808080808081919 -> [+25, +25, +8, +8, +8, +8, +8, +8]
        //   [3] = 0x0808080808082b08 -> [+8, +43, +8, +8, +8, +8, +8, +8]
        //
        // d = 1, extra = 0 -> db = 0.125. Other sub-groups (ib32=1..8) get
        // grids=0/signs=0 so they emit +1.0 uniformly.

        let mut grids = [[0u8; 4]; 8];
        let mut signs = [[0u8; 4]; 8];
        grids[0] = [0, 1, 2, 3];
        signs[0] = [0, 1, 2, 3];
        let block = pack_block(1.0, grids, signs, [0u8; 8]);
        let mut dst = vec![0.0f32; 256];
        dequant_iq2_xxs(&block, &mut dst).unwrap();

        // l=0: grid 0, signs 0  -> 8 × (+8 × +1 × 0.125) = 8 × 1.0
        for v in &dst[0..8] {
            assert!((v - 1.0).abs() < 1e-3, "l=0: got {v}");
        }
        // l=1: grid 1 = [+43, +8, +8, +8, +8, +8, +8, +8], signs 1 negates
        // positions 0 and 7.
        //   pos 0: -43 × 0.125 = -5.375
        //   pos 1..7: +8 × 0.125 = +1.0
        //   pos 7: -8 × 0.125 = -1.0
        let l1 = &dst[8..16];
        assert!((l1[0] - (-5.375)).abs() < 1e-3, "l=1 pos 0: got {}", l1[0]);
        for (j, v) in l1.iter().enumerate().skip(1).take(6) {
            assert!((v - 1.0).abs() < 1e-3, "l=1 pos {j}: got {v}");
        }
        assert!((l1[7] - (-1.0)).abs() < 1e-3, "l=1 pos 7: got {}", l1[7]);
        // l=2: grid 2 = [+25, +25, +8, +8, +8, +8, +8, +8], signs 2 negates
        // positions 1 and 7.
        //   pos 0: +25 × 0.125 = +3.125
        //   pos 1: -25 × 0.125 = -3.125
        //   pos 2..6: +1.0
        //   pos 7: -1.0
        let l2 = &dst[16..24];
        assert!((l2[0] - 3.125).abs() < 1e-3, "l=2 pos 0: got {}", l2[0]);
        assert!((l2[1] - (-3.125)).abs() < 1e-3, "l=2 pos 1: got {}", l2[1]);
        for (j, v) in l2.iter().enumerate().skip(2).take(5) {
            assert!((v - 1.0).abs() < 1e-3, "l=2 pos {j}: got {v}");
        }
        assert!((l2[7] - (-1.0)).abs() < 1e-3, "l=2 pos 7: got {}", l2[7]);
        // l=3: grid 3 = [+8, +43, +8, +8, +8, +8, +8, +8], signs 3 negates
        // positions 0 and 1.
        //   pos 0: -8 × 0.125 = -1.0
        //   pos 1: -43 × 0.125 = -5.375
        //   pos 2..7: +1.0
        let l3 = &dst[24..32];
        assert!((l3[0] - (-1.0)).abs() < 1e-3, "l=3 pos 0: got {}", l3[0]);
        assert!((l3[1] - (-5.375)).abs() < 1e-3, "l=3 pos 1: got {}", l3[1]);
        for (j, v) in l3.iter().enumerate().skip(2) {
            assert!((v - 1.0).abs() < 1e-3, "l=3 pos {j}: got {v}");
        }
        // Other 7 sub-groups should be untouched: +1.0 uniform.
        for v in &dst[32..] {
            assert!((v - 1.0).abs() < 1e-3, "ib32>0 leak: got {v}");
        }
    }

    #[test]
    fn distinct_subgroups_with_distinct_grids_and_extras() {
        // Combine extras + grid variation across all 8 sub-groups to catch
        // any out_off / aux32 read alignment bug.
        // - sub-group ib32 uses grid index = ib32 (all 4 l-slots), signs = 0,
        //   extra = ib32.
        // db(ib32) = 1 × (0.5 + ib32) × 0.25
        // Within a sub-group, all 32 outputs share the same db × grid_byte.
        //
        // IQ2XXS_GRID[0] = 0x0808080808080808 → all 8 bytes are +8
        //   db=0.125 → all outputs = 1.0
        //
        // IQ2XXS_GRID[1] = 0x080808080808082b → bytes (LE) [43, 8, 8, 8, 8, 8, 8, 8]
        //   db = 1 × 1.5 × 0.25 = 0.375
        //   per l-slot: position 0 = 43 × 0.375 = 16.125, positions 1-7 = 3.0
        //
        // IQ2XXS_GRID[4] = 0x0808080808082b2b → bytes (LE) [43, 43, 8, 8, 8, 8, 8, 8]
        //   db = 1 × 4.5 × 0.25 = 1.125
        //   per l-slot: positions 0,1 = 43 × 1.125 = 48.375, positions 2-7 = 9.0
        let grids = std::array::from_fn(|ib32| [ib32 as u8; 4]);
        let signs = [[0u8; 4]; 8];
        let extras = std::array::from_fn(|i| i as u8);
        let block = pack_block(1.0, grids, signs, extras);
        let mut dst = vec![0.0f32; 256];
        dequant_iq2_xxs(&block, &mut dst).unwrap();

        // ib32=0: all 32 outputs = 1.0
        for v in &dst[0..32] {
            assert!((v - 1.0).abs() < 1e-3, "ib32=0: got {v}");
        }
        // ib32=1: 4 l-slots, each: [16.125, 3, 3, 3, 3, 3, 3, 3]
        for l in 0..4 {
            let base = 32 + l * 8;
            assert!(
                (dst[base] - 16.125).abs() < 1e-3,
                "ib32=1 l={l} pos 0: got {}, want 16.125",
                dst[base]
            );
            for j in 1..8 {
                let v = dst[base + j];
                assert!((v - 3.0).abs() < 1e-3, "ib32=1 l={l} j={j}: got {v}");
            }
        }
        // ib32=4: 4 l-slots, each: [48.375, 48.375, 9, 9, 9, 9, 9, 9]
        for l in 0..4 {
            let base = 4 * 32 + l * 8;
            for j in 0..2 {
                let v = dst[base + j];
                assert!(
                    (v - 48.375).abs() < 1e-3,
                    "ib32=4 l={l} j={j}: got {v}, want 48.375"
                );
            }
            for j in 2..8 {
                let v = dst[base + j];
                assert!((v - 9.0).abs() < 1e-3, "ib32=4 l={l} j={j}: got {v}");
            }
        }
    }

    #[test]
    fn wrong_src_size_errors() {
        let src = vec![0u8; 65];
        let mut dst = vec![0.0f32; 256];
        assert!(matches!(
            dequant_iq2_xxs(&src, &mut dst),
            Err(Error::DequantSizeMismatch { .. })
        ));
    }
}
