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
//! the parser reads `qs[ib32*4 .. ib32*4+4]` as two `u32`s (`aux32[0]`,
//! `aux32[1]`):
//!
//! - `aux32[0]` low byte 0..3 = 4 grid indices into `IQ2XXS_GRID` (4 × 8
//!   element patterns = 32 elements per sub-group)
//! - `aux32[1]` low 28 bits = 4 × 7-bit sign indices into `KSIGNS_IQ2XS`
//! - `aux32[1]` top 4 bits = per-sub-group extra exponent
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
    fn wrong_src_size_errors() {
        let src = vec![0u8; 65];
        let mut dst = vec![0.0f32; 256];
        assert!(matches!(
            dequant_iq2_xxs(&src, &mut dst),
            Err(Error::DequantSizeMismatch { .. })
        ));
    }
}
