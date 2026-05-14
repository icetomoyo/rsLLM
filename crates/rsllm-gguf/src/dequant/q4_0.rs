//! Q4_0 and Q4_1 dequantization (legacy 4-bit formats).
//!
//! Decoders ported by reference from `ggml-quants.c` (`block_q4_0` /
//! `dequantize_row_q4_0` and `block_q4_1` / `dequantize_row_q4_1`,
//! MIT, ggml authors). rsLLM does not link against ggml; the byte layout
//! and arithmetic are replicated here from the public format definition.
//!
//! ## Q4_0 block layout (32 elements / 18 bytes)
//!
//! ```text
//! struct block_q4_0 {
//!     fp16    d;        // 2 bytes — per-block scale
//!     uint8_t qs[16];   // 16 bytes — packed 4-bit quants
//! }
//! ```
//!
//! For each `j` in `0..16`:
//! - low nibble `qs[j] & 0x0F` → element `j`        (unsigned 0..15) − 8
//! - high nibble `qs[j] >> 4`  → element `j + 16`   (unsigned 0..15) − 8
//!
//! Decode: `dst[i] = d * (nibble - 8)`.
//!
//! ## Q4_1 block layout (32 elements / 20 bytes)
//!
//! ```text
//! struct block_q4_1 {
//!     fp16    d;        // 2 bytes — scale
//!     fp16    m;        // 2 bytes — min
//!     uint8_t qs[16];   // 16 bytes — packed 4-bit quants
//! }
//! ```
//!
//! Decode: `dst[i] = d * nibble + m` (no `-8` offset; the bias is absorbed
//! by `m`).

use half::f16;

use crate::error::Error;

const ELEMENTS_PER_BLOCK: usize = 32;
const Q4_0_BYTES: usize = 18;
const Q4_1_BYTES: usize = 20;

pub(crate) fn dequant_q4_0(src: &[u8], dst: &mut [f32]) -> Result<(), Error> {
    check_block_layout(src.len(), dst.len(), Q4_0_BYTES)?;

    let n_blocks = dst.len() / ELEMENTS_PER_BLOCK;
    for b in 0..n_blocks {
        let block_off = b * Q4_0_BYTES;
        let d = f16::from_le_bytes([src[block_off], src[block_off + 1]]).to_f32();
        let qs_off = block_off + 2;
        let out_off = b * ELEMENTS_PER_BLOCK;

        for j in 0..16 {
            let byte = src[qs_off + j];
            let x0 = (byte & 0x0F) as i32 - 8;
            let x1 = (byte >> 4) as i32 - 8;
            dst[out_off + j] = (x0 as f32) * d;
            dst[out_off + j + 16] = (x1 as f32) * d;
        }
    }

    Ok(())
}

pub(crate) fn dequant_q4_1(src: &[u8], dst: &mut [f32]) -> Result<(), Error> {
    check_block_layout(src.len(), dst.len(), Q4_1_BYTES)?;

    let n_blocks = dst.len() / ELEMENTS_PER_BLOCK;
    for b in 0..n_blocks {
        let block_off = b * Q4_1_BYTES;
        let d = f16::from_le_bytes([src[block_off], src[block_off + 1]]).to_f32();
        let m = f16::from_le_bytes([src[block_off + 2], src[block_off + 3]]).to_f32();
        let qs_off = block_off + 4;
        let out_off = b * ELEMENTS_PER_BLOCK;

        for j in 0..16 {
            let byte = src[qs_off + j];
            let x0 = f32::from(byte & 0x0F);
            let x1 = f32::from(byte >> 4);
            dst[out_off + j] = x0 * d + m;
            dst[out_off + j + 16] = x1 * d + m;
        }
    }

    Ok(())
}

fn check_block_layout(src_len: usize, dst_len: usize, bytes_per_block: usize) -> Result<(), Error> {
    if !dst_len.is_multiple_of(ELEMENTS_PER_BLOCK) {
        return Err(Error::DequantSizeMismatch {
            src_bytes: src_len,
            expected_bytes: dst_len.div_ceil(ELEMENTS_PER_BLOCK) * bytes_per_block,
        });
    }
    let expected = (dst_len / ELEMENTS_PER_BLOCK) * bytes_per_block;
    if src_len != expected {
        return Err(Error::DequantSizeMismatch {
            src_bytes: src_len,
            expected_bytes: expected,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pack one Q4_0 block: `d` scale, plus 32 raw nibbles (each value in 0..16).
    fn pack_q4_0(d: f32, nibbles: [u8; 32]) -> Vec<u8> {
        let mut out = f16::from_f32(d).to_le_bytes().to_vec();
        for j in 0..16 {
            // Low nibble = element j; high nibble = element j+16.
            let lo = nibbles[j] & 0x0F;
            let hi = nibbles[j + 16] & 0x0F;
            out.push((hi << 4) | lo);
        }
        out
    }

    /// Pack one Q4_1 block: `d` scale, `m` min, plus 32 raw nibbles.
    fn pack_q4_1(d: f32, m: f32, nibbles: [u8; 32]) -> Vec<u8> {
        let mut out = f16::from_f32(d).to_le_bytes().to_vec();
        out.extend_from_slice(&f16::from_f32(m).to_le_bytes());
        for j in 0..16 {
            let lo = nibbles[j] & 0x0F;
            let hi = nibbles[j + 16] & 0x0F;
            out.push((hi << 4) | lo);
        }
        out
    }

    #[test]
    fn q4_0_sizes_match_ds4_table() {
        assert_eq!(Q4_0_BYTES, 18);
        assert_eq!(Q4_1_BYTES, 20);
    }

    #[test]
    fn q4_0_zero_block_yields_negative_eight_times_scale() {
        // All nibbles = 0; dequant = d * (0 - 8) = -8 * d.
        let block = pack_q4_0(1.0, [0u8; 32]);
        let mut dst = vec![0.0f32; 32];
        dequant_q4_0(&block, &mut dst).unwrap();
        for v in &dst {
            assert!((v - (-8.0)).abs() < 1e-4);
        }
    }

    #[test]
    fn q4_0_full_range_block() {
        // Nibbles 0..16 across the 32 element layout.
        let mut nibbles = [0u8; 32];
        for (i, n) in nibbles.iter_mut().enumerate() {
            *n = (i % 16) as u8;
        }
        let block = pack_q4_0(1.0, nibbles);
        let mut dst = vec![0.0f32; 32];
        dequant_q4_0(&block, &mut dst).unwrap();
        for i in 0..32 {
            let want = (nibbles[i] as f32) - 8.0;
            assert!(
                (dst[i] - want).abs() < 1e-4,
                "i={i}, got {}, want {want}",
                dst[i]
            );
        }
    }

    #[test]
    fn q4_0_low_high_nibble_ordering() {
        // Distinct values on low vs high nibbles to confirm element indexing.
        let mut nibbles = [0u8; 32];
        nibbles[0] = 1; // element 0 → low nibble of qs[0]
        nibbles[16] = 15; // element 16 → high nibble of qs[0]
        let block = pack_q4_0(0.5, nibbles);
        let mut dst = vec![0.0f32; 32];
        dequant_q4_0(&block, &mut dst).unwrap();
        assert!((dst[0] - (0.5 * (1.0 - 8.0))).abs() < 1e-4);
        assert!((dst[16] - (0.5 * (15.0 - 8.0))).abs() < 1e-4);
    }

    #[test]
    fn q4_1_min_only_offset() {
        // d = 0 → dst = m everywhere. Equivalent to a constant tensor.
        let block = pack_q4_1(0.0, 3.5, [7u8; 32]);
        let mut dst = vec![0.0f32; 32];
        dequant_q4_1(&block, &mut dst).unwrap();
        for v in &dst {
            assert!((v - 3.5).abs() < 1e-3);
        }
    }

    #[test]
    fn q4_1_linear_with_offset() {
        // d = 0.5, m = 1.0; nibble n → 0.5*n + 1.0.
        let mut nibbles = [0u8; 32];
        for (i, n) in nibbles.iter_mut().enumerate() {
            *n = (i % 16) as u8;
        }
        let block = pack_q4_1(0.5, 1.0, nibbles);
        let mut dst = vec![0.0f32; 32];
        dequant_q4_1(&block, &mut dst).unwrap();
        for i in 0..32 {
            let want = 0.5 * (nibbles[i] as f32) + 1.0;
            assert!(
                (dst[i] - want).abs() < 1e-3,
                "i={i}, got {}, want {want}",
                dst[i]
            );
        }
    }

    #[test]
    fn q4_0_multiple_blocks() {
        let mut src = pack_q4_0(1.0, [0u8; 32]);
        src.extend(pack_q4_0(2.0, [15u8; 32]));
        let mut dst = vec![0.0f32; 64];
        dequant_q4_0(&src, &mut dst).unwrap();
        // Block 0: -8 * 1 = -8
        // Block 1: (15 - 8) * 2 = 14
        for v in &dst[..32] {
            assert!((v - (-8.0)).abs() < 1e-4);
        }
        for v in &dst[32..64] {
            assert!((v - 14.0).abs() < 1e-4);
        }
    }

    #[test]
    fn wrong_src_size_errors() {
        let src = vec![0u8; 17]; // Q4_0 needs 18
        let mut dst = vec![0.0f32; 32];
        assert!(matches!(
            dequant_q4_0(&src, &mut dst),
            Err(Error::DequantSizeMismatch { .. })
        ));
    }

    #[test]
    fn partial_block_dst_errors() {
        let src = vec![0u8; 18];
        let mut dst = vec![0.0f32; 16];
        assert!(matches!(
            dequant_q4_0(&src, &mut dst),
            Err(Error::DequantSizeMismatch { .. })
        ));
    }
}
