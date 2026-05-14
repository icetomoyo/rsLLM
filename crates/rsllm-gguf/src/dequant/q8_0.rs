//! Q8_0 dequantization.
//!
//! Block layout (32 elements / 34 bytes), ported by reference from
//! `ggml-quants.c` (`block_q8_0` / `dequantize_row_q8_0`, MIT, ggml authors):
//!
//! ```text
//! struct block_q8_0 {
//!     fp16   d;        // 2 bytes — per-block scale
//!     int8_t qs[32];   // 32 bytes — quantized values
//! }; // total: 34 bytes per 32 logical elements
//! ```
//!
//! Decode: `dst[i] = d * (qs[i] as f32)`.
//!
//! This is the simplest "with-scale" quantization in the GGUF ecosystem and
//! provides a clean shape for the more complex K-quants that follow.

use half::f16;

use crate::error::Error;

const ELEMENTS_PER_BLOCK: usize = 32;
const BYTES_PER_BLOCK: usize = 34;

pub(crate) fn dequant_q8_0(src: &[u8], dst: &mut [f32]) -> Result<(), Error> {
    if !dst.len().is_multiple_of(ELEMENTS_PER_BLOCK) {
        // Partial-block decode is not supported; callers must size `dst`
        // to a multiple of 32.
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
        let block = &src[block_off..block_off + BYTES_PER_BLOCK];

        // 2-byte LE half-precision scale.
        let d = f16::from_le_bytes([block[0], block[1]]).to_f32();

        // 32 signed-byte quantized values.
        let qs = &block[2..BYTES_PER_BLOCK];

        let out_off = b * ELEMENTS_PER_BLOCK;
        for j in 0..ELEMENTS_PER_BLOCK {
            // `qs[j]` is `u8`; reinterpret as `i8` to honour the on-disk
            // signed-integer representation.
            dst[out_off + j] = f32::from(qs[j] as i8) * d;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build one Q8_0 block (34 bytes) given `d` (scale) and 32 signed bytes.
    fn pack_block(d: f32, qs: [i8; 32]) -> Vec<u8> {
        let mut out = f16::from_f32(d).to_le_bytes().to_vec();
        out.extend(qs.iter().map(|&v| v as u8));
        out
    }

    #[test]
    fn block_size_matches_ds4_table() {
        assert_eq!(BYTES_PER_BLOCK, 34);
        assert_eq!(ELEMENTS_PER_BLOCK, 32);
    }

    #[test]
    fn single_block_identity_scale() {
        let qs: [i8; 32] = std::array::from_fn(|i| i as i8 - 16);
        // d = 1.0 → dst[i] = qs[i] as f32
        let block = pack_block(1.0, qs);
        let mut dst = vec![0.0f32; 32];
        dequant_q8_0(&block, &mut dst).unwrap();
        for (i, &v) in dst.iter().enumerate() {
            assert!((v - f32::from(qs[i])).abs() < 1e-4);
        }
    }

    #[test]
    fn single_block_scaled() {
        let qs: [i8; 32] = std::array::from_fn(|i| i as i8 + 1);
        let block = pack_block(2.0, qs);
        let mut dst = vec![0.0f32; 32];
        dequant_q8_0(&block, &mut dst).unwrap();
        // f16(2.0) is exact; dst[i] should equal 2 * (i+1).
        for (i, v) in dst.iter().enumerate() {
            assert!((v - (2.0 * (i as f32 + 1.0))).abs() < 1e-4);
        }
    }

    #[test]
    fn negative_scale_and_values() {
        let mut qs = [0i8; 32];
        let pattern = [-128i8, -64, -1, 0, 1, 64, 127, 0];
        for (i, slot) in qs.iter_mut().enumerate() {
            *slot = pattern[i % pattern.len()];
        }
        let block = pack_block(-0.5, qs);
        let mut dst = vec![0.0f32; 32];
        dequant_q8_0(&block, &mut dst).unwrap();
        for (i, &v) in dst.iter().enumerate() {
            let want = -0.5 * f32::from(qs[i]);
            assert!((v - want).abs() < 1e-3, "got {v}, want {want}");
        }
    }

    #[test]
    fn multiple_blocks() {
        let qs1: [i8; 32] = std::array::from_fn(|i| i as i8);
        let qs2: [i8; 32] = std::array::from_fn(|i| -(i as i8));
        let mut src = pack_block(1.0, qs1);
        src.extend(pack_block(2.0, qs2));
        let mut dst = vec![0.0f32; 64];
        dequant_q8_0(&src, &mut dst).unwrap();
        for i in 0..32 {
            assert!((dst[i] - f32::from(qs1[i])).abs() < 1e-4);
        }
        for i in 0..32 {
            assert!((dst[32 + i] - 2.0 * f32::from(qs2[i])).abs() < 1e-4);
        }
    }

    #[test]
    fn partial_block_dst_errors() {
        // dst.len() = 16 is not a multiple of 32.
        let src = vec![0u8; 34];
        let mut dst = vec![0.0f32; 16];
        match dequant_q8_0(&src, &mut dst) {
            Err(Error::DequantSizeMismatch { .. }) => {}
            other => panic!("expected DequantSizeMismatch, got {other:?}"),
        }
    }

    #[test]
    fn wrong_src_size_errors() {
        let src = vec![0u8; 33]; // missing 1 byte for one block
        let mut dst = vec![0.0f32; 32];
        match dequant_q8_0(&src, &mut dst) {
            Err(Error::DequantSizeMismatch {
                src_bytes,
                expected_bytes,
            }) => {
                assert_eq!(src_bytes, 33);
                assert_eq!(expected_bytes, 34);
            }
            other => panic!("expected DequantSizeMismatch, got {other:?}"),
        }
    }

    #[test]
    fn empty_buffers_ok() {
        let src: Vec<u8> = vec![];
        let mut dst: Vec<f32> = vec![];
        dequant_q8_0(&src, &mut dst).unwrap();
    }
}
