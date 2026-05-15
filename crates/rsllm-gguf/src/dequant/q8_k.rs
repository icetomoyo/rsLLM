//! Q8_K dequantization (8-bit K-quants, **temporary activation block format**).
//!
//! Q8_K is used as the **quantized activation** type during matmul
//! intermediates — never as an on-disk weight format. DeepSeek V4 Flash's
//! batched Q8_0 matmul (`ds4.c:3277-3297`) generates Q8_K blocks from f32
//! activations, then runs `vec_dot_qXX_q8_K` against Q2_K / Q4_K / IQ2_XXS
//! weights. We implement the dequant path here for completeness and
//! verification — the actual hot path will be in `rsllm-backend-cpu`.
//!
//! Ported by reference from `ggml-quants.c` (`block_q8_K`, MIT, ggml authors).
//!
//! ## Block layout (256 elements / 292 bytes)
//!
//! ```text
//! struct block_q8_K {
//!     float   d;             // 4 bytes — super-block scale (note: f32, not f16!)
//!     int8_t  qs[256];       // 256 bytes — quants
//!     int16_t bsums[16];     // 32 bytes — partial row sums (16 × 16 elements);
//!                            // used by integer matmul for fast bias accumulation;
//!                            // we ignore them for dequant since `d * qs[i]` is
//!                            // sufficient for f32 output.
//! }
//! ```
//!
//! Decode: `dst[i] = d * qs[i]` for `i in 0..256`.

use crate::error::Error;

const ELEMENTS_PER_BLOCK: usize = 256;
const BYTES_PER_BLOCK: usize = 292;

pub(crate) fn dequant_q8_k(src: &[u8], dst: &mut [f32]) -> Result<(), Error> {
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
        let d = f32::from_le_bytes([
            src[block_off],
            src[block_off + 1],
            src[block_off + 2],
            src[block_off + 3],
        ]);
        let qs_off = block_off + 4;
        let out_off = b * ELEMENTS_PER_BLOCK;

        for (i, slot) in dst[out_off..out_off + ELEMENTS_PER_BLOCK]
            .iter_mut()
            .enumerate()
        {
            // qs is int8, stored as u8 on disk; reinterpret as i8.
            let q = src[qs_off + i] as i8;
            *slot = d * f32::from(q);
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pack one Q8_K block. `bsums` are filled with zero — they are not used by
    /// the dequant path.
    fn pack_block(d: f32, qs: [i8; 256]) -> Vec<u8> {
        let mut out = d.to_le_bytes().to_vec();
        for q in &qs {
            out.push(*q as u8);
        }
        // 32 bytes of bsums (16 × i16), filled with 0.
        out.extend(std::iter::repeat_n(0u8, 32));
        out
    }

    #[test]
    fn block_size_matches_spec() {
        assert_eq!(BYTES_PER_BLOCK, 292);
        assert_eq!(ELEMENTS_PER_BLOCK, 256);
    }

    #[test]
    fn unit_scale_passes_through_qs() {
        // d = 1, every qs = its index mod 127.
        let qs: [i8; 256] = std::array::from_fn(|i| (i % 127) as i8);
        let block = pack_block(1.0, qs);
        let mut dst = vec![0.0f32; 256];
        dequant_q8_k(&block, &mut dst).unwrap();
        for (i, v) in dst.iter().enumerate() {
            assert!(
                (v - f32::from(qs[i])).abs() < 1e-4,
                "i={i}: got {v}, want {}",
                qs[i]
            );
        }
    }

    #[test]
    fn negative_scale_inverts_sign() {
        let qs = [5i8; 256];
        let block = pack_block(-0.5, qs);
        let mut dst = vec![0.0f32; 256];
        dequant_q8_k(&block, &mut dst).unwrap();
        for v in &dst {
            assert!((v - (-2.5)).abs() < 1e-4, "got {v}, want -2.5");
        }
    }

    #[test]
    fn boundary_quant_values() {
        let mut qs = [0i8; 256];
        qs[0] = i8::MIN;
        qs[1] = i8::MAX;
        qs[2] = 0;
        qs[3] = -1;
        let block = pack_block(0.01, qs);
        let mut dst = vec![0.0f32; 256];
        dequant_q8_k(&block, &mut dst).unwrap();
        assert!((dst[0] - 0.01 * -128.0).abs() < 1e-4);
        assert!((dst[1] - 0.01 * 127.0).abs() < 1e-4);
        assert!(dst[2].abs() < 1e-4);
        assert!((dst[3] - -0.01).abs() < 1e-4);
    }

    #[test]
    fn wrong_src_size_errors() {
        let src = vec![0u8; 291];
        let mut dst = vec![0.0f32; 256];
        assert!(matches!(
            dequant_q8_k(&src, &mut dst),
            Err(Error::DequantSizeMismatch { .. })
        ));
    }

    #[test]
    fn multi_block_decode() {
        // qs_a wraps through i8 range; qs_b is the wrapping-negation to avoid
        // i8::MIN negation overflow.
        let qs_a: [i8; 256] = std::array::from_fn(|i| i as i8);
        let qs_b: [i8; 256] = std::array::from_fn(|i| (i as i8).wrapping_neg());
        let mut src = pack_block(2.0, qs_a);
        src.extend(pack_block(-1.0, qs_b));
        let mut dst = vec![0.0f32; 512];
        dequant_q8_k(&src, &mut dst).unwrap();
        for (i, v) in dst[..256].iter().enumerate() {
            assert!((v - (2.0 * f32::from(qs_a[i]))).abs() < 1e-4);
        }
        for (i, v) in dst[256..].iter().enumerate() {
            assert!((v - (-f32::from(qs_b[i]))).abs() < 1e-4);
        }
    }
}
