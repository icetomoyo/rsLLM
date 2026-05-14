//! Dequantization of the un-quantized float types: F32, F16, BF16.
//!
//! These are byte-level transmutes plus, for F16/BF16, a widening conversion
//! into `f32`. No reference implementation is required; the format is the
//! IEEE-754 / Brain-Float bit layout.

use half::{bf16, f16};

use crate::error::Error;

/// Copy `f32` little-endian source bytes into `dst`.
///
/// On any little-endian host this is effectively a `memcpy`; we go through
/// `from_le_bytes` per element so big-endian targets (theoretically) also
/// produce correct results.
pub(crate) fn dequant_f32(src: &[u8], dst: &mut [f32]) -> Result<(), Error> {
    let expected = dst.len() * 4;
    check_size(src.len(), expected)?;
    for (i, slot) in dst.iter_mut().enumerate() {
        let off = i * 4;
        let arr = [src[off], src[off + 1], src[off + 2], src[off + 3]];
        *slot = f32::from_le_bytes(arr);
    }
    Ok(())
}

/// Decode `dst.len()` little-endian `f16` half-precision values into `dst`
/// as `f32`. Widening is exact (every `f16` is representable as `f32`).
pub(crate) fn dequant_f16(src: &[u8], dst: &mut [f32]) -> Result<(), Error> {
    let expected = dst.len() * 2;
    check_size(src.len(), expected)?;
    for (i, slot) in dst.iter_mut().enumerate() {
        let off = i * 2;
        let arr = [src[off], src[off + 1]];
        *slot = f16::from_le_bytes(arr).to_f32();
    }
    Ok(())
}

/// Decode `dst.len()` little-endian `bf16` brain-float values into `dst`
/// as `f32`. Widening is exact (every `bf16` is representable as `f32`).
pub(crate) fn dequant_bf16(src: &[u8], dst: &mut [f32]) -> Result<(), Error> {
    let expected = dst.len() * 2;
    check_size(src.len(), expected)?;
    for (i, slot) in dst.iter_mut().enumerate() {
        let off = i * 2;
        let arr = [src[off], src[off + 1]];
        *slot = bf16::from_le_bytes(arr).to_f32();
    }
    Ok(())
}

fn check_size(actual: usize, expected: usize) -> Result<(), Error> {
    if actual == expected {
        Ok(())
    } else {
        Err(Error::DequantSizeMismatch {
            src_bytes: actual,
            expected_bytes: expected,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f32_to_bytes(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    fn f16_to_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|v| f16::from_f32(*v).to_le_bytes())
            .collect()
    }

    fn bf16_to_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|v| bf16::from_f32(*v).to_le_bytes())
            .collect()
    }

    #[test]
    fn f32_roundtrip_exact() {
        let values = [0.0_f32, 1.0, -1.0, 1.5, f32::MIN, f32::MAX];
        let src = f32_to_bytes(&values);
        let mut dst = vec![0.0; values.len()];
        dequant_f32(&src, &mut dst).unwrap();
        for (got, want) in dst.iter().zip(values.iter()) {
            assert_eq!(got.to_bits(), want.to_bits());
        }
    }

    #[test]
    fn f16_roundtrip_low_precision() {
        // f16 has only ~3 decimal digits of precision; pick values that
        // are exactly representable.
        let values = [0.0_f32, 1.0, -1.0, 0.5, 2048.0];
        let src = f16_to_bytes(&values);
        let mut dst = vec![0.0; values.len()];
        dequant_f16(&src, &mut dst).unwrap();
        for (got, want) in dst.iter().zip(values.iter()) {
            assert!((got - want).abs() < 1e-3, "got {got}, want {want}");
        }
    }

    #[test]
    fn bf16_roundtrip_low_precision() {
        // bf16 has ~2-3 decimal digits, same f32 exponent range.
        let values = [0.0_f32, 1.0, -1.0, 0.5, 2.0, 100.0];
        let src = bf16_to_bytes(&values);
        let mut dst = vec![0.0; values.len()];
        dequant_bf16(&src, &mut dst).unwrap();
        for (got, want) in dst.iter().zip(values.iter()) {
            assert!((got - want).abs() < (want.abs() * 0.01) + 1e-3);
        }
    }

    #[test]
    fn f32_size_mismatch_reports_error() {
        let src = vec![0u8; 6]; // not a multiple of 4
        let mut dst = vec![0.0; 2]; // expects 8 bytes
        match dequant_f32(&src, &mut dst) {
            Err(Error::DequantSizeMismatch {
                src_bytes,
                expected_bytes,
            }) => {
                assert_eq!(src_bytes, 6);
                assert_eq!(expected_bytes, 8);
            }
            other => panic!("expected DequantSizeMismatch, got {other:?}"),
        }
    }

    #[test]
    fn empty_buffers_are_ok() {
        let src: Vec<u8> = vec![];
        let mut dst: Vec<f32> = vec![];
        dequant_f32(&src, &mut dst).unwrap();
        dequant_f16(&src, &mut dst).unwrap();
        dequant_bf16(&src, &mut dst).unwrap();
    }
}
