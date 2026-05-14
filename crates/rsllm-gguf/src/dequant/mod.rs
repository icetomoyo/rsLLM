//! Dequantization: convert on-disk GGUF tensor bytes to `f32` arrays.
//!
//! Public entry point is [`dequant_to_f32`]. Per-type implementations live in
//! sibling modules and are not exposed directly to avoid an unstable surface.
//!
//! ## v0.1.0 supported types
//!
//! | Type   | Module       | Reference        |
//! |--------|--------------|------------------|
//! | F32    | `float`      | trivial          |
//! | F16    | `float`      | `half::f16`      |
//! | BF16   | `float`      | `half::bf16`     |
//! | Q8_0   | `q8_0`       | ggml `ggml-quants.c` (MIT, ggml authors) |
//! | Q4_0   | `q4_0`       | same             |
//! | Q4_1   | `q4_0`       | same             |
//! | Q4_K   | `q4_k`       | same             |
//! | Q5_K   | `q5_k`       | same             |
//! | Q6_K   | `q6_k`       | same             |
//!
//! Phase 4 of FEATURE_002 lands these one module at a time. Modules that
//! have not yet been implemented are listed in [`GgmlType::is_decodable_v0_1_0`]
//! as `false`.

use crate::error::Error;
use crate::tensor::GgmlType;

mod float;
mod q4_0;
mod q4_k;
mod q5_k;
mod q6_k;
mod q8_0;

/// Extract the 6-bit scale and 6-bit min for sub-block `j` (0..8) from a
/// K-quant style packed `scales` array (12 bytes).
///
/// Direct port of `get_scale_min_k4` in `ggml-quants.c` (MIT, ggml authors).
/// Shared by Q4_K and Q5_K — both use the identical scale-packing scheme.
fn q4_k_scale_min(j: usize, q: &[u8; 12]) -> (u8, u8) {
    if j < 4 {
        let sc = q[j] & 63;
        let m = q[j + 4] & 63;
        (sc, m)
    } else {
        let sc = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
        let m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
        (sc, m)
    }
}

/// Dequantize `src` bytes (the on-disk representation of a tensor) into the
/// `dst` `f32` slice. `dst.len()` is interpreted as the logical element count.
///
/// Returns:
/// - `Ok(())` on success.
/// - [`Error::DequantSizeMismatch`] if `src.len()` does not match the expected
///   byte count for `dtype` and `dst.len()`.
/// - [`Error::UnsupportedDequant`] if `dtype` is not yet implemented in this
///   build (see [`GgmlType::is_decodable_v0_1_0`]).
pub fn dequant_to_f32(dtype: GgmlType, src: &[u8], dst: &mut [f32]) -> Result<(), Error> {
    match dtype {
        GgmlType::F32 => float::dequant_f32(src, dst),
        GgmlType::F16 => float::dequant_f16(src, dst),
        GgmlType::BF16 => float::dequant_bf16(src, dst),
        GgmlType::Q4_0 => q4_0::dequant_q4_0(src, dst),
        GgmlType::Q4_1 => q4_0::dequant_q4_1(src, dst),
        GgmlType::Q4_K => q4_k::dequant_q4_k(src, dst),
        GgmlType::Q5_K => q5_k::dequant_q5_k(src, dst),
        GgmlType::Q6_K => q6_k::dequant_q6_k(src, dst),
        GgmlType::Q8_0 => q8_0::dequant_q8_0(src, dst),
        _ => Err(Error::UnsupportedDequant(dtype.name())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unsupported_dtype_reports_error() {
        // Q2_K is recognized but not decodable in v0.1.0.
        let src = vec![0u8; 84]; // 1 block of Q2_K
        let mut dst = vec![0.0f32; 256];
        match dequant_to_f32(GgmlType::Q2_K, &src, &mut dst) {
            Err(Error::UnsupportedDequant(name)) => assert_eq!(name, "q2_k"),
            other => panic!("expected UnsupportedDequant, got {other:?}"),
        }
    }

    #[test]
    fn f32_dispatch_roundtrips() {
        let src: Vec<u8> = [1.0f32, -2.5, 1.5, 0.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let mut dst = vec![0.0f32; 4];
        dequant_to_f32(GgmlType::F32, &src, &mut dst).unwrap();
        assert_eq!(dst, vec![1.0, -2.5, 1.5, 0.0]);
    }
}
