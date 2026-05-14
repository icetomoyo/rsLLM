//! Tensor descriptors and the GGML quantization type enum.
//!
//! Type IDs and block layouts mirror `gguf_types[]` in `ds4.c:836-866`
//! (MIT, The ds4.c authors), which in turn match the on-disk format used
//! by `llama.cpp` / `ggml` (MIT). rsLLM does not link against either
//! project; this enum exists purely to interpret bytes that the GGUF spec
//! places in the tensor directory.

use crate::error::Error;
use crate::reader::Reader;

/// Maximum number of dimensions per tensor (matches `DS4_MAX_DIMS` in ds4).
pub const MAX_DIMS: usize = 4;

/// The full set of GGML/GGUF tensor element types recognized by rsLLM.
///
/// All variants in this enum can be **parsed** (i.e. rsLLM can compute the
/// byte size and locate the bytes in the file). Whether a given variant can
/// be **dequantized** depends on the build: v0.1.0 implements decode for a
/// subset; the others are recognized so that `inspect`-style tooling does
/// not fail on unknown tensors. See FEATURE_002 / FEATURE_014 in
/// `docs/features/v0.1.0.md`.
#[allow(non_camel_case_types)]
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgmlType {
    /// 32-bit IEEE-754 float, 1 element / 4 bytes per block.
    F32 = 0,
    /// 16-bit IEEE-754 float (`half::f16`), 1 element / 2 bytes per block.
    F16 = 1,
    /// 4-bit quantization, 32 elements / 18 bytes per block.
    Q4_0 = 2,
    /// 4-bit quantization with per-block offset, 32 elements / 20 bytes per block.
    Q4_1 = 3,
    /// 5-bit quantization, 32 elements / 22 bytes per block.
    Q5_0 = 6,
    /// 5-bit quantization with per-block offset, 32 elements / 24 bytes per block.
    Q5_1 = 7,
    /// 8-bit quantization, 32 elements / 34 bytes per block.
    Q8_0 = 8,
    /// 8-bit quantization with offset, 32 elements / 40 bytes per block.
    Q8_1 = 9,
    /// 2-bit K-quants, 256 elements / 84 bytes per superblock.
    Q2_K = 10,
    /// 3-bit K-quants, 256 elements / 110 bytes per superblock.
    Q3_K = 11,
    /// 4-bit K-quants, 256 elements / 144 bytes per superblock.
    Q4_K = 12,
    /// 5-bit K-quants, 256 elements / 176 bytes per superblock.
    Q5_K = 13,
    /// 6-bit K-quants, 256 elements / 210 bytes per superblock.
    Q6_K = 14,
    /// 8-bit K-quants (used internally by ggml for accumulators).
    Q8_K = 15,
    /// "I-quants" 2-bit XXS, 256 elements / 66 bytes per superblock.
    IQ2_XXS = 16,
    /// "I-quants" 2-bit XS, 256 elements / 74 bytes per superblock.
    IQ2_XS = 17,
    /// "I-quants" 3-bit XXS, 256 elements / 98 bytes per superblock.
    IQ3_XXS = 18,
    /// "I-quants" 1-bit S, 256 elements / 110 bytes per superblock.
    IQ1_S = 19,
    /// "I-quants" 4-bit non-linear, 256 elements / 50 bytes per superblock.
    IQ4_NL = 20,
    /// "I-quants" 3-bit S, 256 elements / 110 bytes per superblock.
    IQ3_S = 21,
    /// "I-quants" 2-bit S, 256 elements / 82 bytes per superblock.
    IQ2_S = 22,
    /// "I-quants" 4-bit XS, 256 elements / 136 bytes per superblock.
    IQ4_XS = 23,
    /// 8-bit signed integer.
    I8 = 24,
    /// 16-bit signed integer.
    I16 = 25,
    /// 32-bit signed integer.
    I32 = 26,
    /// 64-bit signed integer.
    I64 = 27,
    /// 64-bit IEEE-754 float.
    F64 = 28,
    /// "I-quants" 1-bit M, 256 elements / 56 bytes per superblock.
    IQ1_M = 29,
    /// brain-float 16, 1 element / 2 bytes per block.
    BF16 = 30,
}

impl GgmlType {
    /// Convert a raw `u32` (as stored in the GGUF tensor directory) into a
    /// typed `GgmlType`. Returns `None` for unrecognized type IDs.
    pub fn from_u32(v: u32) -> Option<Self> {
        Some(match v {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            3 => Self::Q4_1,
            6 => Self::Q5_0,
            7 => Self::Q5_1,
            8 => Self::Q8_0,
            9 => Self::Q8_1,
            10 => Self::Q2_K,
            11 => Self::Q3_K,
            12 => Self::Q4_K,
            13 => Self::Q5_K,
            14 => Self::Q6_K,
            15 => Self::Q8_K,
            16 => Self::IQ2_XXS,
            17 => Self::IQ2_XS,
            18 => Self::IQ3_XXS,
            19 => Self::IQ1_S,
            20 => Self::IQ4_NL,
            21 => Self::IQ3_S,
            22 => Self::IQ2_S,
            23 => Self::IQ4_XS,
            24 => Self::I8,
            25 => Self::I16,
            26 => Self::I32,
            27 => Self::I64,
            28 => Self::F64,
            29 => Self::IQ1_M,
            30 => Self::BF16,
            _ => return None,
        })
    }

    /// Number of logical elements per block. For unquantized types this is 1.
    /// For block-quantized types this is the block dimension (32 for legacy
    /// `Q*_0/Q*_1`, 256 for K-quants and I-quants).
    pub fn block_elements(self) -> u32 {
        match self {
            Self::F32
            | Self::F16
            | Self::BF16
            | Self::F64
            | Self::I8
            | Self::I16
            | Self::I32
            | Self::I64 => 1,
            Self::Q4_0 | Self::Q4_1 | Self::Q5_0 | Self::Q5_1 | Self::Q8_0 | Self::Q8_1 => 32,
            Self::Q2_K | Self::Q3_K | Self::Q4_K | Self::Q5_K | Self::Q6_K | Self::Q8_K => 256,
            Self::IQ2_XXS
            | Self::IQ2_XS
            | Self::IQ3_XXS
            | Self::IQ1_S
            | Self::IQ4_NL
            | Self::IQ3_S
            | Self::IQ2_S
            | Self::IQ4_XS
            | Self::IQ1_M => 256,
        }
    }

    /// Number of on-disk bytes per block, matching `gguf_types[]` in ds4.
    pub fn block_bytes(self) -> u32 {
        match self {
            Self::F32 => 4,
            Self::F16 => 2,
            Self::BF16 => 2,
            Self::F64 => 8,
            Self::I8 => 1,
            Self::I16 => 2,
            Self::I32 => 4,
            Self::I64 => 8,
            Self::Q4_0 => 18,
            Self::Q4_1 => 20,
            Self::Q5_0 => 22,
            Self::Q5_1 => 24,
            Self::Q8_0 => 34,
            Self::Q8_1 => 40,
            Self::Q2_K => 84,
            Self::Q3_K => 110,
            Self::Q4_K => 144,
            Self::Q5_K => 176,
            Self::Q6_K => 210,
            Self::Q8_K => 292,
            Self::IQ2_XXS => 66,
            Self::IQ2_XS => 74,
            Self::IQ3_XXS => 98,
            Self::IQ1_S | Self::IQ3_S => 110,
            Self::IQ4_NL => 50,
            Self::IQ2_S => 82,
            Self::IQ4_XS => 136,
            Self::IQ1_M => 56,
        }
    }

    /// Compute the on-disk byte size for a tensor of this type with `elements`
    /// logical elements. Returns `None` on overflow.
    pub fn byte_size(self, elements: u64) -> Option<u64> {
        let be = u64::from(self.block_elements());
        let bb = u64::from(self.block_bytes());
        // Number of blocks = ceil(elements / be).
        let blocks = elements.checked_add(be.checked_sub(1)?)? / be;
        blocks.checked_mul(bb)
    }

    /// Stable short name used by `rsllm inspect` and log messages.
    pub fn name(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F16 => "f16",
            Self::Q4_0 => "q4_0",
            Self::Q4_1 => "q4_1",
            Self::Q5_0 => "q5_0",
            Self::Q5_1 => "q5_1",
            Self::Q8_0 => "q8_0",
            Self::Q8_1 => "q8_1",
            Self::Q2_K => "q2_k",
            Self::Q3_K => "q3_k",
            Self::Q4_K => "q4_k",
            Self::Q5_K => "q5_k",
            Self::Q6_K => "q6_k",
            Self::Q8_K => "q8_k",
            Self::IQ2_XXS => "iq2_xxs",
            Self::IQ2_XS => "iq2_xs",
            Self::IQ3_XXS => "iq3_xxs",
            Self::IQ1_S => "iq1_s",
            Self::IQ4_NL => "iq4_nl",
            Self::IQ3_S => "iq3_s",
            Self::IQ2_S => "iq2_s",
            Self::IQ4_XS => "iq4_xs",
            Self::I8 => "i8",
            Self::I16 => "i16",
            Self::I32 => "i32",
            Self::I64 => "i64",
            Self::F64 => "f64",
            Self::IQ1_M => "iq1_m",
            Self::BF16 => "bf16",
        }
    }

    /// Whether v0.1.0 of rsLLM can dequantize this type to `f32`.
    ///
    /// Phase 4 of FEATURE_002 will flip more of these to `true`.
    pub fn is_decodable_v0_1_0(self) -> bool {
        matches!(
            self,
            Self::F32
                | Self::F16
                | Self::BF16
                | Self::Q4_0
                | Self::Q4_1
                | Self::Q4_K
                | Self::Q5_K
                | Self::Q6_K
                | Self::Q8_0
        )
    }
}

/// Description of a single tensor inside a GGUF file.
///
/// Field layout mirrors `ds4_tensor` in `ds4.c:884-893`, but stores ownership
/// of the name (Rust convention) and exposes both the recognized `GgmlType`
/// and the raw `u32` so that unrecognized types still round-trip through
/// `inspect`-style tooling.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorInfo {
    /// Tensor name as stored in the GGUF directory (e.g. `blk.0.attn_q.weight`).
    pub name: String,

    /// Decoded element type. `None` if the raw type ID is unrecognized.
    pub dtype: Option<GgmlType>,

    /// Raw `u32` type ID, retained for diagnostics regardless of `dtype`.
    pub raw_type: u32,

    /// Tensor shape, most-significant dimension last (GGUF convention).
    /// At most [`MAX_DIMS`] entries.
    pub shape: Vec<u64>,

    /// Total element count = product of `shape`.
    pub elements: u64,

    /// Offset of this tensor's bytes **relative to** the tensor data area
    /// (i.e. the first byte after the metadata + tensor directory + alignment
    /// padding). Add `GgufFile::tensor_data_offset()` to get the absolute
    /// file offset.
    pub relative_offset: u64,

    /// Byte size of this tensor's payload on disk. `None` if `dtype` is
    /// unknown or the size computation would overflow.
    pub byte_size: Option<u64>,
}

impl TensorInfo {
    /// Parse a single tensor directory entry from the current reader position.
    pub(crate) fn parse(reader: &mut Reader<'_>) -> Result<Self, Error> {
        let name = reader.read_str()?.to_owned();
        let ndim = reader.read_u32_le()?;
        if ndim == 0 || ndim as usize > MAX_DIMS {
            // Use a generic Truncated as a stand-in; a dedicated error variant
            // can be added if real-world GGUF files trip this.
            return Err(Error::Truncated {
                pos: reader.pos(),
                need: 0,
                have: ndim as u64,
            });
        }

        let mut shape = Vec::with_capacity(ndim as usize);
        let mut elements: u64 = 1;
        for _ in 0..ndim {
            let d = reader.read_u64_le()?;
            elements = elements.checked_mul(d.max(1)).ok_or(Error::ArrayTooLarge {
                len: elements,
                item_size: d,
            })?;
            shape.push(d);
        }

        let raw_type = reader.read_u32_le()?;
        let dtype = GgmlType::from_u32(raw_type);
        let relative_offset = reader.read_u64_le()?;
        let byte_size = dtype.and_then(|t| t.byte_size(elements));

        Ok(Self {
            name,
            dtype,
            raw_type,
            shape,
            elements,
            relative_offset,
            byte_size,
        })
    }
}

/// Parse exactly `n_tensors` tensor entries from the reader, returning them
/// in directory order.
pub(crate) fn parse_tensor_directory(
    reader: &mut Reader<'_>,
    n_tensors: u64,
) -> Result<Vec<TensorInfo>, Error> {
    let mut tensors =
        Vec::with_capacity(
            usize::try_from(n_tensors).map_err(|_| Error::ArrayTooLarge {
                len: n_tensors,
                item_size: 1,
            })?,
        );
    for _ in 0..n_tensors {
        tensors.push(TensorInfo::parse(reader)?);
    }
    Ok(tensors)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ggml_type_roundtrip_known_ids() {
        for raw in [0u32, 1, 2, 3, 6, 7, 8, 12, 14, 30] {
            let ty = GgmlType::from_u32(raw).expect("known");
            assert_eq!(ty as u32, raw);
        }
    }

    #[test]
    fn ggml_type_unknown_id_returns_none() {
        assert!(GgmlType::from_u32(99).is_none());
        // Gaps in the table (e.g. 4, 5) are also unrecognized.
        assert!(GgmlType::from_u32(4).is_none());
        assert!(GgmlType::from_u32(5).is_none());
    }

    #[test]
    fn block_layout_matches_ds4_table() {
        // Cross-check key entries against ds4.c:836-866.
        assert_eq!(GgmlType::F32.block_elements(), 1);
        assert_eq!(GgmlType::F32.block_bytes(), 4);

        assert_eq!(GgmlType::Q4_0.block_elements(), 32);
        assert_eq!(GgmlType::Q4_0.block_bytes(), 18);

        assert_eq!(GgmlType::Q4_K.block_elements(), 256);
        assert_eq!(GgmlType::Q4_K.block_bytes(), 144);

        assert_eq!(GgmlType::Q6_K.block_elements(), 256);
        assert_eq!(GgmlType::Q6_K.block_bytes(), 210);

        assert_eq!(GgmlType::IQ2_XXS.block_elements(), 256);
        assert_eq!(GgmlType::IQ2_XXS.block_bytes(), 66);

        assert_eq!(GgmlType::BF16.block_elements(), 1);
        assert_eq!(GgmlType::BF16.block_bytes(), 2);
    }

    #[test]
    fn byte_size_basic() {
        // 4096 f32 elements = 4096 blocks of 4 bytes = 16384 bytes.
        assert_eq!(GgmlType::F32.byte_size(4096), Some(16_384));
        // 4096 q4_0 elements = ceil(4096/32) = 128 blocks of 18 bytes = 2304 bytes.
        assert_eq!(GgmlType::Q4_0.byte_size(4096), Some(2_304));
        // Non-multiple-of-32 element count rounds up.
        assert_eq!(GgmlType::Q4_0.byte_size(33), Some(36)); // 2 blocks
        // Q4_K: 4096 elements = 16 blocks of 144 bytes = 2304 bytes.
        assert_eq!(GgmlType::Q4_K.byte_size(4096), Some(2_304));
    }

    #[test]
    fn decodable_v010_set() {
        for t in [
            GgmlType::F32,
            GgmlType::F16,
            GgmlType::BF16,
            GgmlType::Q4_0,
            GgmlType::Q4_1,
            GgmlType::Q4_K,
            GgmlType::Q5_K,
            GgmlType::Q6_K,
            GgmlType::Q8_0,
        ] {
            assert!(t.is_decodable_v0_1_0(), "{} should be decodable", t.name());
        }
        for t in [
            GgmlType::Q2_K,
            GgmlType::Q3_K,
            GgmlType::IQ2_XXS,
            GgmlType::IQ3_XXS,
            GgmlType::Q5_0,
            GgmlType::Q5_1,
        ] {
            assert!(
                !t.is_decodable_v0_1_0(),
                "{} should not yet be decodable",
                t.name()
            );
        }
    }

    /// Encode a single tensor directory entry for tests:
    /// `len u64 + name bytes + ndim u32 + dim u64* + type u32 + rel_offset u64`.
    fn pack_tensor(name: &str, ttype: u32, shape: &[u64], rel_offset: u64) -> Vec<u8> {
        let mut out = (name.len() as u64).to_le_bytes().to_vec();
        out.extend_from_slice(name.as_bytes());
        out.extend_from_slice(&(shape.len() as u32).to_le_bytes());
        for &d in shape {
            out.extend_from_slice(&d.to_le_bytes());
        }
        out.extend_from_slice(&ttype.to_le_bytes());
        out.extend_from_slice(&rel_offset.to_le_bytes());
        out
    }

    #[test]
    fn parse_single_tensor() {
        let bytes = pack_tensor("blk.0.attn_q.weight", 12 /* Q4_K */, &[4096, 4096], 0);
        let mut reader = Reader::new(&bytes);
        let info = TensorInfo::parse(&mut reader).unwrap();
        assert_eq!(info.name, "blk.0.attn_q.weight");
        assert_eq!(info.dtype, Some(GgmlType::Q4_K));
        assert_eq!(info.raw_type, 12);
        assert_eq!(info.shape, vec![4096, 4096]);
        assert_eq!(info.elements, 4096 * 4096);
        assert_eq!(info.relative_offset, 0);
        // 4096*4096 = 16_777_216 elements; / 256 = 65_536 blocks; * 144 = 9_437_184 bytes.
        assert_eq!(info.byte_size, Some(9_437_184));
    }

    #[test]
    fn parse_unknown_type_preserves_raw_id() {
        let bytes = pack_tensor("weird.tensor", 999, &[16], 0);
        let mut reader = Reader::new(&bytes);
        let info = TensorInfo::parse(&mut reader).unwrap();
        assert_eq!(info.dtype, None);
        assert_eq!(info.raw_type, 999);
        assert_eq!(info.byte_size, None);
    }

    #[test]
    fn parse_tensor_directory_multiple() {
        let mut bytes = pack_tensor("t1", 0 /* F32 */, &[4], 0);
        bytes.extend(pack_tensor("t2", 1 /* F16 */, &[8, 8], 16));
        bytes.extend(pack_tensor("t3", 8 /* Q8_0 */, &[32], 144));

        let mut reader = Reader::new(&bytes);
        let dir = parse_tensor_directory(&mut reader, 3).unwrap();
        assert_eq!(dir.len(), 3);
        assert_eq!(dir[0].name, "t1");
        assert_eq!(dir[0].dtype, Some(GgmlType::F32));
        assert_eq!(dir[1].name, "t2");
        assert_eq!(dir[1].dtype, Some(GgmlType::F16));
        assert_eq!(dir[2].name, "t3");
        assert_eq!(dir[2].dtype, Some(GgmlType::Q8_0));
    }

    #[test]
    fn rejects_zero_ndim() {
        let mut bytes = (b"x".len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(b"x");
        bytes.extend_from_slice(&0u32.to_le_bytes()); // ndim = 0
        bytes.extend_from_slice(&0u32.to_le_bytes()); // type
        bytes.extend_from_slice(&0u64.to_le_bytes()); // offset

        let mut reader = Reader::new(&bytes);
        assert!(TensorInfo::parse(&mut reader).is_err());
    }

    #[test]
    fn rejects_too_many_dims() {
        let mut bytes = (b"x".len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(b"x");
        bytes.extend_from_slice(&((MAX_DIMS as u32) + 1).to_le_bytes()); // ndim > MAX
        let mut reader = Reader::new(&bytes);
        assert!(TensorInfo::parse(&mut reader).is_err());
    }
}
