//! Errors produced by the GGUF parser.

use std::io;
use thiserror::Error;

/// Errors returned by GGUF parsing operations.
#[derive(Debug, Error)]
pub enum Error {
    /// Wrapper around any I/O error encountered while opening or memory-mapping
    /// a GGUF file.
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),

    /// The file's first four bytes did not match the GGUF magic `GGUF`.
    #[error("not a GGUF file (expected magic 'GGUF', found {found:?})")]
    BadMagic {
        /// The four bytes that were actually read.
        found: [u8; 4],
    },

    /// The file's GGUF version is not supported by this build.
    ///
    /// rsLLM v0.1.0 supports GGUF v3 only — see ADR-0001 and `FEATURE_002`.
    #[error("unsupported GGUF version {0} (this build supports v3)")]
    UnsupportedVersion(u32),

    /// A read was attempted past the end of the byte slice.
    #[error("truncated read at offset {pos}: need {need} bytes, only {have} remaining")]
    Truncated {
        /// Reader position at which the truncated read started.
        pos: u64,
        /// Number of bytes requested.
        need: u64,
        /// Number of bytes that were actually available.
        have: u64,
    },

    /// The file is too small to even contain a complete GGUF header.
    #[error("file too small to be GGUF (size {0} bytes, need at least 32)")]
    TooSmall(u64),

    /// A metadata value declared a `value_type` that does not match any of the
    /// 13 recognized GGUF metadata types.
    #[error("unknown metadata value type {0}")]
    UnknownValueType(u32),

    /// Metadata array nesting exceeded the recursion limit (8).
    ///
    /// This protects against pathological or hostile GGUF inputs.
    #[error("metadata array nesting too deep (max 8)")]
    NestingTooDeep,

    /// A length-prefixed string contained bytes that are not valid UTF-8.
    #[error("invalid UTF-8 in string at offset {0}")]
    InvalidUtf8(u64),

    /// An array declared a `length * item_size` product that would overflow
    /// `u64`, indicating either corruption or a hostile input.
    #[error("metadata array too large: {len} × {item_size} bytes overflows")]
    ArrayTooLarge {
        /// Declared number of elements in the array.
        len: u64,
        /// Per-element byte size.
        item_size: u64,
    },

    /// Source byte slice does not have the size implied by `dst.len()` and the
    /// dtype. Indicates the caller passed a mis-sized buffer pair.
    #[error("dequant size mismatch: src has {src_bytes} bytes, expected {expected_bytes}")]
    DequantSizeMismatch {
        /// Number of bytes actually present in the source slice.
        src_bytes: usize,
        /// Number of bytes expected for the given dtype + element count.
        expected_bytes: usize,
    },

    /// Caller asked for dequantization of a dtype this build does not yet
    /// implement. See `GgmlType::is_decodable_v0_1_0` for the supported set.
    #[error("dtype `{0}` is not yet supported for dequantization in this build")]
    UnsupportedDequant(&'static str),

    /// A tensor directory entry declared a dimension count outside the legal
    /// range (`1..=MAX_DIMS`). Either the file is corrupt or it uses a GGUF
    /// extension this build does not handle.
    #[error("tensor `{name}` has invalid dimension count {ndim} (must be 1..={max})")]
    InvalidTensorDims {
        /// Tensor name from the directory entry.
        name: String,
        /// Declared `ndim`.
        ndim: u32,
        /// Maximum supported dimensions ([`crate::tensor::MAX_DIMS`]).
        max: u32,
    },

    /// The `general.alignment` metadata key, or the default alignment, is
    /// outside the safe range (1..=`MAX_ALIGNMENT`). Protects against hostile
    /// inputs that would otherwise overflow `align_up`.
    #[error("invalid tensor data alignment {0} (must be 1..=65536, power of two recommended)")]
    InvalidAlignment(u64),
}
