//! # rsllm-gguf
//!
//! Self-contained GGUF v3 file format parser and quantized tensor decoder.
//!
//! ## Status
//!
//! This crate is under active construction. The current scope (v0.1.0,
//! [`FEATURE_002`](https://github.com/icetomoyo/rsLLM/blob/main/docs/features/v0.1.0.md))
//! is:
//!
//! - **Phase 1 ✅** — error types, low-level reader, KV metadata parsing
//! - **Phase 2** — tensor info parsing, mmap loader, model fingerprint (SHA-256)
//! - **Phase 3** — quantized tensor dequantization (F32 / F16 / BF16 / Q4_0 /
//!   Q4_1 / Q4_K / Q5_K / Q6_K / Q8_0)
//! - **Phase 4** — end-to-end test against a real Llama 7B Q4_K_M GGUF
//!
//! ## Acknowledgements
//!
//! This crate does **not** link against `ggml` or `llama.cpp`. The GGUF
//! format definitions, value type enum, and byte-cursor reader API are
//! ported from `ds4.c` (MIT, The ds4.c authors); see specific line references
//! in each source file's header comment. Quantization block layouts and
//! decode lookup tables borrowed from `ggml` are credited under MIT in the
//! corresponding `quants/*.rs` files (to land in Phase 3).
//!
//! Rust-side structural choices (`GgufFile`, `Metadata`, `Value`) take
//! inspiration from HuggingFace's `candle` `gguf_file.rs` (Apache-2.0).
//!
//! See [`NOTICE.md`](https://github.com/icetomoyo/rsLLM/blob/main/NOTICE.md)
//! for the full attribution policy.

pub mod dequant;
pub mod error;
pub mod file;
pub mod metadata;
mod reader;
pub mod tensor;

pub use dequant::dequant_to_f32;
pub use error::Error;
pub use file::GgufFile;
pub use metadata::{Array, Metadata, Value, ValueType};
pub use tensor::{GgmlType, MAX_DIMS, TensorInfo};

/// The 4-byte file magic that identifies a GGUF file.
pub const MAGIC: [u8; 4] = *b"GGUF";

/// The GGUF format version supported by this build.
pub const SUPPORTED_VERSION: u32 = 3;

/// Returns the supported GGUF format version (currently 3).
///
/// Provided as a function in addition to the [`SUPPORTED_VERSION`] constant
/// for parity with the legacy stub call site that earlier crates depend on.
#[must_use]
pub fn supported_gguf_version() -> u32 {
    SUPPORTED_VERSION
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn supported_version_is_v3() {
        assert_eq!(supported_gguf_version(), 3);
        assert_eq!(SUPPORTED_VERSION, 3);
    }

    #[test]
    fn magic_is_four_bytes_gguf() {
        assert_eq!(&MAGIC, b"GGUF");
    }
}
