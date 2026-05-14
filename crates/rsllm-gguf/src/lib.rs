//! # rsllm-gguf
//!
//! Self-contained GGUF file format parser and quantized tensor decoder.
//!
//! ## Acknowledgements
//!
//! This crate does **not** link against `ggml` or `llama.cpp`, but the GGUF
//! file format, the quantization block layouts, and several decode lookup
//! tables originate from those projects. Source files that contain code
//! ported or adapted from upstream carry dual copyright headers per MIT.
//!
//! Rust-side structural choices (`GgufFile`, `TensorInfo`, `Metadata`) are
//! influenced by the `gguf_file.rs` module in HuggingFace's `candle`
//! crate (Apache-2.0).
//!
//! See [`docs/features/v0.1.0.md#feature_002`](https://github.com/icetomoyo/rsLLM/blob/main/docs/features/v0.1.0.md)
//! for the full design and [`NOTICE.md`](https://github.com/icetomoyo/rsLLM/blob/main/NOTICE.md)
//! for license attributions.

/// Placeholder error type. Will be expanded in FEATURE_002 implementation.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Stand-in variant so the enum is constructible during the skeleton phase.
    #[error("not yet implemented")]
    NotYetImplemented,
}

/// Returns the supported GGUF format version range for this build.
///
/// rsLLM v0.1.0 targets GGUF v3 (the version produced by current llama.cpp).
#[must_use]
pub fn supported_gguf_version() -> u32 {
    3
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn supported_version_is_v3() {
        assert_eq!(supported_gguf_version(), 3);
    }
}
