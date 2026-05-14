//! # rsllm-tokenizer
//!
//! Tokenizer integration for rsLLM. Will wrap HuggingFace's `tokenizers`
//! crate and add chat-template rendering (via `minijinja`) and
//! GGUF-metadata-driven tokenizer reconstruction.
//!
//! See [`docs/features/v0.1.0.md#feature_003`](https://github.com/icetomoyo/rsLLM/blob/main/docs/features/v0.1.0.md)
//! for the full design.

/// Placeholder error type. Will be expanded in FEATURE_003 implementation.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Stand-in variant so the enum is constructible during the skeleton phase.
    #[error("not yet implemented")]
    NotYetImplemented,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_displays() {
        let err = Error::NotYetImplemented;
        assert_eq!(err.to_string(), "not yet implemented");
    }
}
