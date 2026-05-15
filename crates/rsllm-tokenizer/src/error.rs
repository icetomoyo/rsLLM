//! Errors produced by the rsllm-tokenizer crate.

use thiserror::Error;

/// Errors returned by tokenizer construction and encode / decode operations.
#[derive(Debug, Error)]
pub enum Error {
    /// A required GGUF metadata key for tokenizer construction is missing
    /// from the file. Includes the key path that was missing.
    #[error("GGUF metadata is missing required tokenizer key `{0}`")]
    MissingKey(&'static str),

    /// A GGUF metadata key was present but had the wrong type (e.g.
    /// `tokenizer.ggml.tokens` was not a String array).
    #[error("GGUF metadata key `{key}` has the wrong type ({reason})")]
    WrongMetadataType {
        /// The metadata key that had the wrong type.
        key: &'static str,
        /// Human-readable description of the mismatch.
        reason: &'static str,
    },

    /// The GGUF file declares a pre-tokenizer name that this build does not
    /// implement. v0.1.0 only supports `"joyai-llm"` (DeepSeek V4 Flash).
    #[error("unsupported tokenizer.ggml.pre value `{0}` (v0.1.0 only supports `joyai-llm`)")]
    UnsupportedPreTokenizer(String),

    /// The vocab loaded from GGUF metadata does not contain a special token
    /// that the DeepSeek V4 Flash chat protocol requires (e.g. BOS, EOS,
    /// `<｜User｜>`, `<｜Assistant｜>`).
    #[error("required special token `{0}` is missing from vocab")]
    MissingSpecialToken(&'static str),

    /// A merge entry in `tokenizer.ggml.merges` is malformed (not exactly
    /// two space-separated parts).
    #[error("merge entry `{0}` is malformed (expected `lhs rhs`)")]
    MalformedMerge(String),

    /// Decoding produced a byte sequence that is not valid UTF-8.
    /// Can happen mid-stream when only part of a multi-byte codepoint has
    /// been emitted; callers should buffer until the next decode call.
    #[error("decoded byte stream is not valid UTF-8 yet (partial codepoint?)")]
    DecodePartialUtf8,

    /// The vocab or merge table declares more entries than fit in `u32`,
    /// which is the token-id type. v0.1.0 caps both at `u32::MAX`.
    #[error("`{key}` has {len} entries; maximum supported is u32::MAX")]
    TableTooLarge {
        /// The metadata key whose table is too large.
        key: &'static str,
        /// The declared length.
        len: usize,
    },
}
