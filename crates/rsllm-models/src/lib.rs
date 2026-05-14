//! # rsllm-models
//!
//! Model architecture implementations for rsLLM.
//!
//! Each model family lives in its own module:
//!
//! - `llama` — Llama 2 / 3 / 3.1 / 3.2 (FEATURE_005, v0.1.0)
//! - `qwen` — Qwen 2 / 2.5 (FEATURE_013, v0.1.3)
//! - `mistral` — Mistral / Mixtral (FEATURE_013/014)
//! - `deepseek` — DeepSeek V2 / V3 / R1 with MLA + MoE (FEATURE_021, v0.2.0)
//!
//! Models are written against `rsllm-cal` traits and are backend-agnostic.
//! Forward-pass code is influenced by HuggingFace's `candle` crate
//! (Apache-2.0) and `llama.cpp`'s `llama_decode_internal` (MIT). All
//! source files that contain ported code carry attribution headers.

/// Identifier of the model architecture family.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Architecture {
    /// Llama 2 / 3 / 3.1 / 3.2 family.
    Llama,
    /// Qwen 2 / 2.5 family.
    Qwen,
    /// Mistral / Mixtral family.
    Mistral,
    /// DeepSeek V2 / V3 / R1 family.
    DeepSeek,
}

/// Placeholder error type. Will be expanded as model implementations land.
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
    fn architectures_distinct() {
        assert_ne!(Architecture::Llama, Architecture::Qwen);
    }
}
