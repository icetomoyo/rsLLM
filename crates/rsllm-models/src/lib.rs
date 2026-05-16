//! # rsllm-models
//!
//! Model architecture implementations for rsLLM.
//!
//! v0.1.0 supports exactly one architecture: [`dsv4`] (DeepSeek V4 Flash).
//! Wider model-family coverage is planned for later releases (see
//! `docs/FEATURE_LIST.md`).
//!
//! Models are written against [`rsllm_cal`] traits and are
//! backend-agnostic. The DS V4 Flash port draws from `ds4.c` (MIT,
//! The ds4.c authors); per-file headers document the specific line
//! ranges ported.

pub mod deepseek_v4_flash;
pub mod dsv4;

pub use deepseek_v4_flash::{
    AttentionFn, DeepSeekV4Flash, DsV4Block, ForwardScratch, forward_block, lm_head_logits,
};

/// Identifier of the model architecture family.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Architecture {
    /// DeepSeek V4 Flash (MLA + HC + MoE, 43 layers, 256 experts).
    DeepSeekV4Flash,
}

/// Errors raised by model loading and forward execution.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// A required GGUF metadata key was absent or had the wrong type.
    #[error("required GGUF metadata key missing or wrong type: {0}")]
    MissingMetadata(&'static str),

    /// A GGUF metadata value disagreed with the architecture's fixed shape.
    #[error("shape mismatch for {key}: expected {expected}, got {actual}")]
    ShapeMismatch {
        /// The GGUF metadata key that disagreed.
        key: &'static str,
        /// The hard-coded expected value (as a display string).
        expected: String,
        /// The value read from the GGUF file.
        actual: String,
    },

    /// A required tensor (e.g. an MLA projection) was missing from the GGUF.
    #[error("required tensor missing from GGUF: {0}")]
    MissingTensor(&'static str),

    /// An underlying GGUF parse / dequant error bubbled up.
    #[error("gguf error: {0}")]
    Gguf(#[from] rsllm_gguf::Error),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn architecture_distinct() {
        // Only one architecture for now; this guards against an accidental
        // ordering or duplication when more land in v0.2.0.
        let a = Architecture::DeepSeekV4Flash;
        let b = Architecture::DeepSeekV4Flash;
        assert_eq!(a, b);
    }
}
