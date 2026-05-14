//! # rsllm-kvcache
//!
//! KV cache management for rsLLM.
//!
//! Supports multiple layouts:
//!
//! - `Dense` — contiguous buffer per layer (FEATURE_006, v0.1.0)
//! - `Paged` — vLLM-style block management (FEATURE_010, v0.1.1)
//! - `Mla` — DeepSeek Multi-Latent Attention (FEATURE_021, v0.2.0)
//! - `SlidingWindow` — Mistral / Phi style
//!
//! On-disk persistence uses the `KVC v2` binary format, which extends the
//! file layout pioneered by `ds4.c` with a version field, model fingerprint,
//! and incremental update support. See `docs/features/v0.2.0.md#feature_022`.

/// KV cache layout strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvLayout {
    /// Contiguous per-layer buffer. Simplest, used in v0.1.0.
    Dense,
    /// Paged blocks of fixed token count. vLLM-style.
    Paged,
    /// DeepSeek Multi-Latent Attention compressed layout.
    Mla,
    /// Sliding-window cache (Mistral / Phi).
    SlidingWindow,
}

/// Placeholder error type. Will be expanded in FEATURE_006 implementation.
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
    fn layouts_distinct() {
        assert_ne!(KvLayout::Dense, KvLayout::Paged);
        assert_ne!(KvLayout::Mla, KvLayout::SlidingWindow);
    }
}
