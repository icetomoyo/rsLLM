//! # rsllm-kvcache
//!
//! KV cache management for rsLLM.
//!
//! Layouts:
//!
//! - [`dsv4`] — three-tier KV cache for DeepSeek V4 Flash (FEATURE_006, v0.1.0)
//!
//! Future layouts (v0.1.x+, not yet implemented):
//!
//! - `Dense` — contiguous buffer per layer (FEATURE_010 et al.)
//! - `Paged` — vLLM-style block management
//! - `SlidingWindow` — Mistral / Phi style
//!
//! On-disk persistence (`KVC v2` format) borrows the file layout from
//! `ds4.c` under MIT and is planned for FEATURE_022 (v0.2.0). See
//! `docs/features/v0.2.0.md`.

pub mod dsv4;

/// KV cache layout strategy. v0.1.0 only ships [`KvLayout::DsV4ThreeTier`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvLayout {
    /// DeepSeek V4 Flash three-tier: SWA ring + compressed pool + indexer.
    DsV4ThreeTier,
}

/// Errors raised by the KV cache.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// A buffer length doesn't match the expected shape.
    #[error("{what}: expected {expected}, got {actual}")]
    ShapeMismatch {
        /// Operation / field whose shape disagreed.
        what: &'static str,
        /// Expected length.
        expected: usize,
        /// Actual length passed by the caller.
        actual: usize,
    },

    /// An index access went past the live length of a buffer.
    #[error("{what}: index {idx} out of range (len = {len})")]
    IndexOutOfRange {
        /// Operation / field whose index was out of range.
        what: &'static str,
        /// Requested index.
        idx: usize,
        /// Live length of the buffer.
        len: usize,
    },

    /// A layer index exceeded `DSV4_N_LAYER`.
    #[error("layer index {idx} out of range (max {max})")]
    InvalidLayer {
        /// Requested layer index.
        idx: usize,
        /// Upper bound (typically `DSV4_N_LAYER`).
        max: usize,
    },

    /// A compressed-KV pool emission would exceed the pool's capacity.
    /// Either the context length exceeded the sizing assumption, or
    /// the caller forgot to `clear()` after a session boundary.
    #[error("compressed KV pool is full (capacity {cap})")]
    CompressedPoolFull {
        /// Configured maximum row capacity.
        cap: usize,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_distinct() {
        let a = KvLayout::DsV4ThreeTier;
        let b = KvLayout::DsV4ThreeTier;
        assert_eq!(a, b);
    }
}
