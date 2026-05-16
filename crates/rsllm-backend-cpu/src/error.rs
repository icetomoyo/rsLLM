//! Errors produced by CPU kernels.

use thiserror::Error;

/// Errors returned by `rsllm-backend-cpu` kernels.
#[derive(Debug, Error)]
pub enum Error {
    /// A kernel argument had a length / shape that doesn't match the
    /// other arguments. Includes a human-readable description of the
    /// mismatch.
    #[error("kernel argument shape mismatch: {0}")]
    ShapeMismatch(&'static str),

    /// A kernel argument's element count doesn't divide evenly by the
    /// expected block size (e.g. RMSNorm hidden_size that's not a
    /// multiple of the SIMD vector width).
    #[error("`{what}` length {actual} is not a multiple of {block}")]
    NotBlockAligned {
        /// Argument name in the kernel signature.
        what: &'static str,
        /// Actual element count.
        actual: usize,
        /// Required block size.
        block: usize,
    },

    /// A kernel input contained a non-finite value (NaN or ±∞) where a
    /// finite value was required. Most kernels accept any `f32`, but the
    /// Q8_0 quantizer would silently emit zero quants for NaN inputs;
    /// callers see this error so data corruption is signalled rather
    /// than masked.
    #[error("`{0}` contains a non-finite value (NaN or Inf)")]
    NonFiniteInput(&'static str),

    /// A caller asked the CPU backend to use a SIMD tier that the host
    /// CPU does not support. Returned by
    /// [`crate::CpuBackend::try_with_tier`] — calling an `unsafe
    /// #[target_feature]` SIMD kernel without the underlying CPU
    /// extension would produce a SIGILL at runtime, so we refuse at
    /// construction time instead.
    #[error("SIMD tier `{0}` is not supported on this host")]
    UnsupportedTier(&'static str),
}
