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
}
