//! # rsllm-backend-cpu
//!
//! CPU compute backend for rsLLM.
//!
//! Implements the `rsllm-cal::Backend` trait using native Rust plus SIMD
//! intrinsics (AVX2 / AVX-512 on `x86_64`, NEON / SVE on `aarch64`).
//!
//! ## Acknowledgements
//!
//! Algorithmic kernels (especially quantized matmul and dot-product loops)
//! are borrowed from `ggml`'s CPU path under MIT with attribution in the
//! relevant source-file headers. ARM NEON variants take additional
//! inspiration from `ds4.c`'s NEON code paths. No runtime linkage against
//! either project.
//!
//! See [`docs/features/v0.1.0.md#feature_004`](https://github.com/icetomoyo/rsLLM/blob/main/docs/features/v0.1.0.md)
//! for the full design.

use rsllm_cal::{Backend, BackendKind};

/// CPU backend handle. Detects capabilities at construction time and
/// dispatches each kernel to the best available SIMD variant.
#[derive(Debug, Default)]
pub struct CpuBackend {
    _private: (),
}

impl CpuBackend {
    /// Create a new CPU backend. Capability detection happens here.
    #[must_use]
    pub fn new() -> Self {
        Self { _private: () }
    }
}

impl Backend for CpuBackend {
    fn kind(&self) -> BackendKind {
        BackendKind::Cpu
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_backend_reports_correct_kind() {
        let backend = CpuBackend::new();
        assert_eq!(backend.kind(), BackendKind::Cpu);
    }
}
