//! # rsllm-backend-cpu
//!
//! CPU compute backend for rsLLM.
//!
//! Implements the [`rsllm_cal::Backend`] trait using native Rust plus
//! SIMD intrinsics (AVX2 / AVX-512 VNNI on `x86_64`, NEON dotprod on
//! `aarch64`). Capability detection picks the best tier at construction
//! time; missing extensions fall back through a clear preference order
//! down to plain scalar.
//!
//! Kernel coverage lands across the FEATURE_004 sub-phases (A-E). The
//! current phase A delivers:
//!
//! - Backend handle + capability detection ([`CpuBackend::new`]).
//! - `ops::*` module structure (scalar / NEON / AVX-512 stubs).
//! - Skeleton scalar kernels for [`ops::rmsnorm`] and [`ops::argmax`]
//!   so model graph code can compile against the API immediately.
//!
//! ## Acknowledgements
//!
//! Algorithmic kernels (especially quantized matmul and dot-product
//! loops) are borrowed from `ggml`'s CPU path under MIT with attribution
//! in the relevant source-file headers. ARM NEON variants take
//! additional inspiration from `ds4.c`'s NEON code paths. No runtime
//! linkage against either project.
//!
//! See [`docs/features/v0.1.0.md#feature_004`](https://github.com/icetomoyo/rsLLM/blob/main/docs/features/v0.1.0.md)
//! for the full design.

mod capability;
pub mod error;
pub mod ops;
pub mod parallel;

use rsllm_cal::{Backend, BackendCapability, BackendKind};

pub use capability::{SimdTier, detect};
pub use error::Error;

/// CPU backend handle. Detects capabilities at construction time and
/// dispatches each kernel to the best available SIMD variant.
#[derive(Debug, Clone)]
pub struct CpuBackend {
    tier: SimdTier,
}

impl CpuBackend {
    /// Create a new CPU backend, detecting the best SIMD tier on the
    /// host. Construction is cheap; the result is safe to share across
    /// threads (it carries no allocated state).
    #[must_use]
    pub fn new() -> Self {
        Self { tier: detect() }
    }

    /// Construct with an explicit SIMD tier. Useful for unit tests that
    /// want to exercise the scalar fallback on a host that would
    /// otherwise pick NEON / AVX-512.
    #[must_use]
    pub fn with_tier(tier: SimdTier) -> Self {
        Self { tier }
    }

    /// The SIMD tier this backend was constructed for.
    pub fn tier(&self) -> SimdTier {
        self.tier
    }
}

impl Default for CpuBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl Backend for CpuBackend {
    fn kind(&self) -> BackendKind {
        BackendKind::Cpu
    }

    fn capability(&self) -> BackendCapability {
        capability::capability_for(self.tier)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_backend_reports_correct_kind() {
        let backend = CpuBackend::new();
        assert_eq!(backend.kind(), BackendKind::Cpu);
        assert_eq!(backend.capability().backend, "cpu");
    }

    #[test]
    fn with_tier_forces_scalar() {
        let backend = CpuBackend::with_tier(SimdTier::Scalar);
        assert_eq!(backend.tier(), SimdTier::Scalar);
        assert_eq!(backend.capability().arch, "scalar");
    }
}
