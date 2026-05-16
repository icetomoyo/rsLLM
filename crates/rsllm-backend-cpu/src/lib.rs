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

    /// Construct with an explicit SIMD tier, validating that the host
    /// actually supports it. Use this to force scalar in tests or to
    /// down-tier the dispatcher for benchmarking.
    ///
    /// # Errors
    /// Returns [`Error::UnsupportedTier`] if `tier` requires CPU
    /// extensions the host lacks. Calling an `unsafe #[target_feature]`
    /// kernel without the underlying extension would SIGILL at the
    /// first SIMD instruction, so we refuse at construction time.
    pub fn try_with_tier(tier: SimdTier) -> Result<Self, Error> {
        if !tier.is_supported_on_host() {
            return Err(Error::UnsupportedTier(tier.name()));
        }
        Ok(Self { tier })
    }

    /// Construct with an explicit SIMD tier without checking host
    /// support. **Only safe for [`SimdTier::Scalar`]** (which is
    /// universally supported). Other tiers will SIGILL on hosts that
    /// lack the corresponding extension. Prefer
    /// [`Self::try_with_tier`] in production code.
    #[must_use]
    pub fn with_tier(tier: SimdTier) -> Self {
        debug_assert!(
            tier.is_supported_on_host(),
            "with_tier({tier:?}) called on a host without the required extension; \
             use try_with_tier for runtime-checked construction",
        );
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

    #[test]
    fn try_with_tier_scalar_succeeds() {
        let backend = CpuBackend::try_with_tier(SimdTier::Scalar).unwrap();
        assert_eq!(backend.tier(), SimdTier::Scalar);
    }

    #[test]
    fn try_with_tier_rejects_unsupported() {
        // At least one of NEON / AVX-512 / AVX2 is unsupported on the
        // current host (they're cross-architecture exclusive), so one
        // try_with_tier call should fail.
        let candidates = [SimdTier::Neon, SimdTier::Avx512, SimdTier::Avx2];
        let any_rejected = candidates
            .iter()
            .any(|&t| matches!(CpuBackend::try_with_tier(t), Err(Error::UnsupportedTier(_))));
        assert!(
            any_rejected,
            "expected at least one of {candidates:?} to be rejected on this host"
        );
    }
}
