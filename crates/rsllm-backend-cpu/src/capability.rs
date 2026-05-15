//! Runtime CPU capability detection.
//!
//! Picks one of `"neon"` (aarch64 with the `dotprod` extension),
//! `"avx512"` (x86_64 with `avx512f` + `avx512vnni`), `"avx2"` (x86_64
//! with `avx2`), or `"scalar"`. The selection is computed once at
//! `CpuBackend::new()` and used to gate SIMD kernel dispatch.
//!
//! The intent is **runtime** detection so a single binary built with
//! `cargo build --release` can run on a baseline x86_64 host without
//! AVX-512 yet still take advantage of it on a Strix Halo. Targets
//! that lack the underlying instruction set entirely (e.g. building
//! for `wasm32`) bottom out at the scalar tier.

use rsllm_cal::BackendCapability;

/// SIMD / extension tier picked by [`detect`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimdTier {
    /// aarch64 + `dotprod`. Implies `vdotq_s32` is available.
    Neon,
    /// x86_64 + `avx512f` + `avx512vnni`. Implies `_mm512_dpbusd_epi32`.
    Avx512,
    /// x86_64 + `avx2`. No VNNI — uses 16×16 → 32-bit shift-and-add
    /// idiom for Q8_0 dot products.
    Avx2,
    /// Plain scalar fallback. Works everywhere.
    Scalar,
}

impl SimdTier {
    /// Stable string tier name used in [`BackendCapability::arch`].
    pub const fn name(self) -> &'static str {
        match self {
            Self::Neon => "neon",
            Self::Avx512 => "avx512",
            Self::Avx2 => "avx2",
            Self::Scalar => "scalar",
        }
    }
}

/// Detect the best SIMD tier available on the host at runtime.
///
/// The check is a `std::arch::is_*_feature_detected!` cascade in
/// preference order: NEON dotprod > AVX-512 VNNI > AVX2 > scalar. On
/// targets that aren't aarch64 or x86_64, returns [`SimdTier::Scalar`].
pub fn detect() -> SimdTier {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("dotprod") {
            return SimdTier::Neon;
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512vnni") {
            return SimdTier::Avx512;
        }
        if is_x86_feature_detected!("avx2") {
            return SimdTier::Avx2;
        }
    }
    SimdTier::Scalar
}

/// Build the public capability descriptor from a [`SimdTier`].
pub(crate) fn capability_for(tier: SimdTier) -> BackendCapability {
    BackendCapability {
        backend: "cpu",
        arch: tier.name(),
        // All four kernels land across F004 phases B-D. The capability
        // descriptor advertises what's *implemented*, not just what's
        // architecturally possible — so updated as each kernel ships.
        q4k_matmul: false,
        q2k_matmul: false,
        iq2xxs_matmul: false,
        fp8_kv_cache: false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_returns_valid_tier() {
        // Whatever tier the host returns must round-trip through
        // capability_for with the same name.
        let tier = detect();
        let cap = capability_for(tier);
        assert_eq!(cap.arch, tier.name());
        assert_eq!(cap.backend, "cpu");
    }

    #[test]
    fn tier_names_are_stable() {
        assert_eq!(SimdTier::Neon.name(), "neon");
        assert_eq!(SimdTier::Avx512.name(), "avx512");
        assert_eq!(SimdTier::Avx2.name(), "avx2");
        assert_eq!(SimdTier::Scalar.name(), "scalar");
    }
}
