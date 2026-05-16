//! Runtime CPU capability detection.
//!
//! Picks one of `"neon"` (aarch64 with the `dotprod` extension),
//! `"avx512"` (x86_64 with `avx512f` + `avx512bw` — the extensions the
//! `_mm512_cvtepi8_epi16` + `_mm512_madd_epi16` kernel actually uses;
//! VNNI is a future optimization tier), `"avx2"` (x86_64 with `avx2`),
//! or `"scalar"`. The selection is computed once at
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
    /// x86_64 + `avx512f` + `avx512bw`. Implies `_mm512_cvtepi8_epi16` +
    /// `_mm512_madd_epi16` are available. The optional `avx512vnni`
    /// promotion (for `_mm512_dpbusd_epi32`) lands as a future
    /// optimization in phase E.
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

    /// `true` if the current host actually supports this tier. Used by
    /// [`crate::CpuBackend::with_tier`] to refuse to construct a backend
    /// pinned to a tier the CPU cannot execute (which would result in a
    /// SIGILL at the first SIMD kernel call).
    ///
    /// [`SimdTier::Scalar`] is universally supported.
    pub fn is_supported_on_host(self) -> bool {
        match self {
            Self::Scalar => true,
            Self::Neon => {
                #[cfg(target_arch = "aarch64")]
                {
                    std::arch::is_aarch64_feature_detected!("dotprod")
                }
                #[cfg(not(target_arch = "aarch64"))]
                {
                    false
                }
            }
            Self::Avx512 => {
                #[cfg(target_arch = "x86_64")]
                {
                    is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw")
                }
                #[cfg(not(target_arch = "x86_64"))]
                {
                    false
                }
            }
            Self::Avx2 => {
                #[cfg(target_arch = "x86_64")]
                {
                    is_x86_feature_detected!("avx2")
                }
                #[cfg(not(target_arch = "x86_64"))]
                {
                    false
                }
            }
        }
    }
}

/// Detect the best SIMD tier available on the host at runtime.
///
/// The check is a `std::arch::is_*_feature_detected!` cascade in
/// preference order: NEON dotprod > AVX-512F+BW > AVX2 > scalar. On
/// targets that aren't aarch64 or x86_64, returns [`SimdTier::Scalar`].
///
/// `avx512bw` (not `avx512vnni`) is the load-bearing extension for the
/// current AVX-512 Q8_0 kernel — it uses `_mm512_cvtepi8_epi16` /
/// `_mm512_madd_epi16`, both of which require BW. VNNI promotion is
/// future work.
pub fn detect() -> SimdTier {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("dotprod") {
            return SimdTier::Neon;
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw") {
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

    #[test]
    fn scalar_always_supported() {
        assert!(SimdTier::Scalar.is_supported_on_host());
    }

    #[test]
    fn detected_tier_supported_on_host() {
        // The tier `detect()` chooses must satisfy is_supported_on_host.
        let tier = detect();
        assert!(
            tier.is_supported_on_host(),
            "detect() returned {tier:?} but is_supported_on_host says no"
        );
    }
}
