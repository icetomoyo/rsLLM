//! CPU compute kernels for DeepSeek V4 Flash inference.
//!
//! Module layout follows a three-tier pattern: each kernel has a
//! scalar reference (in [`scalar`]) plus optional NEON / AVX-512
//! specializations gated by `cfg(target_arch = ...)`. The public
//! kernel functions in this module live at the top level and pick
//! the best variant at call time using the [`crate::SimdTier`]
//! carried by the [`crate::CpuBackend`].
//!
//! Kernel coverage rolls out across F004 sub-phases (A-E); see the
//! crate-level docs for the schedule.

pub mod q8_0;
pub mod scalar;

#[cfg(target_arch = "aarch64")]
pub mod neon;

#[cfg(target_arch = "x86_64")]
pub mod avx2;

#[cfg(target_arch = "x86_64")]
pub mod avx512;

use crate::SimdTier;
use crate::error::Error;

/// Compute Root-Mean-Square LayerNorm: `out[i] = x[i] * weight[i] * rsqrt(mean(x^2) + eps)`.
///
/// Per-token normalization used at the start of every MLA / FFN block
/// in DS V4 Flash with `eps = 1e-6`.
///
/// # Errors
/// Returns [`Error::ShapeMismatch`] if `out`, `x`, and `weight` have
/// different lengths, or if they're empty.
pub fn rmsnorm(
    out: &mut [f32],
    x: &[f32],
    weight: &[f32],
    eps: f32,
    tier: SimdTier,
) -> Result<(), Error> {
    if x.len() != weight.len() || x.len() != out.len() {
        return Err(Error::ShapeMismatch(
            "rmsnorm: out / x / weight length mismatch",
        ));
    }
    if x.is_empty() {
        return Err(Error::ShapeMismatch("rmsnorm: empty input"));
    }
    let _ = tier; // future: dispatch to neon::rmsnorm / avx512::rmsnorm
    scalar::rmsnorm(out, x, weight, eps);
    Ok(())
}

/// Return the index of the maximum-valued element in `logits`. Ties
/// break to the lowest index (mirrors `ds4_engine_generate_argmax`,
/// `ds4.c:14183-14194`).
///
/// # Errors
/// Returns [`Error::ShapeMismatch`] if `logits` is empty.
pub fn argmax(logits: &[f32], tier: SimdTier) -> Result<u32, Error> {
    if logits.is_empty() {
        return Err(Error::ShapeMismatch("argmax: empty input"));
    }
    let _ = tier;
    Ok(scalar::argmax(logits))
}

/// SwiGLU activation: `out[i] = silu(gate[i]) * up[i]`. See
/// [`scalar::swiglu`].
///
/// # Errors
/// Returns [`Error::ShapeMismatch`] if `out`, `gate`, `up` differ in
/// length.
pub fn swiglu(out: &mut [f32], gate: &[f32], up: &[f32], tier: SimdTier) -> Result<(), Error> {
    if out.len() != gate.len() || out.len() != up.len() {
        return Err(Error::ShapeMismatch(
            "swiglu: out / gate / up length mismatch",
        ));
    }
    let _ = tier;
    scalar::swiglu(out, gate, up);
    Ok(())
}

/// In-place numerically stable softmax. See [`scalar::softmax`].
///
/// Empty input is a no-op (returns `Ok(())`).
pub fn softmax(x: &mut [f32], tier: SimdTier) -> Result<(), Error> {
    let _ = tier;
    scalar::softmax(x);
    Ok(())
}

/// Sink-aware attention softmax. See [`scalar::softmax_attn`]. Returns
/// the implied sink weight.
pub fn softmax_attn(scores: &mut [f32], sink: f32, tier: SimdTier) -> Result<f32, Error> {
    let _ = tier;
    Ok(scalar::softmax_attn(scores, sink))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rmsnorm_rejects_mismatched_lengths() {
        let mut out = vec![0.0; 4];
        let x = vec![1.0; 4];
        let weight = vec![1.0; 8];
        let err = rmsnorm(&mut out, &x, &weight, 1e-6, SimdTier::Scalar).unwrap_err();
        assert!(matches!(err, Error::ShapeMismatch(_)));
    }

    #[test]
    fn rmsnorm_rejects_empty() {
        let mut out: Vec<f32> = Vec::new();
        let x: Vec<f32> = Vec::new();
        let w: Vec<f32> = Vec::new();
        let err = rmsnorm(&mut out, &x, &w, 1e-6, SimdTier::Scalar).unwrap_err();
        assert!(matches!(err, Error::ShapeMismatch(_)));
    }

    #[test]
    fn argmax_rejects_empty() {
        let err = argmax(&[], SimdTier::Scalar).unwrap_err();
        assert!(matches!(err, Error::ShapeMismatch(_)));
    }

    #[test]
    fn argmax_dispatches_through_to_scalar() {
        let logits = vec![1.0, 3.0, 2.0, 3.0];
        // Tie at indices 1 and 3 — argmax returns the smaller index.
        assert_eq!(argmax(&logits, SimdTier::Scalar).unwrap(), 1);
    }

    #[test]
    fn swiglu_rejects_mismatched_lengths() {
        let gate = vec![1.0_f32; 4];
        let up = vec![1.0_f32; 4];
        let mut out = vec![0.0_f32; 8];
        let err = swiglu(&mut out, &gate, &up, SimdTier::Scalar).unwrap_err();
        assert!(matches!(err, Error::ShapeMismatch(_)));
    }

    #[test]
    fn softmax_empty_is_ok() {
        let mut x: Vec<f32> = Vec::new();
        softmax(&mut x, SimdTier::Scalar).unwrap();
        assert!(x.is_empty());
    }
}
