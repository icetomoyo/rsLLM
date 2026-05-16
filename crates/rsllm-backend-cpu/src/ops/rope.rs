//! RoPE-YaRN rotary positional embedding.
//!
//! DeepSeek V4 Flash applies RoPE only to the **tail** of each attention
//! head — the first `n_nope = head_dim - n_rot` lanes carry "no-position"
//! latent features and the last `n_rot` lanes carry the rotated
//! position. The rotation uses YaRN frequency scaling for long-context
//! extrapolation (`freq_base = 10000`, `original_context = 65536`,
//! `scale_factor = 16`, `beta_fast = 32`, `beta_slow = 1`).
//!
//! Ported by reference from `ds4.c:4529-4596` (MIT, The ds4.c authors):
//!
//! - `rope_yarn_ramp` — smooth interpolation window
//! - `rope_yarn_corr_dim` — dimension at which `freq * n_ctx_orig` completes one full rotation
//! - `rope_yarn_corr_dims` — `[start, end]` of the ramp
//! - `rope_tail_ext_inplace` — in-place rotation of each head's tail

use core::f32::consts::PI;

use crate::SimdTier;
use crate::error::Error;

/// Smooth ramp from 1 (at `i0/2 ≤ low`) to 0 (at `i0/2 ≥ high`). Used
/// by YaRN to mix the linear-interpolation frequency with the
/// frequency-extrapolation result, so the transition between the two
/// regimes is smooth across dimensions.
#[inline]
fn yarn_ramp(low: f32, high: f32, i0: i32) -> f32 {
    let y = ((i0 / 2) as f32 - low) / (high - low).max(0.001);
    1.0 - y.clamp(0.0, 1.0)
}

/// The dimension index at which a sinusoidal frequency `n_rot` would
/// complete `n_ctx_orig / (2π)` rotations across `n_ctx_orig` tokens.
/// Mirrors `rope_yarn_corr_dim` (`ds4.c:4534-4536`).
#[inline]
fn yarn_corr_dim(n_dims: i32, n_ctx_orig: u64, n_rot: f32, base: f32) -> f32 {
    (n_dims as f32) * (n_ctx_orig as f32 / (n_rot * 2.0 * PI)).ln() / (2.0 * base.ln())
}

/// Compute the `[start, end]` ramp window for YaRN frequency mixing.
fn yarn_corr_dims(
    n_dims: i32,
    n_ctx_orig: u64,
    freq_base: f32,
    beta_fast: f32,
    beta_slow: f32,
) -> [f32; 2] {
    let start = yarn_corr_dim(n_dims, n_ctx_orig, beta_fast, freq_base).floor();
    let end = yarn_corr_dim(n_dims, n_ctx_orig, beta_slow, freq_base).ceil();
    [start.max(0.0), end.min((n_dims - 1) as f32)]
}

/// Parameter bundle for RoPE-YaRN tail rotation. Mirrors the
/// twelve-arg signature of `ds4.c:4548-4561` minus the `inverse` flag
/// (handled by [`RoPEParams::inverse`]).
#[derive(Debug, Clone, Copy)]
pub struct RoPEParams {
    /// Number of attention heads to rotate.
    pub n_head: u32,
    /// Total dimension per head (must be > `n_rot`).
    pub head_dim: u32,
    /// Tail length per head that gets rotated. Must be even.
    pub n_rot: u32,
    /// Token position (sequence index).
    pub pos: u32,
    /// Original training context length (e.g. 65536 for DS V4 Flash).
    pub n_ctx_orig: u64,
    /// Base frequency. DS V4 Flash uses `10000.0` for dense layers and
    /// a different base for compressed layers.
    pub freq_base: f32,
    /// Linear frequency scale (e.g. `1.0 / scale_factor`).
    pub freq_scale: f32,
    /// YaRN extrapolation factor; `0.0` disables the ramp / mscale.
    pub ext_factor: f32,
    /// Attention scaling factor (typically `1.0`).
    pub attn_factor: f32,
    /// Frequencies above this dimension extrapolate purely.
    pub beta_fast: f32,
    /// Frequencies below this dimension interpolate purely.
    pub beta_slow: f32,
    /// `true` to rotate back (used by attn-output projection).
    pub inverse: bool,
}

/// In-place RoPE-YaRN rotation of the tail (`n_rot` lanes) of each
/// attention head.
///
/// `x` is a flat `[n_head × head_dim]` buffer; only the last `n_rot`
/// lanes per head are touched. The first `n_nope = head_dim - n_rot`
/// lanes are passed through unchanged.
///
/// # Errors
/// Returns [`Error::ShapeMismatch`] if
///   * `x.len() != n_head * head_dim`
///   * `n_rot > head_dim`
///   * `n_rot` is odd (RoPE rotates `(x0, x1)` pairs)
pub fn rope_yarn_tail(x: &mut [f32], params: &RoPEParams, tier: SimdTier) -> Result<(), Error> {
    let _ = tier;
    let n = params.n_head as usize * params.head_dim as usize;
    if x.len() != n {
        return Err(Error::ShapeMismatch(
            "rope_yarn_tail: x must be n_head*head_dim",
        ));
    }
    if params.n_rot > params.head_dim {
        return Err(Error::ShapeMismatch(
            "rope_yarn_tail: n_rot must not exceed head_dim",
        ));
    }
    if !params.n_rot.is_multiple_of(2) {
        return Err(Error::ShapeMismatch(
            "rope_yarn_tail: n_rot must be even (rotates 2-lane pairs)",
        ));
    }
    // When YaRN ramp / mscale is engaged, `1.0 / freq_scale` is taken;
    // a non-positive freq_scale would produce inf / NaN that silently
    // corrupts every position embedding. Reject up front.
    if params.ext_factor != 0.0 && !(params.freq_scale > 0.0 && params.freq_scale.is_finite()) {
        return Err(Error::NonFiniteInput(
            "rope_yarn_tail: freq_scale must be positive and finite when ext_factor != 0",
        ));
    }
    scalar_rope_yarn_tail(x, params);
    Ok(())
}

fn scalar_rope_yarn_tail(x: &mut [f32], p: &RoPEParams) {
    let n_nope = (p.head_dim - p.n_rot) as usize;
    let theta_scale = p.freq_base.powf(-2.0 / p.n_rot as f32);
    let sin_sign: f32 = if p.inverse { -1.0 } else { 1.0 };

    let corr_dims = if p.ext_factor != 0.0 {
        yarn_corr_dims(
            p.n_rot as i32,
            p.n_ctx_orig,
            p.freq_base,
            p.beta_fast,
            p.beta_slow,
        )
    } else {
        [0.0, 0.0]
    };

    let head_dim = p.head_dim as usize;
    let n_rot = p.n_rot as usize;
    for h in 0..p.n_head as usize {
        let tail_start = h * head_dim + n_nope;
        let mut theta_extrap = p.pos as f32;
        let mut i = 0usize;
        while i < n_rot {
            let theta_interp = p.freq_scale * theta_extrap;
            let mut theta = theta_interp;
            let mut mscale = p.attn_factor;

            if p.ext_factor != 0.0 {
                let ramp_mix = yarn_ramp(corr_dims[0], corr_dims[1], i as i32) * p.ext_factor;
                theta = theta_interp * (1.0 - ramp_mix) + theta_extrap * ramp_mix;
                // ds4.c:4582 — m-scale correction once freq_scale != 1.
                mscale *= 1.0 + 0.1 * (1.0 / p.freq_scale).ln();
            }

            let c = theta.cos() * mscale;
            let s = sin_sign * theta.sin() * mscale;
            let x0 = x[tail_start + i];
            let x1 = x[tail_start + i + 1];
            x[tail_start + i] = x0 * c - x1 * s;
            x[tail_start + i + 1] = x0 * s + x1 * c;

            theta_extrap *= theta_scale;
            i += 2;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn baseline_params(n_head: u32, head_dim: u32, n_rot: u32, pos: u32) -> RoPEParams {
        RoPEParams {
            n_head,
            head_dim,
            n_rot,
            pos,
            n_ctx_orig: 65536,
            freq_base: 10000.0,
            freq_scale: 1.0,
            ext_factor: 0.0,
            attn_factor: 1.0,
            beta_fast: 32.0,
            beta_slow: 1.0,
            inverse: false,
        }
    }

    #[test]
    fn rejects_odd_n_rot() {
        let mut x = vec![0.0_f32; 8];
        let mut p = baseline_params(1, 8, 7, 0);
        p.n_rot = 7;
        let err = rope_yarn_tail(&mut x, &p, SimdTier::Scalar).unwrap_err();
        assert!(matches!(err, Error::ShapeMismatch(_)));
    }

    #[test]
    fn rejects_nrot_exceeds_head_dim() {
        let mut x = vec![0.0_f32; 8];
        let mut p = baseline_params(1, 8, 16, 0);
        p.n_rot = 16;
        let err = rope_yarn_tail(&mut x, &p, SimdTier::Scalar).unwrap_err();
        assert!(matches!(err, Error::ShapeMismatch(_)));
    }

    #[test]
    fn rejects_wrong_shape() {
        let mut x = vec![0.0_f32; 7]; // not n_head*head_dim
        let p = baseline_params(1, 8, 4, 0);
        let err = rope_yarn_tail(&mut x, &p, SimdTier::Scalar).unwrap_err();
        assert!(matches!(err, Error::ShapeMismatch(_)));
    }

    #[test]
    fn pos_zero_is_identity() {
        // At pos=0 theta_extrap=0 → theta=0 → cos=1, sin=0 → identity.
        let head_dim = 8;
        let n_rot = 4;
        let mut x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let orig = x.clone();
        let p = baseline_params(1, head_dim, n_rot, 0);
        rope_yarn_tail(&mut x, &p, SimdTier::Scalar).unwrap();
        for i in 0..head_dim as usize {
            assert!(
                (x[i] - orig[i]).abs() < 1e-6,
                "pos=0 should be identity, i={i}: got {}, want {}",
                x[i],
                orig[i]
            );
        }
    }

    #[test]
    fn preserves_norm_per_pair() {
        // Rotation must preserve the norm of every (x0, x1) pair (with
        // attn_factor=1, ext_factor=0). Test on a random-ish input.
        let head_dim = 16;
        let n_rot = 8;
        let n_nope = (head_dim - n_rot) as usize;
        let mut x: Vec<f32> = (0..head_dim as usize)
            .map(|i| ((i as f32) * 0.3 - 1.0).sin())
            .collect();
        let orig = x.clone();
        let p = baseline_params(1, head_dim, n_rot, 7);
        rope_yarn_tail(&mut x, &p, SimdTier::Scalar).unwrap();

        // The first n_nope lanes are unchanged.
        for i in 0..n_nope {
            assert!((x[i] - orig[i]).abs() < 1e-6);
        }
        // Each (x0, x1) pair in the tail has the same magnitude.
        for i in (n_nope..head_dim as usize).step_by(2) {
            let want = (orig[i] * orig[i] + orig[i + 1] * orig[i + 1]).sqrt();
            let got = (x[i] * x[i] + x[i + 1] * x[i + 1]).sqrt();
            assert!(
                (got - want).abs() < 1e-4,
                "pair i={i}: got {got}, want {want}"
            );
        }
    }

    #[test]
    fn inverse_round_trips() {
        // Forward then inverse rotation must recover the original input
        // (with the same parameters). This is the property MLA depends
        // on at the attn-output projection.
        let head_dim = 16;
        let n_rot = 8;
        let mut x: Vec<f32> = (0..head_dim as usize)
            .map(|i| ((i as f32) * 0.27 + 0.4).cos())
            .collect();
        let orig = x.clone();
        let mut p = baseline_params(1, head_dim, n_rot, 13);
        rope_yarn_tail(&mut x, &p, SimdTier::Scalar).unwrap();
        p.inverse = true;
        rope_yarn_tail(&mut x, &p, SimdTier::Scalar).unwrap();
        for i in 0..head_dim as usize {
            assert!(
                (x[i] - orig[i]).abs() < 1e-4,
                "round-trip mismatch at i={i}: got {}, want {}",
                x[i],
                orig[i]
            );
        }
    }

    #[test]
    fn rejects_zero_freq_scale_with_ext_factor() {
        // Guard against silent NaN poisoning when freq_scale = 0 and
        // ext_factor != 0 (1/0 → inf, ln(inf) → inf, mscale → inf).
        let mut x = vec![1.0_f32; 16];
        let mut p = baseline_params(1, 16, 8, 0);
        p.ext_factor = 1.0;
        p.freq_scale = 0.0;
        let err = rope_yarn_tail(&mut x, &p, SimdTier::Scalar).unwrap_err();
        assert!(matches!(err, Error::NonFiniteInput(_)), "got {err:?}");
    }

    #[test]
    fn rejects_negative_freq_scale_with_ext_factor() {
        let mut x = vec![1.0_f32; 16];
        let mut p = baseline_params(1, 16, 8, 0);
        p.ext_factor = 1.0;
        p.freq_scale = -1.0;
        let err = rope_yarn_tail(&mut x, &p, SimdTier::Scalar).unwrap_err();
        assert!(matches!(err, Error::NonFiniteInput(_)));
    }

    #[test]
    fn accepts_zero_freq_scale_when_ext_factor_zero() {
        // Without YaRN, freq_scale = 0 is benign (just multiplies theta
        // by 0, so every rotation is by 0 = identity). Not useful but
        // not corrupt — we should not error.
        let mut x = vec![1.0_f32; 16];
        let mut p = baseline_params(1, 16, 8, 5);
        p.ext_factor = 0.0;
        p.freq_scale = 0.0;
        rope_yarn_tail(&mut x, &p, SimdTier::Scalar).unwrap();
        assert!(x.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn yarn_extrapolation_smoke() {
        // With ext_factor=1 and a large pos beyond original context,
        // the rotation must still produce finite output. Just a smoke
        // test that the YaRN ramp / mscale paths don't NaN out.
        let head_dim = 32;
        let n_rot = 16;
        let mut x = vec![1.0_f32; head_dim as usize];
        let mut p = baseline_params(1, head_dim, n_rot, 200_000);
        p.ext_factor = 1.0;
        p.freq_scale = 1.0 / 16.0; // DS V4 Flash scale_factor = 16
        rope_yarn_tail(&mut x, &p, SimdTier::Scalar).unwrap();
        assert!(x.iter().all(|v| v.is_finite()));
    }
}
