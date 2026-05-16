//! Sinkhorn–Knopp 20-iter mix-weight kernel for DeepSeek V4 Flash's
//! Hidden Compression (HC) hyper-connection layer.
//!
//! HC routes each token through `N_HC = 4` parallel residual streams.
//! Each layer's "pre" step needs:
//!
//!   * a per-stream sigmoid gate `g[h]` (n_hc lanes),
//!   * a per-stream tanh-like gate `p[h]` (n_hc lanes),
//!   * a doubly-stochastic `n_hc × n_hc` permutation matrix `c[dst,src]`
//!     that smoothly redistributes contributions among the four streams.
//!
//! The last matrix is the Sinkhorn output. Computing a "soft permutation"
//! analytically would require an SVD; ds4 uses 20 rounds of alternating
//! row / column normalization (the Sinkhorn–Knopp algorithm) to get a
//! cheap, GPU-friendly approximation. With a fixed iteration count the
//! kernel is fully deterministic.
//!
//! Ported by reference from `ds4.c:4040-4117` (MIT, The ds4.c authors).
//!
//! Constants (from `ds4.c:103-104`):
//!
//!   * `N_HC = 4`
//!   * `N_HC_SINKHORN_ITER = 20`

use crate::SimdTier;
use crate::error::Error;

/// Hyper-connection stream count for DS V4 Flash.
pub const N_HC: usize = 4;

/// Default number of Sinkhorn iterations used by DS V4 Flash.
pub const N_HC_SINKHORN_ITER: usize = 20;

/// Compute the HC mix weights for one token.
///
/// Layout of `out` (length `2*n_hc + n_hc*n_hc`):
/// - `out[0..n_hc]` — sigmoid gates g[h]
/// - `out[n_hc..2*n_hc]` — `2*sigmoid(...)` post-gates p[h]
/// - `out[2*n_hc..]` — doubly-stochastic matrix c[dst,src], flattened row-major (src + dst*n_hc)
///
/// Layout of `mix` and `base` (both length `2*n_hc + n_hc*n_hc`):
/// - `[0..n_hc]` — pre-sigmoid logits
/// - `[n_hc..2*n_hc]` — post-sigmoid logits
/// - `[2*n_hc..]` — Sinkhorn matrix input logits
///
/// `scale` is the per-channel scale vector `[pre_scale, post_scale,
/// comb_scale]`. `iters` is the Sinkhorn iteration count
/// (typically `N_HC_SINKHORN_ITER = 20`). `eps` is a small numerical
/// floor (DS V4 Flash uses `1e-6`).
///
/// # Errors
/// Returns [`Error::ShapeMismatch`] if buffer lengths don't match.
#[allow(clippy::too_many_arguments)]
pub fn hc_split_sinkhorn(
    out: &mut [f32],
    mix: &[f32],
    scale: &[f32; 3],
    base: &[f32],
    n_hc: usize,
    iters: usize,
    eps: f32,
    tier: SimdTier,
) -> Result<(), Error> {
    let _ = tier;

    if n_hc == 0 {
        return Err(Error::ShapeMismatch("hc_split_sinkhorn: n_hc must be > 0"));
    }
    if n_hc > 16 {
        return Err(Error::ShapeMismatch(
            "hc_split_sinkhorn: n_hc > 16 exceeds ds4's stack-buffer reference",
        ));
    }
    let need = 2 * n_hc + n_hc * n_hc;
    if out.len() != need || mix.len() != need || base.len() != need {
        return Err(Error::ShapeMismatch(
            "hc_split_sinkhorn: buffers must be 2*n_hc + n_hc*n_hc",
        ));
    }

    let pre_scale = scale[0];
    let post_scale = scale[1];
    let comb_scale = scale[2];

    // 1. sigmoid pre-gates g[h] = sigmoid(mix[h]*pre + base[h]) + eps
    for i in 0..n_hc {
        let z = mix[i] * pre_scale + base[i];
        out[i] = crate::ops::scalar::sigmoid(z) + eps;
    }

    // 2. tanh-ish post-gates p[h] = 2 * sigmoid(...) (no +eps; ds4.c:4060)
    for i in 0..n_hc {
        let off = n_hc + i;
        let z = mix[off] * post_scale + base[off];
        out[off] = 2.0 * crate::ops::scalar::sigmoid(z);
    }

    // 3. Stack-allocate `c[n_hc * n_hc]`. ds4 caps at 16×16. We mirror.
    let mut c = [0.0_f32; 16 * 16];

    // First Sinkhorn iteration row pass uses a stable softmax. ds4
    // unrolls the first iter as a max-subtract softmax (ds4.c:4065-4096).
    for dst in 0..n_hc {
        let mut row_max = f32::NEG_INFINITY;
        for src in 0..n_hc {
            let idx = src + dst * n_hc;
            let off = 2 * n_hc + idx;
            let v = mix[off] * comb_scale + base[off];
            c[idx] = v;
            if v > row_max {
                row_max = v;
            }
        }
        let mut row_sum = 0.0_f32;
        for src in 0..n_hc {
            let idx = src + dst * n_hc;
            let v = (c[idx] - row_max).exp();
            c[idx] = v;
            row_sum += v;
        }
        let inv = 1.0 / row_sum;
        for src in 0..n_hc {
            let idx = src + dst * n_hc;
            c[idx] = c[idx] * inv + eps;
        }
    }

    // First column normalize.
    for src in 0..n_hc {
        let mut sum = 0.0_f32;
        for dst in 0..n_hc {
            sum += c[src + dst * n_hc];
        }
        let inv = 1.0 / (sum + eps);
        for dst in 0..n_hc {
            c[src + dst * n_hc] *= inv;
        }
    }

    // Remaining `iters - 1` Sinkhorn iterations: row then column.
    for _ in 1..iters {
        for dst in 0..n_hc {
            let mut sum = 0.0_f32;
            for src in 0..n_hc {
                sum += c[src + dst * n_hc];
            }
            let inv = 1.0 / (sum + eps);
            for src in 0..n_hc {
                c[src + dst * n_hc] *= inv;
            }
        }
        for src in 0..n_hc {
            let mut sum = 0.0_f32;
            for dst in 0..n_hc {
                sum += c[src + dst * n_hc];
            }
            let inv = 1.0 / (sum + eps);
            for dst in 0..n_hc {
                c[src + dst * n_hc] *= inv;
            }
        }
    }

    // Write the final matrix into `out`.
    let mat_off = 2 * n_hc;
    let count = n_hc * n_hc;
    out[mat_off..mat_off + count].copy_from_slice(&c[..count]);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit_input(n_hc: usize) -> (Vec<f32>, Vec<f32>) {
        let len = 2 * n_hc + n_hc * n_hc;
        // mix all zeros → sigmoid(0) = 0.5 for pre/post gates,
        // all matrix entries equal → uniform row before Sinkhorn.
        let mix = vec![0.0_f32; len];
        let base = vec![0.0_f32; len];
        (mix, base)
    }

    #[test]
    fn rejects_wrong_size() {
        let mut out = vec![0.0_f32; 10];
        let err = hc_split_sinkhorn(
            &mut out,
            &[0.0; 24], // wrong size for n_hc=4 (would need 24 OK actually)
            &[1.0, 1.0, 1.0],
            &[0.0; 10],
            4,
            20,
            1e-6,
            SimdTier::Scalar,
        )
        .unwrap_err();
        assert!(matches!(err, Error::ShapeMismatch(_)));
    }

    #[test]
    fn rejects_n_hc_zero() {
        let mut out = Vec::<f32>::new();
        let err = hc_split_sinkhorn(
            &mut out,
            &[],
            &[1.0, 1.0, 1.0],
            &[],
            0,
            20,
            1e-6,
            SimdTier::Scalar,
        )
        .unwrap_err();
        assert!(matches!(err, Error::ShapeMismatch(_)));
    }

    #[test]
    fn pre_gates_have_correct_layout() {
        // With mix=0, base=0, pre_scale=1: pre gate = sigmoid(0) + eps = 0.5 + eps
        let n_hc = N_HC;
        let len = 2 * n_hc + n_hc * n_hc;
        let mut out = vec![99.0_f32; len];
        let (mix, base) = unit_input(n_hc);
        hc_split_sinkhorn(
            &mut out,
            &mix,
            &[1.0, 1.0, 1.0],
            &base,
            n_hc,
            N_HC_SINKHORN_ITER,
            1e-6,
            SimdTier::Scalar,
        )
        .unwrap();

        for (i, &v) in out.iter().take(n_hc).enumerate() {
            // sigmoid(0) = 0.5, plus eps = 1e-6.
            assert!((v - 0.5_f32).abs() < 1e-4, "out[{i}] = {v}");
        }
        // Post gates: 2 * sigmoid(0) = 1.0.
        for &v in out.iter().skip(n_hc).take(n_hc) {
            assert!((v - 1.0_f32).abs() < 1e-4);
        }
    }

    #[test]
    fn sinkhorn_matrix_is_doubly_stochastic() {
        // After 20 iterations of Sinkhorn-Knopp on a uniform-ish input,
        // every row and column of the n_hc x n_hc matrix must sum to ≈ 1.
        let n_hc = N_HC;
        let len = 2 * n_hc + n_hc * n_hc;
        let mut out = vec![0.0_f32; len];
        // Use a slightly non-uniform input so the algorithm has work to do.
        let mut mix = vec![0.0_f32; len];
        for (i, v) in mix.iter_mut().enumerate().skip(2 * n_hc) {
            *v = ((i % 7) as f32) * 0.3 - 1.0;
        }
        let base = vec![0.0_f32; len];

        hc_split_sinkhorn(
            &mut out,
            &mix,
            &[1.0, 1.0, 1.0],
            &base,
            n_hc,
            N_HC_SINKHORN_ITER,
            1e-6,
            SimdTier::Scalar,
        )
        .unwrap();

        let mat = &out[2 * n_hc..];
        // Row sums.
        for dst in 0..n_hc {
            let s: f32 = (0..n_hc).map(|src| mat[src + dst * n_hc]).sum();
            assert!((s - 1.0).abs() < 1e-3, "row {dst} sum {s} not ≈ 1",);
        }
        // Column sums.
        for src in 0..n_hc {
            let s: f32 = (0..n_hc).map(|dst| mat[src + dst * n_hc]).sum();
            assert!((s - 1.0).abs() < 1e-3, "col {src} sum {s} not ≈ 1",);
        }
    }

    #[test]
    fn output_is_finite() {
        // Even with extreme inputs, output must stay finite (eps guard).
        let n_hc = N_HC;
        let len = 2 * n_hc + n_hc * n_hc;
        let mut out = vec![0.0_f32; len];
        let mix: Vec<f32> = (0..len).map(|i| ((i as f32) - 12.0) * 5.0).collect();
        let base = vec![0.0_f32; len];

        hc_split_sinkhorn(
            &mut out,
            &mix,
            &[1.0, 1.0, 1.0],
            &base,
            n_hc,
            N_HC_SINKHORN_ITER,
            1e-6,
            SimdTier::Scalar,
        )
        .unwrap();
        assert!(out.iter().all(|v| v.is_finite()));
    }
}
