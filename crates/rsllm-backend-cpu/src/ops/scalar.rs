//! Pure-Rust scalar implementations of every DS V4 Flash kernel.
//!
//! These are the reference implementations: SIMD variants in sibling
//! modules must produce bit-equivalent output (within a documented
//! `1e-4` tolerance per F004 verification criteria, where bit-equivalence
//! is not achievable due to FMA / reduction-order differences).
//!
//! Pre-conditions: every function here trusts its inputs. Length /
//! shape validation is done by the public wrappers in [`crate::ops`].

/// Scalar RMSNorm reference. See [`crate::ops::rmsnorm`].
///
/// Computes `out[i] = x[i] * weight[i] * (1 / sqrt(mean(x^2) + eps))`.
/// Sum is accumulated in `f32`; for the DS V4 Flash hidden_size of
/// 7168 the precision is sufficient (see ds4.c which uses `float`
/// throughout the rmsnorm).
pub fn rmsnorm(out: &mut [f32], x: &[f32], weight: &[f32], eps: f32) {
    debug_assert_eq!(out.len(), x.len());
    debug_assert_eq!(out.len(), weight.len());

    let n = x.len() as f32;
    let mut sumsq = 0.0_f32;
    for &v in x {
        sumsq += v * v;
    }
    let rms_recip = 1.0 / (sumsq / n + eps).sqrt();
    for i in 0..x.len() {
        out[i] = x[i] * weight[i] * rms_recip;
    }
}

/// Scalar argmax. See [`crate::ops::argmax`]. Ties break to the
/// lowest index — mirrors ds4's `sample_argmax` (`ds4.c:14183-14194`).
pub fn argmax(logits: &[f32]) -> u32 {
    debug_assert!(!logits.is_empty());
    let mut best_i: u32 = 0;
    let mut best_v: f32 = logits[0];
    for (i, &v) in logits.iter().enumerate().skip(1) {
        if v > best_v {
            best_v = v;
            best_i = i as u32;
        }
    }
    best_i
}

/// Numerically stable sigmoid: chooses the branch that avoids `exp` of
/// a large positive operand. Mirrors `ds4.c:4739-4747`.
#[inline]
pub fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        let e = (-x).exp();
        1.0 / (1.0 + e)
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// SiLU activation: `silu(x) = x · sigmoid(x)`. Used inside SwiGLU.
#[inline]
pub fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}

/// SwiGLU activation: `out[i] = silu(gate[i]) * up[i]`.
///
/// Ported from `ds4.c:4876-4880`. DS V4 Flash uses this with a clamp
/// applied upstream on the gate input (`DS4_SWIGLU_CLAMP_EXP = 10.0`,
/// `ds4.c:55`); the clamp is the caller's responsibility — it's applied
/// during the gate-projection matmul before this kernel runs.
pub fn swiglu(out: &mut [f32], gate: &[f32], up: &[f32]) {
    debug_assert_eq!(out.len(), gate.len());
    debug_assert_eq!(out.len(), up.len());
    for i in 0..out.len() {
        out[i] = silu(gate[i]) * up[i];
    }
}

/// In-place numerically stable softmax: subtract the row max before
/// exponentiating, then divide by the sum. Mirrors the pattern in
/// `ds4.c:4076-4083` and the attention softmax in `ds4.c:4773-4786`.
///
/// Empty input is a no-op; an all-NaN or all-`-inf` row produces NaN.
pub fn softmax(x: &mut [f32]) {
    if x.is_empty() {
        return;
    }
    let mut max_v = x[0];
    for &v in &x[1..] {
        if v > max_v {
            max_v = v;
        }
    }
    let mut sum = 0.0_f32;
    for v in x.iter_mut() {
        *v = (*v - max_v).exp();
        sum += *v;
    }
    let inv = 1.0 / sum;
    for v in x.iter_mut() {
        *v *= inv;
    }
}

/// Sink-aware attention softmax: applies the standard
/// `softmax(scores - max(scores, sink))` but adds the sink logit into
/// the denominator without producing an output weight for it.
///
/// Mirrors `ds4.c:4773-4786`. The sink token represents "no attention
/// target" and shifts the denominator so that low-confidence rows
/// down-weight all their attention targets uniformly. Returns the
/// implied sink weight `exp(sink - max) / sum`, which callers can use
/// to detect when the row mostly attends to the sink.
pub fn softmax_attn(scores: &mut [f32], sink: f32) -> f32 {
    if scores.is_empty() {
        return 1.0;
    }
    let mut max_v = sink;
    for &v in scores.iter() {
        if v > max_v {
            max_v = v;
        }
    }
    let sink_weight = (sink - max_v).exp();
    let mut sum = sink_weight;
    for v in scores.iter_mut() {
        *v = (*v - max_v).exp();
        sum += *v;
    }
    let inv = 1.0 / sum;
    for v in scores.iter_mut() {
        *v *= inv;
    }
    sink_weight * inv
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rmsnorm_unit_weights() {
        // For weight = all ones, output i = x[i] / rms.
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let w = vec![1.0; 4];
        let mut out = vec![0.0; 4];
        rmsnorm(&mut out, &x, &w, 0.0);
        // rms = sqrt((1+4+9+16)/4) = sqrt(7.5) ≈ 2.7386
        let expected_rms = (30.0f32 / 4.0).sqrt();
        for (i, &v) in out.iter().enumerate() {
            let want = x[i] / expected_rms;
            assert!((v - want).abs() < 1e-6, "out[{i}] = {v}, want {want}");
        }
    }

    #[test]
    fn rmsnorm_eps_prevents_divzero() {
        let x = vec![0.0; 8];
        let w = vec![1.0; 8];
        let mut out = vec![99.0; 8];
        rmsnorm(&mut out, &x, &w, 1e-6);
        // With all-zero input, output is also all-zero (because x[i] = 0
        // makes the product zero regardless of the rsqrt term).
        assert!(out.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn rmsnorm_weight_scales_output() {
        let x = vec![1.0; 4]; // rms = 1, so out = weight elementwise
        let w = vec![2.0, 3.0, 5.0, 7.0];
        let mut out = vec![0.0; 4];
        rmsnorm(&mut out, &x, &w, 0.0);
        for i in 0..4 {
            assert!((out[i] - w[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn argmax_unique_max() {
        assert_eq!(argmax(&[1.0, 2.0, 3.0, 2.5]), 2);
    }

    #[test]
    fn argmax_ties_pick_lowest_index() {
        assert_eq!(argmax(&[3.0, 3.0, 3.0]), 0);
        assert_eq!(argmax(&[1.0, 3.0, 2.0, 3.0]), 1);
    }

    #[test]
    fn argmax_single_element() {
        assert_eq!(argmax(&[42.0]), 0);
    }

    #[test]
    fn argmax_handles_negatives() {
        assert_eq!(argmax(&[-3.0, -1.0, -2.0]), 1);
    }

    #[test]
    fn sigmoid_matches_reference() {
        // Known values; tolerance loose enough for f32 expf precision.
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
        assert!((sigmoid(2.0) - 0.880_797).abs() < 1e-4);
        assert!((sigmoid(-2.0) - 0.119_203).abs() < 1e-4);
        // Saturation at large |x|.
        assert!((sigmoid(20.0) - 1.0).abs() < 1e-6);
        assert!((sigmoid(-20.0) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn sigmoid_does_not_overflow_for_large_negative() {
        // The naive `1 / (1 + exp(-x))` overflows for x = -100 because
        // exp(100) overflows to inf. The stable branch must not.
        let y = sigmoid(-100.0);
        assert!(y.is_finite());
        assert!(y < 1e-30);
    }

    #[test]
    fn silu_zero_at_zero() {
        assert_eq!(silu(0.0), 0.0);
    }

    #[test]
    fn silu_asymptotes() {
        // silu(x) → x as x → +∞ (sigmoid → 1), → 0 as x → -∞.
        assert!((silu(20.0) - 20.0).abs() < 1e-4);
        assert!(silu(-20.0).abs() < 1e-4);
    }

    #[test]
    fn swiglu_basic() {
        // gate = 2.0, up = 3.0 → silu(2) * 3
        let gate = vec![2.0_f32];
        let up = vec![3.0_f32];
        let mut out = vec![0.0_f32; 1];
        swiglu(&mut out, &gate, &up);
        let expected = silu(2.0) * 3.0;
        assert!((out[0] - expected).abs() < 1e-6);
    }

    #[test]
    fn softmax_sums_to_one() {
        let mut x = vec![1.0_f32, 2.0, 3.0, 4.0];
        softmax(&mut x);
        let sum: f32 = x.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        // Strictly monotonic.
        assert!(x[0] < x[1] && x[1] < x[2] && x[2] < x[3]);
    }

    #[test]
    fn softmax_stable_for_large_values() {
        // Without max subtraction, exp(1000) would overflow to inf.
        let mut x = vec![1000.0_f32; 4];
        softmax(&mut x);
        for v in &x {
            assert!((v - 0.25).abs() < 1e-6, "uniform input -> uniform output");
        }
    }

    #[test]
    fn softmax_attn_zero_sink_recovers_softmax() {
        // sink at -inf is the same as a regular softmax with negligible
        // sink contribution.
        let mut a = vec![1.0_f32, 2.0, 3.0];
        let mut b = a.clone();
        softmax(&mut a);
        let _ = softmax_attn(&mut b, f32::NEG_INFINITY);
        for i in 0..3 {
            assert!((a[i] - b[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn softmax_attn_dominant_sink_collapses_output() {
        // sink at +∞ relative to scores → sink_weight → 1, scores → 0.
        let mut scores = vec![1.0_f32, 2.0, 3.0];
        let sink_w = softmax_attn(&mut scores, 100.0);
        assert!(sink_w > 0.999);
        for v in &scores {
            assert!(*v < 1e-30);
        }
    }
}
