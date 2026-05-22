//! DeepSeek V4 Flash quantisation-aware-training (QAT) simulation
//! kernels: Hadamard-128 + FP4 (E2M1) activation quantisation.
//!
//! The official DeepSeek V4 graph rotates indexer activations with a
//! 128-wide Hadamard transform and immediately runs the FP4
//! activation-simulation round trip before top-K compressed-row
//! selection. Without this step the indexer scores diverge from the
//! model's reference graph (see `ds4.c:1711-1714`).
//!
//! Ported by reference from `ds4.c` (MIT, The ds4.c authors), commit
//! 5bc1e6d (2026-05-17 "Apply Flash graph correctness fixes"):
//!
//! | Upstream symbol | `ds4.c:line` | Rust counterpart |
//! |---|---|---|
//! | `dsv4_e2m1fn_value_cpu` | `:1655-1660` | [`e2m1fn_value`] |
//! | `dsv4_e2m1fn_dequant_cpu` | `:1662-1675` | [`e2m1fn_dequant`] |
//! | `dsv4_hadamard128_inplace_cpu` | `:1677-1689` | [`hadamard128_inplace`] |
//! | `dsv4_fp4_act_quantize_row_inplace_cpu` | `:1691-1709` | [`fp4_act_quantize_row_inplace`] |
//! | `dsv4_indexer_qat_row_inplace_cpu` | `:1715-1719` | [`indexer_qat_row_inplace`] |

use crate::error::Error;

/// One Hadamard-128 row's worth of lanes. Used by [`hadamard128_inplace`]
/// and [`indexer_qat_row_inplace`].
pub const HADAMARD128_DIM: usize = 128;

/// FP4 activation quantisation group size — the per-row `amax` scaling
/// applies in 32-element chunks. Used by [`fp4_act_quantize_row_inplace`].
pub const FP4_GROUP: usize = 32;

/// `1 / sqrt(128)` — the Hadamard-128 normalisation factor. Stored as
/// a float literal exact to F32 to match `ds4.c:1688`.
const HADAMARD128_NORM: f32 = 0.088_388_346_f32;

/// Round-to-zero (relative to `1e-37`) lower bound on the `amax` used
/// to derive the FP4 scale. Mirrors `ds4.c:1700`.
const FP4_AMAX_FLOOR: f32 = 7.052_966e-38_f32;

/// The 8 unsigned FP4 (E2M1) reconstruction values, indexed by the
/// low 3 bits of a code. The 4th bit is the sign, applied by
/// [`e2m1fn_dequant`]. Source: `ds4.c:1657-1659`.
const E2M1FN_VALUES: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

/// Look up the unsigned magnitude of one FP4 (E2M1) code.
///
/// The low 3 bits of `i` index the [`E2M1FN_VALUES`] table; bits above
/// that are ignored (mirrors `i & 7` in `ds4.c:1659`).
#[inline]
#[must_use]
pub fn e2m1fn_value(i: u32) -> f32 {
    E2M1FN_VALUES[(i & 7) as usize]
}

/// Round a real-valued `x` to the nearest FP4 (E2M1) representable
/// magnitude, preserving sign, with the upstream tie-breaking rule
/// (prefer the *even*-coded representative when two candidates tie).
///
/// Mirrors `dsv4_e2m1fn_dequant_cpu` at `ds4.c:1662-1675`. The
/// magnitude search caps at `|x| <= 6.0` (the largest representable),
/// so saturating inputs always quantise to the +/-6.0 endpoint.
#[must_use]
pub fn e2m1fn_dequant(x: f32) -> f32 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let ax = x.abs().min(6.0);
    let mut best: u32 = 0;
    let mut best_diff = (ax - e2m1fn_value(0)).abs();
    for i in 1..8_u32 {
        let diff = (ax - e2m1fn_value(i)).abs();
        // Upstream tie-break: on equal distance, prefer the
        // even-coded candidate when the current `best` is odd. The
        // intent matches `ds4.c:1669`.
        if diff < best_diff || (diff == best_diff && (i & 1) == 0 && (best & 1) != 0) {
            best = i;
            best_diff = diff;
        }
    }
    sign * e2m1fn_value(best)
}

/// Apply the in-place 128-wide Hadamard transform with the
/// `1/sqrt(128)` normalisation.
///
/// The transform is its own inverse up to scale: applying
/// `hadamard128_inplace` twice multiplies the input by `128 *
/// HADAMARD128_NORM^2 = 1.0` (exact in real arithmetic, F32-rounded
/// in practice). Used by [`indexer_qat_row_inplace`].
///
/// # Errors
/// Returns [`Error::ShapeMismatch`] when `x.len() != HADAMARD128_DIM`.
pub fn hadamard128_inplace(x: &mut [f32]) -> Result<(), Error> {
    if x.len() != HADAMARD128_DIM {
        return Err(Error::ShapeMismatch(
            "hadamard128_inplace: x must have length 128",
        ));
    }
    // Sylvester construction: 7 butterfly stages, doubling the stride
    // each pass. After stage k, every output is the sum/difference of
    // 2^(k+1) input lanes. Stops when stride == 64 (the last pass
    // straddles cols 0..64 and 64..128).
    let mut stride = 1_usize;
    while stride < HADAMARD128_DIM {
        let block = 2 * stride;
        let mut base = 0_usize;
        while base < HADAMARD128_DIM {
            for i in 0..stride {
                let a = x[base + i];
                let b = x[base + stride + i];
                x[base + i] = a + b;
                x[base + stride + i] = a - b;
            }
            base += block;
        }
        stride = block;
    }
    for v in x.iter_mut() {
        *v *= HADAMARD128_NORM;
    }
    Ok(())
}

/// Per-row FP4 activation simulation: split `x` into 32-element
/// groups, derive a power-of-two `scale` from the group's `amax`, and
/// round each lane to the nearest E2M1 representative * scale.
///
/// Mirrors `dsv4_fp4_act_quantize_row_inplace_cpu` at `ds4.c:1691-1709`.
///
/// # Errors
/// Returns [`Error::ShapeMismatch`] when `x.len()` is not a positive
/// multiple of [`FP4_GROUP`].
pub fn fp4_act_quantize_row_inplace(x: &mut [f32]) -> Result<(), Error> {
    if x.is_empty() || !x.len().is_multiple_of(FP4_GROUP) {
        return Err(Error::ShapeMismatch(
            "fp4_act_quantize_row_inplace: length must be a positive multiple of 32",
        ));
    }
    for chunk in x.chunks_exact_mut(FP4_GROUP) {
        // Find per-group amax.
        let mut amax = 0.0_f32;
        for &v in chunk.iter() {
            let av = v.abs();
            if av > amax {
                amax = av;
            }
        }
        // Floor amax to a tiny positive so log2 stays finite even when
        // every lane is zero (`ds4.c:1700`).
        if amax < FP4_AMAX_FLOOR {
            amax = FP4_AMAX_FLOOR;
        }
        // scale = 2^ceil(log2(amax / 6)) — the smallest power-of-two
        // multiplier of the 6.0 endpoint that contains amax.
        let exp = (amax / 6.0).log2().ceil() as i32;
        let scale = libm_ldexpf(1.0, exp);
        // Saturate to [-6, +6] in pre-scale space, then snap to E2M1.
        // `clamp` is safe here because the bounds are finite and ordered.
        for v in chunk.iter_mut() {
            let t = (*v / scale).clamp(-6.0, 6.0);
            *v = e2m1fn_dequant(t) * scale;
        }
    }
    Ok(())
}

/// Compose Hadamard-128 + FP4 activation simulation for one
/// `[HADAMARD128_DIM = 128]`-wide indexer row.
///
/// Mirrors `dsv4_indexer_qat_row_inplace_cpu` at `ds4.c:1715-1719`.
/// The transform is destructive: callers wanting to preserve the
/// pre-QAT row should copy first.
///
/// # Errors
/// Returns [`Error::ShapeMismatch`] when `x.len() != HADAMARD128_DIM`.
pub fn indexer_qat_row_inplace(x: &mut [f32]) -> Result<(), Error> {
    if x.len() != HADAMARD128_DIM {
        return Err(Error::ShapeMismatch(
            "indexer_qat_row_inplace: expects 128-wide indexer rows",
        ));
    }
    hadamard128_inplace(x)?;
    fp4_act_quantize_row_inplace(x)
}

/// Apply [`indexer_qat_row_inplace`] to `rows` consecutive rows.
///
/// Mirrors `dsv4_indexer_qat_rows_inplace_cpu` at `ds4.c:1721-1725`.
///
/// # Errors
/// Returns [`Error::ShapeMismatch`] when `x.len() != rows *
/// HADAMARD128_DIM`.
pub fn indexer_qat_rows_inplace(x: &mut [f32], rows: usize) -> Result<(), Error> {
    let expected = rows
        .checked_mul(HADAMARD128_DIM)
        .ok_or(Error::ShapeMismatch(
            "indexer_qat_rows_inplace: rows * 128 overflows",
        ))?;
    if x.len() != expected {
        return Err(Error::ShapeMismatch(
            "indexer_qat_rows_inplace: x.len() must equal rows * 128",
        ));
    }
    for row in x.chunks_exact_mut(HADAMARD128_DIM) {
        indexer_qat_row_inplace(row)?;
    }
    Ok(())
}

/// `ldexpf(x, n) = x * 2^n`. Rust stdlib only exposes the inverse
/// (`frexp`); for the forward direction we hand-roll a small wrapper
/// to keep the kernel `no_std`-compatible without a libm dep.
///
/// The implementation just constructs the bit pattern: an F32 has 8
/// exponent bits biased by 127, so multiplying by `2^n` is a single
/// add to the exponent — but only when the result stays in the
/// normal range. For the FP4 scale path the exponent always lands in
/// `[-126, +127]` (amax in [1e-37, 6e30]), well inside the normal
/// regime, so the simple path suffices.
fn libm_ldexpf(x: f32, n: i32) -> f32 {
    // Clamp `n` to the normal F32 exponent range. Outside the clamp
    // we'd hit subnormals or overflow; for the FP4 quant path that
    // would already imply a degenerate input. The behaviour matches
    // the C `ldexpf` saturation in those cases.
    let n = n.clamp(-126, 127);
    let bits: u32 = ((n + 127) as u32) << 23;
    x * f32::from_bits(bits)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Sanity: ldexpf wrapper matches the obvious `x * 2^n` formula.
    #[test]
    fn ldexpf_matches_pow2() {
        for &(x, n) in &[(1.0_f32, 0), (3.5, 4), (-1.0, -3), (0.125, 7)] {
            let expected = x * (2.0_f32).powi(n);
            assert!(
                (libm_ldexpf(x, n) - expected).abs() < 1e-6,
                "ldexpf({x}, {n}) = {} != {expected}",
                libm_ldexpf(x, n)
            );
        }
    }

    #[test]
    fn e2m1fn_value_low_three_bits() {
        for i in 0..8 {
            assert_eq!(e2m1fn_value(i), E2M1FN_VALUES[i as usize]);
        }
        // Higher bits ignored.
        assert_eq!(e2m1fn_value(8), E2M1FN_VALUES[0]);
        assert_eq!(e2m1fn_value(255), E2M1FN_VALUES[7]);
    }

    #[test]
    fn e2m1fn_dequant_exact_representatives() {
        // Every representative E2M1 value must round to itself.
        for &v in &E2M1FN_VALUES {
            assert_eq!(e2m1fn_dequant(v), v, "positive {v} round-trip");
            assert_eq!(e2m1fn_dequant(-v), -v, "negative {v} round-trip");
        }
    }

    #[test]
    fn e2m1fn_dequant_saturates_at_six() {
        // |x| > 6.0 collapses to ±6.0 (the largest representable).
        assert_eq!(e2m1fn_dequant(100.0), 6.0);
        assert_eq!(e2m1fn_dequant(-100.0), -6.0);
    }

    #[test]
    fn e2m1fn_dequant_midpoint_picks_nearest() {
        // 0.25 sits midway between 0.0 and 0.5 — tie-break picks the
        // even-indexed candidate (`0` is even, `1` is odd → keep `0`).
        assert_eq!(e2m1fn_dequant(0.25), 0.0);
        // 0.75 ties between 0.5 (code 1) and 1.0 (code 2): code 2 even
        // → 1.0.
        assert_eq!(e2m1fn_dequant(0.75), 1.0);
        // 1.25 ties between 1.0 (code 2) and 1.5 (code 3): code 2 even
        // → 1.0.
        assert_eq!(e2m1fn_dequant(1.25), 1.0);
    }

    #[test]
    fn hadamard128_inplace_constant_input() {
        // H * [c; 128] has all-zero output except lane 0 = 128 * c.
        // After * (1/sqrt(128)) the lane-0 value is c * sqrt(128).
        let mut x = vec![3.0_f32; HADAMARD128_DIM];
        hadamard128_inplace(&mut x).unwrap();
        let expected_lane0 = 3.0_f32 * (HADAMARD128_DIM as f32).sqrt();
        assert!(
            (x[0] - expected_lane0).abs() < 1e-3,
            "lane 0 = {}, expected {expected_lane0}",
            x[0]
        );
        for &v in &x[1..] {
            assert!(v.abs() < 1e-3, "expected zero away from lane 0, got {v}");
        }
    }

    #[test]
    fn hadamard128_inplace_involution() {
        // H_n is its own inverse when normalised: applying it twice
        // reconstructs the original (up to F32 rounding). Use a known
        // non-symmetric input to catch any normalisation drift.
        let original: Vec<f32> = (0..HADAMARD128_DIM).map(|i| (i as f32 + 1.0) * 0.1).collect();
        let mut x = original.clone();
        hadamard128_inplace(&mut x).unwrap();
        hadamard128_inplace(&mut x).unwrap();
        for (i, (&got, &want)) in x.iter().zip(original.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-3,
                "lane {i}: round-trip {got} != {want}"
            );
        }
    }

    #[test]
    fn hadamard128_inplace_rejects_wrong_length() {
        let mut x = vec![0.0_f32; 64];
        assert!(matches!(
            hadamard128_inplace(&mut x).unwrap_err(),
            Error::ShapeMismatch(_)
        ));
    }

    #[test]
    fn fp4_act_quantize_row_inplace_constant_input() {
        // 32 lanes all set to 1.0 — amax = 1.0, scale = 2^ceil(log2(1/6))
        // = 2^-2 = 0.25. Each lane normalises to 1/0.25 = 4.0 (an exact
        // E2M1 representative), so the result is 4.0 * 0.25 = 1.0.
        let mut x = vec![1.0_f32; FP4_GROUP];
        fp4_act_quantize_row_inplace(&mut x).unwrap();
        for &v in &x {
            assert!(
                (v - 1.0).abs() < 1e-6,
                "expected 1.0 after exact-rep round-trip, got {v}"
            );
        }
    }

    #[test]
    fn fp4_act_quantize_row_inplace_saturates_outliers() {
        // amax = 100, scale = 2^ceil(log2(100/6)) = 2^5 = 32.
        // Other lanes are tiny; their normalised value is far below
        // the smallest non-zero representative (0.5), so they round to 0.
        let mut x = vec![0.0_f32; FP4_GROUP];
        x[0] = 100.0;
        for v in x.iter_mut().skip(1) {
            *v = 0.001;
        }
        fp4_act_quantize_row_inplace(&mut x).unwrap();
        // The outlier saturates: 100/32 = 3.125 → snaps to E2M1 = 3.0
        // → output = 3.0 * 32 = 96.0.
        assert!(
            (x[0] - 96.0).abs() < 1e-4,
            "outlier lane = {}, expected ≈96",
            x[0]
        );
        // The tiny lanes round to 0.
        for &v in &x[1..] {
            assert_eq!(v, 0.0, "small lane should round to 0, got {v}");
        }
    }

    #[test]
    fn fp4_act_quantize_row_inplace_handles_two_groups_independently() {
        // 64-lane input — two groups of 32, independent amax.
        let mut x = vec![0.0_f32; 2 * FP4_GROUP];
        x[0] = 1.0; // group 0 amax = 1.0
        x[FP4_GROUP] = 100.0; // group 1 amax = 100.0
        fp4_act_quantize_row_inplace(&mut x).unwrap();
        // Group 0: lane 0 = exact round-trip to 1.0.
        assert!((x[0] - 1.0).abs() < 1e-6, "group 0 lane 0 = {}", x[0]);
        // Group 1: lane 0 saturates to 96.0 (same calculation as the
        // saturate-outlier test).
        assert!(
            (x[FP4_GROUP] - 96.0).abs() < 1e-4,
            "group 1 lane 0 = {}",
            x[FP4_GROUP]
        );
    }

    #[test]
    fn fp4_act_quantize_row_inplace_rejects_non_multiple_of_32() {
        let mut x = vec![1.0_f32; 33];
        assert!(matches!(
            fp4_act_quantize_row_inplace(&mut x).unwrap_err(),
            Error::ShapeMismatch(_)
        ));
        let mut empty: Vec<f32> = Vec::new();
        assert!(matches!(
            fp4_act_quantize_row_inplace(&mut empty).unwrap_err(),
            Error::ShapeMismatch(_)
        ));
    }

    #[test]
    fn indexer_qat_row_inplace_composes_both_steps() {
        // The composer is the pipeline used by `compressor_decode_one`
        // for indexer rows (F012.C) and by `project_indexer_query`. With
        // a constant input, Hadamard produces a single concentrated
        // lane; FP4 then quantises that lane against its 32-element
        // group's amax.
        let mut x = vec![1.0_f32; HADAMARD128_DIM];
        indexer_qat_row_inplace(&mut x).unwrap();
        // The Hadamard result has lane 0 ≈ sqrt(128) ≈ 11.3137 and the
        // remaining lanes at 0. The first FP4 group (lanes 0..32) has
        // amax = 11.3137 → scale = 2^ceil(log2(11.3137/6)) = 2^1 = 2.0.
        // Lane 0 normalised = 5.6568 → snaps to E2M1 = 6.0 → output =
        // 6.0 * 2.0 = 12.0. Lanes 1..32 are 0 → stay 0. Groups 1..4
        // (lanes 32..128) are all-zero — amax = floor → scale tiny →
        // output stays 0.
        assert!(
            (x[0] - 12.0).abs() < 1e-4,
            "lane 0 post-QAT = {}, expected ≈12.0",
            x[0]
        );
        for (i, &v) in x.iter().enumerate().skip(1) {
            assert_eq!(v, 0.0, "lane {i} should be 0, got {v}");
        }
    }

    #[test]
    fn indexer_qat_row_inplace_rejects_wrong_length() {
        let mut x = vec![0.0_f32; 100];
        assert!(matches!(
            indexer_qat_row_inplace(&mut x).unwrap_err(),
            Error::ShapeMismatch(_)
        ));
    }

    #[test]
    fn indexer_qat_rows_inplace_applies_to_each_row() {
        // Three constant rows — each independently transforms to
        // [12.0, 0, 0, ..., 0].
        let mut x = vec![1.0_f32; 3 * HADAMARD128_DIM];
        indexer_qat_rows_inplace(&mut x, 3).unwrap();
        for r in 0..3 {
            let off = r * HADAMARD128_DIM;
            assert!(
                (x[off] - 12.0).abs() < 1e-4,
                "row {r} lane 0 = {}",
                x[off]
            );
            for j in 1..HADAMARD128_DIM {
                assert_eq!(x[off + j], 0.0, "row {r} lane {j} = {}", x[off + j]);
            }
        }
    }

    #[test]
    fn indexer_qat_rows_inplace_rejects_size_mismatch() {
        let mut x = vec![0.0_f32; 200];
        assert!(matches!(
            indexer_qat_rows_inplace(&mut x, 3).unwrap_err(),
            Error::ShapeMismatch(_)
        ));
    }
}
