//! DeepSeek V4 Flash quantisation-aware-training (QAT) simulation
//! kernels: Hadamard-128 + FP4 (E2M1) activation quantisation, plus
//! FP8 (E4M3) KV quantisation.
//!
//! The official DeepSeek V4 graph rotates indexer activations with a
//! 128-wide Hadamard transform and immediately runs the FP4
//! activation-simulation round trip before top-K compressed-row
//! selection. Without this step the indexer scores diverge from the
//! model's reference graph (see `ds4.c:1711-1714`).
//!
//! The same graph also stores the non-RoPE portion of every compressed
//! KV row through an E4M3 FP8 round trip (`ds4.c:1632-1652`). The
//! attention compressor's emit path runs the E4M3 simulator over the
//! first `head_dim - n_rot` lanes; the trailing n_rot RoPE-rotated
//! lanes are left at full F32 precision.
//!
//! Ported by reference from `ds4.c` (MIT, The ds4.c authors), commit
//! 5bc1e6d (2026-05-17 "Apply Flash graph correctness fixes"):
//!
//! | Upstream symbol | `ds4.c:line` | Rust counterpart |
//! |---|---|---|
//! | `dsv4_e4m3fn_value_cpu` | `:1590-1603` | [`e4m3fn_value`] |
//! | `dsv4_e4m3fn_dequant_cpu` | `:1605-1630` | [`e4m3fn_dequant`] |
//! | `dsv4_fp8_kv_quantize_row_inplace_cpu` | `:1635-1653` | [`fp8_kv_quantize_row_inplace`] |
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

/// FP8 (E4M3) KV-quantisation group size — the non-RoPE prefix is
/// scaled per 64-lane group. Used by [`fp8_kv_quantize_row_inplace`].
pub const FP8_KV_GROUP: usize = 64;

/// Largest finite magnitude representable by E4M3 (`ds4.c:1607` clamp).
/// Code 127 would round to ~480, but the upstream binary search caps
/// `hi = 126`, so 448 is the de-facto saturation point.
pub const FP8_E4M3_MAX: f32 = 448.0;

/// `1 / sqrt(128)` — the Hadamard-128 normalisation factor. Stored as
/// a float literal exact to F32 to match `ds4.c:1688`.
const HADAMARD128_NORM: f32 = 0.088_388_346_f32;

/// Round-to-zero (relative to `1e-37`) lower bound on the `amax` used
/// to derive the FP4 scale. Mirrors `ds4.c:1700`.
const FP4_AMAX_FLOOR: f32 = 7.052_966e-38_f32;

/// Lower bound on the per-group `amax` for the FP8 KV path. The E4M3
/// dynamic range is wide, so this floor is much higher than the FP4
/// one (`ds4.c:1644`).
const FP8_KV_AMAX_FLOOR: f32 = 1.0e-4_f32;

/// Largest valid E4M3 code consumed by the binary-search dequant.
/// `ds4.c:1610` uses `hi = 126`; code 127 maps to a NaN-ish slot that
/// the kernel deliberately skips.
const FP8_E4M3_CODE_MAX: i32 = 126;

/// Per-exponent scale lookup for the E4M3 normal regime
/// (`ds4.c:1591-1596`). `exp = 0` is unused (subnormal path runs
/// instead); kept as `0.0` for index parity with upstream.
const E4M3FN_EXP_SCALE: [f32; 16] = [
    0.0, 0.015_625, 0.031_25, 0.062_5, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0,
    128.0, 256.0,
];

/// Subnormal step size for E4M3 (`exp == 0`): mantissa lane stride is
/// `1/512 = 0.001953125` (`ds4.c:1601`).
const E4M3FN_SUBNORMAL_STEP: f32 = 0.001_953_125;

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

/// Look up the unsigned magnitude of one E4M3 (FP8) code.
///
/// `i` decomposes into 4 exponent bits (`(i >> 3) & 0xf`) and 3
/// mantissa bits (`i & 7`). Subnormals (`exp == 0`) advance in
/// `1/512`-sized steps; normals follow `(1 + mant/8) * 2^(exp - 7)`.
/// Mirrors `ds4.c:1590-1603`.
#[inline]
#[must_use]
pub fn e4m3fn_value(i: u32) -> f32 {
    let exp = ((i >> 3) & 0xf) as usize;
    let mant = (i & 7) as f32;
    if exp == 0 {
        mant * E4M3FN_SUBNORMAL_STEP
    } else {
        (1.0 + mant * 0.125) * E4M3FN_EXP_SCALE[exp]
    }
}

/// Round a real-valued `x` to the nearest E4M3 (FP8) representable
/// magnitude, preserving sign, with the upstream tie-breaking rule
/// (prefer the *even*-coded representative when two candidates tie).
///
/// Mirrors `dsv4_e4m3fn_dequant_cpu` at `ds4.c:1605-1630`. The
/// magnitude search caps at `|x| <= 448.0` and `code <= 126` (code 127
/// is the NaN-ish slot upstream skips), so any input above 448
/// quantises to `±448.0`.
#[must_use]
pub fn e4m3fn_dequant(x: f32) -> f32 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let ax = x.abs().min(FP8_E4M3_MAX);

    // Binary search for the largest code with value <= ax.
    let mut lo: i32 = 0;
    let mut hi: i32 = FP8_E4M3_CODE_MAX;
    while lo < hi {
        let mid = (lo + hi + 1) >> 1;
        #[allow(clippy::cast_sign_loss)]
        let mid_val = e4m3fn_value(mid as u32);
        if mid_val <= ax {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }

    let mut best = lo;
    if best < FP8_E4M3_CODE_MAX {
        #[allow(clippy::cast_sign_loss)]
        let best_diff = (ax - e4m3fn_value(best as u32)).abs();
        #[allow(clippy::cast_sign_loss)]
        let next_diff = (ax - e4m3fn_value((best + 1) as u32)).abs();
        // Upstream tie-break (`ds4.c:1624`): on equal distance, prefer
        // the even-coded next candidate.
        if next_diff < best_diff
            || (next_diff == best_diff && ((best + 1) & 1) == 0 && (best & 1) != 0)
        {
            best += 1;
        }
    }
    #[allow(clippy::cast_sign_loss)]
    let mag = e4m3fn_value(best as u32);
    sign * mag
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

/// In-place FP8 (E4M3) round-trip over the non-RoPE prefix of one
/// compressed KV row.
///
/// The first `head_dim - n_rot` lanes are split into 64-lane groups;
/// each group derives a power-of-two `scale` from its amax, clamps to
/// `[-448, +448]` in pre-scale space, snaps to the nearest E4M3
/// representative, and multiplies back by scale. The trailing `n_rot`
/// RoPE-rotated lanes are left untouched.
///
/// Mirrors `dsv4_fp8_kv_quantize_row_inplace_cpu` at `ds4.c:1635-1653`.
///
/// # Arguments
/// - `x` — one full compressed KV row of length `head_dim`.
/// - `head_dim` — total lanes in the row (e.g. `DSV4_HEAD_DIM = 512`).
/// - `n_rot` — RoPE tail width (e.g. `DSV4_N_ROT = 64`); the last
///   `n_rot` lanes are preserved as-is.
///
/// # Errors
/// Returns [`Error::ShapeMismatch`] when:
/// - `x.len() != head_dim`,
/// - `n_rot > head_dim`, or
/// - the non-RoPE prefix `head_dim - n_rot` is not a positive multiple
///   of [`FP8_KV_GROUP`].
pub fn fp8_kv_quantize_row_inplace(
    x: &mut [f32],
    head_dim: usize,
    n_rot: usize,
) -> Result<(), Error> {
    if x.len() != head_dim {
        return Err(Error::ShapeMismatch(
            "fp8_kv_quantize_row_inplace: x.len() != head_dim",
        ));
    }
    if n_rot > head_dim {
        return Err(Error::ShapeMismatch(
            "fp8_kv_quantize_row_inplace: n_rot > head_dim",
        ));
    }
    let n_nope = head_dim - n_rot;
    if n_nope == 0 || !n_nope.is_multiple_of(FP8_KV_GROUP) {
        return Err(Error::ShapeMismatch(
            "fp8_kv_quantize_row_inplace: (head_dim - n_rot) must be a positive multiple of 64",
        ));
    }
    for chunk in x[..n_nope].chunks_exact_mut(FP8_KV_GROUP) {
        let mut amax = 0.0_f32;
        for &v in chunk.iter() {
            let av = v.abs();
            if av > amax {
                amax = av;
            }
        }
        if amax < FP8_KV_AMAX_FLOOR {
            amax = FP8_KV_AMAX_FLOOR;
        }
        // scale = 2^ceil(log2(amax / 448)) — smallest power-of-two
        // multiplier of the 448 endpoint that contains amax
        // (`ds4.c:1645`).
        let exp = (amax / FP8_E4M3_MAX).log2().ceil() as i32;
        let scale = libm_ldexpf(1.0, exp);
        for v in chunk.iter_mut() {
            let t = (*v / scale).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX);
            *v = e4m3fn_dequant(t) * scale;
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

    // ---- FP8 (E4M3) KV quant path (F012.D) ---------------------------------

    #[test]
    fn e4m3fn_value_subnormal_low_lanes() {
        // exp == 0 → mantissa lanes step by 1/512.
        assert_eq!(e4m3fn_value(0), 0.0);
        assert_eq!(e4m3fn_value(1), E4M3FN_SUBNORMAL_STEP);
        assert_eq!(e4m3fn_value(7), 7.0 * E4M3FN_SUBNORMAL_STEP);
    }

    #[test]
    fn e4m3fn_value_normal_endpoints() {
        // Code 8 (exp=1, mant=0): (1.0 + 0) * 0.015625 = 0.015625.
        assert_eq!(e4m3fn_value(8), 0.015_625);
        // Code 56 (exp=7, mant=0): (1.0 + 0) * 1.0 = 1.0.
        assert_eq!(e4m3fn_value(56), 1.0);
        // Code 120 (exp=15, mant=0): (1.0 + 0) * 256 = 256.0.
        assert_eq!(e4m3fn_value(120), 256.0);
        // Code 126 (exp=15, mant=6): (1.0 + 6 * 0.125) * 256 = 448.0.
        assert_eq!(e4m3fn_value(126), 448.0);
    }

    #[test]
    fn e4m3fn_dequant_saturates_at_448() {
        assert_eq!(e4m3fn_dequant(1.0e6), 448.0);
        assert_eq!(e4m3fn_dequant(-1.0e6), -448.0);
    }

    #[test]
    fn e4m3fn_dequant_round_trips_exact_representatives() {
        // Spot-check a handful of representable values.
        for &v in &[
            0.0_f32, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 448.0,
        ] {
            assert_eq!(e4m3fn_dequant(v), v, "+{v} round-trip");
            assert_eq!(e4m3fn_dequant(-v), -v, "-{v} round-trip");
        }
    }

    #[test]
    fn fp8_kv_quantize_row_inplace_constant_input() {
        // 64-lane non-RoPE prefix at v = 1.0. amax = 1.0 → scale =
        // 2^ceil(log2(1/448)) = 2^-8 = 1/256. Normalised lane = 256, which
        // is an exact E4M3 representative → snaps to 256 → output = 1.0.
        let head_dim = 128_usize;
        let n_rot = 64_usize;
        let mut x = vec![1.0_f32; head_dim];
        fp8_kv_quantize_row_inplace(&mut x, head_dim, n_rot).unwrap();
        for (i, &v) in x.iter().enumerate().take(head_dim - n_rot) {
            assert!(
                (v - 1.0).abs() < 1e-6,
                "lane {i} (non-RoPE): {v} != 1.0 (constant-input round-trip)"
            );
        }
        // RoPE tail untouched (still 1.0 because we initialised to 1.0).
        for (i, &v) in x.iter().enumerate().skip(head_dim - n_rot) {
            assert_eq!(v, 1.0, "RoPE-tail lane {i} should be preserved");
        }
    }

    #[test]
    fn fp8_kv_quantize_row_inplace_preserves_rope_tail() {
        // Mark the non-RoPE prefix with a saturating outlier and fill
        // the tail with values that have no E4M3 representative
        // (e.g. 0.123, 0.456). The kernel must NOT touch the tail.
        let head_dim = 128_usize;
        let n_rot = 64_usize;
        let mut x = vec![0.0_f32; head_dim];
        x[0] = 1000.0; // saturating outlier in prefix
        for (i, v) in x.iter_mut().enumerate().take(head_dim).skip(n_rot) {
            *v = (i as f32) * 0.123;
        }
        let tail_before: Vec<f32> = x[head_dim - n_rot..].to_vec();
        fp8_kv_quantize_row_inplace(&mut x, head_dim, n_rot).unwrap();
        let tail_after: &[f32] = &x[head_dim - n_rot..];
        assert_eq!(
            tail_before, tail_after,
            "FP8 KV quant must not touch the RoPE tail"
        );
        // Prefix lane 0 still saturates: 1000 / scale clamps to 448,
        // then E4M3-snaps and remultiplies. amax=1000, scale =
        // 2^ceil(log2(1000/448)) = 2^2 = 4. v = 1000/4 = 250 → snaps to
        // 256 (nearest E4M3 representative) → output = 256 * 4 = 1024.
        assert!(
            (x[0] - 1024.0).abs() < 1e-3,
            "saturated lane: got {}, expected ≈1024",
            x[0]
        );
    }

    #[test]
    fn fp8_kv_quantize_row_inplace_handles_multiple_groups() {
        // 2 non-RoPE groups (128 lanes) + 64 RoPE tail = head_dim 192.
        let head_dim = 192_usize;
        let n_rot = 64_usize;
        let mut x = vec![0.0_f32; head_dim];
        x[0] = 1.0; // group 0 amax = 1.0
        x[FP8_KV_GROUP] = 100.0; // group 1 amax = 100.0
        fp8_kv_quantize_row_inplace(&mut x, head_dim, n_rot).unwrap();
        // Group 0 lane 0 = 1.0 (exact representative round-trip).
        assert!((x[0] - 1.0).abs() < 1e-6, "group 0 lane 0 = {}", x[0]);
        // Group 1 lane 0: amax=100, scale = 2^ceil(log2(100/448)) = 2^-2
        // = 0.25. v = 100/0.25 = 400 → snaps to 384 (nearest E4M3
        // representative under 400) → output = 384 * 0.25 = 96.
        // (E4M3 reps near 400: code 125 = 1.75*256 = 448, code 124 =
        // 1.5*256 = 384.)
        assert!(
            (x[FP8_KV_GROUP] - 96.0).abs() < 1e-3,
            "group 1 lane 0 = {}, expected ≈96",
            x[FP8_KV_GROUP]
        );
    }

    #[test]
    fn fp8_kv_quantize_row_inplace_rejects_wrong_length() {
        let mut x = vec![0.0_f32; 100];
        assert!(matches!(
            fp8_kv_quantize_row_inplace(&mut x, 128, 64).unwrap_err(),
            Error::ShapeMismatch(_)
        ));
    }

    #[test]
    fn fp8_kv_quantize_row_inplace_rejects_n_rot_too_big() {
        let mut x = vec![0.0_f32; 64];
        assert!(matches!(
            fp8_kv_quantize_row_inplace(&mut x, 64, 128).unwrap_err(),
            Error::ShapeMismatch(_)
        ));
    }

    #[test]
    fn fp8_kv_quantize_row_inplace_rejects_non_multiple_of_64_prefix() {
        // head_dim 96, n_rot 64 → prefix 32 → not a multiple of 64.
        let mut x = vec![0.0_f32; 96];
        assert!(matches!(
            fp8_kv_quantize_row_inplace(&mut x, 96, 64).unwrap_err(),
            Error::ShapeMismatch(_)
        ));
    }

    #[test]
    fn fp8_kv_quantize_row_inplace_rejects_zero_prefix() {
        // n_rot == head_dim → prefix == 0 → must error.
        let mut x = vec![0.0_f32; 64];
        assert!(matches!(
            fp8_kv_quantize_row_inplace(&mut x, 64, 64).unwrap_err(),
            Error::ShapeMismatch(_)
        ));
    }
}
