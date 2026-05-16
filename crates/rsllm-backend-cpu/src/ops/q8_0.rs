//! Q8_0 batched matmul and activation quantization.
//!
//! This is the **core idiom** of ds4's CPU performance on ARM: weights
//! are stored in 32-element Q8_0 blocks (1 × f16 scale + 32 × i8 quants
//! = 34 bytes), activations are quantized per-block at compute time
//! into the same i8 layout, and the inner dot product becomes a chain
//! of `vdotq_s32` (NEON) or `_mm512_dpbusd_epi32` (AVX-512 VNNI) calls
//! that produce an `i32` accumulator. The accumulator is then converted
//! to `f32` and multiplied by the per-block weight scale × per-block
//! activation scale.
//!
//! Ported by reference from:
//! - `ds4.c:2977-3000` — activation quantization
//! - `ds4.c:2726-2801` — `dot_q8_0_row` (NEON path + scalar fallback)
//! - `ds4.c:3277-3297` — batched matmul outer loop
//!
//! Phase C delivers the **scalar** path. NEON / AVX-512 specializations
//! land alongside as `cfg(target_arch = ...)` modules; the public
//! `matmul_q8_0_batch` selects via [`crate::SimdTier`] at call time.

use bytemuck::cast_slice;
use half::f16;

use crate::SimdTier;
use crate::error::Error;
use crate::parallel::for_each_row_mut;

/// Number of elements per Q8_0 block. Matches GGUF spec.
pub const Q8_0_BLOCK: usize = 32;

/// Size in bytes of a single Q8_0 block: `f16` scale + 32 × `i8` quants.
pub const Q8_0_BLOCK_BYTES: usize = 34;

/// Quantize one row of activations into Q8_0 (per-block absmax / 127.0).
///
/// `x.len()` may not be a multiple of 32; the tail is zero-padded into
/// the last block so the dot-product can still operate on 32 elements
/// at a time. This matches `ds4.c:2996-2998`.
///
/// `xq` must have length `ceil(x.len() / 32) * 32`; `scale` must have
/// length `ceil(x.len() / 32)`.
pub fn quantize_q8_0_activation(x: &[f32], xq: &mut [i8], scale: &mut [f32]) -> Result<(), Error> {
    let n = x.len();
    let blocks = n.div_ceil(Q8_0_BLOCK);

    if xq.len() != blocks * Q8_0_BLOCK {
        return Err(Error::ShapeMismatch(
            "quantize_q8_0: xq length must be ceil(n/32)*32",
        ));
    }
    if scale.len() != blocks {
        return Err(Error::ShapeMismatch(
            "quantize_q8_0: scale length must be ceil(n/32)",
        ));
    }

    for (b, scale_b) in scale.iter_mut().enumerate().take(blocks) {
        let i0 = b * Q8_0_BLOCK;
        let bn = (n - i0).min(Q8_0_BLOCK);
        let mut amax = 0.0_f32;
        for i in 0..bn {
            let v = x[i0 + i];
            // NaN poisoning guard: `NaN.abs() = NaN`, and `NaN > amax`
            // is always `false`, so a naive max-fold would silently
            // skip NaN inputs and produce zero quants — a corrupt
            // model output with no error signal. Fail fast instead.
            if !v.is_finite() {
                return Err(Error::NonFiniteInput("quantize_q8_0: activation"));
            }
            let ax = v.abs();
            if ax > amax {
                amax = ax;
            }
        }
        let d = amax / 127.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        *scale_b = d;
        for i in 0..bn {
            let v = (x[i0 + i] * id).round_ties_even() as i32;
            xq[i0 + i] = v.clamp(-128, 127) as i8;
        }
        // Zero-pad the tail of the last block.
        for i in bn..Q8_0_BLOCK {
            xq[i0 + i] = 0;
        }
    }
    Ok(())
}

/// Quantize a batch of token activations row-by-row (parallel).
///
/// `x` is `[n_tok × in_dim]` row-major; `xq` is the packed per-block
/// i8 buffer (length `n_tok × blocks × 32`); `scale` is the per-token
/// per-block scale buffer (length `n_tok × blocks`).
pub fn quantize_q8_0_batch(
    x: &[f32],
    xq: &mut [i8],
    scale: &mut [f32],
    n_tok: usize,
    in_dim: usize,
) -> Result<(), Error> {
    let blocks = in_dim.div_ceil(Q8_0_BLOCK);
    if x.len() != n_tok * in_dim {
        return Err(Error::ShapeMismatch(
            "quantize_q8_0_batch: x must be n_tok*in_dim",
        ));
    }
    if xq.len() != n_tok * blocks * Q8_0_BLOCK {
        return Err(Error::ShapeMismatch(
            "quantize_q8_0_batch: xq must be n_tok*blocks*32",
        ));
    }
    if scale.len() != n_tok * blocks {
        return Err(Error::ShapeMismatch(
            "quantize_q8_0_batch: scale must be n_tok*blocks",
        ));
    }

    // Reuse single-row routine across tokens. Rayon parallelism happens
    // at the batch level — each token is independent.
    use rayon::prelude::*;
    xq.par_chunks_mut(blocks * Q8_0_BLOCK)
        .zip(scale.par_chunks_mut(blocks))
        .enumerate()
        .try_for_each(|(t, (xq_row, scale_row))| {
            let x_row = &x[t * in_dim..(t + 1) * in_dim];
            quantize_q8_0_activation(x_row, xq_row, scale_row)
        })
}

/// Decode a Q8_0 block's scale (first 2 bytes, little-endian `f16`).
#[inline]
fn block_scale(block: &[u8]) -> f32 {
    debug_assert!(block.len() >= 2);
    let bits = u16::from_le_bytes([block[0], block[1]]);
    f16::from_bits(bits).to_f32()
}

/// Read a Q8_0 block's scale and coerce non-finite values to `0.0`.
/// Shared by the scalar and SIMD dot-product paths so corruption
/// handling stays consistent across tiers.
#[inline]
pub(crate) fn block_scale_finite(block: &[u8]) -> f32 {
    let s = block_scale(block);
    if s.is_finite() { s } else { 0.0 }
}

/// Compute the dot product of one packed weight row against quantized
/// activations: `sum_b w_scale[b] * x_scale[b] * dot_i32(w_quants[b], xq[b])`.
///
/// Scalar fallback path used by every tier; SIMD modules override at
/// the public entry-point level.
///
/// `row` is the raw GGUF block bytes (length `blocks * 34`). Each
/// block is `[f16 scale (2 bytes) | i8 quants × 32 (32 bytes)]`. We
/// reinterpret the 32-byte quant region as `&[i8]` via `bytemuck` so
/// the dot-product loop sees the spec's signed semantics directly
/// instead of relying on a `u8 as i8` bit-reinterpret cast.
///
/// Non-finite f16 block scales (which can occur in a corrupted GGUF)
/// are coerced to `0.0` so they contribute nothing rather than
/// poisoning the accumulator with NaN / Inf.
pub fn dot_q8_0_row_scalar(
    row: &[u8],     // packed Q8_0 row, length blocks * 34
    xq: &[i8],      // pre-quantized activations, length blocks * 32
    xscale: &[f32], // per-block activation scales, length blocks
    in_dim: usize,
) -> f32 {
    let blocks = in_dim.div_ceil(Q8_0_BLOCK);
    debug_assert_eq!(row.len(), blocks * Q8_0_BLOCK_BYTES);
    debug_assert_eq!(xq.len(), blocks * Q8_0_BLOCK);
    debug_assert_eq!(xscale.len(), blocks);

    let mut acc = 0.0_f32;
    for b in 0..blocks {
        let block = &row[b * Q8_0_BLOCK_BYTES..(b + 1) * Q8_0_BLOCK_BYTES];
        let wscale = block_scale_finite(block);
        let wq: &[i8] = cast_slice(&block[2..]);
        let xqb = &xq[b * Q8_0_BLOCK..(b + 1) * Q8_0_BLOCK];

        let i0 = b * Q8_0_BLOCK;
        let bn = (in_dim - i0).min(Q8_0_BLOCK);

        // `dot` max magnitude: 32 × 127 × 127 = 516_128, exactly
        // representable in f32 (well under 2^24). `as f32` is safe.
        let mut dot: i32 = 0;
        for i in 0..bn {
            dot += i32::from(wq[i]) * i32::from(xqb[i]);
        }
        acc += wscale * xscale[b] * dot as f32;
    }
    acc
}

/// Batched Q8_0 matmul: `out[t, o] = sum_i W[o, i] * x[t, i]`, with
/// pre-quantized activations.
///
/// Shapes:
///   * `weights`: `[out_dim × blocks × 34]` row-major (Q8_0 block stride 34).
///   * `xq`     : `[n_tok × blocks × 32]` row-major activations.
///   * `xscale` : `[n_tok × blocks]`.
///   * `out`    : `[n_tok × out_dim]`.
///
/// **Weight packing contract**: when `in_dim` is not a multiple of 32,
/// the tail bytes of each weight row's last block must be zero-padded
/// (i.e. the unused i8 quant slots set to `0`). The dot-product loop
/// only iterates `bn = in_dim - i0` lanes per block, so garbage in the
/// tail is skipped by the index bound — but if `bn` is ever widened
/// (e.g. by a future SIMD specialization that processes 32 lanes at
/// once), unpadded tails would silently corrupt the result. Activation
/// quantization already zero-pads its tails.
///
/// `tier` is currently unused — phase C ships the scalar reference;
/// NEON / AVX-512 paths land in phase D after benchmarking validates
/// the scalar baseline.
#[allow(clippy::too_many_arguments)]
pub fn matmul_q8_0_batch(
    out: &mut [f32],
    weights: &[u8],
    xq: &[i8],
    xscale: &[f32],
    n_tok: usize,
    in_dim: usize,
    out_dim: usize,
    tier: SimdTier,
) -> Result<(), Error> {
    let blocks = in_dim.div_ceil(Q8_0_BLOCK);

    if weights.len() != out_dim * blocks * Q8_0_BLOCK_BYTES {
        return Err(Error::ShapeMismatch(
            "matmul_q8_0_batch: weights must be out_dim*blocks*34",
        ));
    }
    if xq.len() != n_tok * blocks * Q8_0_BLOCK {
        return Err(Error::ShapeMismatch(
            "matmul_q8_0_batch: xq must be n_tok*blocks*32",
        ));
    }
    if xscale.len() != n_tok * blocks {
        return Err(Error::ShapeMismatch(
            "matmul_q8_0_batch: xscale must be n_tok*blocks",
        ));
    }
    if out.len() != n_tok * out_dim {
        return Err(Error::ShapeMismatch(
            "matmul_q8_0_batch: out must be n_tok*out_dim",
        ));
    }

    let row_stride_bytes = blocks * Q8_0_BLOCK_BYTES;
    let xq_row_stride = blocks * Q8_0_BLOCK;

    for_each_row_mut(out, out_dim, |t, out_row| {
        let xq_row = &xq[t * xq_row_stride..(t + 1) * xq_row_stride];
        let xscale_row = &xscale[t * blocks..(t + 1) * blocks];
        for o in 0..out_dim {
            let w_row = &weights[o * row_stride_bytes..(o + 1) * row_stride_bytes];
            out_row[o] = dot_q8_0_row_dispatch(w_row, xq_row, xscale_row, in_dim, tier);
        }
    });

    Ok(())
}

/// Pick the best dot-product implementation for the current
/// [`SimdTier`]. Each SIMD branch is `#[cfg]`-gated so it's only
/// compiled into the binary on the matching target arch — but the
/// runtime tier may still be `Scalar` even on a SIMD-capable arch if
/// the host CPU lacks the required extension.
#[inline]
fn dot_q8_0_row_dispatch(
    row: &[u8],
    xq: &[i8],
    xscale: &[f32],
    in_dim: usize,
    tier: SimdTier,
) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        if matches!(tier, SimdTier::Neon) {
            // SAFETY: `tier == Neon` is only set when `dotprod` is
            // confirmed present by [`crate::detect`]. The slice length
            // contracts are checked once at the top of
            // [`matmul_q8_0_batch`] and inherited here.
            return unsafe { super::neon::dot_q8_0_row_neon(row, xq, xscale, in_dim) };
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if matches!(tier, SimdTier::Avx512) {
            // SAFETY: `tier == Avx512` is only set when both `avx512f`
            // and `avx512bw` are confirmed present at construction.
            return unsafe { super::avx512::dot_q8_0_row_avx512(row, xq, xscale, in_dim) };
        }
    }
    let _ = tier; // unused on non-aarch64, non-x86_64 builds
    dot_q8_0_row_scalar(row, xq, xscale, in_dim)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pack a row of f32 weights into Q8_0 blocks (mirroring quantize).
    fn pack_q8_0_row(x: &[f32]) -> Vec<u8> {
        let blocks = x.len().div_ceil(Q8_0_BLOCK);
        let mut out = vec![0u8; blocks * Q8_0_BLOCK_BYTES];
        let mut tmp_q = vec![0i8; blocks * Q8_0_BLOCK];
        let mut tmp_s = vec![0.0_f32; blocks];
        quantize_q8_0_activation(x, &mut tmp_q, &mut tmp_s).unwrap();
        for b in 0..blocks {
            let dst = &mut out[b * Q8_0_BLOCK_BYTES..(b + 1) * Q8_0_BLOCK_BYTES];
            let scale_bits = f16::from_f32(tmp_s[b]).to_bits().to_le_bytes();
            dst[0] = scale_bits[0];
            dst[1] = scale_bits[1];
            for i in 0..Q8_0_BLOCK {
                dst[2 + i] = tmp_q[b * Q8_0_BLOCK + i] as u8;
            }
        }
        out
    }

    #[test]
    fn quantize_round_trip_within_tolerance() {
        // For values well within the dynamic range of one block, the
        // f32 → i8 → f32 round-trip error should be < 1/127 of the
        // block max (Q8_0 step size).
        let x: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.1).collect();
        let mut xq = vec![0i8; 64];
        let mut scale = vec![0.0_f32; 2];
        quantize_q8_0_activation(&x, &mut xq, &mut scale).unwrap();
        for b in 0..2 {
            for i in 0..32 {
                let recovered = xq[b * 32 + i] as f32 * scale[b];
                let want = x[b * 32 + i];
                let err = (recovered - want).abs();
                assert!(
                    err < 0.05,
                    "block {b} idx {i}: want {want}, got {recovered}, err {err}"
                );
            }
        }
    }

    #[test]
    fn quantize_zero_input_is_zero() {
        let x = vec![0.0_f32; 32];
        let mut xq = vec![1i8; 32];
        let mut scale = vec![0.0_f32; 1];
        quantize_q8_0_activation(&x, &mut xq, &mut scale).unwrap();
        assert_eq!(scale[0], 0.0);
        assert!(xq.iter().all(|&v| v == 0));
    }

    #[test]
    fn quantize_pads_partial_block_to_zero() {
        let x = vec![1.0_f32; 20]; // not a multiple of 32
        let mut xq = vec![99i8; 32];
        let mut scale = vec![0.0_f32; 1];
        quantize_q8_0_activation(&x, &mut xq, &mut scale).unwrap();
        // First 20 quantize to ~127; tail is zero-padded.
        for &v in &xq[20..] {
            assert_eq!(v, 0);
        }
    }

    #[test]
    fn dot_q8_0_row_matches_full_precision() {
        // Compare scalar Q8_0 dot to an f32 reference for a 96-element
        // vector (3 blocks). Tolerance loose because Q8_0 has limited
        // dynamic range per block.
        let w: Vec<f32> = (0..96).map(|i| ((i as f32) * 0.07 - 3.0).sin()).collect();
        let x: Vec<f32> = (0..96).map(|i| ((i as f32) * 0.11 + 1.0).cos()).collect();

        let row = pack_q8_0_row(&w);

        let mut xq = vec![0i8; 96];
        let mut xscale = vec![0.0_f32; 3];
        quantize_q8_0_activation(&x, &mut xq, &mut xscale).unwrap();

        let got = dot_q8_0_row_scalar(&row, &xq, &xscale, 96);
        let want: f32 = w.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
        assert!(
            (got - want).abs() < 0.01 * want.abs().max(1.0),
            "got {got}, want {want}"
        );
    }

    #[test]
    fn matmul_q8_0_batch_matches_naive() {
        let in_dim: usize = 64;
        let out_dim: usize = 4;
        let n_tok: usize = 2;
        let blocks = in_dim.div_ceil(Q8_0_BLOCK);

        // Random-ish but deterministic weights and activations.
        let weights_f32: Vec<f32> = (0..out_dim * in_dim)
            .map(|i| ((i as f32) * 0.013 + 0.1).sin())
            .collect();
        let x: Vec<f32> = (0..n_tok * in_dim)
            .map(|i| ((i as f32) * 0.027 - 0.2).cos())
            .collect();

        // Pack weights row by row.
        let mut weights = Vec::with_capacity(out_dim * blocks * Q8_0_BLOCK_BYTES);
        for o in 0..out_dim {
            let row = pack_q8_0_row(&weights_f32[o * in_dim..(o + 1) * in_dim]);
            weights.extend_from_slice(&row);
        }

        // Quantize activations.
        let mut xq = vec![0i8; n_tok * blocks * Q8_0_BLOCK];
        let mut xscale = vec![0.0_f32; n_tok * blocks];
        quantize_q8_0_batch(&x, &mut xq, &mut xscale, n_tok, in_dim).unwrap();

        // Run matmul.
        let mut out = vec![0.0_f32; n_tok * out_dim];
        matmul_q8_0_batch(
            &mut out,
            &weights,
            &xq,
            &xscale,
            n_tok,
            in_dim,
            out_dim,
            SimdTier::Scalar,
        )
        .unwrap();

        // Naive f32 reference.
        for t in 0..n_tok {
            for o in 0..out_dim {
                let mut dot = 0.0_f32;
                for i in 0..in_dim {
                    dot += weights_f32[o * in_dim + i] * x[t * in_dim + i];
                }
                let got = out[t * out_dim + o];
                assert!(
                    (got - dot).abs() < 0.01 * dot.abs().max(1.0),
                    "t {t} o {o}: got {got}, want {dot}"
                );
            }
        }
    }

    #[test]
    fn matmul_rejects_shape_mismatch() {
        let mut out = vec![0.0; 4];
        let err = matmul_q8_0_batch(&mut out, &[], &[], &[], 1, 32, 4, SimdTier::Scalar);
        assert!(err.is_err());
    }

    #[test]
    fn quantize_rejects_nan_input() {
        // CRITICAL-1 fix: NaN must not silently quantize to zero.
        let x = vec![1.0, 2.0, f32::NAN, 3.0];
        let mut xq = vec![0i8; 32];
        let mut scale = vec![0.0_f32; 1];
        let err = quantize_q8_0_activation(&x, &mut xq, &mut scale).unwrap_err();
        assert!(matches!(err, Error::NonFiniteInput(_)));
    }

    #[test]
    fn quantize_rejects_inf_input() {
        let x = vec![1.0, f32::INFINITY, 2.0];
        let mut xq = vec![0i8; 32];
        let mut scale = vec![0.0_f32; 1];
        let err = quantize_q8_0_activation(&x, &mut xq, &mut scale).unwrap_err();
        assert!(matches!(err, Error::NonFiniteInput(_)));
    }

    #[test]
    fn matmul_dispatch_runs_on_detected_tier() {
        // Dispatch test: build a real-sized matmul and run it through
        // the dispatcher with `SimdTier::detect()` — whatever the host
        // supports. The result must match the explicit scalar path
        // within 1e-4 relative tolerance.
        use crate::detect;

        let in_dim: usize = 128;
        let out_dim: usize = 2;
        let n_tok: usize = 1;
        let blocks = in_dim / Q8_0_BLOCK;

        let weights_f32: Vec<f32> = (0..out_dim * in_dim)
            .map(|i| ((i as f32) * 0.05).sin())
            .collect();
        let x: Vec<f32> = (0..n_tok * in_dim)
            .map(|i| ((i as f32) * 0.07).cos())
            .collect();

        let mut weights = Vec::with_capacity(out_dim * blocks * Q8_0_BLOCK_BYTES);
        for o in 0..out_dim {
            weights.extend_from_slice(&pack_q8_0_row(&weights_f32[o * in_dim..(o + 1) * in_dim]));
        }
        let mut xq = vec![0i8; n_tok * blocks * Q8_0_BLOCK];
        let mut xscale = vec![0.0_f32; n_tok * blocks];
        quantize_q8_0_batch(&x, &mut xq, &mut xscale, n_tok, in_dim).unwrap();

        let detected_tier = detect();
        let mut out_detected = vec![0.0_f32; n_tok * out_dim];
        matmul_q8_0_batch(
            &mut out_detected,
            &weights,
            &xq,
            &xscale,
            n_tok,
            in_dim,
            out_dim,
            detected_tier,
        )
        .unwrap();

        let mut out_scalar = vec![0.0_f32; n_tok * out_dim];
        matmul_q8_0_batch(
            &mut out_scalar,
            &weights,
            &xq,
            &xscale,
            n_tok,
            in_dim,
            out_dim,
            SimdTier::Scalar,
        )
        .unwrap();

        for (a, b) in out_detected.iter().zip(out_scalar.iter()) {
            let denom = b.abs().max(1.0);
            assert!(
                (a - b).abs() / denom < 1e-4,
                "tier={detected_tier:?}: got {a}, scalar {b}"
            );
        }
    }

    #[test]
    fn dot_q8_0_row_ignores_nonfinite_scale() {
        // Build a row whose block scale bits decode to f16 NaN
        // (0x7E00 is a quiet NaN in f16 representation).
        let mut row = vec![0u8; Q8_0_BLOCK_BYTES];
        row[0] = 0x00;
        row[1] = 0x7E; // f16 NaN scale (LE)
        // Quants are arbitrary; the NaN scale must zero them out.
        for i in 0..Q8_0_BLOCK {
            row[2 + i] = 127;
        }
        let xq = vec![1i8; Q8_0_BLOCK];
        let xscale = vec![1.0_f32; 1];
        let got = dot_q8_0_row_scalar(&row, &xq, &xscale, Q8_0_BLOCK);
        // With NaN scale forced to 0.0, the block contributes nothing.
        assert!(got.is_finite());
        assert_eq!(got, 0.0);
    }
}
