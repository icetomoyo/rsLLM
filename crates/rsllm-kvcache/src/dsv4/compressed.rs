//! Compressed KV pool with per-dimension softmax-weighted aggregation.
//!
//! Every `compress_ratio` tokens, the per-layer cache emits one
//! compressed row to the long-history pool. The compression rule
//! (`ds4.c:6376-6420` `compressor_pool_decode_state`) is a per-output-
//! dimension softmax weighted by token-specific scores:
//!
//! ```text
//! for each output dim j:
//!     max_j = max over (r in 0..ratio) of score[r, j]
//!     denom = sum over r of exp(score[r, j] - max_j)
//!     out[j] = sum over r of exp(score[r, j] - max_j) * kv[r, j] / denom
//! ```
//!
//! This is **not** a global softmax — each output dimension is its own
//! softmax over the ratio-sized window. The result is a single
//! `head_dim`-wide row that the caller then appends to the pool.
//!
//! Per-dimension state for one layer:
//!
//! - `state_kv`     : `[ratio × head_dim]` row-major
//! - `state_score`  : `[ratio × head_dim]` row-major, initialised to `-INF`
//! - `count`        : how many of the `ratio` slots are populated
//! - `compressed`   : the long-history pool, append-only
//!
//! v0.1.0 implements the **primary attention compression lane only**.
//! Ratio-4 layers additionally maintain an indexer compression lane;
//! that lane has identical structure (different `head_dim`) and is
//! handled by [`super::indexer`] as a separate pool instance.
//!
//! Ported by reference from `ds4.c:6322-6420` (MIT, The ds4.c authors).
//! Line numbers pinned to ds4 commit `ef0a490` (2026-05-17).

use crate::Error;

use super::shape::DSV4_HEAD_DIM;

/// `-inf` sentinel used to initialise unused score slots (`ds4.c:6275`).
const NEG_INF: f32 = f32::NEG_INFINITY;

/// Per-layer compressed-KV pool. Holds the long-history aggregated
/// rows plus the in-progress state for the next emission.
#[derive(Debug)]
pub struct CompressedKvPool {
    /// Output rows, `[n_comp × head_dim]` row-major. Append-only.
    compressed: Vec<f32>,
    /// In-progress aggregation state, `[ratio × head_dim]` row-major.
    /// Cleared whenever an emission fires.
    state_kv: Vec<f32>,
    /// Per-dimension score history, `[ratio × head_dim]` row-major.
    /// Slots filled by tokens after the current emission boundary;
    /// rest hold `NEG_INF` so the softmax ignores them.
    state_score: Vec<f32>,
    /// Number of compressed rows currently in `compressed`.
    n_comp: usize,
    /// Maximum number of compressed rows the pool can hold.
    cap_comp: usize,
    /// How many of the `ratio` state slots have been filled since the
    /// last emission. Wraps to 0 after each emission.
    state_count: u32,
    /// Compression ratio for this layer (4 or 128; 0 layers do not own
    /// a compressed pool and should not be constructed).
    ratio: u32,
    /// Per-row dimension (typically [`DSV4_HEAD_DIM`] for attention,
    /// `DSV4_N_INDEXER_HEAD_DIM` for the indexer).
    head_dim: usize,
}

impl CompressedKvPool {
    /// Create an empty pool sized for `cap_comp` aggregated rows.
    ///
    /// `head_dim` is the per-row dimension. `ratio` is the layer's
    /// compression ratio (must be `> 0`).
    #[must_use]
    pub fn new(cap_comp: usize, ratio: u32, head_dim: usize) -> Self {
        assert!(ratio > 0, "compression ratio must be > 0");
        assert!(head_dim > 0);
        let r = ratio as usize;
        Self {
            compressed: vec![0.0_f32; cap_comp * head_dim],
            state_kv: vec![0.0_f32; r * head_dim],
            state_score: vec![NEG_INF; r * head_dim],
            n_comp: 0,
            cap_comp,
            state_count: 0,
            ratio,
            head_dim,
        }
    }

    /// Create a pool with DS V4 Flash attention defaults
    /// (`head_dim = HEAD_DIM`). `cap_comp` is typically `ctx_size / ratio + 2`.
    #[must_use]
    pub fn with_dsv4_attn(cap_comp: usize, ratio: u32) -> Self {
        Self::new(cap_comp, ratio, DSV4_HEAD_DIM)
    }

    /// Number of long-history aggregated rows.
    #[must_use]
    pub fn len(&self) -> usize {
        self.n_comp
    }

    /// Returns `true` if no rows have been emitted yet.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.n_comp == 0
    }

    /// Maximum capacity for aggregated rows.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.cap_comp
    }

    /// Compression ratio for this pool.
    #[must_use]
    pub fn ratio(&self) -> u32 {
        self.ratio
    }

    /// Per-row dimension.
    #[must_use]
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Number of state slots filled since the last emission.
    #[must_use]
    pub fn state_count(&self) -> u32 {
        self.state_count
    }

    /// Read-only slice of all aggregated rows in chronological order.
    /// Shape `[n_comp × head_dim]` row-major.
    #[must_use]
    pub fn rows(&self) -> &[f32] {
        &self.compressed[..self.n_comp * self.head_dim]
    }

    /// Append one token's (kv, score) pair to the state and, if the
    /// state is now full, run the per-dim softmax aggregation, append
    /// the result to the compressed pool, and reset the state.
    ///
    /// Returns the index of the newly emitted compressed row when the
    /// state was just filled, or `None` if more tokens are still
    /// needed before emission.
    ///
    /// # Errors
    /// - `Error::ShapeMismatch` if `kv.len() != head_dim` or `score.len() != head_dim`.
    /// - `Error::CompressedPoolFull` if an emission would exceed `cap_comp`.
    pub fn accumulate(&mut self, kv: &[f32], score: &[f32]) -> Result<Option<usize>, Error> {
        if kv.len() != self.head_dim {
            return Err(Error::ShapeMismatch {
                what: "CompressedKvPool::accumulate: kv",
                expected: self.head_dim,
                actual: kv.len(),
            });
        }
        if score.len() != self.head_dim {
            return Err(Error::ShapeMismatch {
                what: "CompressedKvPool::accumulate: score",
                expected: self.head_dim,
                actual: score.len(),
            });
        }

        // Write current token's (kv, score) into state row `state_count`.
        let dim = self.head_dim;
        let slot = self.state_count as usize;
        let off = slot * dim;
        self.state_kv[off..off + dim].copy_from_slice(kv);
        self.state_score[off..off + dim].copy_from_slice(score);
        self.state_count += 1;

        if self.state_count < self.ratio {
            return Ok(None);
        }

        // State is full: aggregate and emit.
        if self.n_comp >= self.cap_comp {
            return Err(Error::CompressedPoolFull {
                cap: self.cap_comp,
            });
        }
        let comp_off = self.n_comp * dim;
        per_dim_softmax_aggregate(
            &self.state_kv,
            &self.state_score,
            self.ratio as usize,
            dim,
            &mut self.compressed[comp_off..comp_off + dim],
        );

        // Reset state for next ratio-sized window.
        self.state_kv.fill(0.0);
        self.state_score.fill(NEG_INF);
        self.state_count = 0;

        let emitted_idx = self.n_comp;
        self.n_comp += 1;
        Ok(Some(emitted_idx))
    }

    /// Clean up unused state rows after prefill, matching ds4's
    /// `compressor_finish_prefill_state_cpu` (`ds4.c:6331-6351`). The
    /// state slots that streaming decode would never have touched are
    /// reset so the post-prefill cache is byte-identical to a stream
    /// that processed the same prefix one token at a time.
    ///
    /// `n_tokens` is the total prefill length. Only state slots beyond
    /// `n_tokens % ratio` need clearing.
    pub fn finish_prefill_state(&mut self, n_tokens: usize) {
        let r = self.ratio as usize;
        let rem = n_tokens % r;
        // ds4 has a ratio-4-specific `clear_start = ratio + rem`, but
        // that applies to the **2× state** layout where ratio-4 layers
        // keep two lanes in one buffer. We track the indexer lane in a
        // separate pool, so for our single-lane buffer we just clear
        // anything past `rem`.
        let dim = self.head_dim;
        for row in rem..r {
            let off = row * dim;
            self.state_kv[off..off + dim].fill(0.0);
            self.state_score[off..off + dim].fill(NEG_INF);
        }
        // `state_count` should also be aligned with `rem` so that decode
        // resumes at the right slot. Any in-progress accumulation past
        // `rem` was speculative and is now discarded.
        self.state_count = rem as u32;
    }

    /// Discard all rows and reset state.
    pub fn clear(&mut self) {
        self.n_comp = 0;
        self.state_count = 0;
        self.state_kv.fill(0.0);
        self.state_score.fill(NEG_INF);
    }
}

/// Per-dimension softmax aggregation. For each output dim `j`:
///
/// ```text
/// max_j  = max over r of score[r * dim + j]
/// denom  = sum over r of exp(score[r * dim + j] - max_j)
/// out[j] = sum over r of exp(score[r * dim + j] - max_j) * kv[r * dim + j] / denom
/// ```
///
/// If every `score[*, j]` is `-inf`, `out[j]` is set to `0` (no
/// observed contribution). This matches `ds4.c:6385-6420`.
fn per_dim_softmax_aggregate(
    state_kv: &[f32],
    state_score: &[f32],
    ratio: usize,
    head_dim: usize,
    out: &mut [f32],
) {
    debug_assert_eq!(state_kv.len(), ratio * head_dim);
    debug_assert_eq!(state_score.len(), ratio * head_dim);
    debug_assert_eq!(out.len(), head_dim);

    for j in 0..head_dim {
        // 1. find per-dim max.
        let mut max_s = NEG_INF;
        for r in 0..ratio {
            let s = state_score[r * head_dim + j];
            if s > max_s {
                max_s = s;
            }
        }
        if !max_s.is_finite() {
            out[j] = 0.0;
            continue;
        }
        // 2. softmax weighted sum.
        let mut denom = 0.0_f32;
        let mut numer = 0.0_f32;
        for r in 0..ratio {
            let s = state_score[r * head_dim + j];
            let w = (s - max_s).exp();
            denom += w;
            numer += w * state_kv[r * head_dim + j];
        }
        out[j] = if denom > 0.0 { numer / denom } else { 0.0 };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn const_row(v: f32, dim: usize) -> Vec<f32> {
        vec![v; dim]
    }

    #[test]
    fn empty_pool_has_no_rows() {
        let p = CompressedKvPool::new(8, 4, 6);
        assert!(p.is_empty());
        assert_eq!(p.len(), 0);
        assert_eq!(p.state_count(), 0);
        assert_eq!(p.ratio(), 4);
        assert_eq!(p.head_dim(), 6);
        assert!(p.rows().is_empty());
    }

    #[test]
    fn accumulate_before_boundary_returns_none() {
        let mut p = CompressedKvPool::new(8, 4, 4);
        for t in 0..3 {
            let r = p.accumulate(&const_row(t as f32, 4), &const_row(1.0, 4)).unwrap();
            assert!(r.is_none(), "step {t} should not emit");
            assert_eq!(p.state_count(), (t + 1) as u32);
        }
        assert_eq!(p.len(), 0);
    }

    #[test]
    fn accumulate_emits_at_boundary() {
        let mut p = CompressedKvPool::new(8, 4, 4);
        for t in 0..3 {
            p.accumulate(&const_row(t as f32, 4), &const_row(1.0, 4)).unwrap();
        }
        let emitted =
            p.accumulate(&const_row(3.0_f32, 4), &const_row(1.0, 4)).unwrap();
        assert_eq!(emitted, Some(0));
        assert_eq!(p.len(), 1);
        // After emission, state should be reset.
        assert_eq!(p.state_count(), 0);
    }

    #[test]
    fn uniform_scores_give_simple_average() {
        // All scores equal → softmax weights = 1/ratio → output = mean of kv.
        let mut p = CompressedKvPool::new(8, 4, 4);
        for t in 0..4 {
            p.accumulate(&const_row(t as f32, 4), &const_row(2.5_f32, 4)).unwrap();
        }
        // mean(0,1,2,3) = 1.5
        for &v in &p.rows()[..4] {
            assert!((v - 1.5).abs() < 1e-5, "got {v}");
        }
    }

    #[test]
    fn high_score_dominates() {
        // One token has score much higher than the others → output ≈ its kv.
        let mut p = CompressedKvPool::new(8, 4, 2);
        p.accumulate(&const_row(1.0, 2), &const_row(0.0, 2)).unwrap();
        p.accumulate(&const_row(2.0, 2), &const_row(0.0, 2)).unwrap();
        p.accumulate(&const_row(100.0, 2), &const_row(20.0, 2)).unwrap(); // dominator
        p.accumulate(&const_row(4.0, 2), &const_row(0.0, 2)).unwrap();
        for &v in &p.rows()[..2] {
            assert!((v - 100.0).abs() < 1e-3, "expected ~100, got {v}");
        }
    }

    #[test]
    fn neg_inf_scores_give_zero_output() {
        // If every score for a dim is left as NEG_INF (we wrote score=NEG_INF in all states),
        // the softmax falls back to 0. Use a small dim and don't fill at all — but that
        // means no emission. Instead push 4 rows with score=NEG_INF for one dim and
        // finite for another. Force pool's per-dim fallback.
        let mut p = CompressedKvPool::new(2, 4, 2);
        for _ in 0..4 {
            // dim 0 has finite score, dim 1 is NEG_INF.
            let kv = vec![5.0_f32, 7.0_f32];
            let mut score = vec![0.0_f32; 2];
            score[1] = NEG_INF;
            p.accumulate(&kv, &score).unwrap();
        }
        let row = &p.rows()[..2];
        assert!((row[0] - 5.0).abs() < 1e-5);
        assert_eq!(row[1], 0.0); // NEG_INF fallback
    }

    #[test]
    fn rejects_wrong_kv_length() {
        let mut p = CompressedKvPool::new(2, 4, 8);
        let err = p.accumulate(&[1.0; 6], &[0.0; 8]).unwrap_err();
        assert!(matches!(err, Error::ShapeMismatch { .. }));
    }

    #[test]
    fn rejects_wrong_score_length() {
        let mut p = CompressedKvPool::new(2, 4, 8);
        let err = p.accumulate(&[1.0; 8], &[0.0; 6]).unwrap_err();
        assert!(matches!(err, Error::ShapeMismatch { .. }));
    }

    #[test]
    fn errors_when_pool_is_full() {
        let mut p = CompressedKvPool::new(1, 2, 4); // cap=1
        // First 2 tokens fill, emit 1 compressed row.
        p.accumulate(&const_row(0.0, 4), &const_row(0.0, 4)).unwrap();
        p.accumulate(&const_row(1.0, 4), &const_row(0.0, 4)).unwrap();
        assert_eq!(p.len(), 1);
        // Next emission would overflow.
        p.accumulate(&const_row(2.0, 4), &const_row(0.0, 4)).unwrap();
        let err = p.accumulate(&const_row(3.0, 4), &const_row(0.0, 4)).unwrap_err();
        assert!(matches!(err, Error::CompressedPoolFull { .. }));
    }

    #[test]
    fn clear_resets_pool_and_state() {
        let mut p = CompressedKvPool::new(8, 4, 2);
        for _ in 0..6 {
            p.accumulate(&const_row(1.0, 2), &const_row(0.0, 2)).unwrap();
        }
        assert!(!p.is_empty());
        assert!(p.state_count() > 0);
        p.clear();
        assert_eq!(p.len(), 0);
        assert_eq!(p.state_count(), 0);
    }

    #[test]
    fn finish_prefill_state_aligns_to_remainder() {
        let mut p = CompressedKvPool::new(8, 4, 2);
        // 7 tokens: emits 1 compressed (4 tokens), state has 3 slots filled.
        for _ in 0..7 {
            p.accumulate(&const_row(1.0, 2), &const_row(0.5, 2)).unwrap();
        }
        assert_eq!(p.len(), 1);
        assert_eq!(p.state_count(), 3);
        // finish_prefill aligns to rem = 7 % 4 = 3 — already there. No change.
        p.finish_prefill_state(7);
        assert_eq!(p.state_count(), 3);
        // Now simulate that some speculative slots had been filled past the
        // remainder via a different code path. We can't do this through the
        // public API, but `finish_prefill_state` should normalize regardless.
        p.finish_prefill_state(2); // rem = 2, count drops to 2.
        assert_eq!(p.state_count(), 2);
    }
}
