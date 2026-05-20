//! Ratio-4 sparse indexer for DS V4 Flash.
//!
//! Only the **ratio-4** subset of layers (`layer_compress_ratio == 4`,
//! ~21 of 43 layers) maintains an indexer. Its job is to choose
//! `N_INDEXER_TOP_K = 512` compressed rows from the long-history pool
//! that contribute most to the current token's attention — so the rest
//! can be skipped, turning O(n_comp) into O(top_k) for the long-history
//! attention sweep.
//!
//! The indexer storage is structurally identical to the primary
//! compressed-KV pool ([`super::compressed::CompressedKvPool`]), with
//! a different per-row dimension (`N_INDEXER_HEAD_DIM = 128` vs the
//! primary `HEAD_DIM = 512`). We reuse the pool type for storage and
//! add a top-K selection routine on top.
//!
//! Selection algorithm (`ds4.c:6868-6914` / `indexer_allowed_*`):
//!
//! ```text
//! q       : [N_INDEXER_HEAD × N_INDEXER_HEAD_DIM]  query latent
//! weights : [N_INDEXER_HEAD]                       per-head soft gate
//! scale   : 1 / sqrt(head_dim * n_head)
//!
//! for each compressed row c in 0..n_comp:
//!     score[c] = 0
//!     for each head h in 0..n_head:
//!         d = dot(kv_c, q_h)
//!         if d < 0: d = 0          # ReLU
//!         score[c] += d * weights[h] * scale
//!
//! top_k_indices = argpartition(score, K=N_INDEXER_TOP_K)
//! ```
//!
//! Note that `weights` is pre-scaled by `1 / sqrt(head_dim * n_head)`
//! in ds4 (`ds4.c:6884-6885`) — we expose [`scale_factor`] for the
//! caller and let them pass pre-scaled weights to keep `select_top_k`
//! score-formula-agnostic.
//!
//! Ported by reference from `ds4.c:6862-6914` (MIT, The ds4.c authors).
//! Line numbers pinned to ds4 commit `ef0a490` (2026-05-17).

use crate::Error;

use super::compressed::CompressedKvPool;
use super::shape::{DSV4_N_INDEXER_HEAD, DSV4_N_INDEXER_HEAD_DIM, DSV4_N_INDEXER_TOP_K};

/// Standard scale factor that the model applies to per-head indexer
/// weights: `1 / sqrt(head_dim * n_head)`. (`ds4.c:6884`.)
///
/// Callers should multiply their raw per-head logits by this scale
/// before passing them to [`IndexerPool::select_top_k`]. Exposed as a
/// `const` to allow compile-time evaluation.
#[must_use]
pub fn scale_factor() -> f32 {
    1.0 / ((DSV4_N_INDEXER_HEAD * DSV4_N_INDEXER_HEAD_DIM) as f32).sqrt()
}

/// Per-layer indexer pool — a CompressedKvPool with the indexer's
/// `head_dim` plus a sparse top-K selection method.
#[derive(Debug)]
pub struct IndexerPool {
    /// Underlying compressed-row storage (`head_dim = N_INDEXER_HEAD_DIM`).
    pool: CompressedKvPool,
}

impl IndexerPool {
    /// Construct an empty indexer pool sized for `cap_comp` rows.
    ///
    /// `ratio` is the layer's compression ratio (must be `4` for an
    /// indexer-bearing layer per [`super::shape::layer_has_indexer`]).
    #[must_use]
    pub fn new(cap_comp: usize, ratio: u32) -> Self {
        debug_assert_eq!(
            ratio, 4,
            "indexer is only valid on ratio-4 layers (got ratio={ratio})",
        );
        Self {
            pool: CompressedKvPool::new(cap_comp, ratio, DSV4_N_INDEXER_HEAD_DIM),
        }
    }

    /// Number of compressed indexer rows currently held.
    #[must_use]
    pub fn len(&self) -> usize {
        self.pool.len()
    }

    /// Returns `true` if no rows have been emitted yet.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.pool.is_empty()
    }

    /// Read-only slice over all indexer rows in chronological order.
    /// Shape `[n_comp × N_INDEXER_HEAD_DIM]` row-major.
    #[must_use]
    pub fn rows(&self) -> &[f32] {
        self.pool.rows()
    }

    /// Append one token's (kv, score) pair to the indexer's compressor
    /// state. Returns `Some(emitted_idx)` when a ratio-4 boundary fires.
    ///
    /// **Tests/internal only.** As of F011.D the production path in
    /// `rsllm-models::dsv4::attention::ThreeTierAttention::run_layer`
    /// drives the indexer via `compressor_decode_one` against
    /// [`Self::inner_mut`] (the wide-row `accumulate_wide` API).
    /// This `head_dim`-wide entry point is retained for unit tests
    /// of [`Self::select_top_k`].
    ///
    /// # Errors
    /// Bubbles up [`CompressedKvPool::accumulate`] errors.
    pub fn accumulate(&mut self, kv: &[f32], score: &[f32]) -> Result<Option<usize>, Error> {
        self.pool.accumulate(kv, score)
    }

    /// Width of one row in the underlying compressor state buffer —
    /// `coff * head_dim`. For an indexer (always ratio-4) this is
    /// `2 * N_INDEXER_HEAD_DIM = 256`.
    #[must_use]
    pub fn width(&self) -> usize {
        self.pool.width()
    }

    /// Mutable reference to the underlying [`CompressedKvPool`].
    /// Exposed (F011.D) so the rsllm-models `compressor_decode_one`
    /// kernel — which operates on `CompressedKvPool` directly — can
    /// drive the indexer's stateful per-token pipeline. The wrapper
    /// stays in place for its `select_top_k` method and for type-safety
    /// at construction sites; this accessor is the seam.
    ///
    /// **Caller contract.** Use only to advance per-token state via
    /// `accumulate_wide` / `compressed_row_mut`. Do NOT invoke
    /// `clear()` or `finish_prefill_state(...)` through this reference
    /// mid-forward-pass — those operations reset the pool's internal
    /// token cursor, which would silently desynchronise the indexer
    /// from the layer's compressor pool and SWA ring. Session-level
    /// resets should go through the cache-wide `ThreeTierKvCache::clear`
    /// / `finish_prefill` entry points, which keep all three tiers
    /// aligned.
    pub fn inner_mut(&mut self) -> &mut CompressedKvPool {
        &mut self.pool
    }

    /// Discard all rows and reset state.
    pub fn clear(&mut self) {
        self.pool.clear();
    }

    /// Run prefill-state cleanup so decode resumes from the right slot.
    pub fn finish_prefill_state(&mut self, n_tokens: usize) {
        self.pool.finish_prefill_state(n_tokens);
    }

    /// Score each compressed row against a query `q` and per-head
    /// weights, then return the indices of the `top_k` highest-scoring
    /// rows (in unspecified order). When fewer than `top_k` rows exist
    /// in the pool, returns all of them.
    ///
    /// `q.len()` must equal `N_INDEXER_HEAD * N_INDEXER_HEAD_DIM`.
    /// `weights.len()` must equal `N_INDEXER_HEAD`. The caller is
    /// responsible for any scaling on `weights` (typically multiplying
    /// by [`scale_factor`]).
    ///
    /// `top_k = 0` returns an empty Vec. `top_k > n_comp` clamps to
    /// `n_comp` (matches ds4's `top_k = MIN(N_INDEXER_TOP_K, n_comp)`).
    ///
    /// # Errors
    /// - [`Error::ShapeMismatch`] if `q` or `weights` have the wrong length.
    pub fn select_top_k(
        &self,
        q: &[f32],
        weights: &[f32],
        top_k: usize,
    ) -> Result<Vec<u32>, Error> {
        let n_head = DSV4_N_INDEXER_HEAD;
        let head_dim = DSV4_N_INDEXER_HEAD_DIM;
        if q.len() != n_head * head_dim {
            return Err(Error::ShapeMismatch {
                what: "IndexerPool::select_top_k: q",
                expected: n_head * head_dim,
                actual: q.len(),
            });
        }
        if weights.len() != n_head {
            return Err(Error::ShapeMismatch {
                what: "IndexerPool::select_top_k: weights",
                expected: n_head,
                actual: weights.len(),
            });
        }

        let n_comp = self.pool.len();
        if n_comp == 0 || top_k == 0 {
            return Ok(Vec::new());
        }
        let effective_k = top_k.min(n_comp);

        // 1. Score every compressed row.
        let mut scores = vec![0.0_f32; n_comp];
        let rows = self.pool.rows();
        for c in 0..n_comp {
            let kv = &rows[c * head_dim..(c + 1) * head_dim];
            let mut s = 0.0_f32;
            for h in 0..n_head {
                let qh = &q[h * head_dim..(h + 1) * head_dim];
                let mut dot = 0.0_f32;
                for i in 0..head_dim {
                    dot += kv[i] * qh[i];
                }
                // ReLU on dot, then weighted accumulate.
                if dot > 0.0 {
                    s += dot * weights[h];
                }
            }
            scores[c] = s;
        }

        // 2. Pick top-`effective_k` indices. ds4 uses an O(top_k * n_comp)
        //    repeated-best-scan (`ds4.c:6899-6909`). We mirror that — it
        //    is the simplest correct algorithm and stays well within
        //    budget at `top_k=512, n_comp ≤ ctx/4`.
        let mut allowed = vec![false; n_comp];
        let mut out = Vec::with_capacity(effective_k);
        for _ in 0..effective_k {
            let mut best_idx = 0_usize;
            let mut best_score = f32::NEG_INFINITY;
            for c in 0..n_comp {
                if !allowed[c] && scores[c] > best_score {
                    best_score = scores[c];
                    best_idx = c;
                }
            }
            allowed[best_idx] = true;
            out.push(best_idx as u32);
        }
        Ok(out)
    }

    /// Returns [`DSV4_N_INDEXER_TOP_K`] = 512, the canonical top-K cap.
    /// Exposed as a method so test/integration code can express
    /// `pool.select_top_k(q, w, IndexerPool::default_top_k())` without
    /// importing the shape constants.
    #[must_use]
    pub fn default_top_k() -> usize {
        DSV4_N_INDEXER_TOP_K
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsv4::shape::DSV4_N_INDEXER_HEAD_DIM;

    fn fill_pool(pool: &mut IndexerPool, n_rows: usize, kv_seed: impl Fn(usize) -> f32) {
        // Each emission requires 4 accumulate calls (ratio=4). We push
        // identical kv and score within each 4-window so the aggregation
        // is deterministic.
        for r in 0..n_rows {
            let kv = vec![kv_seed(r); DSV4_N_INDEXER_HEAD_DIM];
            let score = vec![0.0_f32; DSV4_N_INDEXER_HEAD_DIM]; // uniform score
            for _ in 0..4 {
                pool.accumulate(&kv, &score).unwrap();
            }
        }
    }

    #[test]
    fn empty_pool_returns_empty_selection() {
        let pool = IndexerPool::new(4, 4);
        let q = vec![0.0_f32; DSV4_N_INDEXER_HEAD * DSV4_N_INDEXER_HEAD_DIM];
        let w = vec![0.0_f32; DSV4_N_INDEXER_HEAD];
        let sel = pool.select_top_k(&q, &w, 10).unwrap();
        assert!(sel.is_empty());
    }

    #[test]
    fn top_k_clamps_to_n_comp() {
        let mut pool = IndexerPool::new(4, 4);
        fill_pool(&mut pool, 3, |r| r as f32);
        let q = vec![0.0_f32; DSV4_N_INDEXER_HEAD * DSV4_N_INDEXER_HEAD_DIM];
        // Weights and q are zero → all scores 0; selection still returns 3 indices.
        let w = vec![0.0_f32; DSV4_N_INDEXER_HEAD];
        let sel = pool.select_top_k(&q, &w, 10).unwrap();
        assert_eq!(sel.len(), 3);
    }

    #[test]
    fn top_k_zero_returns_empty() {
        let mut pool = IndexerPool::new(4, 4);
        fill_pool(&mut pool, 2, |r| r as f32);
        let q = vec![1.0_f32; DSV4_N_INDEXER_HEAD * DSV4_N_INDEXER_HEAD_DIM];
        let w = vec![1.0_f32; DSV4_N_INDEXER_HEAD];
        let sel = pool.select_top_k(&q, &w, 0).unwrap();
        assert!(sel.is_empty());
    }

    #[test]
    fn rejects_wrong_q_length() {
        let pool = IndexerPool::new(4, 4);
        let q = vec![0.0_f32; 10];
        let w = vec![0.0_f32; DSV4_N_INDEXER_HEAD];
        let err = pool.select_top_k(&q, &w, 1).unwrap_err();
        assert!(matches!(err, Error::ShapeMismatch { .. }));
    }

    #[test]
    fn rejects_wrong_weights_length() {
        let pool = IndexerPool::new(4, 4);
        let q = vec![0.0_f32; DSV4_N_INDEXER_HEAD * DSV4_N_INDEXER_HEAD_DIM];
        let w = vec![0.0_f32; 10];
        let err = pool.select_top_k(&q, &w, 1).unwrap_err();
        assert!(matches!(err, Error::ShapeMismatch { .. }));
    }

    #[test]
    fn selects_highest_scoring_row() {
        // Build 4 emitted rows with distinct kv values. Query is all ones,
        // weights uniform positive — score is just sum(kv) per row. Top-2
        // should pick the two largest.
        let mut pool = IndexerPool::new(8, 4);
        fill_pool(&mut pool, 4, |r| r as f32); // kvs = 0, 1, 2, 3
        let q = vec![1.0_f32; DSV4_N_INDEXER_HEAD * DSV4_N_INDEXER_HEAD_DIM];
        let w = vec![1.0_f32; DSV4_N_INDEXER_HEAD];
        let sel = pool.select_top_k(&q, &w, 2).unwrap();
        assert_eq!(sel.len(), 2);
        let mut sorted: Vec<u32> = sel.into_iter().collect();
        sorted.sort_unstable();
        assert_eq!(sorted, vec![2, 3]);
    }

    #[test]
    fn relu_filters_negative_dots() {
        // Two rows: row 0 has positive correlation with q, row 1 has negative.
        // With ReLU, row 1's contribution is clipped to 0, so row 0 wins
        // regardless of magnitude.
        let mut pool = IndexerPool::new(4, 4);
        // emit row 0: kv = +1
        for _ in 0..4 {
            pool.accumulate(
                &vec![1.0_f32; DSV4_N_INDEXER_HEAD_DIM],
                &vec![0.0_f32; DSV4_N_INDEXER_HEAD_DIM],
            )
            .unwrap();
        }
        // emit row 1: kv = -100 (very negative)
        for _ in 0..4 {
            pool.accumulate(
                &vec![-100.0_f32; DSV4_N_INDEXER_HEAD_DIM],
                &vec![0.0_f32; DSV4_N_INDEXER_HEAD_DIM],
            )
            .unwrap();
        }
        let q = vec![1.0_f32; DSV4_N_INDEXER_HEAD * DSV4_N_INDEXER_HEAD_DIM];
        let w = vec![1.0_f32; DSV4_N_INDEXER_HEAD];
        let sel = pool.select_top_k(&q, &w, 1).unwrap();
        assert_eq!(sel, vec![0]);
    }

    #[test]
    fn scale_factor_is_positive_finite() {
        let s = scale_factor();
        assert!(s > 0.0 && s.is_finite());
        // Expected: 1 / sqrt(64 * 128) = 1 / sqrt(8192) ≈ 0.01104854
        assert!((s - (1.0 / (8192.0_f32).sqrt())).abs() < 1e-7);
    }

    #[test]
    fn default_top_k_matches_constant() {
        assert_eq!(IndexerPool::default_top_k(), DSV4_N_INDEXER_TOP_K);
    }
}
