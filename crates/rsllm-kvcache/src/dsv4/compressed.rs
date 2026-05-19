//! Compressed KV pool with per-dimension softmax-weighted aggregation.
//!
//! Every `compress_ratio` tokens, the per-layer cache emits one
//! compressed row to the long-history pool. The compression rule
//! (`ds4.c:6376-6427` `compressor_pool_decode_state`) is a per-output-
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
//! softmax over the ratio-sized window.
//!
//! ## State layout
//!
//! Upstream `ds4.c:6258-6290` allocates the per-layer compressor state
//! with a regime-dependent shape:
//!
//! - **Ratio-128 layers** (`coff = 1`): a single-lane buffer sized
//!   `[ratio × head_dim]`. Each token writes one `head_dim`-wide row
//!   at slot `pos % ratio`. Aggregation reads the ratio rows.
//!
//! - **Ratio-4 layers** (`coff = 2`): a **double-buffered** state
//!   sized `[2*ratio × 2*head_dim]`. Each token writes one
//!   `2*head_dim`-wide row at slot `ratio + (pos % ratio)`
//!   (i.e. always into the upper half). On every emission the buffer
//!   rotates: upper half is copied to lower half and back to upper
//!   half, so both halves now carry the just-emitted window. The next
//!   ratio-4 window then overwrites the upper half, and the aggregator
//!   sees a sliding window of `2*ratio = 8` tokens (the previous
//!   ratio-4 window in the lower half + the current window in the
//!   upper half) feeding the softmax. The two halves address two
//!   distinct `head_dim`-wide "lanes" within each `2*head_dim` row,
//!   per `ds4.c:6388-6420`.
//!
//! v0.1.0 implements the **primary attention compression lane only**.
//! Ratio-4 layers additionally maintain an indexer compression lane;
//! that lane has identical structure (different `head_dim`) and is
//! handled by [`super::indexer`] as a separate pool instance.
//!
//! ## API stability
//!
//! The current [`CompressedKvPool::accumulate`] takes `head_dim`-wide
//! `kv` and `score` slices. For ratio-4 layers the value is
//! **replicated** across both lanes of the upper-half row (lane p at
//! columns `[0, head_dim)`, lane c at columns `[head_dim, 2*head_dim)`).
//! Replication preserves the upstream invariant that the compressor
//! matmul output is `width = 2 * head_dim` wide and writes every lane
//! each token — without it, the rotation would move only NEG_INF
//! lane-p data into the lower half and the sliding-window aggregation
//! would collapse to the current ratio-4 window only. Identical
//! replication keeps the per-window numerical output equal to the
//! single-lane case while still giving the rotation a real "previous
//! window" payload to slide. F011.B will introduce a wide-row API
//! once the actual compressor algorithm (`x → kv_cur, sc_cur`
//! projections) is implemented; at that point the two lanes carry
//! independent payloads.
//!
//! Ported by reference from `ds4.c:6258-6420` (MIT, The ds4.c authors).
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
    /// In-progress aggregation state, `[state_rows × width]` row-major.
    /// For ratio-4: `state_rows = 8`, `width = 2 * head_dim`.
    /// For ratio-128: `state_rows = ratio`, `width = head_dim`.
    state_kv: Vec<f32>,
    /// Per-dimension score history, same shape as `state_kv`.
    /// Initial value is `NEG_INF` so unfilled slots are ignored by the
    /// per-dim softmax.
    state_score: Vec<f32>,
    /// Number of compressed rows currently in `compressed`.
    n_comp: usize,
    /// Maximum number of compressed rows the pool can hold.
    cap_comp: usize,
    /// How many tokens have been accepted in the current ratio-sized
    /// window. Wraps to 0 after each emission. Always in `[0, ratio)`.
    state_count: u32,
    /// Compression ratio for this layer (4 or 128; 0 layers do not own
    /// a compressed pool and should not be constructed).
    ratio: u32,
    /// Per-row output dimension (typically [`DSV4_HEAD_DIM`] for
    /// attention, `DSV4_N_INDEXER_HEAD_DIM` for the indexer).
    head_dim: usize,
    /// Multiplier on the state buffer's row count and lane width.
    /// `2` for ratio-4 (double-buffered), `1` otherwise (`ds4.c:6266`).
    coff: usize,
    /// `coff * head_dim` — width of one state row.
    width: usize,
    /// `coff * ratio` — number of state rows.
    state_rows: usize,
}

impl CompressedKvPool {
    /// Create an empty pool sized for `cap_comp` aggregated rows.
    ///
    /// `head_dim` is the per-row dimension. `ratio` is the layer's
    /// compression ratio (must be `> 0`). For `ratio == 4`, the
    /// internal state buffers are allocated double-sized to match
    /// upstream's `coff = 2` layout.
    #[must_use]
    pub fn new(cap_comp: usize, ratio: u32, head_dim: usize) -> Self {
        assert!(ratio > 0, "compression ratio must be > 0");
        assert!(head_dim > 0);
        let coff: usize = if ratio == 4 { 2 } else { 1 };
        let width = coff * head_dim;
        let state_rows = coff * ratio as usize;
        Self {
            compressed: vec![0.0_f32; cap_comp * head_dim],
            state_kv: vec![0.0_f32; state_rows * width],
            state_score: vec![NEG_INF; state_rows * width],
            n_comp: 0,
            cap_comp,
            state_count: 0,
            ratio,
            head_dim,
            coff,
            width,
            state_rows,
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

    /// Number of state slots filled since the last emission. Always
    /// in `[0, ratio)`; the underlying buffer has `coff * ratio` rows
    /// but the state-count is reported in window-relative terms.
    #[must_use]
    pub fn state_count(&self) -> u32 {
        self.state_count
    }

    /// Double-buffer multiplier. `2` for ratio-4 layers, `1` otherwise.
    /// Exposed so the caller of [`Self::accumulate_wide`] can size its
    /// `kv_cur` / `sc_cur` matmul output buffers to `width = coff *
    /// head_dim`.
    #[must_use]
    pub fn coff(&self) -> usize {
        self.coff
    }

    /// State-row width in floats — `coff * head_dim`. The
    /// [`Self::accumulate_wide`] entry point expects `kv` and `score`
    /// slices of exactly this length.
    #[must_use]
    pub fn width(&self) -> usize {
        self.width
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
    /// `kv` and `score` are both `head_dim`-wide. For ratio-4 layers
    /// the value is **replicated** across both `head_dim`-wide lanes
    /// of the upper-half row (lane p at columns `[0, head_dim)`, lane
    /// c at columns `[head_dim, 2*head_dim)`). Replication preserves
    /// the upstream behavior that the compressor matmul output is
    /// `width = 2 * head_dim` wide and writes every lane each token —
    /// without it, post-rotation lane p would stay at its initial
    /// NEG_INF and the sliding-window aggregation would collapse to
    /// the current ratio-4 window only. Replicating identical values
    /// keeps the per-window mean numerically equal to the single-lane
    /// case while giving the rotation a real "previous window"
    /// payload to slide into. F011.B will replace this entry point
    /// with a wide-row API once the compressor's `width`-wide
    /// `wkv·x` / `wgate·x` projections are implemented.
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

        // Place this token's data:
        // - ratio-128 (coff=1): write into row `pos_mod` at column 0.
        // - ratio-4   (coff=2): write into row `ratio + pos_mod`
        //   (upper half), filling the full `width = 2 * head_dim`
        //   row by replicating the `head_dim`-wide input into both
        //   lane p (`[0, head_dim)`) and lane c (`[head_dim, width)`).
        let slot = self.state_count as usize;
        let row = if self.coff == 2 {
            self.ratio as usize + slot
        } else {
            slot
        };
        let row_off = row * self.width;
        self.state_kv[row_off..row_off + self.head_dim].copy_from_slice(kv);
        self.state_score[row_off..row_off + self.head_dim].copy_from_slice(score);
        if self.coff == 2 {
            // Replicate into lane c so the post-rotation sliding-window
            // softmax sees the previous window's data in lane p of the
            // lower half. F011.B's wide-row API will replace this with
            // two distinct lane payloads from the compressor matmul.
            let off_c = row_off + self.head_dim;
            self.state_kv[off_c..off_c + self.head_dim].copy_from_slice(kv);
            self.state_score[off_c..off_c + self.head_dim].copy_from_slice(score);
        }
        self.state_count += 1;

        self.maybe_emit()
    }

    /// Wide-row variant of [`Self::accumulate`] used by the F011.B
    /// stateful compressor algorithm in `rsllm-models::dsv4::compressor`.
    ///
    /// `kv` and `score` are both `width = coff * head_dim`-wide rows
    /// produced by the compressor's `wkv·x` / `wgate·x` matmuls (with
    /// APE bias already added to `score`). The row is written into
    /// the upper-half slot (`ratio + pos_mod` for ratio-4) or the
    /// single-half slot (`pos_mod` for ratio-128), and the same
    /// boundary / rotation logic as [`Self::accumulate`] runs.
    ///
    /// **Post-processing contract.** On emission the per-dim softmax
    /// aggregate is written into the compressed array as a RAW pooled
    /// value, with no RMSNorm or RoPE applied yet. The caller MUST
    /// post-process that row in place via [`Self::compressed_row_mut`]
    /// before any downstream attention read sees it; the upstream
    /// `compressor_decode_one` (`ds4.c:6483-6500`) does exactly this.
    /// A caller that forgets the post-processing will leave a raw
    /// pooled row in `compressed`, which subsequent attention paths
    /// would treat as a final (RoPE-rotated, RMS-normalised) row and
    /// produce corrupt scores. This contract is in service of avoiding
    /// a per-emission scratch allocation; cleaner alternatives are
    /// possible if the perf cost is acceptable.
    ///
    /// Returns `Some(idx)` of the just-emitted (pre-post-processing)
    /// row, or `None` when more tokens are needed.
    ///
    /// # Errors
    /// - `Error::ShapeMismatch` if `kv.len() != width` or `score.len() != width`.
    /// - `Error::CompressedPoolFull` if an emission would exceed `cap_comp`.
    #[must_use = "an emitted row holds the raw pooled value; the caller \
                  must post-process it (RMSNorm + RoPE) via \
                  compressed_row_mut before downstream attention reads it"]
    pub fn accumulate_wide(
        &mut self,
        kv: &[f32],
        score: &[f32],
    ) -> Result<Option<usize>, Error> {
        if kv.len() != self.width {
            return Err(Error::ShapeMismatch {
                what: "CompressedKvPool::accumulate_wide: kv",
                expected: self.width,
                actual: kv.len(),
            });
        }
        if score.len() != self.width {
            return Err(Error::ShapeMismatch {
                what: "CompressedKvPool::accumulate_wide: score",
                expected: self.width,
                actual: score.len(),
            });
        }

        let slot = self.state_count as usize;
        let row = if self.coff == 2 {
            self.ratio as usize + slot
        } else {
            slot
        };
        let row_off = row * self.width;
        self.state_kv[row_off..row_off + self.width].copy_from_slice(kv);
        self.state_score[row_off..row_off + self.width].copy_from_slice(score);
        self.state_count += 1;

        self.maybe_emit()
    }

    /// Mutable access to a just-emitted compressed row, for callers of
    /// [`Self::accumulate_wide`] that need to apply RMSNorm + RoPE
    /// (and optionally FP8 quantize) in place before subsequent
    /// attention reads see the row.
    ///
    /// # Panics
    /// Panics if `idx >= n_comp` (asserting the caller is reading a
    /// row they actually emitted).
    pub fn compressed_row_mut(&mut self, idx: usize) -> &mut [f32] {
        assert!(idx < self.n_comp, "compressed_row_mut: idx out of range");
        let off = idx * self.head_dim;
        &mut self.compressed[off..off + self.head_dim]
    }

    /// Run the boundary check + aggregation + rotation step.
    /// Shared between [`Self::accumulate`] and [`Self::accumulate_wide`].
    fn maybe_emit(&mut self) -> Result<Option<usize>, Error> {
        if self.state_count < self.ratio {
            return Ok(None);
        }

        // State is full: aggregate and emit.
        if self.n_comp >= self.cap_comp {
            return Err(Error::CompressedPoolFull {
                cap: self.cap_comp,
            });
        }
        let comp_off = self.n_comp * self.head_dim;
        per_dim_softmax_aggregate(
            &self.state_kv,
            &self.state_score,
            self.ratio as usize,
            self.head_dim,
            self.coff,
            self.width,
            &mut self.compressed[comp_off..comp_off + self.head_dim],
        );

        // Post-emission state transition (ds4.c:6502-6519).
        if self.coff == 2 {
            // Ratio-4: rotate upper half down to lower, then duplicate
            // lower back to upper. Result: both halves contain the
            // just-emitted window. The next ratio-4 window overwrites
            // upper, while lower preserves the previous window for the
            // sliding-buffer softmax.
            let r = self.ratio as usize;
            for src_row in 0..r {
                let src = (r + src_row) * self.width;
                let dst = src_row * self.width;
                self.state_kv.copy_within(src..src + self.width, dst);
                self.state_score
                    .copy_within(src..src + self.width, dst);
            }
            for src_row in 0..r {
                let src = src_row * self.width;
                let dst = (r + src_row) * self.width;
                self.state_kv.copy_within(src..src + self.width, dst);
                self.state_score
                    .copy_within(src..src + self.width, dst);
            }
        } else {
            // Ratio-128: upstream relies on natural overwriting in the
            // next window, but a defensive clear keeps a misuse of
            // `state_count` from leaking stale data through the next
            // softmax (no upstream divergence — overwritten values are
            // identical either way).
            self.state_kv.fill(0.0);
            self.state_score.fill(NEG_INF);
        }
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
    /// `n_tokens` is the total prefill length. The clear-range matches
    /// upstream exactly:
    /// - ratio-4: clear rows `[ratio + rem, 2*ratio)`. Rows `[0, ratio)`
    ///   carry the rotated previous window; rows `[ratio, ratio + rem)`
    ///   carry the in-flight window's first `rem` tokens.
    /// - ratio-128: clear rows `[rem, ratio)`.
    ///
    /// **Precondition.** `n_tokens` should equal the total number of
    /// tokens actually accumulated (i.e. `pos + 1` at call time). The
    /// function unconditionally overwrites `state_count` with
    /// `n_tokens % ratio`; if the supplied count disagrees with the
    /// caller's true position, subsequent `accumulate` /
    /// `accumulate_wide` calls will write to the wrong state slot and
    /// produce silently wrong compressed rows. No runtime check
    /// enforces the precondition because some callers (e.g. tests
    /// that simulate prefill-truncation paths) intentionally pass
    /// a smaller `n_tokens` to normalise the count downward.
    pub fn finish_prefill_state(&mut self, n_tokens: usize) {
        let r = self.ratio as usize;
        let rem = n_tokens % r;
        let clear_start = if self.coff == 2 { r + rem } else { rem };
        let clear_end = self.state_rows;
        for row in clear_start..clear_end {
            let off = row * self.width;
            self.state_kv[off..off + self.width].fill(0.0);
            self.state_score[off..off + self.width].fill(NEG_INF);
        }
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
/// max_j  = max over r of score[r, j] (across both lanes if coff=2)
/// denom  = sum over r of exp(score[r, j] - max_j)
/// out[j] = sum over r of exp(score[r, j] - max_j) * kv[r, j] / denom
/// ```
///
/// For `coff == 2` (ratio-4), each of the `ratio` rows reads from two
/// `head_dim`-wide lanes:
/// - **Lane p** at `state[r * width + j]` for `r in 0..ratio`.
/// - **Lane c** at `state[(ratio + r) * width + head_dim + j]` for `r in 0..ratio`.
///
/// Both lanes feed the same per-dim softmax, matching `ds4.c:6388-6420`.
/// If every score for a dim is `NEG_INF` (no observed contribution),
/// `out[j]` is set to `0`.
fn per_dim_softmax_aggregate(
    state_kv: &[f32],
    state_score: &[f32],
    ratio: usize,
    head_dim: usize,
    coff: usize,
    width: usize,
    out: &mut [f32],
) {
    let state_rows = coff * ratio;
    debug_assert_eq!(state_kv.len(), state_rows * width);
    debug_assert_eq!(state_score.len(), state_rows * width);
    debug_assert_eq!(out.len(), head_dim);
    debug_assert_eq!(width, coff * head_dim);

    for j in 0..head_dim {
        // 1. Find per-dim max across all contributing slots.
        let mut max_s = NEG_INF;
        if coff == 2 {
            for r in 0..ratio {
                let sp = state_score[r * width + j];
                let sc = state_score[(ratio + r) * width + head_dim + j];
                if sp > max_s {
                    max_s = sp;
                }
                if sc > max_s {
                    max_s = sc;
                }
            }
        } else {
            for r in 0..ratio {
                let s = state_score[r * width + j];
                if s > max_s {
                    max_s = s;
                }
            }
        }
        // Match ds4.c:6402 exactly: short-circuit only when every
        // slot is NEG_INF (the initial state — no observation yet).
        // The previous `!max_s.is_finite()` Rust guard *also* caught
        // +Inf, which is a structural divergence from upstream even
        // though the observable result is the same: +Inf propagates
        // through `expf(+Inf - +Inf) = NaN` and the trailing
        // `denom > 0.0` ternary below catches the NaN and emits 0.0
        // — exactly what upstream produces. Mirroring the upstream
        // guard expression keeps the code structurally portable so
        // future audits read top-to-bottom against ds4.c.
        if max_s <= NEG_INF * 0.5 {
            out[j] = 0.0;
            continue;
        }
        // 2. Softmax-weighted sum.
        let mut denom = 0.0_f32;
        let mut numer = 0.0_f32;
        if coff == 2 {
            for r in 0..ratio {
                let off_p = r * width + j;
                let off_c = (ratio + r) * width + head_dim + j;
                let wp = (state_score[off_p] - max_s).exp();
                let wc = (state_score[off_c] - max_s).exp();
                denom += wp + wc;
                numer += wp * state_kv[off_p] + wc * state_kv[off_c];
            }
        } else {
            for r in 0..ratio {
                let off = r * width + j;
                let w = (state_score[off] - max_s).exp();
                denom += w;
                numer += w * state_kv[off];
            }
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
    fn ratio4_pool_allocates_double_buffer() {
        // Internal sanity: a ratio-4 pool with head_dim=6 must hold
        // 2*4*2*6 = 96 floats for kv + 96 for score. Inspect via the
        // accumulate path (which would panic / mis-write if sizes were
        // off).
        let mut p = CompressedKvPool::new(2, 4, 6);
        for _ in 0..4 {
            p.accumulate(&const_row(0.0, 6), &const_row(0.0, 6)).unwrap();
        }
        // No panic = buffer was big enough.
        assert_eq!(p.len(), 1);
    }

    #[test]
    fn ratio128_pool_keeps_single_buffer() {
        // ratio-128 is the only other valid setting in DS V4 Flash.
        // Buffer should be 128 * head_dim, not 2*128*2*head_dim.
        let mut p = CompressedKvPool::new(1, 128, 4);
        for _ in 0..128 {
            p.accumulate(&const_row(0.0, 4), &const_row(0.0, 4)).unwrap();
        }
        assert_eq!(p.len(), 1);
    }

    #[test]
    fn coff_and_width_track_ratio() {
        let r4 = CompressedKvPool::new(2, 4, 8);
        assert_eq!(r4.coff(), 2);
        assert_eq!(r4.width(), 16); // 2 * 8
        let r128 = CompressedKvPool::new(1, 128, 8);
        assert_eq!(r128.coff(), 1);
        assert_eq!(r128.width(), 8);
    }

    #[test]
    fn accumulate_wide_emits_at_boundary_with_distinct_lanes_ratio4() {
        // F011.B-style entry: caller supplies width-wide (= 2*head_dim)
        // kv/score rows where lane p and lane c carry DISTINCT values.
        // The dual-lane softmax should mix both lanes per dim.
        let head_dim = 2;
        let width = 4; // 2 * head_dim, ratio-4
        let mut p = CompressedKvPool::new(2, 4, head_dim);
        // Score uniform 0.0 across all lanes — pure mean across all 8
        // contributing slots (4 lane-p + 4 lane-c, but for the first
        // window lane-p of the lower half is NEG_INF; the upstream
        // softmax then collapses to the 4 upper-half rows × 2 lanes.
        let score = vec![0.0_f32; width];
        // Token 0: kv lane p = [10, 10], lane c = [20, 20]
        let mut kv0 = vec![0.0_f32; width];
        kv0[..head_dim].copy_from_slice(&[10.0, 10.0]);
        kv0[head_dim..].copy_from_slice(&[20.0, 20.0]);
        for _ in 0..4 {
            p.accumulate_wide(&kv0, &score).unwrap();
        }
        // First window aggregation reads:
        // - Lane p (lower-half rows 0..4 col 0..head_dim): NEG_INF score,
        //   contributes 0.
        // - Lane c (upper-half rows 4..7 col head_dim..2*head_dim):
        //   4 rows × kv=20, score=0 → softmax-mean = 20.
        // So output should be 20 for each dim.
        let row = &p.rows()[..head_dim];
        for &v in row {
            assert!((v - 20.0).abs() < 1e-5, "expected 20.0, got {v}");
        }
    }

    #[test]
    fn accumulate_wide_rejects_wrong_width() {
        let mut p = CompressedKvPool::new(2, 4, 6);
        // width = 12 for ratio-4. Passing 6 (head_dim) should fail.
        let err = p.accumulate_wide(&[0.0; 6], &[0.0; 12]).unwrap_err();
        assert!(matches!(err, Error::ShapeMismatch { .. }));
    }

    #[test]
    fn compressed_row_mut_allows_in_place_post_processing() {
        let head_dim = 4;
        let mut p = CompressedKvPool::new(2, 128, head_dim);
        for _ in 0..128 {
            p.accumulate(&const_row(2.0, head_dim), &const_row(0.0, head_dim))
                .unwrap();
        }
        // Just-emitted row holds the raw pool value (here, 2.0). Caller
        // overwrites in place — mimics what F011.B will do for RMSNorm
        // and RoPE.
        let row = p.compressed_row_mut(0);
        for v in row.iter_mut() {
            *v = 99.0;
        }
        let final_row = &p.rows()[..head_dim];
        assert!(final_row.iter().all(|&v| v == 99.0));
    }

    #[test]
    fn compressed_row_mut_panics_on_unemitted_idx() {
        let mut p = CompressedKvPool::new(2, 128, 4);
        // No emissions yet — n_comp == 0, idx 0 is out of range.
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            p.compressed_row_mut(0);
        }));
        assert!(result.is_err());
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
        // After emission, state-count should be reset.
        assert_eq!(p.state_count(), 0);
    }

    #[test]
    fn uniform_scores_give_simple_average() {
        // All lane-c scores equal → softmax weights = 1/ratio → output
        // = mean of kv values. Lane p stays at NEG_INF so it contributes
        // zero to the softmax. Identical numerical result to a single-
        // lane pool.
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
        // If every score for a dim is NEG_INF on BOTH lanes, the softmax
        // falls back to 0. We test the dim-1 fallback by writing NEG_INF
        // for that dim alone.
        let mut p = CompressedKvPool::new(2, 4, 2);
        for _ in 0..4 {
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
        let mut p = CompressedKvPool::new(1, 2, 4); // cap=1, ratio=2 → coff=1.
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
        // Now normalise to a different remainder.
        p.finish_prefill_state(2); // rem = 2, count drops to 2.
        assert_eq!(p.state_count(), 2);
    }

    #[test]
    fn ratio4_second_window_sees_previous_window_via_lower_half() {
        // After the first ratio-4 emission, ds4.c rotates the upper
        // half to the lower half (then duplicates it back), so the
        // lower half carries the just-emitted window. The next 4 tokens
        // overwrite the upper half but the aggregator sees a sliding
        // window of 8 tokens (4 from the previous window + 4 new).
        //
        // Test setup: emit two windows with distinct, distinguishable
        // scores so the post-rotation aggregation's lane-p contribution
        // is observable.
        let mut p = CompressedKvPool::new(8, 4, 2);
        // Window 1: kv = 10, score = 0.0 (uniform). Mean kv ≈ 10.
        for _ in 0..4 {
            p.accumulate(&const_row(10.0, 2), &const_row(0.0, 2)).unwrap();
        }
        let row0 = p.rows()[..2].to_vec();
        for &v in &row0 {
            assert!((v - 10.0).abs() < 1e-5);
        }

        // Window 2: kv = 20, score = 0.0 (uniform). If only the new
        // window were aggregated, output would be 20.0. With the
        // sliding 8-token buffer, lane p (= rotated previous window,
        // kv = 10, score = 0.0) and lane c (= new window, kv = 20,
        // score = 0.0) both contribute equally → average = 15.0.
        for _ in 0..4 {
            p.accumulate(&const_row(20.0, 2), &const_row(0.0, 2)).unwrap();
        }
        let row1 = &p.rows()[2..4];
        for &v in row1 {
            assert!((v - 15.0).abs() < 1e-5, "expected sliding-window mean 15.0, got {v}");
        }
    }

    #[test]
    fn ratio128_emission_clears_state_for_next_window() {
        // ratio-128 has no rotation; state is cleared after emit.
        let mut p = CompressedKvPool::new(2, 128, 2);
        for _ in 0..128 {
            p.accumulate(&const_row(5.0, 2), &const_row(0.0, 2)).unwrap();
        }
        assert_eq!(p.len(), 1);
        // After emit, all state slots return to NEG_INF / 0 — the next
        // emission must depend purely on the new window's tokens.
        for _ in 0..128 {
            p.accumulate(&const_row(50.0, 2), &const_row(0.0, 2)).unwrap();
        }
        let row1 = &p.rows()[2..4];
        for &v in row1 {
            assert!((v - 50.0).abs() < 1e-3, "ratio-128 should not carry prior window, got {v}");
        }
    }

    #[test]
    fn finish_prefill_clears_correct_rows_for_ratio4() {
        // After 7 tokens on a ratio-4 pool: 1 emit, then 3 in the
        // upper half (rows 4..7). finish_prefill_state(7) should clear
        // rows [ratio + rem, 2*ratio) = [7, 8) — i.e. the single
        // unfilled upper-half slot. The lower half (rotated previous
        // window) must be preserved.
        let mut p = CompressedKvPool::new(8, 4, 2);
        for _ in 0..7 {
            p.accumulate(&const_row(7.0, 2), &const_row(0.0, 2)).unwrap();
        }
        p.finish_prefill_state(7);
        assert_eq!(p.state_count(), 3);
        // Sanity: subsequent token write at slot 3 should land at row
        // ratio + 3 = 7, the just-cleared slot. Then emit fires and
        // both halves contribute. The lower half carries the previous
        // (uniform 7.0) window; the upper half is the partial new
        // window (3 tokens at kv=7.0 from prefill + 1 token at kv=99.0).
        // Per-dim aggregate with uniform zero scores: 8 tokens with kv
        // values (7,7,7,7, 7,7,7,99) → mean = (7*7 + 99) / 8 = 18.5.
        p.accumulate(&const_row(99.0, 2), &const_row(0.0, 2)).unwrap();
        let row = &p.rows()[2..4];
        for &v in row {
            assert!((v - 18.5).abs() < 1e-4, "got {v}");
        }
    }
}
