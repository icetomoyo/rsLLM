//! Top-level three-tier KV cache: SWA ring + compressed pool + indexer.
//!
//! Composes one [`LayerCache`] per transformer block. Each layer's
//! cache holds:
//!
//! - A raw SWA ring ([`super::swa::RawSwaRing`]) — most recent
//!   `N_SWA = 128` token KV-latent rows.
//! - A compressed-KV pool ([`super::compressed::CompressedKvPool`]) —
//!   long-history rows aggregated every `compress_ratio` tokens.
//! - For ratio-4 layers only, an indexer pool ([`super::indexer::IndexerPool`])
//!   for the top-K sparse selection.
//!
//! Dense layers (`il < 2`, `compress_ratio == 0`) own only the SWA ring.
//!
//! Ported by reference from `ds4.c:6068-6371` (MIT, The ds4.c authors).
//! Line numbers pinned to ds4 commit `ef0a490` (2026-05-17).

use crate::Error;

use super::compressed::CompressedKvPool;
use super::indexer::IndexerPool;
use super::shape::{
    DSV4_HEAD_DIM, DSV4_N_INDEXER_HEAD_DIM, DSV4_N_LAYER, DSV4_N_SWA, layer_compress_ratio,
    layer_has_indexer,
};
use super::swa::RawSwaRing;

/// Per-layer cache state. Wraps the three tiers with optional bits
/// according to the layer's compression regime.
#[derive(Debug)]
pub struct LayerCache {
    /// Layer index this cache belongs to (`0..DSV4_N_LAYER`).
    pub il: usize,
    /// Compression ratio for this layer (0 / 4 / 128).
    pub compress_ratio: u32,
    /// Raw KV sliding-window ring (present on every layer).
    pub swa: RawSwaRing,
    /// Compressed-KV pool. `None` for dense layers (`compress_ratio == 0`).
    pub compressed: Option<CompressedKvPool>,
    /// Ratio-4 indexer. `Some` only when `layer_has_indexer(il)`.
    pub indexer: Option<IndexerPool>,
}

/// Hard upper bound on `ctx_size`. Anything larger would imply
/// per-layer allocations in the multi-GiB range and is rejected
/// at construction time. Picked at 1 M tokens — comfortably above
/// any v0.1.0 use case, comfortably below the multiplicative
/// overflow threshold for `ctx_size * head_dim * f32_size`.
pub const DSV4_KVC_MAX_CTX_SIZE: usize = 1 << 20;

impl LayerCache {
    /// Allocate cache state for layer `il`. `ctx_size` is the maximum
    /// supported context length; the compressed-pool capacity is
    /// derived from it (`ctx_size / ratio + 2`, matching ds4's sizing
    /// at `ds4.c:6128`).
    ///
    /// # Panics
    /// Panics if `il >= DSV4_N_LAYER`, `ctx_size == 0`, or
    /// `ctx_size > DSV4_KVC_MAX_CTX_SIZE`. Callers that need a
    /// fallible interface should validate up front.
    #[must_use]
    pub fn new(il: usize, ctx_size: usize) -> Self {
        assert!(il < DSV4_N_LAYER, "layer index {il} >= DSV4_N_LAYER {DSV4_N_LAYER}");
        assert!(ctx_size > 0, "ctx_size must be > 0");
        assert!(
            ctx_size <= DSV4_KVC_MAX_CTX_SIZE,
            "ctx_size {ctx_size} exceeds DSV4_KVC_MAX_CTX_SIZE = {DSV4_KVC_MAX_CTX_SIZE}",
        );
        let ratio = layer_compress_ratio(il);
        let swa = RawSwaRing::new(DSV4_N_SWA.min(ctx_size), DSV4_HEAD_DIM);
        let compressed = if ratio == 0 {
            None
        } else {
            let cap = ctx_size / (ratio as usize) + 2;
            Some(CompressedKvPool::with_dsv4_attn(cap, ratio))
        };
        let indexer = if layer_has_indexer(il) {
            // Indexer shares the same capacity sizing as the primary
            // compressed pool — they emit on the same boundary cadence.
            let cap = ctx_size / 4 + 2;
            Some(IndexerPool::new(cap, 4))
        } else {
            None
        };
        Self {
            il,
            compress_ratio: ratio,
            swa,
            compressed,
            indexer,
        }
    }

    /// Discard all rows + state. Capacity is preserved so the layer
    /// can be re-used across sessions without reallocation.
    pub fn clear(&mut self) {
        self.swa.clear();
        if let Some(c) = self.compressed.as_mut() {
            c.clear();
        }
        if let Some(idx) = self.indexer.as_mut() {
            idx.clear();
        }
    }
}

/// Append-input bundle for one token's contribution to a layer's cache.
///
/// `kv_latent` is the per-token MLA KV latent (`HEAD_DIM = 512` lanes).
/// `compress_score` is the per-dim softmax-score for the compressor
/// (`HEAD_DIM` lanes); `None` skips the compression update — caller
/// should pass `None` for the first 2 dense layers OR pass real scores
/// for compressed layers.
/// `indexer_kv` and `indexer_score` populate the indexer pool, both
/// length `N_INDEXER_HEAD_DIM = 128`. Pass `None` to skip (also fine
/// on non-indexer layers).
#[derive(Debug, Clone, Copy)]
pub struct LayerAppend<'a> {
    /// Required: `HEAD_DIM` lanes, the per-token KV latent for the SWA ring.
    pub kv_latent: &'a [f32],
    /// Optional: per-dim score for compressed-pool aggregation.
    pub compress_score: Option<&'a [f32]>,
    /// Optional: per-token indexer KV row (`N_INDEXER_HEAD_DIM` lanes).
    pub indexer_kv: Option<&'a [f32]>,
    /// Optional: per-dim score for the indexer compressed pool.
    pub indexer_score: Option<&'a [f32]>,
}

/// Top-level cache: 43 layer caches, current cursor, layer compress ratios.
#[derive(Debug)]
pub struct ThreeTierKvCache {
    /// One cache per transformer block.
    pub layers: Vec<LayerCache>,
    /// Total tokens appended (cumulative across layers — append is
    /// per-layer but we track logical position centrally for
    /// finish_prefill alignment).
    current_pos: usize,
    /// Maximum supported context length this cache was sized for.
    ctx_size: usize,
}

impl ThreeTierKvCache {
    /// Allocate a cache sized for `ctx_size` tokens.
    ///
    /// # Panics
    /// Panics if `ctx_size == 0` or `ctx_size > DSV4_KVC_MAX_CTX_SIZE`
    /// (see [`LayerCache::new`]).
    #[must_use]
    pub fn new(ctx_size: usize) -> Self {
        let layers = (0..DSV4_N_LAYER).map(|il| LayerCache::new(il, ctx_size)).collect();
        Self {
            layers,
            current_pos: 0,
            ctx_size,
        }
    }

    /// Current logical position (cumulative tokens seen).
    #[must_use]
    pub fn current_pos(&self) -> usize {
        self.current_pos
    }

    /// Maximum supported context length.
    #[must_use]
    pub fn ctx_size(&self) -> usize {
        self.ctx_size
    }

    /// Append one token's full contribution to layer `il`.
    ///
    /// Writes the kv_latent to the SWA ring unconditionally. If the
    /// layer is compressed, accumulates `compress_score` into the
    /// compressed pool. If the layer has an indexer, accumulates
    /// `indexer_kv` + `indexer_score` into the indexer pool.
    ///
    /// **Does not** advance `current_pos`; call [`Self::advance_pos`]
    /// after appending to all layers for the same token. This split
    /// lets prefill batch multiple layers in parallel without needing
    /// to interleave position bumps.
    ///
    /// # Errors
    /// - [`Error::InvalidLayer`] if `il >= DSV4_N_LAYER`.
    /// - [`Error::ShapeMismatch`] if any input length disagrees with
    ///   the corresponding tier's `head_dim`.
    pub fn append_layer(&mut self, il: usize, input: LayerAppend<'_>) -> Result<(), Error> {
        if il >= self.layers.len() {
            return Err(Error::InvalidLayer {
                idx: il,
                max: self.layers.len(),
            });
        }
        let layer = &mut self.layers[il];

        // 1. Always write to the SWA ring.
        layer.swa.append(input.kv_latent)?;

        // 2. Compressed pool (if present).
        if let Some(pool) = layer.compressed.as_mut() {
            let score = input.compress_score.ok_or(Error::ShapeMismatch {
                what: "ThreeTierKvCache::append_layer: missing compress_score for compressed layer",
                expected: DSV4_HEAD_DIM,
                actual: 0,
            })?;
            pool.accumulate(input.kv_latent, score)?;
        }

        // 3. Indexer pool (if present).
        if let Some(idx_pool) = layer.indexer.as_mut() {
            let idx_kv = input.indexer_kv.ok_or(Error::ShapeMismatch {
                what: "ThreeTierKvCache::append_layer: missing indexer_kv for ratio-4 layer",
                expected: DSV4_N_INDEXER_HEAD_DIM,
                actual: 0,
            })?;
            let idx_score = input.indexer_score.ok_or(Error::ShapeMismatch {
                what: "ThreeTierKvCache::append_layer: missing indexer_score for ratio-4 layer",
                expected: DSV4_N_INDEXER_HEAD_DIM,
                actual: 0,
            })?;
            idx_pool.accumulate(idx_kv, idx_score)?;
        }

        Ok(())
    }

    /// Advance the cache's logical token cursor. Call once per token
    /// after appending all layers (or once per prefill chunk).
    pub fn advance_pos(&mut self, n: usize) {
        self.current_pos += n;
    }

    /// Normalize per-layer compressor state after a prefill so decode
    /// resumes from the same partial-window state a streaming run
    /// would have produced (`ds4.c:6353-6371`).
    ///
    /// `n_tokens` is the total prefill length (== `current_pos` after
    /// the prefill batch).
    pub fn finish_prefill(&mut self, n_tokens: usize) {
        for layer in &mut self.layers {
            if let Some(pool) = layer.compressed.as_mut() {
                pool.finish_prefill_state(n_tokens);
            }
            if let Some(idx) = layer.indexer.as_mut() {
                idx.finish_prefill_state(n_tokens);
            }
        }
    }

    /// Discard every layer's state, keep capacities. Use this between
    /// independent sessions.
    pub fn clear(&mut self) {
        for layer in &mut self.layers {
            layer.clear();
        }
        self.current_pos = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsv4::shape::DSV4_HEAD_DIM;

    fn const_kv(v: f32) -> Vec<f32> {
        vec![v; DSV4_HEAD_DIM]
    }
    fn const_indexer_kv(v: f32) -> Vec<f32> {
        vec![v; DSV4_N_INDEXER_HEAD_DIM]
    }

    #[test]
    fn new_cache_has_n_layer_blocks() {
        let cache = ThreeTierKvCache::new(1024);
        assert_eq!(cache.layers.len(), DSV4_N_LAYER);
        assert_eq!(cache.current_pos(), 0);
        assert_eq!(cache.ctx_size(), 1024);
    }

    #[test]
    fn dense_layers_have_no_compressed_pool_or_indexer() {
        let cache = ThreeTierKvCache::new(1024);
        assert!(cache.layers[0].compressed.is_none());
        assert!(cache.layers[1].compressed.is_none());
        assert!(cache.layers[0].indexer.is_none());
        assert!(cache.layers[1].indexer.is_none());
        assert_eq!(cache.layers[0].compress_ratio, 0);
    }

    #[test]
    fn ratio_4_layers_have_indexer() {
        let cache = ThreeTierKvCache::new(1024);
        // layer 2 (first ratio-4 layer): compressed + indexer.
        assert!(cache.layers[2].compressed.is_some());
        assert!(cache.layers[2].indexer.is_some());
        assert_eq!(cache.layers[2].compress_ratio, 4);
    }

    #[test]
    fn ratio_128_layers_have_compressed_but_no_indexer() {
        let cache = ThreeTierKvCache::new(1024);
        assert!(cache.layers[3].compressed.is_some()); // odd ≥ 2 → ratio 128
        assert!(cache.layers[3].indexer.is_none());
        assert_eq!(cache.layers[3].compress_ratio, 128);
    }

    #[test]
    fn append_layer_to_dense_writes_swa_only() {
        let mut cache = ThreeTierKvCache::new(64);
        let kv = const_kv(7.0);
        cache
            .append_layer(
                0,
                LayerAppend {
                    kv_latent: &kv,
                    compress_score: None,
                    indexer_kv: None,
                    indexer_score: None,
                },
            )
            .unwrap();
        assert_eq!(cache.layers[0].swa.len(), 1);
    }

    #[test]
    fn append_layer_to_ratio4_needs_indexer_inputs() {
        let mut cache = ThreeTierKvCache::new(64);
        let kv = const_kv(0.5);
        let comp_score = const_kv(0.0);
        // Missing indexer_kv/score should error.
        let err = cache
            .append_layer(
                2,
                LayerAppend {
                    kv_latent: &kv,
                    compress_score: Some(&comp_score),
                    indexer_kv: None,
                    indexer_score: None,
                },
            )
            .unwrap_err();
        assert!(matches!(err, Error::ShapeMismatch { .. }));
    }

    #[test]
    fn append_layer_full_ratio4_succeeds() {
        let mut cache = ThreeTierKvCache::new(64);
        let kv = const_kv(0.5);
        let comp_score = const_kv(0.0);
        let idx_kv = const_indexer_kv(0.3);
        let idx_score = const_indexer_kv(0.0);
        cache
            .append_layer(
                2,
                LayerAppend {
                    kv_latent: &kv,
                    compress_score: Some(&comp_score),
                    indexer_kv: Some(&idx_kv),
                    indexer_score: Some(&idx_score),
                },
            )
            .unwrap();
        assert_eq!(cache.layers[2].swa.len(), 1);
        // First token → no emission yet (need 4 to fire ratio-4 boundary).
        assert_eq!(cache.layers[2].compressed.as_ref().unwrap().len(), 0);
        assert_eq!(cache.layers[2].indexer.as_ref().unwrap().len(), 0);
    }

    #[test]
    fn finish_prefill_normalizes_compressor_state() {
        let mut cache = ThreeTierKvCache::new(64);
        let kv = const_kv(0.5);
        let cs = const_kv(0.1);
        let ikv = const_indexer_kv(0.2);
        let is = const_indexer_kv(0.1);
        // Push 7 tokens through layer 2 → 1 compressed emission, 3 in-flight state slots.
        for _ in 0..7 {
            cache
                .append_layer(
                    2,
                    LayerAppend {
                        kv_latent: &kv,
                        compress_score: Some(&cs),
                        indexer_kv: Some(&ikv),
                        indexer_score: Some(&is),
                    },
                )
                .unwrap();
        }
        let comp_ref = cache.layers[2].compressed.as_ref().unwrap();
        assert_eq!(comp_ref.len(), 1);
        assert_eq!(comp_ref.state_count(), 3);
        cache.finish_prefill(7);
        // After finish_prefill(7), state count should still be 7 % 4 = 3.
        let comp_after = cache.layers[2].compressed.as_ref().unwrap();
        assert_eq!(comp_after.state_count(), 3);
    }

    #[test]
    fn invalid_layer_idx_errors() {
        let mut cache = ThreeTierKvCache::new(64);
        let kv = const_kv(0.0);
        let err = cache
            .append_layer(
                99,
                LayerAppend {
                    kv_latent: &kv,
                    compress_score: None,
                    indexer_kv: None,
                    indexer_score: None,
                },
            )
            .unwrap_err();
        assert!(matches!(err, Error::InvalidLayer { .. }));
    }

    #[test]
    fn clear_resets_every_layer() {
        let mut cache = ThreeTierKvCache::new(64);
        let kv = const_kv(1.0);
        cache
            .append_layer(
                0,
                LayerAppend {
                    kv_latent: &kv,
                    compress_score: None,
                    indexer_kv: None,
                    indexer_score: None,
                },
            )
            .unwrap();
        cache.advance_pos(1);
        assert_eq!(cache.current_pos(), 1);
        cache.clear();
        assert_eq!(cache.current_pos(), 0);
        assert_eq!(cache.layers[0].swa.len(), 0);
    }

    #[test]
    #[should_panic(expected = "ctx_size must be > 0")]
    fn ctx_size_zero_panics() {
        let _ = ThreeTierKvCache::new(0);
    }

    #[test]
    #[should_panic(expected = "exceeds DSV4_KVC_MAX_CTX_SIZE")]
    fn ctx_size_above_cap_panics() {
        let _ = ThreeTierKvCache::new(DSV4_KVC_MAX_CTX_SIZE + 1);
    }
}
