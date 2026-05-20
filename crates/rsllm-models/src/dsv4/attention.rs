//! Three-tier attention adapter — bridges F005's [`AttentionFn`] callback
//! with F006's [`rsllm_kvcache::dsv4::three_tier::ThreeTierKvCache`].
//!
//! For each token in the input batch and one layer index `il`:
//!
//! 1. Append the per-token KV latent to the cache. Compressed layers
//!    receive a placeholder per-dim score (zeros → uniform pooling);
//!    ratio-4 indexer inputs are likewise placeholdered. Real scoring
//!    weights are produced by `attn_compressor` / `attn_indexer` LoRAs,
//!    which are part of the F008 (numerical parity) extension.
//! 2. Compute MLA-absorbed attention against the cached SWA window:
//!    for each head `h ∈ [0, N_HEAD)`, softmax(Q_h · KV_k^T / √d) · KV_k.
//!    Compressed-pool and indexer-selected rows are *additional* keys
//!    that decode would attend to — those are wired here as a `TODO(F008)`
//!    extension; F006's contract is the cache plumbing.
//!
//! The attention sink (per-head virtual logit, `attn_sinks [N_HEAD]`)
//! is **not** applied here — it modifies the softmax denominator only,
//! and the absorbed-form weight stays separate from the cache itself.
//! That detail is consumed at the post-projection stage in F008.
//!
//! Ported by reference from `ds4.c:6310-6371` (MIT, The ds4.c authors).
//! Line numbers pinned to ds4 commit `ef0a490` (2026-05-17).

use rsllm_backend_cpu::SimdTier;
use rsllm_kvcache::dsv4::shape::{
    DSV4_HEAD_DIM as KV_HEAD_DIM, DSV4_N_INDEXER_HEAD_DIM, DSV4_N_LAYER,
};
use rsllm_kvcache::dsv4::three_tier::{LayerAppend, ThreeTierKvCache};

use crate::Error;
use crate::dsv4::compressor::{
    CompressorWeights, IndexerWeights, compressor_decode_one,
};
use crate::dsv4::shape::{DSV4_HEAD_DIM, DSV4_N_HEAD};

/// Maximum compressor `width = coff * head_dim` across all DS V4 Flash
/// layers. Ratio-4 layers use `coff = 2` → `width = 1024`; ratio-128
/// layers use `coff = 1` → `width = 512`. Pre-sizing the per-token
/// scratch to the max avoids per-layer re-allocation.
const COMPRESSOR_SCRATCH_MAX_WIDTH: usize = 2 * DSV4_HEAD_DIM;

const _: () = assert!(KV_HEAD_DIM == DSV4_HEAD_DIM);

/// Per-layer LoRA bundle the adapter borrows to produce real
/// compress / indexer scores from the residual stream `x`. A `None`
/// entry on any sub-field means "use placeholder zeros for that
/// signal" — useful for partial wiring and for the F006-era tests
/// that don't yet hand in real weights.
#[derive(Debug, Clone, Copy, Default)]
pub struct LayerLoRAs<'a> {
    pub compressor: Option<&'a CompressorWeights<'a>>,
    pub indexer: Option<&'a IndexerWeights<'a>>,
}

/// Stateful adapter that satisfies F005's [`crate::AttentionFn`] surface
/// by routing reads/writes through a [`ThreeTierKvCache`].
///
/// One adapter is created per forward pass and dropped at the end. The
/// caller is responsible for [`ThreeTierKvCache::advance_pos`] /
/// [`ThreeTierKvCache::finish_prefill`] after the pass.
///
/// LoRA weights are optional — without them the compressed-pool /
/// indexer-pool stores zero-score placeholders (the F006 behavior).
/// Supply weights via [`Self::with_loras`] to enable real numerical
/// scoring (F008.C.2). The adapter borrows the slice immutably; the
/// caller (typically the CLI's decode loop) owns the storage.
pub struct ThreeTierAttention<'cache, 'lora> {
    cache: &'cache mut ThreeTierKvCache,
    /// One entry per transformer block (length 0 = no LoRAs at all,
    /// matching `new()`). When non-empty, must have exactly
    /// `DSV4_N_LAYER` entries.
    loras: &'lora [LayerLoRAs<'lora>],
    /// SIMD tier for the LoRA matmul kernels. Defaults to Scalar
    /// — set via [`Self::with_tier`].
    tier: SimdTier,
    /// Reusable scratch buffers for `compressor_decode_one` (F011.C).
    /// Each holds one token's `width`-wide row; we re-use across the
    /// per-token loop and across layers. Sized to the max width
    /// (`COMPRESSOR_SCRATCH_MAX_WIDTH`); per-layer width is queried
    /// from the pool and the buffer is sliced down before each call.
    scratch_kv_cur: Vec<f32>,
    scratch_sc_cur: Vec<f32>,
    scratch_ape_col: Vec<f32>,
    /// Reusable scratch — `[n_tok × N_INDEXER_HEAD_DIM]` indexer KV rows.
    scratch_indexer_kv: Vec<f32>,
    /// Reusable scratch — `[n_tok × N_INDEXER_HEAD_DIM]` indexer scores.
    scratch_indexer_score: Vec<f32>,
}

/// Empty slice used by [`ThreeTierAttention::new`] so the `loras`
/// field can keep the same `&[LayerLoRAs]` type whether or not
/// weights are supplied.
const EMPTY_LORAS: &[LayerLoRAs<'static>] = &[];

impl<'cache, 'lora> ThreeTierAttention<'cache, 'lora> {
    /// Construct an adapter around the given cache. Without LoRA
    /// weights — falls back to F006's zero-placeholder behavior.
    ///
    /// `EMPTY_LORAS` is `&'static [...]`, which coerces to any
    /// `&'lora [...]` automatically (Rust's reborrow / variance
    /// rules); no explicit lifetime bound needed.
    #[must_use]
    pub fn new(cache: &'cache mut ThreeTierKvCache) -> Self {
        Self {
            cache,
            loras: EMPTY_LORAS,
            tier: SimdTier::Scalar,
            scratch_kv_cur: vec![0.0; COMPRESSOR_SCRATCH_MAX_WIDTH],
            scratch_sc_cur: vec![0.0; COMPRESSOR_SCRATCH_MAX_WIDTH],
            scratch_ape_col: vec![0.0; COMPRESSOR_SCRATCH_MAX_WIDTH],
            scratch_indexer_kv: Vec::new(),
            scratch_indexer_score: Vec::new(),
        }
    }

    /// Construct with per-layer LoRA weights. `loras.len()` must
    /// equal `DSV4_N_LAYER`; otherwise the first `run_layer` call
    /// will error out.
    #[must_use]
    pub fn with_loras(
        cache: &'cache mut ThreeTierKvCache,
        loras: &'lora [LayerLoRAs<'lora>],
    ) -> Self {
        Self {
            cache,
            loras,
            tier: SimdTier::Scalar,
            scratch_kv_cur: vec![0.0; COMPRESSOR_SCRATCH_MAX_WIDTH],
            scratch_sc_cur: vec![0.0; COMPRESSOR_SCRATCH_MAX_WIDTH],
            scratch_ape_col: vec![0.0; COMPRESSOR_SCRATCH_MAX_WIDTH],
            scratch_indexer_kv: Vec::new(),
            scratch_indexer_score: Vec::new(),
        }
    }

    /// Pick the SIMD tier used by the per-token LoRA projections.
    /// Defaults to [`SimdTier::Scalar`].
    pub fn with_tier(mut self, tier: SimdTier) -> Self {
        self.tier = tier;
        self
    }

    /// Surrender mutable access to the underlying cache (so the caller
    /// can advance positions / finish prefill after the forward pass).
    pub fn into_cache(self) -> &'cache mut ThreeTierKvCache {
        self.cache
    }

    /// Execute attention for one layer over a batch of `n_tok` tokens.
    ///
    /// Inputs (matches [`crate::AttentionFn`]):
    ///   - `q`: `[n_tok × DSV4_N_HEAD × DSV4_HEAD_DIM]` (RoPE'd query).
    ///   - `kv`: `[n_tok × DSV4_HEAD_DIM]` (1-head MLA KV latent).
    ///   - `x`: `[n_tok × DSV4_N_EMBD]` post-RMSNorm hidden state.
    ///     Currently ignored — F008.C.2 will use it to call the
    ///     per-layer `attn_compressor` / `attn_indexer_*` LoRAs and
    ///     replace the zero-placeholder compress/indexer scores. The
    ///     signature is locked in now so the AttentionFn ABI is
    ///     stable across F008.C.1 → F008.C.2.
    ///   - `layer_idx`: 0-based block index.
    ///   - `attn_out`: `[n_tok × DSV4_N_HEAD × DSV4_HEAD_DIM]` (write-only).
    ///
    /// The attention math used here is the MLA-absorbed form: K and V
    /// are both reconstructed from the same KV latent so the dot
    /// product `Q_h · KV_k^T` doubles as the score, and the value-side
    /// linear is `KV_k` itself (the absorbed projection moves into the
    /// downstream `attn_output_a/b` stage).
    ///
    /// # Errors
    /// - [`Error::ShapeMismatch`] on input length disagreements.
    /// - [`Error::KvCache`] if the cache rejects an append.
    pub fn run_layer(
        &mut self,
        q: &[f32],
        kv: &[f32],
        x: &[f32],
        layer_idx: usize,
        attn_out: &mut [f32],
    ) -> Result<(), Error> {
        if layer_idx >= DSV4_N_LAYER {
            return Err(Error::KvCache(rsllm_kvcache::Error::InvalidLayer {
                idx: layer_idx,
                max: DSV4_N_LAYER,
            }));
        }
        let head_dim = DSV4_HEAD_DIM;
        let n_head = DSV4_N_HEAD;
        if !kv.len().is_multiple_of(head_dim) {
            return Err(Error::ShapeMismatch {
                key: "ThreeTierAttention::run_layer.kv",
                expected: format!("multiple of HEAD_DIM = {head_dim}"),
                actual: format!("{}", kv.len()),
            });
        }
        let n_tok = kv.len() / head_dim;
        // Validate the residual stream's length now that we accept it
        // — F008.C.2 will read it during LoRA projection.
        let expected_x = n_tok
            .checked_mul(crate::dsv4::shape::DSV4_N_EMBD)
            .ok_or(Error::ShapeMismatch {
                key: "ThreeTierAttention::run_layer.x",
                expected: format!("n_tok × N_EMBD (overflow with n_tok={n_tok})"),
                actual: format!("{}", x.len()),
            })?;
        if x.len() != expected_x {
            return Err(Error::ShapeMismatch {
                key: "ThreeTierAttention::run_layer.x",
                expected: format!("{expected_x}"),
                actual: format!("{}", x.len()),
            });
        }
        let q_stride = n_head * head_dim;
        // Use checked_mul so a maliciously large `n_tok` (derived from
        // `kv.len() / head_dim`) cannot wrap and bypass the shape check.
        let expected_q = n_tok.checked_mul(q_stride).ok_or(Error::ShapeMismatch {
            key: "ThreeTierAttention::run_layer.q",
            expected: format!("n_tok * q_stride (overflow with n_tok={n_tok})"),
            actual: format!("{}", q.len()),
        })?;
        if q.len() != expected_q {
            return Err(Error::ShapeMismatch {
                key: "ThreeTierAttention::run_layer.q",
                expected: format!("{expected_q}"),
                actual: format!("{}", q.len()),
            });
        }
        if attn_out.len() != expected_q {
            return Err(Error::ShapeMismatch {
                key: "ThreeTierAttention::run_layer.attn_out",
                expected: format!("{expected_q}"),
                actual: format!("{}", attn_out.len()),
            });
        }

        let compress_ratio = self.cache.layers[layer_idx].compress_ratio;
        let has_indexer = self.cache.layers[layer_idx].indexer.is_some();

        // Locate this layer's LoRA bundle (if any). An empty `loras`
        // slice means "use placeholder zeros" — the F006 behavior.
        let layer_loras = if self.loras.is_empty() {
            LayerLoRAs::default()
        } else if self.loras.len() != DSV4_N_LAYER {
            return Err(Error::ShapeMismatch {
                key: "ThreeTierAttention.loras.len",
                expected: format!("{DSV4_N_LAYER}"),
                actual: format!("{}", self.loras.len()),
            });
        } else {
            self.loras[layer_idx]
        };

        // F011.C: the compressor pool is now updated INSIDE the
        // per-token loop below via `compressor_decode_one`, not via a
        // single batched matmul + per-token append. The new path
        // operates on the layer's compressor pool state directly,
        // applies APE bias + per-dim softmax pooling + RMSNorm + RoPE
        // on every emission, and skips the pool write entirely on
        // non-boundary tokens. When no compressor weights are supplied
        // (`layer_loras.compressor == None`), the pool stays empty —
        // matching the F006 zero-placeholder semantics.
        if has_indexer {
            self.scratch_indexer_kv
                .resize(n_tok * DSV4_N_INDEXER_HEAD_DIM, 0.0);
            self.scratch_indexer_score
                .resize(n_tok * DSV4_N_INDEXER_HEAD_DIM, 0.0);
            // TODO(F011): indexer pipeline — project_indexer_* intentionally absent.
            // F011 will wire the per-position APE bias, gate sigmoid, pool reduction,
            // and RMSNorm using `layer_loras.indexer`. Until then, zero-fill both
            // scratch buffers so the cache append is exercised but scores are inert.
            for v in self.scratch_indexer_kv.iter_mut() {
                *v = 0.0;
            }
            for v in self.scratch_indexer_score.iter_mut() {
                *v = 0.0;
            }
        }

        let scale = 1.0_f32 / (head_dim as f32).sqrt();

        // Absolute sequence position of token 0 in this batch.
        let batch_start_pos = self.cache.current_pos();

        for t in 0..n_tok {
            let kv_row = &kv[t * head_dim..(t + 1) * head_dim];

            // 1a. Compressor pool update for this token (F011.C).
            // Only fires when (a) the layer has a compressor pool
            // (compress_ratio > 0) AND (b) LoRA weights were supplied.
            // Without weights the pool stays empty, matching the F006
            // zero-placeholder semantics.
            if compress_ratio > 0
                && let Some(weights) = layer_loras.compressor
            {
                let x_t = &x[t * crate::dsv4::shape::DSV4_N_EMBD
                    ..(t + 1) * crate::dsv4::shape::DSV4_N_EMBD];
                let pool = self.cache.layers[layer_idx]
                    .compressed
                    .as_mut()
                    .expect("compress_ratio > 0 implies compressed pool present");
                let width = pool.width();
                debug_assert!(width <= COMPRESSOR_SCRATCH_MAX_WIDTH);
                let pos = (batch_start_pos + t) as u32;
                compressor_decode_one(
                    x_t,
                    weights,
                    pool,
                    pos,
                    layer_idx as u32,
                    &mut self.scratch_kv_cur[..width],
                    &mut self.scratch_sc_cur[..width],
                    &mut self.scratch_ape_col[..width],
                    self.tier,
                )?;
            }

            let idx_kv_t = if has_indexer {
                Some(
                    &self.scratch_indexer_kv
                        [t * DSV4_N_INDEXER_HEAD_DIM..(t + 1) * DSV4_N_INDEXER_HEAD_DIM],
                )
            } else {
                None
            };
            let idx_score_t = if has_indexer {
                Some(
                    &self.scratch_indexer_score
                        [t * DSV4_N_INDEXER_HEAD_DIM..(t + 1) * DSV4_N_INDEXER_HEAD_DIM],
                )
            } else {
                None
            };

            // 1b. Append this token's KV to the SWA ring + indexer pool.
            let append = LayerAppend {
                kv_latent: kv_row,
                indexer_kv: idx_kv_t,
                indexer_score: idx_score_t,
            };
            self.cache.append_layer(layer_idx, append)?;

            // 2. Attention over the SWA window. The cache stores rows
            //    in chronological order (oldest → newest), so iterating
            //    by `row(idx)` gives the natural attention key order.
            let layer = &self.cache.layers[layer_idx];
            let n_keys = layer.swa.len();
            if n_keys == 0 {
                // Defensive: append should have written one row already.
                for o in &mut attn_out[t * q_stride..(t + 1) * q_stride] {
                    *o = 0.0;
                }
                continue;
            }

            // Per-head softmax(Q_h · K_k^T) · K_k.
            // Logits scratch is local — `n_keys ≤ N_SWA = 128`, so we
            // can afford a small Vec without instrumenting global state.
            let mut logits = Vec::with_capacity(n_keys);
            for h in 0..n_head {
                let q_h = &q[t * q_stride + h * head_dim..t * q_stride + (h + 1) * head_dim];

                // Compute logits.
                logits.clear();
                let mut max_logit = f32::NEG_INFINITY;
                for k in 0..n_keys {
                    let k_row = layer.swa.row(k)?;
                    let mut dot = 0.0_f32;
                    for d in 0..head_dim {
                        dot += q_h[d] * k_row[d];
                    }
                    let l = dot * scale;
                    if l > max_logit {
                        max_logit = l;
                    }
                    logits.push(l);
                }

                // Weighted-sum target. Zero before either the NaN
                // fallback or the real softmax mass flows in.
                let out_h = &mut attn_out
                    [t * q_stride + h * head_dim..t * q_stride + (h + 1) * head_dim];
                for o in out_h.iter_mut() {
                    *o = 0.0;
                }

                // Guard against non-finite max_logit (all-NaN or all
                // negative-infinity inputs). Mirrors the equivalent
                // guard in `CompressedKvPool::per_dim_softmax_aggregate`
                // so a NaN in upstream weights does not silently
                // propagate to model logits.
                if !max_logit.is_finite() {
                    continue;
                }

                // Softmax (numerically stable).
                let mut denom = 0.0_f32;
                for l in logits.iter_mut() {
                    *l = (*l - max_logit).exp();
                    denom += *l;
                }
                let inv_denom = 1.0_f32 / denom;

                // Weighted sum into attn_out.
                for (k, &logit) in logits.iter().enumerate() {
                    let k_row = layer.swa.row(k)?;
                    let w = logit * inv_denom;
                    for (o, &kv_d) in out_h.iter_mut().zip(k_row.iter()) {
                        *o += w * kv_d;
                    }
                }
            }
            // TODO(F008): also attend to compressed-pool rows and (on
            // ratio-4 layers) indexer-selected rows. Placeholder scores
            // make those tiers numerically inert today, but the cache
            // is being populated so the wiring is exercised.
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_q(n_tok: usize, fill: f32) -> Vec<f32> {
        vec![fill; n_tok * DSV4_N_HEAD * DSV4_HEAD_DIM]
    }
    fn make_kv(n_tok: usize, fill: f32) -> Vec<f32> {
        vec![fill; n_tok * DSV4_HEAD_DIM]
    }
    fn make_x(n_tok: usize) -> Vec<f32> {
        vec![0.0_f32; n_tok * crate::dsv4::shape::DSV4_N_EMBD]
    }

    #[test]
    fn run_layer_writes_attn_out_for_dense_layer() {
        let mut cache = ThreeTierKvCache::new(32);
        let mut attn = ThreeTierAttention::new(&mut cache);
        let q = make_q(1, 0.0);
        let kv = make_kv(1, 1.0);
        let mut out = vec![0.0_f32; DSV4_N_HEAD * DSV4_HEAD_DIM];
        attn.run_layer(&q, &kv, &make_x(kv.len() / DSV4_HEAD_DIM), 0, &mut out).unwrap();
        // With Q=0 and 1 cached KV row of all-ones, softmax gives w=1.0
        // and weighted sum = the KV row itself (all ones).
        assert!(out.iter().all(|&v| (v - 1.0).abs() < 1e-6));
    }

    #[test]
    fn run_layer_attends_to_prior_token() {
        let mut cache = ThreeTierKvCache::new(32);
        let mut attn = ThreeTierAttention::new(&mut cache);
        // Two-token sequence on layer 0 (dense).
        let mut kv = vec![0.0_f32; 2 * DSV4_HEAD_DIM];
        for d in 0..DSV4_HEAD_DIM {
            kv[d] = 1.0;
            kv[DSV4_HEAD_DIM + d] = 2.0;
        }
        let q = make_q(2, 0.0); // uniform → softmax averages keys
        let mut out = vec![0.0_f32; 2 * DSV4_N_HEAD * DSV4_HEAD_DIM];
        attn.run_layer(&q, &kv, &make_x(kv.len() / DSV4_HEAD_DIM), 0, &mut out).unwrap();
        // Token 0 attends only to itself (value 1.0).
        for &v in out.iter().take(DSV4_HEAD_DIM) {
            assert!((v - 1.0).abs() < 1e-5);
        }
        // Token 1 attends uniformly to {1.0, 2.0} → mean 1.5.
        let stride = DSV4_N_HEAD * DSV4_HEAD_DIM;
        for (d, &v) in out[stride..stride + DSV4_HEAD_DIM].iter().enumerate() {
            assert!((v - 1.5).abs() < 1e-5, "token 1 head 0 dim {d} = {v}");
        }
    }

    #[test]
    fn run_layer_populates_swa_ring_for_compressed_layer() {
        let mut cache = ThreeTierKvCache::new(64);
        let mut attn = ThreeTierAttention::new(&mut cache);
        let q = make_q(1, 0.0);
        let kv = make_kv(1, 0.1);
        let mut out = vec![0.0_f32; DSV4_N_HEAD * DSV4_HEAD_DIM];
        // Layer 2 is ratio-4 with an indexer.
        attn.run_layer(&q, &kv, &make_x(1), 2, &mut out).unwrap();
        assert_eq!(cache.layers[2].swa.len(), 1);
        // First token of 4-cycle: no compressed-pool emission yet.
        assert_eq!(cache.layers[2].compressed.as_ref().unwrap().len(), 0);
        assert_eq!(cache.layers[2].indexer.as_ref().unwrap().len(), 0);
    }

    #[test]
    fn run_layer_zeros_output_on_nan_inputs() {
        // NaN-laced Q would normally produce NaN logits → NaN softmax →
        // silent NaN propagation. The guard should zero the output
        // instead so the upstream pipeline can detect bad inputs.
        let mut cache = ThreeTierKvCache::new(8);
        let mut attn = ThreeTierAttention::new(&mut cache);
        let q = vec![f32::NAN; DSV4_N_HEAD * DSV4_HEAD_DIM];
        let kv = make_kv(1, 1.0);
        let mut out = vec![123.0_f32; DSV4_N_HEAD * DSV4_HEAD_DIM];
        attn.run_layer(&q, &kv, &make_x(kv.len() / DSV4_HEAD_DIM), 0, &mut out).unwrap();
        assert!(out.iter().all(|v| *v == 0.0), "expected all zeros, got {:?}", &out[..4]);
    }

    #[test]
    fn run_layer_rejects_invalid_layer_idx() {
        let mut cache = ThreeTierKvCache::new(32);
        let mut attn = ThreeTierAttention::new(&mut cache);
        let q = make_q(1, 0.0);
        let kv = make_kv(1, 0.0);
        let mut out = vec![0.0_f32; DSV4_N_HEAD * DSV4_HEAD_DIM];
        let err = attn.run_layer(&q, &kv, &make_x(1), DSV4_N_LAYER, &mut out).unwrap_err();
        assert!(matches!(
            err,
            Error::KvCache(rsllm_kvcache::Error::InvalidLayer { .. })
        ));
    }

    #[test]
    fn run_layer_rejects_x_length_mismatch() {
        // F008.C.1 adds the residual-stream length check; a mismatched
        // x must fire before the LoRA projections in F008.C.2.
        let mut cache = ThreeTierKvCache::new(32);
        let mut attn = ThreeTierAttention::new(&mut cache);
        let q = make_q(1, 0.0);
        let kv = make_kv(1, 0.0);
        let bad_x = vec![0.0_f32; crate::dsv4::shape::DSV4_N_EMBD + 1];
        let mut out = vec![0.0_f32; DSV4_N_HEAD * DSV4_HEAD_DIM];
        let err = attn.run_layer(&q, &kv, &bad_x, 0, &mut out).unwrap_err();
        assert!(matches!(
            err,
            Error::ShapeMismatch { key, .. } if key == "ThreeTierAttention::run_layer.x"
        ));
    }

    #[test]
    fn run_layer_rejects_kv_length_mismatch() {
        let mut cache = ThreeTierKvCache::new(32);
        let mut attn = ThreeTierAttention::new(&mut cache);
        let q = make_q(1, 0.0);
        let bad_kv = vec![0.0_f32; DSV4_HEAD_DIM - 1];
        let mut out = vec![0.0_f32; DSV4_N_HEAD * DSV4_HEAD_DIM];
        let err = attn.run_layer(&q, &bad_kv, &make_x(0), 0, &mut out).unwrap_err();
        assert!(matches!(err, Error::ShapeMismatch { .. }));
    }

    #[test]
    fn run_layer_usable_as_attention_fn_closure() {
        // Demonstrate the adapter satisfies the F005 callback shape via
        // a thin `&mut |..| ..` wrapper.
        let mut cache = ThreeTierKvCache::new(16);
        let mut attn = ThreeTierAttention::new(&mut cache);
        let q = make_q(1, 0.0);
        let kv = make_kv(1, 1.0);
        let mut out = vec![0.0_f32; DSV4_N_HEAD * DSV4_HEAD_DIM];

        let mut closure = |q: &[f32],
                           kv: &[f32],
                           x: &[f32],
                           il: usize,
                           o: &mut [f32]|
         -> Result<(), Error> { attn.run_layer(q, kv, x, il, o) };
        let attn_fn: crate::AttentionFn<'_> = &mut closure;
        let x = make_x(1);
        attn_fn(&q, &kv, &x, 0, &mut out).unwrap();
        assert!(out.iter().all(|&v| (v - 1.0).abs() < 1e-6));
    }

    #[test]
    fn loras_path_writes_real_compressor_score_into_pool() {
        // Run 4 tokens through a ratio-4 layer (il=2) with a non-zero
        // compressor weight. The compressed pool should fire exactly
        // one emission. F011.C now routes every token through
        // `compressor_decode_one`, so the pool count is the structural
        // signal that the wiring is intact (the numeric content is
        // covered by the F011.B compressor unit tests).
        //
        // Compressor weight shapes for a ratio-4 layer:
        // - kv / gate: [comp_width × n_embd] with comp_width = 2*HEAD_DIM
        // - ape: [comp_width × ratio] = [comp_width × 4]
        // - norm: [HEAD_DIM]
        let n_embd = crate::dsv4::shape::DSV4_N_EMBD;
        let comp_width = 2 * DSV4_HEAD_DIM;
        let mut comp_kv = vec![0.0_f32; comp_width * n_embd];
        for o in 0..comp_width {
            comp_kv[o * n_embd + o] = 1.0;
        }
        let comp_gate = vec![0.0_f32; comp_width * n_embd];
        let comp_ape = vec![0.0_f32; comp_width * 4];
        let comp_norm = vec![1.0_f32; DSV4_HEAD_DIM];
        let compressor = CompressorWeights {
            kv: crate::dsv4::weight::WeightBlob::F32(&comp_kv),
            gate: crate::dsv4::weight::WeightBlob::F32(&comp_gate),
            ape: crate::dsv4::weight::WeightBlob::F32(&comp_ape),
            norm: &comp_norm,
        };

        // Indexer weights: all-zero bundle (path is exercised so the cache
        // append succeeds; real scoring is F011). Six tensors required by
        // IndexerWeights; shapes match the upstream constants.
        let index_width = 2 * DSV4_N_INDEXER_HEAD_DIM; // 256
        let idx_attn_q_b =
            vec![0.0_f32; crate::dsv4::shape::DSV4_N_LORA_Q * crate::dsv4::shape::DSV4_N_INDEXER_HEAD * DSV4_N_INDEXER_HEAD_DIM];
        let idx_proj = vec![0.0_f32; n_embd * crate::dsv4::shape::DSV4_N_INDEXER_HEAD];
        let idx_comp_ape = vec![0.0_f32; index_width * 4];
        let idx_comp_kv = vec![0.0_f32; n_embd * index_width];
        let idx_comp_gate = vec![0.0_f32; n_embd * index_width];
        let idx_comp_norm = vec![1.0_f32; DSV4_N_INDEXER_HEAD_DIM];
        let indexer_weights_bundle = IndexerWeights {
            attn_q_b: crate::dsv4::weight::WeightBlob::F32(&idx_attn_q_b),
            proj: crate::dsv4::weight::WeightBlob::F32(&idx_proj),
            comp_ape: crate::dsv4::weight::WeightBlob::F32(&idx_comp_ape),
            comp_kv: crate::dsv4::weight::WeightBlob::F32(&idx_comp_kv),
            comp_gate: crate::dsv4::weight::WeightBlob::F32(&idx_comp_gate),
            comp_norm: &idx_comp_norm,
        };

        let mut loras = vec![LayerLoRAs::default(); DSV4_N_LAYER];
        for (il, slot) in loras.iter_mut().enumerate() {
            let ratio = rsllm_kvcache::dsv4::shape::layer_compress_ratio(il);
            if ratio > 0 {
                slot.compressor = Some(&compressor);
            }
            if rsllm_kvcache::dsv4::shape::layer_has_indexer(il) {
                slot.indexer = Some(&indexer_weights_bundle);
            }
        }

        let mut cache = ThreeTierKvCache::new(64);
        {
            let mut attn = ThreeTierAttention::with_loras(&mut cache, &loras);
            // Push 4 tokens (= ratio-4 boundary) on layer 2 as a single
            // batched call so `pos = batch_start_pos + t` correctly
            // advances 0..4 within the run. (Per-call advance_pos is
            // the caller's job per the adapter docstring, but a single
            // batched call keeps this test focused on the wiring.)
            let q = vec![0.0_f32; 4 * DSV4_N_HEAD * DSV4_HEAD_DIM];
            let kv = vec![1.0_f32; 4 * DSV4_HEAD_DIM];
            // Token t gets value (t+1) at lane 0 so the score is non-trivial.
            let mut x = vec![0.0_f32; 4 * n_embd];
            for t in 0..4 {
                x[t * n_embd] = (t as f32) + 1.0;
            }
            let mut out = vec![0.0_f32; 4 * DSV4_N_HEAD * DSV4_HEAD_DIM];
            attn.run_layer(&q, &kv, &x, 2, &mut out).unwrap();
        }
        // After 4 tokens on a ratio-4 layer: exactly one compressed
        // pool emission.
        assert_eq!(cache.layers[2].compressed.as_ref().unwrap().len(), 1);
    }

    #[test]
    fn loras_with_wrong_length_errors() {
        let mut cache = ThreeTierKvCache::new(16);
        // Build a too-short loras slice (1 entry instead of N_LAYER).
        let loras = vec![LayerLoRAs::default(); 1];
        let mut attn = ThreeTierAttention::with_loras(&mut cache, &loras);
        let q = make_q(1, 0.0);
        let kv = make_kv(1, 1.0);
        let x = make_x(1);
        let mut out = vec![0.0_f32; DSV4_N_HEAD * DSV4_HEAD_DIM];
        // Dense layer 0 doesn't trigger compress/indexer paths and may
        // still succeed; force a compressed layer to surface the loras
        // length check.
        let err = attn.run_layer(&q, &kv, &x, 2, &mut out).unwrap_err();
        assert!(matches!(
            err,
            Error::ShapeMismatch { key, .. } if key == "ThreeTierAttention.loras.len"
        ));
    }

    #[test]
    fn loras_indexer_emits_one_pool_entry_per_four_tokens() {
        // Exercises the indexer path end-to-end with an IndexerWeights
        // bundle supplied via slot.indexer. The indexer algorithm itself
        // is deferred to F011.D, so the scratch buffers are zero-filled
        // regardless of weight content. The structural guarantee tested
        // here is: 4 tokens on a ratio-4 layer triggers exactly one
        // emission into the indexer compressed pool.
        //
        // TODO(F011.D): once the indexer compressor sub-pipeline is
        // wired, add a non-zero numeric assertion on the emitted pool
        // row to cover the wiring boundary between the layer-loop and
        // the kernel.
        let n_embd = crate::dsv4::shape::DSV4_N_EMBD;
        // Compressor: all-zero, inert in this test (still must have the
        // ratio-4 shapes so `compressor_decode_one` matmuls pass).
        let comp_width = 2 * DSV4_HEAD_DIM;
        let comp_kv = vec![0.0_f32; comp_width * n_embd];
        let comp_gate = vec![0.0_f32; comp_width * n_embd];
        let comp_ape = vec![0.0_f32; comp_width * 4];
        let comp_norm = vec![1.0_f32; DSV4_HEAD_DIM];
        let compressor = CompressorWeights {
            kv: crate::dsv4::weight::WeightBlob::F32(&comp_kv),
            gate: crate::dsv4::weight::WeightBlob::F32(&comp_gate),
            ape: crate::dsv4::weight::WeightBlob::F32(&comp_ape),
            norm: &comp_norm,
        };
        // Indexer weights: six-tensor bundle (shapes matching upstream).
        // All-zero is fine because F011 owns the algorithm; this test
        // only checks the cache-append plumbing.
        let index_width = 2 * DSV4_N_INDEXER_HEAD_DIM; // 256
        let idx_attn_q_b =
            vec![0.0_f32; crate::dsv4::shape::DSV4_N_LORA_Q * crate::dsv4::shape::DSV4_N_INDEXER_HEAD * DSV4_N_INDEXER_HEAD_DIM];
        let idx_proj = vec![0.0_f32; n_embd * crate::dsv4::shape::DSV4_N_INDEXER_HEAD];
        let idx_comp_ape = vec![0.0_f32; index_width * 4];
        let idx_comp_kv = vec![0.0_f32; n_embd * index_width];
        let idx_comp_gate = vec![0.0_f32; n_embd * index_width];
        let idx_comp_norm = vec![1.0_f32; DSV4_N_INDEXER_HEAD_DIM];
        let indexer_weights_bundle = IndexerWeights {
            attn_q_b: crate::dsv4::weight::WeightBlob::F32(&idx_attn_q_b),
            proj: crate::dsv4::weight::WeightBlob::F32(&idx_proj),
            comp_ape: crate::dsv4::weight::WeightBlob::F32(&idx_comp_ape),
            comp_kv: crate::dsv4::weight::WeightBlob::F32(&idx_comp_kv),
            comp_gate: crate::dsv4::weight::WeightBlob::F32(&idx_comp_gate),
            comp_norm: &idx_comp_norm,
        };

        let mut loras = vec![LayerLoRAs::default(); DSV4_N_LAYER];
        for (il, slot) in loras.iter_mut().enumerate() {
            let ratio = rsllm_kvcache::dsv4::shape::layer_compress_ratio(il);
            if ratio > 0 {
                slot.compressor = Some(&compressor);
            }
            if rsllm_kvcache::dsv4::shape::layer_has_indexer(il) {
                slot.indexer = Some(&indexer_weights_bundle);
            }
        }

        let mut cache = ThreeTierKvCache::new(64);
        {
            let mut attn = ThreeTierAttention::with_loras(&mut cache, &loras);
            // Single batched 4-token call (same reasoning as the
            // sibling `loras_path_writes_real_compressor_score_into_pool`
            // test — `pos` must advance 0..4 within the batch).
            let q = vec![0.0_f32; 4 * DSV4_N_HEAD * DSV4_HEAD_DIM];
            let kv = vec![1.0_f32; 4 * DSV4_HEAD_DIM];
            let x = vec![0.0_f32; 4 * n_embd];
            let mut out = vec![0.0_f32; 4 * DSV4_N_HEAD * DSV4_HEAD_DIM];
            attn.run_layer(&q, &kv, &x, 2, &mut out).unwrap();
        }
        // 4 tokens through a ratio-4 indexer → exactly one emission.
        // Numeric content is zero until F011.D wires the real algorithm.
        let indexer_pool = cache.layers[2].indexer.as_ref().unwrap();
        assert_eq!(indexer_pool.len(), 1);
    }

    #[test]
    fn with_loras_runs_dense_layer_correctly() {
        // Dense layer (il=0) has compress_ratio=0 and no indexer, so
        // neither LoRA path runs. with_loras must still produce the
        // F006-equivalent dense attention output.
        let loras = vec![LayerLoRAs::default(); DSV4_N_LAYER];
        let mut cache = ThreeTierKvCache::new(32);
        let mut attn = ThreeTierAttention::with_loras(&mut cache, &loras);
        let q = make_q(1, 0.0);
        let kv = make_kv(1, 1.0);
        let x = make_x(1);
        let mut out = vec![0.0_f32; DSV4_N_HEAD * DSV4_HEAD_DIM];
        attn.run_layer(&q, &kv, &x, 0, &mut out).unwrap();
        // Same expectation as run_layer_writes_attn_out_for_dense_layer:
        // 1 cached KV row of all-ones, Q=0 → softmax = 1.0 → out = KV.
        assert!(out.iter().all(|&v| (v - 1.0).abs() < 1e-6));
    }

    #[test]
    fn loras_empty_falls_back_to_placeholder() {
        // `new()` uses an empty loras slice; existing behavior must
        // remain identical (this duplicates the dense-layer test for
        // the new code path).
        let mut cache = ThreeTierKvCache::new(32);
        let mut attn = ThreeTierAttention::new(&mut cache);
        let q = make_q(1, 0.0);
        let kv = make_kv(1, 1.0);
        let x = make_x(1);
        let mut out = vec![0.0_f32; DSV4_N_HEAD * DSV4_HEAD_DIM];
        attn.run_layer(&q, &kv, &x, 2, &mut out).unwrap();
        // Compressed pool unchanged after 1 token (still under boundary).
        assert_eq!(cache.layers[2].compressed.as_ref().unwrap().len(), 0);
    }
}
