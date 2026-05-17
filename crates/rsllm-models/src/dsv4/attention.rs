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

use rsllm_kvcache::dsv4::shape::{
    DSV4_HEAD_DIM as KV_HEAD_DIM, DSV4_N_INDEXER_HEAD_DIM, DSV4_N_LAYER,
};
use rsllm_kvcache::dsv4::three_tier::{LayerAppend, ThreeTierKvCache};

use crate::Error;
use crate::dsv4::shape::{DSV4_HEAD_DIM, DSV4_N_HEAD};

const _: () = assert!(KV_HEAD_DIM == DSV4_HEAD_DIM);

/// Stateful adapter that satisfies F005's [`crate::AttentionFn`] surface
/// by routing reads/writes through a [`ThreeTierKvCache`].
///
/// One adapter is created per forward pass and dropped at the end. The
/// caller is responsible for [`ThreeTierKvCache::advance_pos`] /
/// [`ThreeTierKvCache::finish_prefill`] after the pass.
pub struct ThreeTierAttention<'cache> {
    /// Borrowed reference to the three-tier cache. The adapter holds
    /// a `&mut`, so it cannot outlive the cache and only one closure
    /// can write to the cache at a time.
    cache: &'cache mut ThreeTierKvCache,
}

impl<'cache> ThreeTierAttention<'cache> {
    /// Construct an adapter around the given cache. The cache is
    /// borrowed mutably for the lifetime of the adapter.
    #[must_use]
    pub fn new(cache: &'cache mut ThreeTierKvCache) -> Self {
        Self { cache }
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

        // Placeholder per-dim scores (uniform softmax). Real
        // `attn_compressor` weights land in F008.
        let zero_kv_score = vec![0.0_f32; head_dim];
        let zero_idx_kv = vec![0.0_f32; DSV4_N_INDEXER_HEAD_DIM];
        let zero_idx_score = vec![0.0_f32; DSV4_N_INDEXER_HEAD_DIM];

        let scale = 1.0_f32 / (head_dim as f32).sqrt();

        for t in 0..n_tok {
            let kv_row = &kv[t * head_dim..(t + 1) * head_dim];

            // 1. Append this token's KV to the appropriate tiers.
            let append = LayerAppend {
                kv_latent: kv_row,
                compress_score: if compress_ratio > 0 {
                    Some(&zero_kv_score)
                } else {
                    None
                },
                indexer_kv: if has_indexer { Some(&zero_idx_kv) } else { None },
                indexer_score: if has_indexer {
                    Some(&zero_idx_score)
                } else {
                    None
                },
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
}
