//! Concrete [`Engine`] / [`Session`] implementation for DeepSeek V4 Flash.
//!
//! Owns a loaded [`DeepSeekV4Flash`] borrowing the GGUF mmap and
//! exposes a CPU-tier prefill / decode_one cycle that wires together:
//!
//! - F005 [`forward_block`] (per-layer transformer body)
//! - F006 [`ThreeTierKvCache`] (SWA ring + compressed pool + indexer)
//! - F008.B [`LayerLoRAs`] (real attn_compressor / attn_indexer scoring)
//! - F007 [`Sampler`] (multinomial draw)
//!
//! ## Scope of v0.1.0 (F008.C.3.d)
//!
//! The structural plumbing is complete: prefill walks all 43 layers,
//! advances the KV cache, and produces logits via [`lm_head_logits`];
//! decode_one feeds one token at a time and samples via [`Sampler`].
//! Numerical correctness against ds4 is gated on F008.C.3.e
//! (`tests/dsv4-vectors/` replay), which will surface any remaining
//! TODO(ds4) discrepancies (HC scale metadata, attn_compressor tensor
//! name, final-output HC merge convention).

use rsllm_backend_cpu::SimdTier;
use rsllm_core::{
    DecodeStep, Engine, EngineError, Sampler, SamplingParams, Session,
};
use rsllm_gguf::dequant_to_f32;
use rsllm_kvcache::dsv4::three_tier::ThreeTierKvCache;

use crate::deepseek_v4_flash::{
    DeepSeekV4Flash, ForwardScratch, forward_block, lm_head_logits,
};
use crate::dsv4::attention::{LayerLoRAs, ThreeTierAttention};
use crate::dsv4::shape::{DSV4_N_EMBD, DSV4_N_LAYER, DSV4_N_VOCAB};
use crate::dsv4::weight::WeightBlob;

/// Concrete engine: owns the loaded model (which borrows the GGUF
/// mmap).
///
/// `'gguf` is the lifetime of the underlying [`rsllm_gguf::GgufFile`].
/// The engine is created once per process and dropped at shutdown;
/// sessions borrow the engine via `&'engine`.
///
/// Note: the per-layer [`LayerLoRAs`] slice is rebuilt on every
/// forward pass rather than cached on the engine. Each `LayerLoRAs`
/// is a Copy struct of `Option<&...>` references; collecting 43 of
/// them costs one small heap alloc, far below the cost of the
/// forward pass itself, and it sidesteps the self-referential-struct
/// problem (the slice would otherwise need to borrow from the same
/// struct that owns `model`).
pub struct DsV4FlashEngine<'gguf> {
    model: DeepSeekV4Flash<'gguf>,
    tier: SimdTier,
}

impl<'gguf> DsV4FlashEngine<'gguf> {
    /// Construct an engine from an already-loaded model.
    ///
    /// Use [`crate::dsv4::loader::load_dsv4_flash`] to build the
    /// model, then hand it here.
    #[must_use]
    pub fn new(model: DeepSeekV4Flash<'gguf>) -> Self {
        Self {
            model,
            tier: SimdTier::Scalar,
        }
    }

    /// Pick the SIMD tier for the forward pass. Defaults to
    /// [`SimdTier::Scalar`]. Higher tiers (`NeonF16`, `AvxBf16`,
    /// `Avx2I8`) are selected at runtime by `rsllm-backend-cpu` based
    /// on detected CPU features.
    pub fn with_tier(mut self, tier: SimdTier) -> Self {
        self.tier = tier;
        self
    }

    /// Borrow the loaded model (for `rsllm inspect` and similar
    /// read-only introspection).
    #[must_use]
    pub fn model(&self) -> &DeepSeekV4Flash<'gguf> {
        &self.model
    }

    /// Build a `Vec<LayerLoRAs>` from the model's per-block weight
    /// references. Called once per forward pass; the slice lives only
    /// for the duration of the call.
    fn collect_layer_loras<'a>(&'a self) -> Vec<LayerLoRAs<'a>> {
        self.model
            .blocks
            .iter()
            .map(|block| LayerLoRAs {
                compressor: block.compressor.as_ref(),
                indexer_write: block.indexer_write.as_ref(),
                indexer_read: block.indexer_read.as_ref(),
            })
            .collect()
    }
}

impl<'gguf> Engine for DsV4FlashEngine<'gguf> {
    type Session<'engine>
        = DsV4FlashSession<'engine, 'gguf>
    where
        Self: 'engine;

    fn architecture(&self) -> &'static str {
        "deepseek-v4-flash"
    }

    fn vocab_size(&self) -> usize {
        DSV4_N_VOCAB
    }

    fn max_ctx_size(&self) -> usize {
        rsllm_kvcache::dsv4::three_tier::DSV4_KVC_MAX_CTX_SIZE
    }

    fn start_session(
        &self,
        ctx_size: usize,
        params: SamplingParams,
    ) -> Result<Self::Session<'_>, EngineError> {
        if ctx_size == 0 {
            return Err(EngineError::ShapeMismatch {
                what: "session.ctx_size".into(),
                expected: "> 0".into(),
                actual: "0".into(),
            });
        }
        if ctx_size > self.max_ctx_size() {
            return Err(EngineError::ShapeMismatch {
                what: "session.ctx_size".into(),
                expected: format!("<= {}", self.max_ctx_size()),
                actual: format!("{ctx_size}"),
            });
        }
        let cache = ThreeTierKvCache::new(ctx_size);
        tracing::info!(
            target: "rsllm_models::engine_impl",
            ctx_size,
            tier = ?self.tier,
            "session started",
        );
        Ok(DsV4FlashSession {
            engine: self,
            cache,
            sampler: Sampler::new(params),
            scratch: ForwardScratch::new(1),
            current_n_tok: 1,
            position: 0,
            capacity: ctx_size,
            logits_buf: vec![0.0_f32; DSV4_N_VOCAB],
        })
    }
}

/// Concrete session. Holds the KV cache, sampler, scratch buffers,
/// and position cursor. Created via [`DsV4FlashEngine::start_session`].
pub struct DsV4FlashSession<'engine, 'gguf: 'engine> {
    engine: &'engine DsV4FlashEngine<'gguf>,
    cache: ThreeTierKvCache,
    sampler: Sampler,
    scratch: ForwardScratch,
    /// Currently-allocated scratch capacity (in tokens). When the
    /// next prefill / decode call has a different n_tok, resize.
    current_n_tok: usize,
    position: usize,
    capacity: usize,
    /// Reusable logits buffer so a high-rate decode loop doesn't
    /// allocate per token.
    logits_buf: Vec<f32>,
}

impl<'engine, 'gguf: 'engine> DsV4FlashSession<'engine, 'gguf> {
    /// Resize scratch buffers in-place for the next forward pass.
    fn ensure_scratch_for(&mut self, n_tok: usize) {
        if self.current_n_tok != n_tok {
            self.scratch.resize(n_tok);
            self.current_n_tok = n_tok;
        }
    }

    /// Forward pass over `n_tok` tokens already embedded into
    /// `scratch.streams`. Runs `forward_block` × `DSV4_N_LAYER`,
    /// applies the F006 ThreeTierAttention closure (with real LoRA
    /// scoring when weights are present), and writes the final
    /// hidden state for the **last** token into `out_last_hidden`
    /// (length `[DSV4_N_EMBD]`).
    fn forward_pass(
        &mut self,
        n_tok: usize,
        position_offset: usize,
        out_last_hidden: &mut [f32],
    ) -> Result<(), EngineError> {
        let tier = self.engine.tier;
        {
            // Bind attention adapter to the cache + per-layer LoRAs.
            // It lives for the whole layer loop so the scratch buffers
            // inside it are reused across all 43 blocks. The LoRA
            // slice is rebuilt per pass; see [`DsV4FlashEngine`].
            let loras = self.engine.collect_layer_loras();
            let mut attn =
                ThreeTierAttention::with_loras(&mut self.cache, &loras).with_tier(tier);
            for il in 0..DSV4_N_LAYER {
                let block = &self.engine.model.blocks[il];
                let pos_off = position_offset as u32;
                let mut closure =
                    |q: &[f32],
                     kv: &[f32],
                     x: &[f32],
                     layer_idx: usize,
                     out: &mut [f32]|
                     -> Result<(), crate::Error> {
                        attn.run_layer(q, kv, x, layer_idx, out)
                    };
                forward_block(
                    block,
                    il,
                    &[], // token_ids unused by forward_block's MLA path today
                    &mut self.scratch,
                    n_tok,
                    |t| pos_off + t as u32,
                    &mut closure,
                    tier,
                )
                .map_err(map_model_err)?;
            }
        }
        // Advance the cache's logical position cursor once per pass.
        self.cache.advance_pos(n_tok);

        // Final HC merge: reduce the last token's `[N_HC × N_EMBD]`
        // residual streams to a single `[N_EMBD]` vector via the
        // learned output_hc gate (ds4.c:7916-7944). Mirrors
        // `output_hc_head_one` exactly; replaces the F008.C.3.d
        // stream-0 placeholder that was algorithmically wrong.
        let last_tok = n_tok - 1;
        let hc_dim = crate::dsv4::hc::HC_DIM;
        let stream_off = last_tok * hc_dim;
        let last_inp_hc = &self.scratch.streams[stream_off..stream_off + hc_dim];
        let oh = &self.engine.model.output_hc;
        crate::dsv4::hc::output_hc_collapse(
            last_inp_hc,
            out_last_hidden,
            &oh.mix_fn,
            oh.scale,
            oh.base,
            tier,
        )
        .map_err(map_model_err)?;
        Ok(())
    }
}

impl Session for DsV4FlashSession<'_, '_> {
    fn position(&self) -> usize {
        self.position
    }

    fn capacity(&self) -> usize {
        self.capacity
    }

    fn prefill(&mut self, tokens: &[u32]) -> Result<Vec<f32>, EngineError> {
        if tokens.is_empty() {
            return Err(EngineError::ShapeMismatch {
                what: "prefill.tokens".into(),
                expected: "non-empty slice".into(),
                actual: "empty".into(),
            });
        }
        if self.position + tokens.len() > self.capacity {
            tracing::warn!(
                target: "rsllm_models::engine_impl",
                filled = self.position,
                needed = tokens.len(),
                capacity = self.capacity,
                "context full — prefill rejected",
            );
            return Err(EngineError::ContextFull {
                filled: self.position,
                capacity: self.capacity,
            });
        }
        let n_tok = tokens.len();
        let _span = tracing::debug_span!(
            target: "rsllm_models::engine_impl",
            "prefill",
            n_tok,
            position_before = self.position,
        )
        .entered();
        let started = std::time::Instant::now();
        self.ensure_scratch_for(n_tok);

        // Zero the whole `streams` region before embedding. `Vec::resize`
        // only zero-fills *newly grown* elements, so when a second prefill
        // has `n_tok <= current_n_tok` the resize is a no-op and HC
        // streams 1..3 still hold the previous pass's outputs. Those are
        // *outputs* of the hyper-connection ops, not inputs, so they
        // must be zero on entry to forward_block.
        let total = n_tok * rsllm_backend_cpu::ops::sinkhorn::N_HC * DSV4_N_EMBD;
        self.scratch.streams[..total].fill(0.0);

        // Embed every token into stream 0. Streams 1..3 stay zero.
        for (t, &id) in tokens.iter().enumerate() {
            let stream0_off = t * rsllm_backend_cpu::ops::sinkhorn::N_HC * DSV4_N_EMBD;
            embed_token(
                &self.engine.model.embed_tokens,
                id,
                &mut self.scratch.streams[stream0_off..stream0_off + DSV4_N_EMBD],
            )?;
        }

        let position_offset = self.position;
        let mut last_hidden = vec![0.0_f32; DSV4_N_EMBD];
        self.forward_pass(n_tok, position_offset, &mut last_hidden)?;
        self.position += n_tok;
        // Normalize per-layer compressor state so the next decode_one
        // resumes from the same partial-window state a streaming run
        // would produce (ds4.c:6353-6371 — see ThreeTierKvCache docs).
        self.cache.finish_prefill(self.position);

        let tier = self.engine.tier;
        lm_head_logits(
            &self.engine.model,
            &last_hidden,
            &mut self.logits_buf,
            tier,
        )
        .map_err(map_model_err)?;
        tracing::info!(
            target: "rsllm_models::engine_impl",
            n_tok,
            elapsed_ms = started.elapsed().as_millis() as u64,
            position_after = self.position,
            "prefill complete",
        );
        Ok(self.logits_buf.clone())
    }

    fn decode_one(&mut self, last_token: u32) -> Result<DecodeStep, EngineError> {
        if self.position >= self.capacity {
            tracing::warn!(
                target: "rsllm_models::engine_impl",
                filled = self.position,
                capacity = self.capacity,
                "context full — decode_one rejected",
            );
            return Err(EngineError::ContextFull {
                filled: self.position,
                capacity: self.capacity,
            });
        }
        let started = std::time::Instant::now();
        self.ensure_scratch_for(1);
        let stream0_len = rsllm_backend_cpu::ops::sinkhorn::N_HC * DSV4_N_EMBD;
        embed_token(
            &self.engine.model.embed_tokens,
            last_token,
            &mut self.scratch.streams[..DSV4_N_EMBD],
        )?;
        // Zero the rest of stream 0..N_HC × N_EMBD for safety
        // (resize keeps stale data past the embed lane).
        for v in self.scratch.streams[DSV4_N_EMBD..stream0_len].iter_mut() {
            *v = 0.0;
        }
        let position_offset = self.position;
        let mut last_hidden = vec![0.0_f32; DSV4_N_EMBD];
        self.forward_pass(1, position_offset, &mut last_hidden)?;
        self.position += 1;

        let tier = self.engine.tier;
        lm_head_logits(
            &self.engine.model,
            &last_hidden,
            &mut self.logits_buf,
            tier,
        )
        .map_err(map_model_err)?;
        let token_id = self.sampler.sample(&mut self.logits_buf);
        tracing::trace!(
            target: "rsllm_models::engine_impl",
            last_token,
            sampled = token_id,
            position = self.position,
            elapsed_us = started.elapsed().as_micros() as u64,
            "decode step",
        );
        Ok(DecodeStep {
            token_id,
            probs: self.logits_buf.clone(),
        })
    }

    fn reset(&mut self) {
        let prior_position = self.position;
        self.cache.clear();
        self.position = 0;
        // Force the next prefill / decode to resize scratch, which
        // zero-fills any newly grown elements. This is belt-and-
        // suspenders alongside the explicit `streams[..].fill(0.0)` in
        // `prefill` — if a caller skips prefill and goes straight to
        // decode after reset, the scratch is still hot from a prior
        // session.
        self.current_n_tok = 0;
        tracing::debug!(
            target: "rsllm_models::engine_impl",
            prior_position,
            "session reset",
        );
    }
}

/// Copy / dequant one embedding row (`token_id`) into `out`
/// (length `[DSV4_N_EMBD]`).
fn embed_token(
    embed: &WeightBlob<'_>,
    token_id: u32,
    out: &mut [f32],
) -> Result<(), EngineError> {
    if out.len() != DSV4_N_EMBD {
        return Err(EngineError::ShapeMismatch {
            what: "embed_token.out".into(),
            expected: format!("{DSV4_N_EMBD}"),
            actual: format!("{}", out.len()),
        });
    }
    let tid = token_id as usize;
    if tid >= DSV4_N_VOCAB {
        return Err(EngineError::ShapeMismatch {
            what: "embed_token.token_id".into(),
            expected: format!("< {DSV4_N_VOCAB}"),
            actual: format!("{tid}"),
        });
    }
    match embed {
        WeightBlob::F32(s) => {
            // checked_mul guards 32-bit targets (v0.1.0 ships 64-bit
            // only, but adding the check is free and protects fuzz /
            // future ports). On 64-bit the product fits trivially:
            // DSV4_N_VOCAB * DSV4_N_EMBD ≈ 9.3e8 << usize::MAX.
            let start = tid.checked_mul(DSV4_N_EMBD).ok_or_else(|| {
                EngineError::ShapeMismatch {
                    what: "embed_token.f32_start".into(),
                    expected: "tid * N_EMBD fits in usize".into(),
                    actual: format!("tid={tid} N_EMBD={DSV4_N_EMBD}"),
                }
            })?;
            let end = start
                .checked_add(DSV4_N_EMBD)
                .ok_or_else(|| EngineError::ShapeMismatch {
                    what: "embed_token.f32_end".into(),
                    expected: "start + N_EMBD fits in usize".into(),
                    actual: format!("start={start} N_EMBD={DSV4_N_EMBD}"),
                })?;
            // Use `.get(...)` rather than `&s[..]` so an undersized
            // F32 blob (e.g. constructed in a test or by a future
            // API that bypasses the loader's check_shape guard)
            // returns Err instead of panicking.
            let row = s.get(start..end).ok_or_else(|| EngineError::ShapeMismatch {
                what: "embed_token.f32_row".into(),
                expected: format!("F32 storage >= {end} elements"),
                actual: format!("{}", s.len()),
            })?;
            out.copy_from_slice(row);
            Ok(())
        }
        WeightBlob::Quant { data, dtype } => {
            // Per-row dequant. ds4 typically ships embeddings as
            // F16 or Q4_K; either case has a fixed row-byte stride.
            //
            // `byte_size` returns u64 to express on-disk sizes that
            // could exceed 32-bit on 32-bit targets; we explicitly
            // catch that overflow via TryFrom rather than `as usize`
            // (which silently truncates).
            let row_bytes_u64 =
                dtype.byte_size(DSV4_N_EMBD as u64).ok_or_else(|| {
                    EngineError::ShapeMismatch {
                        what: "embed_token.row_bytes".into(),
                        expected: "byte_size computable".into(),
                        actual: format!("{dtype:?}"),
                    }
                })?;
            let row_bytes =
                usize::try_from(row_bytes_u64).map_err(|_| EngineError::ShapeMismatch {
                    what: "embed_token.row_bytes_fit".into(),
                    expected: "row_bytes fits in usize".into(),
                    actual: format!("{row_bytes_u64}"),
                })?;
            let row_start = tid.checked_mul(row_bytes).ok_or_else(|| {
                EngineError::ShapeMismatch {
                    what: "embed_token.quant_row_start".into(),
                    expected: "tid * row_bytes fits in usize".into(),
                    actual: format!("tid={tid} row_bytes={row_bytes}"),
                }
            })?;
            let row_end =
                row_start
                    .checked_add(row_bytes)
                    .ok_or_else(|| EngineError::ShapeMismatch {
                        what: "embed_token.quant_row_end".into(),
                        expected: "row_start + row_bytes fits in usize".into(),
                        actual: format!("row_start={row_start} row_bytes={row_bytes}"),
                    })?;
            let src = data.get(row_start..row_end).ok_or_else(|| {
                EngineError::ShapeMismatch {
                    what: "embed_token.row_slice".into(),
                    expected: format!("bytes {row_start}..{row_end}"),
                    actual: format!("data.len() = {}", data.len()),
                }
            })?;
            dequant_to_f32(*dtype, src, out).map_err(|e| EngineError::ShapeMismatch {
                what: "embed_token.dequant".into(),
                expected: "ok".into(),
                actual: format!("{e}"),
            })
        }
    }
}

/// Map a `crate::Error` into an `EngineError`. Keeps the call sites
/// in `Session` clean.
fn map_model_err(e: crate::Error) -> EngineError {
    match e {
        crate::Error::MissingMetadata(k) => EngineError::Missing(k.to_string()),
        crate::Error::MissingTensor(k) => EngineError::Missing(k),
        crate::Error::ShapeMismatch { key, expected, actual } => EngineError::ShapeMismatch {
            what: key.to_string(),
            expected,
            actual,
        },
        // NOTE: a GGUF parse error is semantically a parse failure, not
        // a shape mismatch. EngineError has no generic Parse / Io
        // variant covering it today; using ShapeMismatch here keeps the
        // mapping total at the cost of a slightly misleading CLI error
        // message ("shape mismatch: gguf expected valid, got ..."). A
        // follow-up that adds EngineError::Parse can clean this up.
        crate::Error::Gguf(e) => EngineError::ShapeMismatch {
            what: "gguf".into(),
            expected: "valid".into(),
            actual: format!("{e}"),
        },
        crate::Error::KvCache(e) => EngineError::KvCache(format!("{e}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsv4::weight::WeightBlob;

    #[test]
    fn embed_token_rejects_wrong_out_length() {
        let storage = vec![0.0_f32; DSV4_N_VOCAB * DSV4_N_EMBD];
        let embed = WeightBlob::F32(&storage);
        let mut bad = vec![0.0_f32; DSV4_N_EMBD - 1];
        let err = embed_token(&embed, 0, &mut bad).unwrap_err();
        assert!(matches!(err, EngineError::ShapeMismatch { .. }));
    }

    #[test]
    fn embed_token_rejects_out_of_range_id() {
        let storage = vec![0.0_f32; DSV4_N_VOCAB * DSV4_N_EMBD];
        let embed = WeightBlob::F32(&storage);
        let mut out = vec![0.0_f32; DSV4_N_EMBD];
        let err =
            embed_token(&embed, DSV4_N_VOCAB as u32, &mut out).unwrap_err();
        assert!(matches!(err, EngineError::ShapeMismatch { .. }));
    }

    #[test]
    fn embed_token_f32_copies_correct_row() {
        let mut storage = vec![0.0_f32; DSV4_N_VOCAB * DSV4_N_EMBD];
        // Mark row 5 with sentinel value 42.0 at lane 0.
        storage[5 * DSV4_N_EMBD] = 42.0;
        let embed = WeightBlob::F32(&storage);
        let mut out = vec![0.0_f32; DSV4_N_EMBD];
        embed_token(&embed, 5, &mut out).unwrap();
        assert_eq!(out[0], 42.0);
    }

    #[test]
    fn map_model_err_carries_message() {
        let e = crate::Error::MissingTensor("blk.0.attn_norm.weight".into());
        let mapped = map_model_err(e);
        match mapped {
            EngineError::Missing(s) => assert!(s.contains("blk.0")),
            other => panic!("expected Missing, got {other:?}"),
        }
    }

    #[test]
    fn embed_token_f32_rejects_undersized_storage() {
        // Storage holds *fewer* than DSV4_N_VOCAB rows; the token-id
        // bounds check passes (we ask for row 0), but the F32 slice
        // is too short. The fix in F008.C.3.d review guards this with
        // `.get(start..end)` instead of a panicking index.
        let storage = vec![0.0_f32; DSV4_N_EMBD / 2];
        let embed = WeightBlob::F32(&storage);
        let mut out = vec![0.0_f32; DSV4_N_EMBD];
        let err = embed_token(&embed, 0, &mut out).unwrap_err();
        match err {
            EngineError::ShapeMismatch { what, .. } => {
                assert!(
                    what.contains("f32_row"),
                    "expected 'f32_row' guard, got `what` = {what:?}"
                );
            }
            other => panic!("expected ShapeMismatch::f32_row, got {other:?}"),
        }
    }
}
