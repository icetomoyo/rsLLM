//! DeepSeek V4 Flash model assembly.
//!
//! Holds the per-layer weight structures and the forward dispatcher
//! that wires [`mla`], [`hc`], and [`moe`] into the layered residual
//! computation. The actual attention dot-product + softmax + KV cache
//! interaction is gated on FEATURE_006 (three-tier KV cache) and lives
//! behind the [`AttentionFn`] callback supplied by the caller —
//! v0.1.0's CLI binary will plug in the F006 cache code at that point.
//!
//! Layer body for one of the 43 transformer blocks:
//!
//! ```text
//! streams (4 × n_tok × N_EMBD)
//!   │
//!   ├─ hc_pre(attn) → merged (n_tok × N_EMBD)
//!   │   └─ rmsnorm(attn_norm) → normed
//!   │       └─ mla_projections → (q, kv)
//!   │           └─ attention(q, kv, kv_cache) → attn_out  (F006)
//!   │             └─ output projection (attn_o) → attn_proj
//!   │               └─ hc_post(attn) [in-place add into streams]
//!   │
//!   └─ hc_pre(ffn) → merged
//!       └─ rmsnorm(ffn_norm) → normed
//!           ├─ apply_shared_expert → shared_out
//!           └─ moe_hash_route OR moe_topk_route → moe_out
//!               └─ ffn_out = shared_out + moe_out
//!                 └─ hc_post(ffn) [in-place add into streams]
//! ```
//!
//! Ported by reference from ds4.c (MIT, The ds4.c authors):
//! - HC family: `ds4.c:4186-4310`
//! - MLA Q/KV LoRA: see `dsv4::mla` (anchored on `attn_q_a` family in
//!   ds4.c:2306+ for the layer-weight struct)
//! - MoE routing: `ds4.c:5178-5466`
//!
//! Line numbers pinned to ds4 commit `ef0a490` (2026-05-17).

use rsllm_backend_cpu::SimdTier;
use rsllm_backend_cpu::ops::{rmsnorm, sinkhorn::N_HC};
use rsllm_gguf::Metadata;

use crate::Error;
use crate::dsv4::compressor::{CompressorWeights, IndexerWeights};
use crate::dsv4::hc::{HcScratch, HcSublayerWeights, hc_post, hc_pre};
use crate::dsv4::mla::{MlaOutput, MlaScratch, MlaWeights, mla_projections};
use crate::dsv4::moe::{
    MoeExpertWeights, MoeHashRouter, MoeScratch, MoeTopkRouter, SharedExpertWeights,
    apply_shared_expert, moe_hash_route, moe_topk_route,
};
use crate::dsv4::shape::{
    DSV4_HASH_ROUTE_LAYERS, DSV4_HEAD_DIM, DSV4_N_EMBD, DSV4_N_HEAD, DSV4_N_LAYER, DSV4_N_LORA_O,
    DSV4_N_OUT_GROUP, DSV4_N_VOCAB, DSV4_RMS_EPS, validate_metadata,
};
use crate::dsv4::weight::{WeightBlob, matmul_grouped_lora_down, matmul_weight_f32};

/// Weight package for one transformer block.
///
/// All slices are borrowed views into the mmap'd GGUF. Per-layer
/// RMSNorm scales are kept as `&[f32]` because their byte cost is tiny
/// (4096 lanes × 4 bytes × 4 norms = 64KB / layer); promoting them to
/// f32 at load time keeps the inner loop hot.
#[derive(Debug, Clone, Copy)]
pub struct DsV4Block<'a> {
    /// Pre-attention RMSNorm scale `[N_EMBD]`.
    pub attn_norm: &'a [f32],
    /// MLA Q/KV LoRA projections.
    pub mla: MlaWeights<'a>,
    /// Per-head attention sink values `[N_HEAD]`. Each value plays the
    /// role of a virtual sink-token logit in the attention softmax; it
    /// is mixed into both the running max and the partition-sum denom
    /// per head (`ds4.c:4904-4922`). Required by F006 attention, not
    /// consumed in F005's projection-only forward path.
    pub attn_sinks: &'a [f32],
    /// Grouped LoRA **down**-projection of the attention output.
    /// Logical shape `[N_OUT_GROUP × group_dim × N_LORA_O]` where
    /// `group_dim = HEAD_DIM * (N_HEAD / N_OUT_GROUP) = 4096`.
    /// Each input chunk of `group_dim` lanes projects to its own
    /// `N_LORA_O = 1024` slot; the per-group outputs are concatenated
    /// to form an `out_low_dim = N_OUT_GROUP * N_LORA_O = 8192` vector.
    /// (`ds4.c:2312`, `ds4.c:4960`.)
    pub attn_output_a: WeightBlob<'a>,
    /// Dense LoRA **up**-projection of the attention output.
    /// Shape `[out_low_dim × N_EMBD]` = `[8192 × 4096]`.
    /// (`ds4.c:2313`, `ds4.c:4962`.)
    pub attn_output_b: WeightBlob<'a>,
    /// HC weights for the attention sublayer (`fn` + `scale` + `base`).
    /// The matching `hc_post` step has no weights of its own — it
    /// reuses the per-token Sinkhorn split produced by `hc_pre` and
    /// stored on [`HcScratch::split`] across the `pre → sublayer → post`
    /// boundary.
    pub hc_attn: HcSublayerWeights<'a>,

    /// Per-dim compressed-KV score LoRA. Present iff this layer has
    /// `compress_ratio > 0` (i.e. all layers except the first two
    /// dense ones, per
    /// [`rsllm_kvcache::dsv4::shape::layer_compress_ratio`]).
    /// Consumed by F008.C to replace the zero-placeholder scores in
    /// [`crate::dsv4::attention::ThreeTierAttention`].
    pub compressor: Option<CompressorWeights<'a>>,
    /// Six-tensor indexer weight bundle (ds4.c:2326-2331). Present iff
    /// this layer is ratio-4 (even `il >= 2`). The algorithm using these
    /// is deferred to F011; until then `attention.rs` keeps zero-placeholder
    /// indexer scores.
    pub indexer: Option<IndexerWeights<'a>>,

    /// Pre-FFN RMSNorm scale `[N_EMBD]`.
    pub ffn_norm: &'a [f32],
    /// MoE routed experts (`[N_EXPERT × N_FF_EXP × N_EMBD]`).
    pub moe_experts: MoeExpertWeights<'a>,
    /// Shared (always-on) expert.
    pub shared_expert: SharedExpertWeights<'a>,
    /// Hash router for layers `[0, 3)`; `None` for top-k layers.
    pub hash_router: Option<MoeHashRouter<'a>>,
    /// Top-k router for layers `[3, 43)`; `None` for hash layers.
    pub topk_router: Option<MoeTopkRouter<'a>>,
    /// HC weights for the FFN sublayer. Same shape semantics as
    /// [`Self::hc_attn`]; the post step is weightless and consumes
    /// the matching `hc_pre`'s Sinkhorn split.
    pub hc_ffn: HcSublayerWeights<'a>,
}

impl<'a> DsV4Block<'a> {
    /// Sanity-check internal consistency:
    /// - hash-routed layer must have `hash_router = Some(_)` and `topk_router = None`,
    ///   and vice versa;
    /// - dense layers (`il < 2`) must NOT carry compressor/indexer weights;
    /// - compressed layers must carry a `compressor`;
    /// - ratio-4 layers must carry `indexer = Some(_)`, ratio-128 layers must NOT.
    fn validate(&self, layer_idx: usize) -> Result<(), Error> {
        let is_hash = layer_idx < DSV4_HASH_ROUTE_LAYERS;
        match (is_hash, self.hash_router.is_some(), self.topk_router.is_some()) {
            (true, true, false) | (false, false, true) => {}
            _ => {
                return Err(Error::ShapeMismatch {
                    key: "block.router",
                    expected: if is_hash {
                        "hash_router=Some, topk_router=None".to_string()
                    } else {
                        "hash_router=None, topk_router=Some".to_string()
                    },
                    actual: format!(
                        "hash_router={}, topk_router={}",
                        self.hash_router.is_some(),
                        self.topk_router.is_some()
                    ),
                });
            }
        }

        let ratio = rsllm_kvcache::dsv4::shape::layer_compress_ratio(layer_idx);
        let has_indexer = rsllm_kvcache::dsv4::shape::layer_has_indexer(layer_idx);
        // Compressor: required iff ratio > 0.
        if (ratio > 0) != self.compressor.is_some() {
            return Err(Error::ShapeMismatch {
                key: "block.compressor",
                expected: if ratio > 0 {
                    format!("Some (layer {layer_idx}, ratio={ratio})")
                } else {
                    format!("None (layer {layer_idx} is dense)")
                },
                actual: format!("{}", self.compressor.is_some()),
            });
        }
        // Indexer bundle: required iff this is a ratio-4 layer.
        if has_indexer != self.indexer.is_some() {
            return Err(Error::ShapeMismatch {
                key: "block.indexer",
                expected: if has_indexer {
                    format!("Some (layer {layer_idx} is ratio-4)")
                } else {
                    format!("None (layer {layer_idx} is not ratio-4)")
                },
                actual: format!("{}", self.indexer.is_some()),
            });
        }

        // Deep shape checks on the LoRA byte storage. Catches a
        // wrong-sized GGUF tensor at load time instead of at first
        // matmul (security-review F008.B finding). We only run these
        // on layers that actually carry the weights — dense layers
        // already had `compressor.is_none()` enforced above.
        if let Some(c) = self.compressor.as_ref() {
            // Compute the per-regime `comp_width` once and check all
            // four tensors against it. ratio-4 → comp_width = 1024,
            // ratio-128 → comp_width = 512.
            let ratio_u32 = rsllm_kvcache::dsv4::shape::layer_compress_ratio(layer_idx);
            let ratio = ratio_u32 as usize;
            let coff = if ratio_u32 == 4 { 2 } else { 1 };
            let comp_width = coff * DSV4_HEAD_DIM;
            c.kv.check_shape(comp_width, DSV4_N_EMBD, "block.compressor.kv")?;
            c.gate.check_shape(comp_width, DSV4_N_EMBD, "block.compressor.gate")?;
            c.ape.check_shape(ratio, comp_width, "block.compressor.ape")?;
            if c.norm.len() != DSV4_HEAD_DIM {
                return Err(Error::ShapeMismatch {
                    key: "block.compressor.norm",
                    expected: format!("{DSV4_HEAD_DIM}"),
                    actual: format!("{}", c.norm.len()),
                });
            }
        }
        // Deep shape checks for the six indexer tensors (F010.C).
        // index_width = 2 × N_INDEXER_HEAD_DIM, ratio = 4 (ratio-4 layers only).
        if let Some(iw) = self.indexer.as_ref() {
            let index_width = 2 * crate::dsv4::shape::DSV4_N_INDEXER_HEAD_DIM;
            let indexer_ratio = 4usize;
            iw.attn_q_b.check_shape(
                crate::dsv4::shape::DSV4_N_INDEXER_HEAD
                    * crate::dsv4::shape::DSV4_N_INDEXER_HEAD_DIM,
                crate::dsv4::shape::DSV4_N_LORA_Q,
                "block.indexer.attn_q_b",
            )?;
            iw.proj.check_shape(
                crate::dsv4::shape::DSV4_N_INDEXER_HEAD,
                DSV4_N_EMBD,
                "block.indexer.proj",
            )?;
            iw.comp_ape.check_shape(indexer_ratio, index_width, "block.indexer.comp_ape")?;
            iw.comp_kv.check_shape(index_width, DSV4_N_EMBD, "block.indexer.comp_kv")?;
            iw.comp_gate.check_shape(index_width, DSV4_N_EMBD, "block.indexer.comp_gate")?;
            if iw.comp_norm.len() != crate::dsv4::shape::DSV4_N_INDEXER_HEAD_DIM {
                return Err(Error::ShapeMismatch {
                    key: "block.indexer.comp_norm",
                    expected: format!("{}", crate::dsv4::shape::DSV4_N_INDEXER_HEAD_DIM),
                    actual: format!("{}", iw.comp_norm.len()),
                });
            }
        }
        Ok(())
    }
}

/// Full DeepSeek V4 Flash model.
///
/// Owns no weight bytes itself — every slice points into a parent
/// mmap'd region with lifetime `'a`. Constructing the model is the
/// place where shape mismatches are caught and turned into `Error`s.
#[derive(Debug)]
pub struct DeepSeekV4Flash<'a> {
    /// Token embedding `[N_VOCAB × N_EMBD]`.
    pub embed_tokens: WeightBlob<'a>,
    /// 43 transformer blocks.
    pub blocks: Vec<DsV4Block<'a>>,
    /// Output-head HC merge weights — collapse the 4 residual streams
    /// back into a single `[N_EMBD]` vector before the final RMSNorm
    /// and LM-head projection (ds4.c:7916-7944 `output_hc_head_one`).
    pub output_hc: OutputHcWeights<'a>,
    /// Final RMSNorm scale `[N_EMBD]`.
    pub output_norm: &'a [f32],
    /// LM head `[N_VOCAB × N_EMBD]` (often the embedding tied weight).
    pub lm_head: WeightBlob<'a>,
}

/// Top-level HC head weights — used once at the end of the model to
/// reduce `[N_HC × N_EMBD]` residual streams to a single `[N_EMBD]`
/// vector that feeds the LM head.
///
/// Verified against ds4 commit `ef0a490`:
/// - shapes at `ds4.c:2292-2294`
/// - loader at `ds4.c:2581-2583`
/// - algorithm at `ds4.c:7916-7944`
#[derive(Debug, Clone, Copy)]
pub struct OutputHcWeights<'a> {
    /// `[HC_DIM × N_HC]` = `[16384 × 4]` F16. Projects the flat+normed
    /// residual to per-stream gate logits.
    pub mix_fn: WeightBlob<'a>,
    /// `[1]` F32 scalar applied to all gate logits before
    /// the sigmoid: `w[i] = sigmoid(pre[i] * scale[0] + base[i]) + eps`.
    pub scale: &'a [f32],
    /// `[N_HC]` F32 bias added to the gate logits.
    pub base: &'a [f32],
}

impl<'a> DeepSeekV4Flash<'a> {
    /// Construct a model from already-resolved weight views.
    ///
    /// Validates that all 43 blocks are supplied and that routing
    /// configuration matches the layer index (hash vs. top-k).
    ///
    /// Use [`Self::validate_gguf_metadata`] before calling this with
    /// shapes resolved from a real GGUF file.
    ///
    /// # Errors
    /// Returns [`Error::ShapeMismatch`] for any structural mismatch.
    pub fn new(
        embed_tokens: WeightBlob<'a>,
        blocks: Vec<DsV4Block<'a>>,
        output_hc: OutputHcWeights<'a>,
        output_norm: &'a [f32],
        lm_head: WeightBlob<'a>,
    ) -> Result<Self, Error> {
        if blocks.len() != DSV4_N_LAYER {
            return Err(Error::ShapeMismatch {
                key: "model.block_count",
                expected: format!("{DSV4_N_LAYER}"),
                actual: format!("{}", blocks.len()),
            });
        }
        if output_norm.len() != DSV4_N_EMBD {
            return Err(Error::ShapeMismatch {
                key: "model.output_norm",
                expected: format!("{DSV4_N_EMBD}"),
                actual: format!("{}", output_norm.len()),
            });
        }
        for (i, block) in blocks.iter().enumerate() {
            block.validate(i)?;
            if block.attn_norm.len() != DSV4_N_EMBD {
                return Err(Error::ShapeMismatch {
                    key: "block.attn_norm",
                    expected: format!("{DSV4_N_EMBD}"),
                    actual: format!("{}", block.attn_norm.len()),
                });
            }
            if block.ffn_norm.len() != DSV4_N_EMBD {
                return Err(Error::ShapeMismatch {
                    key: "block.ffn_norm",
                    expected: format!("{DSV4_N_EMBD}"),
                    actual: format!("{}", block.ffn_norm.len()),
                });
            }
            // attn_sinks is consumed by F006 attention but we validate
            // here so a malformed GGUF fails at load time rather than at
            // first forward call (per security review 2026-05-17).
            if block.attn_sinks.len() != DSV4_N_HEAD {
                return Err(Error::ShapeMismatch {
                    key: "block.attn_sinks",
                    expected: format!("{DSV4_N_HEAD}"),
                    actual: format!("{}", block.attn_sinks.len()),
                });
            }
        }
        // Validate output_hc bundle shapes — these are tiny (a few
        // dozen elements aside from mix_fn) so the check is cheap and
        // catches a malformed GGUF before the first forward pass.
        let hc_dim = crate::dsv4::hc::HC_DIM;
        output_hc
            .mix_fn
            .check_shape(N_HC, hc_dim, "model.output_hc.mix_fn")?;
        if output_hc.scale.len() != 1 {
            return Err(Error::ShapeMismatch {
                key: "model.output_hc.scale",
                expected: "1".to_string(),
                actual: format!("{}", output_hc.scale.len()),
            });
        }
        if output_hc.base.len() != N_HC {
            return Err(Error::ShapeMismatch {
                key: "model.output_hc.base",
                expected: format!("{N_HC}"),
                actual: format!("{}", output_hc.base.len()),
            });
        }
        Ok(Self {
            embed_tokens,
            blocks,
            output_hc,
            output_norm,
            lm_head,
        })
    }

    /// Validate a GGUF metadata block against the DS V4 Flash spec.
    /// Convenience re-export so callers don't have to import [`shape`].
    ///
    /// # Errors
    /// Bubbles up [`validate_metadata`] errors.
    pub fn validate_gguf_metadata(meta: &Metadata) -> Result<(), Error> {
        validate_metadata(meta)
    }
}

/// Pluggable attention callback. [`crate::dsv4::attention::ThreeTierAttention`]
/// (F006 + F008) is the v0.1.0 production implementation — wrap it in
/// a `&mut |q, kv, x, il, out| attn.run_layer(q, kv, x, il, out)`
/// closure to satisfy this type.
///
/// The signature is intentionally generic across layers so a single
/// closure can carry both prefill and decode state.
///
/// Inputs:
/// - `q`: `[n_tok × N_HEAD × HEAD_DIM]` RoPE'd query latent.
/// - `kv`: `[n_tok × HEAD_DIM]` RoPE'd KV latent (1-head, MLA).
/// - `x`: `[n_tok × N_EMBD]` the post-RMSNorm hidden state. The
///   attention adapter re-projects this through the per-layer
///   `attn_compressor` / `attn_indexer_*` LoRAs to produce the per-dim
///   compression score and the indexer KV+score (F008.B weight types).
/// - `layer_idx`: 0-based block index, 0..N_LAYER.
///
/// Output:
/// - `attn_out`: `[n_tok × N_HEAD × HEAD_DIM]` (pre output-projection).
pub type AttentionFn<'f> =
    &'f mut dyn FnMut(&[f32], &[f32], &[f32], usize, &mut [f32]) -> Result<(), Error>;

/// Reusable per-forward scratch. Held by the caller so we don't
/// re-allocate large activation buffers between tokens.
#[derive(Debug)]
pub struct ForwardScratch {
    /// `[n_tok × N_HC × N_EMBD]` — the four residual streams.
    pub streams: Vec<f32>,
    /// `[n_tok × N_EMBD]` — HC-merged hidden state.
    pub merged: Vec<f32>,
    /// `[n_tok × N_EMBD]` — RMSNorm-ed hidden state.
    pub normed: Vec<f32>,
    /// `[n_tok × N_HEAD * HEAD_DIM]` — Q latent.
    pub q: Vec<f32>,
    /// `[n_tok × HEAD_DIM]` — KV latent.
    pub kv: Vec<f32>,
    /// `[n_tok × N_HEAD * HEAD_DIM]` — attention raw output.
    pub attn_raw: Vec<f32>,
    /// `[n_tok × N_OUT_GROUP * N_LORA_O]` = `[n_tok × 8192]` — output
    /// of the grouped LoRA down-projection (intermediate latent
    /// between `attn_output_a` and `attn_output_b`).
    pub attn_low: Vec<f32>,
    /// `[n_tok × N_EMBD]` — attention projected output.
    pub attn_proj: Vec<f32>,
    /// `[n_tok × N_EMBD]` — FFN (shared + MoE) output.
    pub ffn_out: Vec<f32>,
    /// `[n_tok × N_EMBD]` — shared expert output buffer.
    pub shared_out: Vec<f32>,
    /// Nested scratches for the kernels that need them.
    pub mla: MlaScratch,
    pub hc: HcScratch,
    pub moe: MoeScratch,
}

impl ForwardScratch {
    /// Allocate scratch sized for `n_tok` tokens.
    #[must_use]
    pub fn new(n_tok: usize) -> Self {
        Self {
            streams: vec![0.0_f32; n_tok * N_HC * DSV4_N_EMBD],
            merged: vec![0.0_f32; n_tok * DSV4_N_EMBD],
            normed: vec![0.0_f32; n_tok * DSV4_N_EMBD],
            q: vec![0.0_f32; n_tok * DSV4_N_HEAD * DSV4_HEAD_DIM],
            kv: vec![0.0_f32; n_tok * DSV4_HEAD_DIM],
            attn_raw: vec![0.0_f32; n_tok * DSV4_N_HEAD * DSV4_HEAD_DIM],
            attn_low: vec![0.0_f32; n_tok * DSV4_N_OUT_GROUP * DSV4_N_LORA_O],
            attn_proj: vec![0.0_f32; n_tok * DSV4_N_EMBD],
            ffn_out: vec![0.0_f32; n_tok * DSV4_N_EMBD],
            shared_out: vec![0.0_f32; n_tok * DSV4_N_EMBD],
            mla: MlaScratch::new(n_tok),
            hc: HcScratch::new(n_tok),
            moe: MoeScratch::new(n_tok),
        }
    }

    /// Resize all sub-buffers for a new `n_tok`.
    pub fn resize(&mut self, n_tok: usize) {
        self.streams.resize(n_tok * N_HC * DSV4_N_EMBD, 0.0);
        self.merged.resize(n_tok * DSV4_N_EMBD, 0.0);
        self.normed.resize(n_tok * DSV4_N_EMBD, 0.0);
        self.q.resize(n_tok * DSV4_N_HEAD * DSV4_HEAD_DIM, 0.0);
        self.kv.resize(n_tok * DSV4_HEAD_DIM, 0.0);
        self.attn_raw
            .resize(n_tok * DSV4_N_HEAD * DSV4_HEAD_DIM, 0.0);
        self.attn_low
            .resize(n_tok * DSV4_N_OUT_GROUP * DSV4_N_LORA_O, 0.0);
        self.attn_proj.resize(n_tok * DSV4_N_EMBD, 0.0);
        self.ffn_out.resize(n_tok * DSV4_N_EMBD, 0.0);
        self.shared_out.resize(n_tok * DSV4_N_EMBD, 0.0);
        self.mla.resize(n_tok);
        self.hc.resize(n_tok);
        self.moe.resize(n_tok);
    }
}

/// Run one transformer block.
///
/// Updates `scratch.streams` in place. `attention` is the F006-supplied
/// callback that turns `(q, kv)` into `attn_raw`. Per-token positions
/// are produced by `position_of`.
#[allow(clippy::too_many_arguments)]
pub fn forward_block(
    block: &DsV4Block<'_>,
    layer_idx: usize,
    token_ids: &[u32],
    scratch: &mut ForwardScratch,
    n_tok: usize,
    position_of: impl Fn(usize) -> u32,
    attention: AttentionFn<'_>,
    tier: SimdTier,
) -> Result<(), Error> {
    // === Attention sub-layer ===
    // hc_pre fills `scratch.merged` (input to the sublayer) AND
    // `scratch.hc.split` (per-token Sinkhorn output used by the
    // matching hc_post below).
    hc_pre(
        &scratch.streams,
        &mut scratch.merged,
        &block.hc_attn,
        &mut scratch.hc,
        n_tok,
        tier,
    )?;
    rmsnorm_per_token(&mut scratch.normed, &scratch.merged, block.attn_norm, n_tok, tier)?;
    {
        let mut mla_out = MlaOutput {
            q: &mut scratch.q,
            kv: &mut scratch.kv,
        };
        mla_projections(
            &block.mla,
            &scratch.normed,
            &mut mla_out,
            &mut scratch.mla,
            n_tok,
            &position_of,
            tier,
        )?;
    }
    attention(
        &scratch.q,
        &scratch.kv,
        &scratch.normed,
        layer_idx,
        &mut scratch.attn_raw,
    )?;
    // Output projection (two-stage grouped LoRA, ds4.c:4960-4962):
    //   attn_raw  [n_tok × N_HEAD * HEAD_DIM = n_tok × 32768]
    //     │
    //     │ attn_output_a  (grouped: 8 × [4096 → 1024])
    //     ▼
    //   attn_low  [n_tok × N_OUT_GROUP * N_LORA_O = n_tok × 8192]
    //     │
    //     │ attn_output_b  (dense: 8192 → 4096)
    //     ▼
    //   attn_proj [n_tok × N_EMBD = n_tok × 4096]
    let group_dim = DSV4_HEAD_DIM * (DSV4_N_HEAD / DSV4_N_OUT_GROUP);
    matmul_grouped_lora_down(
        &mut scratch.attn_low,
        &block.attn_output_a,
        &scratch.attn_raw,
        n_tok,
        DSV4_N_OUT_GROUP,
        group_dim,
        DSV4_N_LORA_O,
        tier,
    )?;
    matmul_weight_f32(
        &mut scratch.attn_proj,
        &block.attn_output_b,
        &scratch.attn_low,
        n_tok,
        DSV4_N_OUT_GROUP * DSV4_N_LORA_O,
        DSV4_N_EMBD,
        tier,
    )?;
    // hc_post owns no weights — it reads the `post` + `comb` slices
    // out of `scratch.hc.split` populated by the attention hc_pre call.
    hc_post(
        &mut scratch.streams,
        &scratch.attn_proj,
        &mut scratch.hc,
        n_tok,
    )?;

    // === FFN sub-layer ===
    // FFN's hc_pre overwrites `scratch.hc.split` with its own Sinkhorn
    // output; the next hc_post will consume that.
    hc_pre(
        &scratch.streams,
        &mut scratch.merged,
        &block.hc_ffn,
        &mut scratch.hc,
        n_tok,
        tier,
    )?;
    rmsnorm_per_token(&mut scratch.normed, &scratch.merged, block.ffn_norm, n_tok, tier)?;
    // `apply_shared_expert` writes `shared_out` via three matmul
    // dispatches whose semantics are **overwrite** (see
    // `rsllm_backend_cpu::ops::quant_matmul::matmul_quant_f32`, which
    // assigns `out_row[o] = dot`, and `weight::matmul_weight_f32`'s
    // F32 branch, which is the same). So pre-clearing `shared_out`
    // is unnecessary — but the caller must not rely on the previous
    // forward's contents either.
    apply_shared_expert(
        &mut scratch.shared_out,
        &scratch.normed,
        &block.shared_expert,
        &mut scratch.moe,
        n_tok,
        tier,
    )?;
    // The MoE routed-experts function accumulates into `ffn_out`, so
    // pre-fill `ffn_out` with the shared-expert contribution as the
    // starting accumulator value (NOT as a temporary buffer).
    scratch.ffn_out.copy_from_slice(&scratch.shared_out);
    if layer_idx < DSV4_HASH_ROUTE_LAYERS {
        let router = block.hash_router.as_ref().expect("hash_router checked by validate()");
        moe_hash_route(
            &mut scratch.ffn_out,
            &scratch.normed,
            token_ids,
            &block.moe_experts,
            router,
            &mut scratch.moe,
            n_tok,
            tier,
        )?;
    } else {
        let router = block.topk_router.as_ref().expect("topk_router checked by validate()");
        moe_topk_route(
            &mut scratch.ffn_out,
            &scratch.normed,
            &block.moe_experts,
            router,
            &mut scratch.moe,
            n_tok,
            tier,
        )?;
    }
    hc_post(
        &mut scratch.streams,
        &scratch.ffn_out,
        &mut scratch.hc,
        n_tok,
    )?;

    Ok(())
}

/// Per-token RMSNorm helper: applies `out[t] = rmsnorm(in[t], weight)`
/// for each of the `n_tok` tokens.
fn rmsnorm_per_token(
    out: &mut [f32],
    x: &[f32],
    weight: &[f32],
    n_tok: usize,
    tier: SimdTier,
) -> Result<(), Error> {
    let dim = weight.len();
    if x.len() != n_tok * dim || out.len() != n_tok * dim {
        return Err(Error::ShapeMismatch {
            key: "rmsnorm_per_token",
            expected: format!("{}", n_tok * dim),
            actual: format!("x={} out={}", x.len(), out.len()),
        });
    }
    for t in 0..n_tok {
        let off = t * dim;
        rmsnorm(
            &mut out[off..off + dim],
            &x[off..off + dim],
            weight,
            DSV4_RMS_EPS,
            tier,
        )
        .map_err(|e| Error::ShapeMismatch {
            key: "rmsnorm_per_token.kernel",
            expected: "ok".to_string(),
            actual: format!("{e}"),
        })?;
    }
    Ok(())
}

/// Compute LM head logits for the last token in a batch (greedy decode
/// only needs the last position). Produces `[N_VOCAB]`.
///
/// Used by the sampler in F007; included here so the model exposes a
/// clean inference entry point.
///
/// # Errors
/// Bubbles up shape errors from the underlying matmul.
pub fn lm_head_logits(
    model: &DeepSeekV4Flash<'_>,
    last_hidden: &[f32],
    out_logits: &mut [f32],
    tier: SimdTier,
) -> Result<(), Error> {
    if last_hidden.len() != DSV4_N_EMBD {
        return Err(Error::ShapeMismatch {
            key: "lm_head.last_hidden",
            expected: format!("{DSV4_N_EMBD}"),
            actual: format!("{}", last_hidden.len()),
        });
    }
    if out_logits.len() != DSV4_N_VOCAB {
        return Err(Error::ShapeMismatch {
            key: "lm_head.out_logits",
            expected: format!("{DSV4_N_VOCAB}"),
            actual: format!("{}", out_logits.len()),
        });
    }
    let mut normed = vec![0.0_f32; DSV4_N_EMBD];
    rmsnorm(&mut normed, last_hidden, model.output_norm, DSV4_RMS_EPS, tier).map_err(|e| {
        Error::ShapeMismatch {
            key: "lm_head.output_norm",
            expected: "ok".to_string(),
            actual: format!("{e}"),
        }
    })?;
    matmul_weight_f32(
        out_logits,
        &model.lm_head,
        &normed,
        1,
        DSV4_N_EMBD,
        DSV4_N_VOCAB,
        tier,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsv4::hc::{HC_DIM, HC_MIX_DIM};
    use crate::dsv4::moe::StackedExperts;
    use crate::dsv4::shape::{DSV4_N_EXPERT, DSV4_N_EXPERT_USED, DSV4_N_FF_EXP, DSV4_N_LORA_Q};

    /// Storage backing for *one* block worth of small / mid-sized
    /// tensors. We deliberately avoid materializing the routed-expert
    /// weights ([N_EXPERT × N_FF_EXP × N_EMBD] f32 ≈ 8 GiB) because a
    /// full GGUF-backed forward path test belongs as an integration
    /// test against a real model file (lands with F008's CLI).
    ///
    /// What we *can* validate here: the structural invariants of
    /// [`DsV4Block`] and [`DeepSeekV4Flash::new`] — these only inspect
    /// shape fields and don't run any matmul.
    struct StubBlockStorage {
        attn_norm: Vec<f32>,
        q_a: Vec<f32>,
        q_a_norm: Vec<f32>,
        q_b: Vec<f32>,
        kv_a: Vec<f32>,
        kv_a_norm: Vec<f32>,
        attn_sinks: Vec<f32>,
        attn_output_a: Vec<u8>,
        attn_output_b: Vec<u8>,
        hc_mix_fn: Vec<f32>,
        hc_scale: Vec<f32>,
        hc_base: Vec<f32>,
        ffn_norm: Vec<f32>,
        // Empty bytes: not exercised by structural tests.
        moe_gate: Vec<u8>,
        moe_up: Vec<u8>,
        moe_down: Vec<u8>,
        shared_gate: Vec<f32>,
        shared_up: Vec<f32>,
        shared_down: Vec<f32>,
        gate_inp: Vec<f32>,
        tid2eid: Vec<i32>,
        // F010.B per-layer LoRA backing buffers. All zero — the
        // structural tests check Option-presence only.
        //
        // Ratio-128 (odd `il >= 2`) sizing: comp_width = HEAD_DIM = 512,
        // ratio = 128. Stubs use ratio-128 sizing for compressor; ratio-4
        // layers (`il = 2`) are special-cased in the block() helper.
        compressor_kv_r128: Vec<f32>,
        compressor_gate_r128: Vec<f32>,
        compressor_ape_r128: Vec<f32>,
        compressor_kv_r4: Vec<f32>,
        compressor_gate_r4: Vec<f32>,
        compressor_ape_r4: Vec<f32>,
        compressor_norm: Vec<f32>,
        // F010.C six-tensor indexer backing buffers (ratio-4 only).
        // index_width = 2 * N_INDEXER_HEAD_DIM = 256, ratio = 4.
        indexer_attn_q_b: Vec<f32>,
        indexer_proj: Vec<f32>,
        indexer_comp_ape: Vec<f32>,
        indexer_comp_kv: Vec<f32>,
        indexer_comp_gate: Vec<f32>,
        indexer_comp_norm: Vec<f32>,
    }

    impl StubBlockStorage {
        fn new() -> Self {
            Self {
                attn_norm: vec![1.0; DSV4_N_EMBD],
                q_a: vec![0.0; DSV4_N_LORA_Q * DSV4_N_EMBD],
                q_a_norm: vec![1.0; DSV4_N_LORA_Q],
                q_b: vec![0.0; DSV4_N_HEAD * DSV4_HEAD_DIM * DSV4_N_LORA_Q],
                kv_a: vec![0.0; DSV4_HEAD_DIM * DSV4_N_EMBD],
                kv_a_norm: vec![1.0; DSV4_HEAD_DIM],
                attn_sinks: vec![0.0; DSV4_N_HEAD],
                // Two-stage LoRA: a is [N_OUT_GROUP × group_dim × N_LORA_O],
                // b is [out_low_dim × N_EMBD]. The stub uses empty bytes
                // because structural tests don't run any matmul.
                attn_output_a: vec![],
                attn_output_b: vec![],
                hc_mix_fn: vec![0.0; HC_DIM * HC_MIX_DIM],
                hc_scale: vec![1.0, 1.0, 1.0],
                hc_base: vec![0.0; HC_MIX_DIM],
                ffn_norm: vec![1.0; DSV4_N_EMBD],
                moe_gate: vec![],
                moe_up: vec![],
                moe_down: vec![],
                shared_gate: vec![0.0; DSV4_N_FF_EXP * DSV4_N_EMBD],
                shared_up: vec![0.0; DSV4_N_FF_EXP * DSV4_N_EMBD],
                shared_down: vec![0.0; DSV4_N_EMBD * DSV4_N_FF_EXP],
                gate_inp: vec![0.0; DSV4_N_EXPERT * DSV4_N_EMBD],
                tid2eid: vec![0_i32; DSV4_N_EXPERT_USED * DSV4_N_VOCAB],
                // Compressor backing buffers, sized per regime:
                //   ratio-128: comp_width = HEAD_DIM = 512
                //   ratio-4:   comp_width = 2 * HEAD_DIM = 1024
                compressor_kv_r128:   vec![0.0; DSV4_HEAD_DIM * DSV4_N_EMBD],
                compressor_gate_r128: vec![0.0; DSV4_HEAD_DIM * DSV4_N_EMBD],
                compressor_ape_r128:  vec![0.0; DSV4_HEAD_DIM * 128],
                compressor_kv_r4:     vec![0.0; 2 * DSV4_HEAD_DIM * DSV4_N_EMBD],
                compressor_gate_r4:   vec![0.0; 2 * DSV4_HEAD_DIM * DSV4_N_EMBD],
                compressor_ape_r4:    vec![0.0; 2 * DSV4_HEAD_DIM * 4],
                compressor_norm:      vec![1.0; DSV4_HEAD_DIM],
                // F010.C indexer — index_width = 2 * N_INDEXER_HEAD_DIM = 256
                indexer_attn_q_b: vec![
                    0.0;
                    crate::dsv4::shape::DSV4_N_LORA_Q
                        * crate::dsv4::shape::DSV4_N_INDEXER_HEAD
                        * crate::dsv4::shape::DSV4_N_INDEXER_HEAD_DIM
                ],
                indexer_proj: vec![
                    0.0;
                    DSV4_N_EMBD * crate::dsv4::shape::DSV4_N_INDEXER_HEAD
                ],
                indexer_comp_ape: vec![
                    0.0;
                    (2 * crate::dsv4::shape::DSV4_N_INDEXER_HEAD_DIM) * 4
                ],
                indexer_comp_kv: vec![
                    0.0;
                    DSV4_N_EMBD * (2 * crate::dsv4::shape::DSV4_N_INDEXER_HEAD_DIM)
                ],
                indexer_comp_gate: vec![
                    0.0;
                    DSV4_N_EMBD * (2 * crate::dsv4::shape::DSV4_N_INDEXER_HEAD_DIM)
                ],
                indexer_comp_norm: vec![1.0; crate::dsv4::shape::DSV4_N_INDEXER_HEAD_DIM],
            }
        }

        fn block(&self, layer_idx: usize) -> DsV4Block<'_> {
            let hc = HcSublayerWeights {
                mix_fn: WeightBlob::F32(&self.hc_mix_fn),
                scale: &self.hc_scale,
                base: &self.hc_base,
            };
            DsV4Block {
                attn_norm: &self.attn_norm,
                mla: MlaWeights {
                    attn_q_a: WeightBlob::F32(&self.q_a),
                    q_a_norm: &self.q_a_norm,
                    attn_q_b: WeightBlob::F32(&self.q_b),
                    attn_kv_a: WeightBlob::F32(&self.kv_a),
                    kv_a_norm: &self.kv_a_norm,
                },
                attn_sinks: &self.attn_sinks,
                attn_output_a: WeightBlob::Quant {
                    data: &self.attn_output_a,
                    dtype: rsllm_gguf::GgmlType::Q8_0,
                },
                attn_output_b: WeightBlob::Quant {
                    data: &self.attn_output_b,
                    dtype: rsllm_gguf::GgmlType::Q8_0,
                },
                hc_attn: hc,
                ffn_norm: &self.ffn_norm,
                moe_experts: MoeExpertWeights {
                    gate: StackedExperts {
                        blob: WeightBlob::Quant {
                            data: &self.moe_gate,
                            dtype: rsllm_gguf::GgmlType::Q4_K,
                        },
                        elements_per_expert: DSV4_N_FF_EXP * DSV4_N_EMBD,
                    },
                    up: StackedExperts {
                        blob: WeightBlob::Quant {
                            data: &self.moe_up,
                            dtype: rsllm_gguf::GgmlType::Q4_K,
                        },
                        elements_per_expert: DSV4_N_FF_EXP * DSV4_N_EMBD,
                    },
                    down: StackedExperts {
                        blob: WeightBlob::Quant {
                            data: &self.moe_down,
                            dtype: rsllm_gguf::GgmlType::Q2_K,
                        },
                        elements_per_expert: DSV4_N_EMBD * DSV4_N_FF_EXP,
                    },
                },
                shared_expert: SharedExpertWeights {
                    gate: WeightBlob::F32(&self.shared_gate),
                    up: WeightBlob::F32(&self.shared_up),
                    down: WeightBlob::F32(&self.shared_down),
                },
                hash_router: if layer_idx < DSV4_HASH_ROUTE_LAYERS {
                    Some(MoeHashRouter {
                        tid2eid: &self.tid2eid,
                        gate_inp: WeightBlob::F32(&self.gate_inp),
                        gate_bias: None,
                    })
                } else {
                    None
                },
                topk_router: if layer_idx >= DSV4_HASH_ROUTE_LAYERS {
                    Some(MoeTopkRouter {
                        gate_inp: WeightBlob::F32(&self.gate_inp),
                        gate_bias: None,
                    })
                } else {
                    None
                },
                hc_ffn: hc,
                compressor: {
                    let ratio = rsllm_kvcache::dsv4::shape::layer_compress_ratio(layer_idx);
                    match ratio {
                        0 => None,
                        4 => Some(CompressorWeights {
                            kv: WeightBlob::F32(&self.compressor_kv_r4),
                            gate: WeightBlob::F32(&self.compressor_gate_r4),
                            ape: WeightBlob::F32(&self.compressor_ape_r4),
                            norm: &self.compressor_norm,
                        }),
                        _ => Some(CompressorWeights {
                            kv: WeightBlob::F32(&self.compressor_kv_r128),
                            gate: WeightBlob::F32(&self.compressor_gate_r128),
                            ape: WeightBlob::F32(&self.compressor_ape_r128),
                            norm: &self.compressor_norm,
                        }),
                    }
                },
                indexer: if rsllm_kvcache::dsv4::shape::layer_has_indexer(layer_idx) {
                    Some(IndexerWeights {
                        attn_q_b: WeightBlob::F32(&self.indexer_attn_q_b),
                        proj: WeightBlob::F32(&self.indexer_proj),
                        comp_ape: WeightBlob::F32(&self.indexer_comp_ape),
                        comp_kv: WeightBlob::F32(&self.indexer_comp_kv),
                        comp_gate: WeightBlob::F32(&self.indexer_comp_gate),
                        comp_norm: &self.indexer_comp_norm,
                    })
                } else {
                    None
                },
            }
        }
    }

    #[test]
    fn block_router_mismatch_rejected() {
        let storage = StubBlockStorage::new();
        let mut block = storage.block(0);
        block.topk_router = None;
        // layer_idx 10 expects top-k; this hash-only block must be rejected.
        let err = block.validate(10).unwrap_err();
        assert!(matches!(err, Error::ShapeMismatch { .. }));
    }

    #[test]
    fn block_passes_validation_at_correct_layer() {
        let storage = StubBlockStorage::new();
        let hash_block = storage.block(0);
        hash_block.validate(0).unwrap();
        let topk_block = storage.block(10);
        topk_block.validate(10).unwrap();
    }

    #[test]
    fn dense_layer_with_compressor_rejected() {
        // Layer 0 is dense — compressor must be `None`. If the GGUF
        // load path accidentally hands one in we want a loud failure.
        // We don't care which regime sizing the unwanted compressor
        // has; ratio-128 stubs are fine.
        let storage = StubBlockStorage::new();
        let mut block = storage.block(0);
        block.compressor = Some(CompressorWeights {
            kv: WeightBlob::F32(&storage.compressor_kv_r128),
            gate: WeightBlob::F32(&storage.compressor_gate_r128),
            ape: WeightBlob::F32(&storage.compressor_ape_r128),
            norm: &storage.compressor_norm,
        });
        let err = block.validate(0).unwrap_err();
        assert!(matches!(err, Error::ShapeMismatch { key, .. } if key == "block.compressor"));
    }

    #[test]
    fn compressed_layer_without_compressor_rejected() {
        // Layer 3 is ratio-128 — compressor required.
        let storage = StubBlockStorage::new();
        let mut block = storage.block(3);
        block.compressor = None;
        let err = block.validate(3).unwrap_err();
        assert!(matches!(err, Error::ShapeMismatch { key, .. } if key == "block.compressor"));
    }

    #[test]
    fn ratio4_layer_without_indexer_rejected() {
        // Layer 2 is the first ratio-4 layer; indexer bundle required.
        let storage = StubBlockStorage::new();
        let mut block = storage.block(2);
        block.indexer = None;
        let err = block.validate(2).unwrap_err();
        assert!(matches!(err, Error::ShapeMismatch { key, .. } if key == "block.indexer"));
    }

    #[test]
    fn ratio128_layer_with_indexer_rejected() {
        // Layer 3 is ratio-128; carrying an indexer bundle should error.
        let storage = StubBlockStorage::new();
        let mut block = storage.block(3);
        block.indexer = Some(IndexerWeights {
            attn_q_b: WeightBlob::F32(&storage.indexer_attn_q_b),
            proj: WeightBlob::F32(&storage.indexer_proj),
            comp_ape: WeightBlob::F32(&storage.indexer_comp_ape),
            comp_kv: WeightBlob::F32(&storage.indexer_comp_kv),
            comp_gate: WeightBlob::F32(&storage.indexer_comp_gate),
            comp_norm: &storage.indexer_comp_norm,
        });
        let err = block.validate(3).unwrap_err();
        assert!(matches!(err, Error::ShapeMismatch { key, .. } if key == "block.indexer"));
    }

    #[test]
    fn compressor_with_wrong_byte_len_rejected() {
        // Load-time byte-length check (security-review F008.B fix,
        // updated for F010.B's 4-tensor compressor). Build a ratio-128
        // layer (il=3, comp_width = HEAD_DIM = 512) whose `kv` backing
        // storage is one element short of the expected
        // comp_width × N_EMBD.
        let storage = StubBlockStorage::new();
        let wrong = vec![0.0_f32; DSV4_HEAD_DIM * DSV4_N_EMBD - 1];
        let mut block = storage.block(3);
        block.compressor = Some(CompressorWeights {
            kv: WeightBlob::F32(&wrong),
            gate: WeightBlob::F32(&storage.compressor_gate_r128),
            ape: WeightBlob::F32(&storage.compressor_ape_r128),
            norm: &storage.compressor_norm,
        });
        let err = block.validate(3).unwrap_err();
        assert!(matches!(
            err,
            Error::ShapeMismatch { key, .. } if key == "block.compressor.kv"
        ));
    }

    #[test]
    fn indexer_attn_q_b_with_wrong_byte_len_rejected() {
        // Verify the deep shape check on attn_q_b fires when the backing
        // storage is one element short of [N_LORA_Q × (N_INDEXER_HEAD × N_INDEXER_HEAD_DIM)].
        let storage = StubBlockStorage::new();
        let attn_q_b_lanes = crate::dsv4::shape::DSV4_N_LORA_Q
            * crate::dsv4::shape::DSV4_N_INDEXER_HEAD
            * crate::dsv4::shape::DSV4_N_INDEXER_HEAD_DIM;
        let wrong_attn_q_b = vec![0.0_f32; attn_q_b_lanes - 1];
        let mut block = storage.block(2);
        block.indexer = Some(IndexerWeights {
            attn_q_b: WeightBlob::F32(&wrong_attn_q_b),
            proj: WeightBlob::F32(&storage.indexer_proj),
            comp_ape: WeightBlob::F32(&storage.indexer_comp_ape),
            comp_kv: WeightBlob::F32(&storage.indexer_comp_kv),
            comp_gate: WeightBlob::F32(&storage.indexer_comp_gate),
            comp_norm: &storage.indexer_comp_norm,
        });
        let err = block.validate(2).unwrap_err();
        assert!(matches!(
            err,
            Error::ShapeMismatch { key, .. } if key == "block.indexer.attn_q_b"
        ));
    }

    /// Build a minimal `OutputHcWeights` for tests. mix_fn is sized
    /// for the F16 [HC_DIM × N_HC] tensor (`16384 × 4 × 2 bytes`).
    fn stub_output_hc<'a>(
        oh_fn_bytes: &'a [u8],
        oh_scale: &'a [f32],
        oh_base: &'a [f32],
    ) -> OutputHcWeights<'a> {
        OutputHcWeights {
            mix_fn: WeightBlob::Quant {
                data: oh_fn_bytes,
                dtype: rsllm_gguf::GgmlType::F16,
            },
            scale: oh_scale,
            base: oh_base,
        }
    }

    fn stub_output_hc_storage() -> (Vec<u8>, Vec<f32>, Vec<f32>) {
        let hc_dim = crate::dsv4::hc::HC_DIM;
        // F16 = 2 bytes/element
        (vec![0u8; hc_dim * N_HC * 2], vec![1.0_f32; 1], vec![0.0_f32; N_HC])
    }

    #[test]
    fn model_requires_all_n_layer_blocks() {
        let storage = StubBlockStorage::new();
        let blocks: Vec<_> = (0..5).map(|i| storage.block(i)).collect();
        let lm_head_bytes: Vec<u8> = vec![];
        let embed_bytes: Vec<u8> = vec![];
        let norm = vec![1.0_f32; DSV4_N_EMBD];
        let (oh_fn, oh_scale, oh_base) = stub_output_hc_storage();
        let oh = stub_output_hc(&oh_fn, &oh_scale, &oh_base);
        let err = DeepSeekV4Flash::new(
            WeightBlob::Quant {
                data: &embed_bytes,
                dtype: rsllm_gguf::GgmlType::F16,
            },
            blocks,
            oh,
            &norm,
            WeightBlob::Quant {
                data: &lm_head_bytes,
                dtype: rsllm_gguf::GgmlType::F16,
            },
        )
        .unwrap_err();
        assert!(matches!(err, Error::ShapeMismatch { .. }));
    }

    #[test]
    fn model_rejects_wrong_attn_sinks_dim() {
        let mut storage = StubBlockStorage::new();
        storage.attn_sinks = vec![0.0_f32; 10]; // wrong: should be N_HEAD = 64
        let blocks: Vec<_> = (0..DSV4_N_LAYER).map(|i| storage.block(i)).collect();
        let norm = vec![1.0_f32; DSV4_N_EMBD];
        let bytes: Vec<u8> = vec![];
        let (oh_fn, oh_scale, oh_base) = stub_output_hc_storage();
        let oh = stub_output_hc(&oh_fn, &oh_scale, &oh_base);
        let err = DeepSeekV4Flash::new(
            WeightBlob::Quant {
                data: &bytes,
                dtype: rsllm_gguf::GgmlType::F16,
            },
            blocks,
            oh,
            &norm,
            WeightBlob::Quant {
                data: &bytes,
                dtype: rsllm_gguf::GgmlType::F16,
            },
        )
        .unwrap_err();
        match err {
            Error::ShapeMismatch { key, .. } => assert_eq!(key, "block.attn_sinks"),
            other => panic!("expected ShapeMismatch.attn_sinks, got {other:?}"),
        }
    }

    #[test]
    fn model_rejects_wrong_output_norm_dim() {
        let storage = StubBlockStorage::new();
        let blocks: Vec<_> = (0..DSV4_N_LAYER).map(|i| storage.block(i)).collect();
        let norm = vec![1.0_f32; 10]; // wrong
        let bytes: Vec<u8> = vec![];
        let (oh_fn, oh_scale, oh_base) = stub_output_hc_storage();
        let oh = stub_output_hc(&oh_fn, &oh_scale, &oh_base);
        let err = DeepSeekV4Flash::new(
            WeightBlob::Quant {
                data: &bytes,
                dtype: rsllm_gguf::GgmlType::F16,
            },
            blocks,
            oh,
            &norm,
            WeightBlob::Quant {
                data: &bytes,
                dtype: rsllm_gguf::GgmlType::F16,
            },
        )
        .unwrap_err();
        assert!(matches!(err, Error::ShapeMismatch { .. }));
    }
}
