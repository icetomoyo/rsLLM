//! Fixed shape constants for DeepSeek V4 Flash.
//!
//! Every value here is **hard-coded** from `ds4.c:85-108` (MIT, The ds4.c
//! authors). The DS V4 Flash architecture is not a tunable family: there
//! is only one set of dimensions, and we fail-fast at GGUF load time on
//! any mismatch. This matches ds4's tensor-layout validation behavior
//! (`tensor_expect_layout` family at `ds4.c:2291-2351`) and is the
//! cheapest way to keep our forward path arithmetic free of
//! dynamic-shape branching.
//!
//! Ported by reference from `ds4.c` (MIT, The ds4.c authors).
//! Line numbers in this file (and across rsllm-models) are pinned to
//! ds4 commit `ef0a490` (2026-05-17). See `docs/research/ds4-analysis.md`
//! §"ds4.c 行号引用基线" for the mapping table; re-audit on upstream
//! bumps via `git -C path/to/ds4 grep -n <anchor>`.

use rsllm_gguf::{Array, Metadata, Value};

use crate::Error;

/// Number of transformer blocks.
pub const DSV4_N_LAYER: usize = 43;

/// Embedding dimension (per-token residual stream width).
pub const DSV4_N_EMBD: usize = 4096;

/// Vocabulary size.
pub const DSV4_N_VOCAB: usize = 129_280;

/// Attention head count (Q heads).
pub const DSV4_N_HEAD: usize = 64;

/// KV head count. DS V4 Flash uses MLA: KV is compressed to 1 head and
/// projected to a per-head latent, so `n_head_kv == 1` at the GGUF level.
pub const DSV4_N_HEAD_KV: usize = 1;

/// Per-head Q/K latent dimension (after LoRA up-projection).
pub const DSV4_HEAD_DIM: usize = 512;

/// Per-head V latent dimension. Same value as [`DSV4_HEAD_DIM`] for
/// DS V4 Flash, but the upstream metadata exposes the two independently
/// (`attention.key_length` and `attention.value_length`) so we validate
/// both. Ref: `ds4.c:93`.
pub const DSV4_N_VALUE_DIM: usize = 512;

/// RoPE-rotated tail length per Q/K head. Only the last `N_ROT` lanes
/// of each `HEAD_DIM` head get the YaRN-scaled RoPE rotation.
pub const DSV4_N_ROT: usize = 64;

/// LoRA Q down-projection rank (the bottleneck width on the Q path).
pub const DSV4_N_LORA_Q: usize = 1024;

/// LoRA output-projection rank (`attn_o`'s down-then-up bottleneck).
pub const DSV4_N_LORA_O: usize = 1024;

/// Number of MoE routed experts per layer.
pub const DSV4_N_EXPERT: usize = 256;

/// Number of routed experts activated per token (top-k MoE).
pub const DSV4_N_EXPERT_USED: usize = 6;

/// Number of shared (always-on) experts per layer. Always 1 for DS V4
/// Flash; encoded as a constant rather than a literal so the
/// shared-expert apply path is self-documenting.
pub const DSV4_N_EXPERT_SHARED: usize = 1;

/// Group count for the attention output LoRA projection. Each layer's
/// post-attention projection is grouped into `N_OUT_GROUP` chunks of
/// `HEAD_DIM * (N_HEAD / N_OUT_GROUP) = 4096`-dim input, each projected
/// to a per-group `N_LORA_O`-rank latent before a final dense up.
/// Ref: `ds4.c:94`.
pub const DSV4_N_OUT_GROUP: usize = 8;

/// Hyper-connection residual stream count.
pub const DSV4_N_HC: usize = 4;

/// Sliding-window attention ring size (most recent N tokens are kept
/// in raw form by the three-tier KV cache, [`rsllm_kvcache`]).
pub const DSV4_N_SWA: usize = 128;

/// Top-k for the ratio-4 sparse KV indexer (used by F006).
pub const DSV4_N_INDEXER_TOP_K: usize = 512;

/// Number of indexer heads (used by F006). Each indexer head is
/// `N_INDEXER_HEAD_DIM`-wide. Ref: `ds4.c:104`.
pub const DSV4_N_INDEXER_HEAD: usize = 64;

/// Per-head dimension for the indexer. Ref: `ds4.c:105`.
pub const DSV4_N_INDEXER_HEAD_DIM: usize = 128;

/// Sinkhorn-Knopp iteration count for the HC split. Upstream pins this
/// at 20 (`ds4.c:108`) and bakes it into every Sinkhorn call site
/// (`ds4.c:4310`, `9057`, `11119`, ...). We validate the GGUF carries
/// the same value so a mis-trained checkpoint can't silently drift the
/// projection toward a different fixed point.
pub const DSV4_N_HC_SINKHORN_ITER: u32 = 20;

/// RoPE frequency base for the standard MLA attention rotation.
/// Ref: `ds4.c:56` (`DS4_ROPE_FREQ_BASE = 10000.0f`). The 160_000.0
/// value used elsewhere is the *compress*-path RoPE base, see
/// [`DSV4_COMPRESS_ROPE_FREQ_BASE`].
pub const DSV4_ROPE_FREQ_BASE: f32 = 10_000.0;

/// RoPE frequency base used by the indexer/compressor path.
/// Ref: `ds4.c:60` (`DS4_COMPRESS_ROPE_FREQ_BASE = 160000.0f`).
pub const DSV4_COMPRESS_ROPE_FREQ_BASE: f32 = 160_000.0;

/// YaRN linear-extrapolation scale factor. Ref: `ds4.c:57`.
pub const DSV4_ROPE_SCALE_FACTOR: f32 = 16.0;

/// YaRN ramp upper bound (fast end). Ref: `ds4.c:58`.
pub const DSV4_ROPE_YARN_BETA_FAST: f32 = 32.0;

/// YaRN ramp lower bound (slow end). Ref: `ds4.c:59`.
pub const DSV4_ROPE_YARN_BETA_SLOW: f32 = 1.0;

/// Original (un-extrapolated) context length the model was trained on,
/// used by YaRN to compute the ramp. Ref: `ds4.c:61`
/// (`DS4_ROPE_ORIG_CTX = 65536`). Encoded as `u64` in GGUF, matching
/// upstream `required_u64` (`ds4.c:2549`).
pub const DSV4_ROPE_ORIG_CTX: u64 = 65_536;

/// MoE expert-weights scale applied to the routed mixture pre-softmax
/// in the post-hash regime. Ref: `ds4.c:54`.
pub const DSV4_EXPERT_WEIGHT_SCALE: f32 = 1.5;

/// RMSNorm epsilon. Same value at every layer.
pub const DSV4_RMS_EPS: f32 = 1e-6;

/// HC RMSNorm epsilon. Numerically identical to [`DSV4_RMS_EPS`] at
/// present but exposed as a separate metadata key by upstream so we
/// validate it independently. Ref: `ds4.c:53`.
pub const DSV4_HC_EPS: f32 = 1e-6;

/// SwiGLU clamp exponent. Bounds the FFN activation magnitude before the
/// SiLU gate, preventing numerical blow-up for outlier activations. Same
/// value at every layer. Ref: `ds4.c:55`.
pub const DSV4_SWIGLU_CLAMP_EXP: f32 = 10.0;

/// Number of leading layers that use **hash routing** for MoE instead
/// of top-k softmax routing (`ds4.c:5002-5050`). Layers `[0, 3)` use
/// the `ffn_gate_tid2eid` lookup table indexed by token id.
pub const DSV4_HASH_ROUTE_LAYERS: usize = 3;

/// Expected GGUF `general.architecture` value. Matches the upstream
/// metadata-key prefix (`deepseek4.*`).
pub const DSV4_ARCH_NAME: &str = "deepseek4";

/// FFN hidden-dim per *expert* in the MoE. Each layer has 256 routed
/// experts plus one shared expert; this is the per-expert FFN width.
/// The shared expert uses the same FFN width per ds4.
pub const DSV4_N_FF_EXP: usize = 2048;

/// Validate that a GGUF metadata block describes a DeepSeek V4 Flash
/// model with **exactly** the shapes we hard-code above.
///
/// This is intentionally strict: any disagreement on a load-bearing
/// dimension is a fail-fast `Error::ShapeMismatch`. The reasoning,
/// from ds4's design notes, is that a wrong shape would not crash
/// loudly — it would silently feed bad activations through every
/// subsequent kernel, often producing plausible-looking but corrupt
/// output. Better to refuse early.
///
/// Metadata key names follow upstream `ds4.c:2493-2573` (commit
/// `ef0a490`). All keys share the `deepseek4.*` prefix.
///
/// # Errors
/// Returns [`Error::MissingMetadata`] if a required key is absent and
/// [`Error::ShapeMismatch`] if any present value disagrees with the
/// constants above.
pub fn validate_metadata(meta: &Metadata) -> Result<(), Error> {
    // 1. Architecture name.
    let arch = meta
        .get_str("general.architecture")
        .ok_or(Error::MissingMetadata("general.architecture"))?;
    if arch != DSV4_ARCH_NAME {
        return Err(Error::ShapeMismatch {
            key: "general.architecture",
            expected: DSV4_ARCH_NAME.to_string(),
            actual: arch.to_string(),
        });
    }

    // 2. Layer count.
    expect_u32(meta, "deepseek4.block_count", DSV4_N_LAYER as u32)?;

    // 3. Embedding dim.
    expect_u32(meta, "deepseek4.embedding_length", DSV4_N_EMBD as u32)?;

    // 4. Vocab size.
    expect_u32(meta, "deepseek4.vocab_size", DSV4_N_VOCAB as u32)?;

    // 5. Head counts.
    expect_u32(meta, "deepseek4.attention.head_count", DSV4_N_HEAD as u32)?;
    expect_u32(
        meta,
        "deepseek4.attention.head_count_kv",
        DSV4_N_HEAD_KV as u32,
    )?;

    // 6. Per-head latent dims (K and V exposed separately upstream).
    expect_u32(
        meta,
        "deepseek4.attention.key_length",
        DSV4_HEAD_DIM as u32,
    )?;
    expect_u32(
        meta,
        "deepseek4.attention.value_length",
        DSV4_N_VALUE_DIM as u32,
    )?;
    expect_u32(meta, "deepseek4.rope.dimension_count", DSV4_N_ROT as u32)?;

    // 7. LoRA ranks.
    expect_u32(
        meta,
        "deepseek4.attention.q_lora_rank",
        DSV4_N_LORA_Q as u32,
    )?;
    expect_u32(
        meta,
        "deepseek4.attention.output_lora_rank",
        DSV4_N_LORA_O as u32,
    )?;
    expect_u32(
        meta,
        "deepseek4.attention.output_group_count",
        DSV4_N_OUT_GROUP as u32,
    )?;

    // 8. MoE shape.
    expect_u32(meta, "deepseek4.expert_count", DSV4_N_EXPERT as u32)?;
    expect_u32(
        meta,
        "deepseek4.expert_used_count",
        DSV4_N_EXPERT_USED as u32,
    )?;
    expect_u32(
        meta,
        "deepseek4.expert_feed_forward_length",
        DSV4_N_FF_EXP as u32,
    )?;
    expect_u32(
        meta,
        "deepseek4.expert_shared_count",
        DSV4_N_EXPERT_SHARED as u32,
    )?;
    expect_u32(
        meta,
        "deepseek4.hash_layer_count",
        DSV4_HASH_ROUTE_LAYERS as u32,
    )?;

    // 9. Sliding window.
    expect_u32(
        meta,
        "deepseek4.attention.sliding_window",
        DSV4_N_SWA as u32,
    )?;

    // 10. Indexer (`ds4.c:2533-2537`); used by F006 KV cache.
    expect_u32(
        meta,
        "deepseek4.attention.indexer.head_count",
        DSV4_N_INDEXER_HEAD as u32,
    )?;
    expect_u32(
        meta,
        "deepseek4.attention.indexer.key_length",
        DSV4_N_INDEXER_HEAD_DIM as u32,
    )?;
    expect_u32(
        meta,
        "deepseek4.attention.indexer.top_k",
        DSV4_N_INDEXER_TOP_K as u32,
    )?;

    // 11. Hyper-connection (`ds4.c:2539-2542`).
    expect_u32(meta, "deepseek4.hyper_connection.count", DSV4_N_HC as u32)?;
    expect_u32(
        meta,
        "deepseek4.hyper_connection.sinkhorn_iterations",
        DSV4_N_HC_SINKHORN_ITER,
    )?;

    // 12. RoPE / YaRN (`ds4.c:2549-2562`).
    expect_u64(
        meta,
        "deepseek4.rope.scaling.original_context_length",
        DSV4_ROPE_ORIG_CTX,
    )?;
    expect_f32_close(meta, "deepseek4.rope.freq_base", DSV4_ROPE_FREQ_BASE)?;
    expect_f32_close(
        meta,
        "deepseek4.rope.scaling.factor",
        DSV4_ROPE_SCALE_FACTOR,
    )?;
    expect_f32_close(
        meta,
        "deepseek4.rope.scaling.yarn_beta_fast",
        DSV4_ROPE_YARN_BETA_FAST,
    )?;
    expect_f32_close(
        meta,
        "deepseek4.rope.scaling.yarn_beta_slow",
        DSV4_ROPE_YARN_BETA_SLOW,
    )?;
    expect_f32_close(
        meta,
        "deepseek4.attention.compress_rope_freq_base",
        DSV4_COMPRESS_ROPE_FREQ_BASE,
    )?;

    // 13. MoE expert-weights scaling and norm flag (`ds4.c:2566-2572`).
    expect_f32_close(
        meta,
        "deepseek4.expert_weights_scale",
        DSV4_EXPERT_WEIGHT_SCALE,
    )?;
    expect_bool(meta, "deepseek4.expert_weights_norm", true)?;

    // 14. RMS / HC epsilons (`ds4.c:2568, 2570`).
    expect_f32_close(
        meta,
        "deepseek4.attention.layer_norm_rms_epsilon",
        DSV4_RMS_EPS,
    )?;
    expect_f32_close(meta, "deepseek4.hyper_connection.epsilon", DSV4_HC_EPS)?;

    // 15. Per-layer compress ratios. Upstream `validate_compress_ratio_metadata`
    // (`ds4.c:2401-2434`) requires a u32/i32 array of length >= N_LAYER whose
    // values match `layer_compress_ratio(il)` for every layer. A mismatch would
    // silently route the wrong layers through the indexer / ratio-128 paths.
    expect_compress_ratios(meta)?;

    // 16. Per-layer SwiGLU clamp exponents (`ds4.c:2436-2462`). Same shape
    // requirement; every layer must use `DSV4_SWIGLU_CLAMP_EXP`. A drifted
    // value would change the FFN activation bound without warning.
    expect_swiglu_clamp(meta)?;

    // 17. Optional expert-group keys (`ds4.c:2511-2529`). DS V4 Flash never
    // uses expert groups: both `expert_group_count` and
    // `expert_group_used_count` must be 0 if present. Missing keys are OK
    // (upstream `model_get_u32` leaves the value at its 0 init), so absence
    // is treated as 0.
    expect_optional_u32_zero(meta, "deepseek4.expert_group_count")?;
    expect_optional_u32_zero(meta, "deepseek4.expert_group_used_count")?;

    Ok(())
}

fn expect_u32(meta: &Metadata, key: &'static str, want: u32) -> Result<(), Error> {
    let got = meta.get_u32(key).ok_or(Error::MissingMetadata(key))?;
    if got == want {
        Ok(())
    } else {
        Err(Error::ShapeMismatch {
            key,
            expected: want.to_string(),
            actual: got.to_string(),
        })
    }
}

fn expect_u64(meta: &Metadata, key: &'static str, want: u64) -> Result<(), Error> {
    let got = meta.get_u64(key).ok_or(Error::MissingMetadata(key))?;
    if got == want {
        Ok(())
    } else {
        Err(Error::ShapeMismatch {
            key,
            expected: want.to_string(),
            actual: got.to_string(),
        })
    }
}

fn expect_bool(meta: &Metadata, key: &'static str, want: bool) -> Result<(), Error> {
    let got = meta.get_bool(key).ok_or(Error::MissingMetadata(key))?;
    if got == want {
        Ok(())
    } else {
        Err(Error::ShapeMismatch {
            key,
            expected: want.to_string(),
            actual: got.to_string(),
        })
    }
}

fn expect_optional_u32_zero(meta: &Metadata, key: &'static str) -> Result<(), Error> {
    match meta.get(key) {
        None => Ok(()),
        Some(v) => match v.as_u32() {
            Some(0) => Ok(()),
            Some(got) => Err(Error::ShapeMismatch {
                key,
                expected: "0 (or absent)".to_string(),
                actual: got.to_string(),
            }),
            None => Err(Error::ShapeMismatch {
                key,
                expected: "u32 or absent".to_string(),
                actual: format!("{:?}", v.ty()),
            }),
        },
    }
}

/// Validate `deepseek4.attention.compress_ratios` matches the per-layer
/// fixed schedule (`ds4.c:411-416`). Accepts both U32 and I32 element
/// arrays for parity with upstream. Length must be at least N_LAYER;
/// surplus tail entries are ignored exactly as upstream does.
fn expect_compress_ratios(meta: &Metadata) -> Result<(), Error> {
    const KEY: &str = "deepseek4.attention.compress_ratios";
    let value = meta.get(KEY).ok_or(Error::MissingMetadata(KEY))?;
    let arr = match value {
        Value::Array(a) => a,
        other => {
            return Err(Error::ShapeMismatch {
                key: KEY,
                expected: "Array(U32|I32)".to_string(),
                actual: format!("{:?}", other.ty()),
            });
        }
    };
    let len = arr.len();
    if len < DSV4_N_LAYER {
        return Err(Error::ShapeMismatch {
            key: KEY,
            expected: format!(">= {DSV4_N_LAYER}"),
            actual: len.to_string(),
        });
    }
    for il in 0..DSV4_N_LAYER {
        let got = match arr {
            Array::U32(v) => v[il],
            Array::I32(v) => {
                let raw = v[il];
                if raw < 0 {
                    return Err(Error::ShapeMismatch {
                        key: KEY,
                        expected: format!("non-negative @ layer {il}"),
                        actual: raw.to_string(),
                    });
                }
                raw as u32
            }
            _ => {
                return Err(Error::ShapeMismatch {
                    key: KEY,
                    expected: "Array(U32|I32)".to_string(),
                    actual: format!("Array({:?})", arr.item_type()),
                });
            }
        };
        let want = rsllm_kvcache::dsv4::shape::layer_compress_ratio(il);
        if got != want {
            return Err(Error::ShapeMismatch {
                key: KEY,
                expected: format!("{want} @ layer {il}"),
                actual: got.to_string(),
            });
        }
    }
    Ok(())
}

/// Validate `deepseek4.swiglu_clamp_exp` is an F32 (or F64) array of
/// length >= N_LAYER, with every entry close to [`DSV4_SWIGLU_CLAMP_EXP`].
/// Ref: `ds4.c:2436-2462`.
fn expect_swiglu_clamp(meta: &Metadata) -> Result<(), Error> {
    const KEY: &str = "deepseek4.swiglu_clamp_exp";
    let value = meta.get(KEY).ok_or(Error::MissingMetadata(KEY))?;
    let arr = match value {
        Value::Array(a) => a,
        other => {
            return Err(Error::ShapeMismatch {
                key: KEY,
                expected: "Array(F32|F64)".to_string(),
                actual: format!("{:?}", other.ty()),
            });
        }
    };
    let len = arr.len();
    if len < DSV4_N_LAYER {
        return Err(Error::ShapeMismatch {
            key: KEY,
            expected: format!(">= {DSV4_N_LAYER}"),
            actual: len.to_string(),
        });
    }
    for il in 0..DSV4_N_LAYER {
        let got: f32 = match arr {
            Array::F32(v) => v[il],
            Array::F64(v) => v[il] as f32,
            _ => {
                return Err(Error::ShapeMismatch {
                    key: KEY,
                    expected: "Array(F32|F64)".to_string(),
                    actual: format!("Array({:?})", arr.item_type()),
                });
            }
        };
        let want = DSV4_SWIGLU_CLAMP_EXP;
        let denom = want.abs().max(1.0);
        if (got - want).abs() / denom >= 1e-3 {
            return Err(Error::ShapeMismatch {
                key: KEY,
                expected: format!("{want} @ layer {il}"),
                actual: format!("{got}"),
            });
        }
    }
    Ok(())
}

fn expect_f32_close(meta: &Metadata, key: &'static str, want: f32) -> Result<(), Error> {
    let got = meta.get_f32(key).ok_or(Error::MissingMetadata(key))?;
    // Floats encoded by trainers can drift in the last few ULPs.
    // 1e-3 relative tolerance is more than enough for shape sanity:
    // ds4 also accepts approximate matches for freq_base / eps.
    let denom = want.abs().max(1.0);
    if (got - want).abs() / denom < 1e-3 {
        Ok(())
    } else {
        Err(Error::ShapeMismatch {
            key,
            expected: format!("{want:e}"),
            actual: format!("{got:e}"),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rsllm_gguf::Value;

    fn good_meta() -> Metadata {
        let mut m = Metadata::new();
        m.insert(
            "general.architecture",
            Value::String(DSV4_ARCH_NAME.to_string()),
        );
        m.insert("deepseek4.block_count", Value::U32(DSV4_N_LAYER as u32));
        m.insert("deepseek4.embedding_length", Value::U32(DSV4_N_EMBD as u32));
        m.insert("deepseek4.vocab_size", Value::U32(DSV4_N_VOCAB as u32));
        m.insert(
            "deepseek4.attention.head_count",
            Value::U32(DSV4_N_HEAD as u32),
        );
        m.insert(
            "deepseek4.attention.head_count_kv",
            Value::U32(DSV4_N_HEAD_KV as u32),
        );
        m.insert(
            "deepseek4.attention.key_length",
            Value::U32(DSV4_HEAD_DIM as u32),
        );
        m.insert(
            "deepseek4.attention.value_length",
            Value::U32(DSV4_N_VALUE_DIM as u32),
        );
        m.insert(
            "deepseek4.rope.dimension_count",
            Value::U32(DSV4_N_ROT as u32),
        );
        m.insert(
            "deepseek4.attention.q_lora_rank",
            Value::U32(DSV4_N_LORA_Q as u32),
        );
        m.insert(
            "deepseek4.attention.output_lora_rank",
            Value::U32(DSV4_N_LORA_O as u32),
        );
        m.insert(
            "deepseek4.attention.output_group_count",
            Value::U32(DSV4_N_OUT_GROUP as u32),
        );
        m.insert("deepseek4.expert_count", Value::U32(DSV4_N_EXPERT as u32));
        m.insert(
            "deepseek4.expert_used_count",
            Value::U32(DSV4_N_EXPERT_USED as u32),
        );
        m.insert(
            "deepseek4.expert_feed_forward_length",
            Value::U32(DSV4_N_FF_EXP as u32),
        );
        m.insert(
            "deepseek4.expert_shared_count",
            Value::U32(DSV4_N_EXPERT_SHARED as u32),
        );
        m.insert(
            "deepseek4.hash_layer_count",
            Value::U32(DSV4_HASH_ROUTE_LAYERS as u32),
        );
        m.insert(
            "deepseek4.attention.sliding_window",
            Value::U32(DSV4_N_SWA as u32),
        );
        m.insert(
            "deepseek4.attention.indexer.head_count",
            Value::U32(DSV4_N_INDEXER_HEAD as u32),
        );
        m.insert(
            "deepseek4.attention.indexer.key_length",
            Value::U32(DSV4_N_INDEXER_HEAD_DIM as u32),
        );
        m.insert(
            "deepseek4.attention.indexer.top_k",
            Value::U32(DSV4_N_INDEXER_TOP_K as u32),
        );
        m.insert("deepseek4.hyper_connection.count", Value::U32(DSV4_N_HC as u32));
        m.insert(
            "deepseek4.hyper_connection.sinkhorn_iterations",
            Value::U32(DSV4_N_HC_SINKHORN_ITER),
        );
        m.insert(
            "deepseek4.rope.scaling.original_context_length",
            Value::U64(DSV4_ROPE_ORIG_CTX),
        );
        m.insert(
            "deepseek4.rope.freq_base",
            Value::F32(DSV4_ROPE_FREQ_BASE),
        );
        m.insert(
            "deepseek4.rope.scaling.factor",
            Value::F32(DSV4_ROPE_SCALE_FACTOR),
        );
        m.insert(
            "deepseek4.rope.scaling.yarn_beta_fast",
            Value::F32(DSV4_ROPE_YARN_BETA_FAST),
        );
        m.insert(
            "deepseek4.rope.scaling.yarn_beta_slow",
            Value::F32(DSV4_ROPE_YARN_BETA_SLOW),
        );
        m.insert(
            "deepseek4.attention.compress_rope_freq_base",
            Value::F32(DSV4_COMPRESS_ROPE_FREQ_BASE),
        );
        m.insert(
            "deepseek4.expert_weights_scale",
            Value::F32(DSV4_EXPERT_WEIGHT_SCALE),
        );
        m.insert("deepseek4.expert_weights_norm", Value::Bool(true));
        m.insert(
            "deepseek4.attention.layer_norm_rms_epsilon",
            Value::F32(DSV4_RMS_EPS),
        );
        m.insert(
            "deepseek4.hyper_connection.epsilon",
            Value::F32(DSV4_HC_EPS),
        );
        // Per-layer arrays — populated to match the upstream schedule.
        let ratios: Vec<u32> = (0..DSV4_N_LAYER)
            .map(rsllm_kvcache::dsv4::shape::layer_compress_ratio)
            .collect();
        m.insert(
            "deepseek4.attention.compress_ratios",
            Value::Array(Array::U32(ratios)),
        );
        let clamps: Vec<f32> = vec![DSV4_SWIGLU_CLAMP_EXP; DSV4_N_LAYER];
        m.insert(
            "deepseek4.swiglu_clamp_exp",
            Value::Array(Array::F32(clamps)),
        );
        m
    }

    #[test]
    fn accepts_canonical_metadata() {
        validate_metadata(&good_meta()).unwrap();
    }

    #[test]
    fn rejects_wrong_arch() {
        let mut m = good_meta();
        m.insert("general.architecture", Value::String("llama".to_string()));
        let err = validate_metadata(&m).unwrap_err();
        assert!(matches!(err, Error::ShapeMismatch { .. }));
    }

    #[test]
    fn rejects_legacy_arch_prefix() {
        // Catches accidental regression to the old `deepseek-v4-flash`
        // architecture string we used before the F010.E prefix migration.
        let mut m = good_meta();
        m.insert(
            "general.architecture",
            Value::String("deepseek-v4-flash".to_string()),
        );
        let err = validate_metadata(&m).unwrap_err();
        assert!(matches!(err, Error::ShapeMismatch { .. }));
    }

    #[test]
    fn rejects_wrong_layer_count() {
        let mut m = good_meta();
        m.insert("deepseek4.block_count", Value::U32(42));
        let err = validate_metadata(&m).unwrap_err();
        match err {
            Error::ShapeMismatch { key, .. } => assert_eq!(key, "deepseek4.block_count"),
            other => panic!("expected ShapeMismatch, got {other:?}"),
        }
    }

    #[test]
    fn rejects_missing_required_key() {
        let mut m = good_meta();
        // `Metadata` exposes no `remove`, so we type-smuggle a String into
        // a numeric key. `get_u32` returns `None` on a String value, which
        // surfaces as `MissingMetadata` — the same code path a truly-absent
        // key would take. If `Metadata` ever gains `remove`, prefer that.
        m.insert(
            "deepseek4.embedding_length",
            Value::String("oops".to_string()),
        );
        let err = validate_metadata(&m).unwrap_err();
        assert!(matches!(err, Error::MissingMetadata(_)));
    }

    #[test]
    fn accepts_rope_base_within_tolerance() {
        let mut m = good_meta();
        // 1e-4 relative drift on a 10000 base = 1, well under the 1e-3 gate.
        m.insert(
            "deepseek4.rope.freq_base",
            Value::F32(DSV4_ROPE_FREQ_BASE + 1.0),
        );
        validate_metadata(&m).unwrap();
    }

    #[test]
    fn rejects_rope_base_far_off() {
        let mut m = good_meta();
        m.insert(
            "deepseek4.rope.freq_base",
            Value::F32(160_000.0), // Compress-path base, way off from standard MLA base.
        );
        let err = validate_metadata(&m).unwrap_err();
        assert!(matches!(err, Error::ShapeMismatch { .. }));
    }

    #[test]
    fn rejects_wrong_sinkhorn_iterations() {
        let mut m = good_meta();
        m.insert(
            "deepseek4.hyper_connection.sinkhorn_iterations",
            Value::U32(15),
        );
        let err = validate_metadata(&m).unwrap_err();
        match err {
            Error::ShapeMismatch { key, .. } => {
                assert_eq!(key, "deepseek4.hyper_connection.sinkhorn_iterations");
            }
            other => panic!("expected ShapeMismatch, got {other:?}"),
        }
    }

    #[test]
    fn rejects_wrong_value_length() {
        let mut m = good_meta();
        m.insert("deepseek4.attention.value_length", Value::U32(256));
        let err = validate_metadata(&m).unwrap_err();
        match err {
            Error::ShapeMismatch { key, .. } => {
                assert_eq!(key, "deepseek4.attention.value_length");
            }
            other => panic!("expected ShapeMismatch, got {other:?}"),
        }
    }

    #[test]
    fn rejects_wrong_expert_weights_norm() {
        let mut m = good_meta();
        m.insert("deepseek4.expert_weights_norm", Value::Bool(false));
        let err = validate_metadata(&m).unwrap_err();
        match err {
            Error::ShapeMismatch { key, .. } => {
                assert_eq!(key, "deepseek4.expert_weights_norm");
            }
            other => panic!("expected ShapeMismatch, got {other:?}"),
        }
    }

    #[test]
    fn rejects_wrong_rope_orig_ctx() {
        let mut m = good_meta();
        m.insert(
            "deepseek4.rope.scaling.original_context_length",
            Value::U64(32_768),
        );
        let err = validate_metadata(&m).unwrap_err();
        match err {
            Error::ShapeMismatch { key, .. } => {
                assert_eq!(key, "deepseek4.rope.scaling.original_context_length");
            }
            other => panic!("expected ShapeMismatch, got {other:?}"),
        }
    }

    #[test]
    fn rejects_missing_compress_ratios_array() {
        let mut m = good_meta();
        m.insert(
            "deepseek4.attention.compress_ratios",
            Value::String("nope".to_string()),
        );
        let err = validate_metadata(&m).unwrap_err();
        match err {
            Error::ShapeMismatch { key, .. } => {
                assert_eq!(key, "deepseek4.attention.compress_ratios");
            }
            other => panic!("expected ShapeMismatch, got {other:?}"),
        }
    }

    #[test]
    fn rejects_short_compress_ratios_array() {
        let mut m = good_meta();
        m.insert(
            "deepseek4.attention.compress_ratios",
            Value::Array(Array::U32(vec![0, 0, 4])),
        );
        let err = validate_metadata(&m).unwrap_err();
        assert!(matches!(err, Error::ShapeMismatch { .. }));
    }

    #[test]
    fn rejects_wrong_compress_ratio_value() {
        let mut m = good_meta();
        // Swap the layer-2 ratio (must be 4) for an unsupported 7.
        let mut bad: Vec<u32> = (0..DSV4_N_LAYER)
            .map(rsllm_kvcache::dsv4::shape::layer_compress_ratio)
            .collect();
        bad[2] = 7;
        m.insert(
            "deepseek4.attention.compress_ratios",
            Value::Array(Array::U32(bad)),
        );
        let err = validate_metadata(&m).unwrap_err();
        match err {
            Error::ShapeMismatch { key, .. } => {
                assert_eq!(key, "deepseek4.attention.compress_ratios");
            }
            other => panic!("expected ShapeMismatch, got {other:?}"),
        }
    }

    #[test]
    fn accepts_compress_ratios_as_i32_array() {
        let mut m = good_meta();
        let ratios: Vec<i32> = (0..DSV4_N_LAYER)
            .map(|il| rsllm_kvcache::dsv4::shape::layer_compress_ratio(il) as i32)
            .collect();
        m.insert(
            "deepseek4.attention.compress_ratios",
            Value::Array(Array::I32(ratios)),
        );
        validate_metadata(&m).unwrap();
    }

    #[test]
    fn rejects_wrong_swiglu_clamp_value() {
        let mut m = good_meta();
        let mut bad = vec![DSV4_SWIGLU_CLAMP_EXP; DSV4_N_LAYER];
        bad[7] = 5.0;
        m.insert("deepseek4.swiglu_clamp_exp", Value::Array(Array::F32(bad)));
        let err = validate_metadata(&m).unwrap_err();
        match err {
            Error::ShapeMismatch { key, .. } => assert_eq!(key, "deepseek4.swiglu_clamp_exp"),
            other => panic!("expected ShapeMismatch, got {other:?}"),
        }
    }

    #[test]
    fn accepts_swiglu_clamp_as_f64_array() {
        let mut m = good_meta();
        let clamps = vec![DSV4_SWIGLU_CLAMP_EXP as f64; DSV4_N_LAYER];
        m.insert(
            "deepseek4.swiglu_clamp_exp",
            Value::Array(Array::F64(clamps)),
        );
        validate_metadata(&m).unwrap();
    }

    #[test]
    fn accepts_missing_optional_expert_group_keys() {
        // good_meta() never inserts the optional keys; absence is fine.
        validate_metadata(&good_meta()).unwrap();
    }

    #[test]
    fn rejects_nonzero_expert_group_count() {
        let mut m = good_meta();
        m.insert("deepseek4.expert_group_count", Value::U32(8));
        let err = validate_metadata(&m).unwrap_err();
        match err {
            Error::ShapeMismatch { key, .. } => assert_eq!(key, "deepseek4.expert_group_count"),
            other => panic!("expected ShapeMismatch, got {other:?}"),
        }
    }

    #[test]
    fn accepts_zero_expert_group_keys_when_present() {
        let mut m = good_meta();
        m.insert("deepseek4.expert_group_count", Value::U32(0));
        m.insert("deepseek4.expert_group_used_count", Value::U32(0));
        validate_metadata(&m).unwrap();
    }
}
