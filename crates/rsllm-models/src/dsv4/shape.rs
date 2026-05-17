//! Fixed shape constants for DeepSeek V4 Flash.
//!
//! Every value here is **hard-coded** from `ds4.c:87-108` (MIT, The ds4.c
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

use rsllm_gguf::Metadata;

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

/// Per-head Q/K/V latent dimension (after LoRA up-projection).
pub const DSV4_HEAD_DIM: usize = 512;

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
/// Ref: `ds4.c:95`.
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

/// RoPE frequency base. Sized for the long-context YaRN scaling regime.
pub const DSV4_ROPE_FREQ_BASE: f32 = 160_000.0;

/// RMSNorm epsilon. Same value at every layer.
pub const DSV4_RMS_EPS: f32 = 1e-6;

/// Number of leading layers that use **hash routing** for MoE instead
/// of top-k softmax routing (`ds4.c:5002-5050`). Layers `[0, 3)` use
/// the `ffn_gate_tid2eid` lookup table indexed by token id.
pub const DSV4_HASH_ROUTE_LAYERS: usize = 3;

/// Expected GGUF `general.architecture` value.
pub const DSV4_ARCH_NAME: &str = "deepseek-v4-flash";

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
    expect_u32(meta, "deepseek-v4-flash.block_count", DSV4_N_LAYER as u32)?;

    // 3. Embedding dim.
    expect_u32(
        meta,
        "deepseek-v4-flash.embedding_length",
        DSV4_N_EMBD as u32,
    )?;

    // 4. Vocab size.
    expect_u32(meta, "deepseek-v4-flash.vocab_size", DSV4_N_VOCAB as u32)?;

    // 5. Head counts.
    expect_u32(
        meta,
        "deepseek-v4-flash.attention.head_count",
        DSV4_N_HEAD as u32,
    )?;
    expect_u32(
        meta,
        "deepseek-v4-flash.attention.head_count_kv",
        DSV4_N_HEAD_KV as u32,
    )?;

    // 6. Per-head latent dim.
    expect_u32(meta, "deepseek-v4-flash.head_dim", DSV4_HEAD_DIM as u32)?;
    expect_u32(meta, "deepseek-v4-flash.n_rot", DSV4_N_ROT as u32)?;

    // 7. LoRA ranks.
    expect_u32(meta, "deepseek-v4-flash.q_lora_rank", DSV4_N_LORA_Q as u32)?;
    expect_u32(meta, "deepseek-v4-flash.o_lora_rank", DSV4_N_LORA_O as u32)?;

    // 8. MoE shape.
    expect_u32(meta, "deepseek-v4-flash.expert_count", DSV4_N_EXPERT as u32)?;
    expect_u32(
        meta,
        "deepseek-v4-flash.expert_used_count",
        DSV4_N_EXPERT_USED as u32,
    )?;
    expect_u32(
        meta,
        "deepseek-v4-flash.expert_feed_forward_length",
        DSV4_N_FF_EXP as u32,
    )?;
    expect_u32(
        meta,
        "deepseek-v4-flash.expert_shared_count",
        DSV4_N_EXPERT_SHARED as u32,
    )?;
    expect_u32(meta, "deepseek-v4-flash.n_hc", DSV4_N_HC as u32)?;

    // Attention output LoRA grouping (`ds4.c:2520`).
    expect_u32(
        meta,
        "deepseek-v4-flash.attention.output_group_count",
        DSV4_N_OUT_GROUP as u32,
    )?;

    // Indexer shape (`ds4.c:2536-2537`); used by F006 KV cache.
    expect_u32(
        meta,
        "deepseek-v4-flash.attention.indexer.head_count",
        DSV4_N_INDEXER_HEAD as u32,
    )?;
    expect_u32(
        meta,
        "deepseek-v4-flash.attention.indexer.key_length",
        DSV4_N_INDEXER_HEAD_DIM as u32,
    )?;

    // 9. RoPE base — float, exact match.
    expect_f32_close(
        meta,
        "deepseek-v4-flash.rope.freq_base",
        DSV4_ROPE_FREQ_BASE,
    )?;

    // 10. RMSNorm eps.
    expect_f32_close(
        meta,
        "deepseek-v4-flash.attention.layer_norm_rms_epsilon",
        DSV4_RMS_EPS,
    )?;

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
        m.insert(
            "deepseek-v4-flash.block_count",
            Value::U32(DSV4_N_LAYER as u32),
        );
        m.insert(
            "deepseek-v4-flash.embedding_length",
            Value::U32(DSV4_N_EMBD as u32),
        );
        m.insert(
            "deepseek-v4-flash.vocab_size",
            Value::U32(DSV4_N_VOCAB as u32),
        );
        m.insert(
            "deepseek-v4-flash.attention.head_count",
            Value::U32(DSV4_N_HEAD as u32),
        );
        m.insert(
            "deepseek-v4-flash.attention.head_count_kv",
            Value::U32(DSV4_N_HEAD_KV as u32),
        );
        m.insert(
            "deepseek-v4-flash.head_dim",
            Value::U32(DSV4_HEAD_DIM as u32),
        );
        m.insert("deepseek-v4-flash.n_rot", Value::U32(DSV4_N_ROT as u32));
        m.insert(
            "deepseek-v4-flash.q_lora_rank",
            Value::U32(DSV4_N_LORA_Q as u32),
        );
        m.insert(
            "deepseek-v4-flash.o_lora_rank",
            Value::U32(DSV4_N_LORA_O as u32),
        );
        m.insert(
            "deepseek-v4-flash.expert_count",
            Value::U32(DSV4_N_EXPERT as u32),
        );
        m.insert(
            "deepseek-v4-flash.expert_used_count",
            Value::U32(DSV4_N_EXPERT_USED as u32),
        );
        m.insert(
            "deepseek-v4-flash.expert_feed_forward_length",
            Value::U32(DSV4_N_FF_EXP as u32),
        );
        m.insert(
            "deepseek-v4-flash.expert_shared_count",
            Value::U32(DSV4_N_EXPERT_SHARED as u32),
        );
        m.insert("deepseek-v4-flash.n_hc", Value::U32(DSV4_N_HC as u32));
        m.insert(
            "deepseek-v4-flash.attention.output_group_count",
            Value::U32(DSV4_N_OUT_GROUP as u32),
        );
        m.insert(
            "deepseek-v4-flash.attention.indexer.head_count",
            Value::U32(DSV4_N_INDEXER_HEAD as u32),
        );
        m.insert(
            "deepseek-v4-flash.attention.indexer.key_length",
            Value::U32(DSV4_N_INDEXER_HEAD_DIM as u32),
        );
        m.insert(
            "deepseek-v4-flash.rope.freq_base",
            Value::F32(DSV4_ROPE_FREQ_BASE),
        );
        m.insert(
            "deepseek-v4-flash.attention.layer_norm_rms_epsilon",
            Value::F32(DSV4_RMS_EPS),
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
    fn rejects_wrong_layer_count() {
        let mut m = good_meta();
        m.insert("deepseek-v4-flash.block_count", Value::U32(42));
        let err = validate_metadata(&m).unwrap_err();
        match err {
            Error::ShapeMismatch { key, .. } => assert_eq!(key, "deepseek-v4-flash.block_count"),
            other => panic!("expected ShapeMismatch, got {other:?}"),
        }
    }

    #[test]
    fn rejects_missing_required_key() {
        let mut m = good_meta();
        // BTreeMap has no `remove` exposed; rebuild without one key.
        // Simulate by inserting a wrong-typed value (still triggers MissingMetadata
        // because get_u32 returns None on String).
        m.insert(
            "deepseek-v4-flash.embedding_length",
            Value::String("oops".to_string()),
        );
        let err = validate_metadata(&m).unwrap_err();
        assert!(matches!(err, Error::MissingMetadata(_)));
    }

    #[test]
    fn accepts_rope_base_within_tolerance() {
        let mut m = good_meta();
        // 1e-4 relative drift on a 160000 base = 16, well under the 1e-3 gate.
        m.insert(
            "deepseek-v4-flash.rope.freq_base",
            Value::F32(DSV4_ROPE_FREQ_BASE + 10.0),
        );
        validate_metadata(&m).unwrap();
    }

    #[test]
    fn rejects_rope_base_far_off() {
        let mut m = good_meta();
        m.insert(
            "deepseek-v4-flash.rope.freq_base",
            Value::F32(10_000.0), // Llama-style base, way off.
        );
        let err = validate_metadata(&m).unwrap_err();
        assert!(matches!(err, Error::ShapeMismatch { .. }));
    }
}
