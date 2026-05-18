//! GGUF → [`crate::DeepSeekV4Flash`] loader.
//!
//! Resolves the per-layer tensor lookups, picks the right `WeightBlob`
//! variant from the GGUF declared type, and threads them through the
//! `DsV4Block` slots defined in F005 (+ F008.B compressor/indexer
//! additions).
//!
//! ## Tensor naming
//!
//! Names follow ds4 upstream (commit `ef0a490`). The full mapping
//! lives in [`tensor_names`] as a single source of truth — if ds4
//! upstream renames anything, this is the only file to patch.
//!
//! The well-known names (`token_embd.weight`, `output_norm.weight`,
//! `blk.N.attn_norm.weight`, the MLA q/kv LoRA family, the MoE
//! `ffn_*_exps.weight` triplet, etc.) are stable across llama.cpp
//! forks. The compressor block (`attn_compressor_{kv,gate,ape,norm}`,
//! F010.B), the indexer block (`indexer.attn_q_b`, `indexer.proj`,
//! and the four `indexer_compressor_*` tensors, F010.C), the HC
//! triple (`hc_{attn,ffn}_{fn,scale,base}`, F010.A), and the
//! top-level `output_hc_*` (F010.D) are all pinned to ds4.c lines
//! 2604-2615. The structural surface (this loader's interface, error
//! shape, per-layer regime dispatch) does not depend on the exact
//! strings.
//!
//! Ported by reference from `ds4.c:1846-1853` and the surrounding
//! layer-weight struct (MIT, The ds4.c authors). Line numbers pinned
//! to ds4 commit `ef0a490` (2026-05-17).

use rsllm_gguf::{GgmlType, GgufFile, Metadata, TensorInfo};
use rsllm_kvcache::dsv4::shape::{layer_compress_ratio, layer_has_indexer};

use crate::Error;
use crate::deepseek_v4_flash::{DeepSeekV4Flash, DsV4Block};
use crate::dsv4::compressor::{CompressorWeights, IndexerWeights};
use crate::dsv4::hc::{HC_DIM, HC_MIX_DIM, HcSublayerWeights};
use crate::dsv4::mla::MlaWeights;
use crate::dsv4::moe::{
    MoeExpertWeights, MoeHashRouter, MoeTopkRouter, SharedExpertWeights, StackedExperts,
};
use crate::dsv4::shape::{
    DSV4_HASH_ROUTE_LAYERS, DSV4_HEAD_DIM, DSV4_N_EMBD, DSV4_N_EXPERT, DSV4_N_EXPERT_USED,
    DSV4_N_FF_EXP, DSV4_N_HEAD, DSV4_N_INDEXER_HEAD, DSV4_N_INDEXER_HEAD_DIM, DSV4_N_LAYER,
    DSV4_N_LORA_Q, DSV4_N_VOCAB, validate_metadata,
};
use crate::dsv4::weight::WeightBlob;

/// Canonical tensor-name strings used by the loader.
///
/// One place to patch if the GGUF naming convention drifts.
pub mod tensor_names {
    /// `[N_VOCAB × N_EMBD]` embedding table.
    pub const TOKEN_EMBD: &str = "token_embd.weight";
    /// `[N_EMBD]` final RMSNorm scale.
    pub const OUTPUT_NORM: &str = "output_norm.weight";
    /// `[N_VOCAB × N_EMBD]` LM head. Some GGUFs tie this to TOKEN_EMBD
    /// — the loader falls back to TOKEN_EMBD if OUTPUT is absent.
    pub const OUTPUT: &str = "output.weight";

    /// Output-head HC merge tensors (verified ds4.c:2581-2583, 2292-2294).
    pub const OUTPUT_HC_FN: &str = "output_hc_fn.weight";
    pub const OUTPUT_HC_SCALE: &str = "output_hc_scale.weight";
    pub const OUTPUT_HC_BASE: &str = "output_hc_base.weight";

    // --- Per-layer suffixes (formatted as `blk.{N}.{SUFFIX}`). ---

    /// Pre-attention RMSNorm.
    pub const ATTN_NORM: &str = "attn_norm.weight";

    // MLA Q/KV LoRA family.
    pub const ATTN_Q_A: &str = "attn_q_a.weight";
    pub const ATTN_Q_A_NORM: &str = "attn_q_a_norm.weight";
    pub const ATTN_Q_B: &str = "attn_q_b.weight";
    pub const ATTN_KV_A: &str = "attn_kv_a_mqa.weight";
    pub const ATTN_KV_A_NORM: &str = "attn_kv_a_norm.weight";

    /// `[N_HEAD]` per-head attention sink (F005 extension).
    pub const ATTN_SINKS: &str = "attn_sinks.weight";

    /// Two-stage grouped LoRA output projection.
    pub const ATTN_OUTPUT_A: &str = "attn_output_a.weight";
    pub const ATTN_OUTPUT_B: &str = "attn_output_b.weight";

    /// HC sublayer weights — verified against ds4 commit `ef0a490`
    /// (line 2591-2620). Each transformer block has TWO sublayers
    /// (attn + ffn), each carrying a `(fn, scale, base)` triplet.
    /// The post step has no weights of its own; it reads the
    /// Sinkhorn split from the matching pre call.
    pub const HC_ATTN_FN: &str = "hc_attn_fn.weight";
    pub const HC_ATTN_SCALE: &str = "hc_attn_scale.weight";
    pub const HC_ATTN_BASE: &str = "hc_attn_base.weight";
    pub const HC_FFN_FN: &str = "hc_ffn_fn.weight";
    pub const HC_FFN_SCALE: &str = "hc_ffn_scale.weight";
    pub const HC_FFN_BASE: &str = "hc_ffn_base.weight";

    pub const FFN_NORM: &str = "ffn_norm.weight";

    /// MoE routed-expert weights (stacked).
    pub const FFN_GATE_EXPS: &str = "ffn_gate_exps.weight";
    pub const FFN_UP_EXPS: &str = "ffn_up_exps.weight";
    pub const FFN_DOWN_EXPS: &str = "ffn_down_exps.weight";

    /// Shared-expert weights.
    pub const FFN_GATE_SHEXP: &str = "ffn_gate_shexp.weight";
    pub const FFN_UP_SHEXP: &str = "ffn_up_shexp.weight";
    pub const FFN_DOWN_SHEXP: &str = "ffn_down_shexp.weight";

    /// MoE router weights.
    pub const FFN_GATE_INP: &str = "ffn_gate_inp.weight";
    /// Optional per-expert gate bias.
    pub const FFN_EXP_PROBS_B: &str = "ffn_exp_probs_b.bias";
    /// Hash-router tid → eid table (only layers `< DSV4_HASH_ROUTE_LAYERS`).
    pub const TID2EID: &str = "ffn_hash_tid2eid";

    /// F010.B compressor tensors — verified against ds4 commit
    /// `ef0a490` at line 2604-2607. Each compressed layer (ratio > 0)
    /// carries all four.
    pub const ATTN_COMPRESSOR_KV: &str = "attn_compressor_kv.weight";
    pub const ATTN_COMPRESSOR_GATE: &str = "attn_compressor_gate.weight";
    pub const ATTN_COMPRESSOR_APE: &str = "attn_compressor_ape.weight";
    pub const ATTN_COMPRESSOR_NORM: &str = "attn_compressor_norm.weight";

    /// F010.C indexer tensors — verified against ds4 commit `ef0a490`
    /// at lines 2326-2331 (struct) and 2610-2615 (load). Only ratio-4
    /// layers carry all six. Note the dot-separated names for the two
    /// LoRA tensors (`indexer.attn_q_b`, `indexer.proj`) — these are
    /// exact upstream names, not a typo.
    pub const INDEXER_ATTN_Q_B: &str = "indexer.attn_q_b.weight";
    pub const INDEXER_PROJ: &str = "indexer.proj.weight";
    pub const INDEXER_COMPRESSOR_APE: &str = "indexer_compressor_ape.weight";
    pub const INDEXER_COMPRESSOR_KV: &str = "indexer_compressor_kv.weight";
    pub const INDEXER_COMPRESSOR_GATE: &str = "indexer_compressor_gate.weight";
    pub const INDEXER_COMPRESSOR_NORM: &str = "indexer_compressor_norm.weight";

    /// Format the per-layer key `"blk.{il}.{suffix}"` once.
    #[must_use]
    pub fn blk(il: usize, suffix: &str) -> String {
        format!("blk.{il}.{suffix}")
    }
}

/// Reinterpret a byte slice from the GGUF mmap as `&[f32]` after
/// validating length and pointer alignment. Shared helper for
/// [`resolve_blob`] and [`resolve_blob_opt`] so the safety
/// reasoning lives in one place.
///
/// # Errors
/// - [`Error::ShapeMismatch`] with `key = "loader.f32.length"` if
///   the byte length is not a multiple of 4 (tensor name carried
///   in the `actual` field).
/// - [`Error::ShapeMismatch`] with `key = "loader.f32.alignment"`
///   if the start pointer is not 4-byte aligned (tensor name
///   carried in the `actual` field).
fn reinterpret_as_f32<'a>(bytes: &'a [u8], name: &str) -> Result<&'a [f32], Error> {
    if !bytes.len().is_multiple_of(4) {
        return Err(Error::ShapeMismatch {
            key: "loader.f32.length",
            expected: "byte length divisible by 4 (F32)".into(),
            actual: format!("{name}: {}", bytes.len()),
        });
    }
    if !(bytes.as_ptr() as usize).is_multiple_of(std::mem::align_of::<f32>()) {
        return Err(Error::ShapeMismatch {
            key: "loader.f32.alignment",
            expected: "4-byte aligned F32 tensor".into(),
            actual: format!(
                "{name}: alignment offset {}",
                bytes.as_ptr() as usize % std::mem::align_of::<f32>()
            ),
        });
    }
    // SAFETY: We just verified
    //   (a) byte length is a multiple of 4 (matching `size_of::<f32>()`),
    //   (b) start pointer is 4-byte aligned (matching `align_of::<f32>()`),
    //   (c) every bit pattern in 4 bytes is a valid f32 (f32 has no
    //       invalid bit patterns — including NaN — per the Rust
    //       Reference's "Behavior considered undefined").
    // The resulting slice's lifetime is the input slice's lifetime,
    // which is tied to the GGUF mmap via the caller's `'a` parameter.
    let slice = unsafe {
        std::slice::from_raw_parts(bytes.as_ptr() as *const f32, bytes.len() / 4)
    };
    Ok(slice)
}

/// Look up a tensor by name. Returns [`Error::MissingTensor`] with
/// the GGUF key if absent.
pub fn lookup<'a>(gguf: &'a GgufFile, name: &str) -> Result<&'a TensorInfo, Error> {
    gguf.tensor(name).ok_or_else(|| Error::MissingTensor(name.to_owned()))
}

/// Resolve a tensor to a borrowed [`WeightBlob`]. Dispatches to the
/// F32 or Quant variant based on the GGUF declared dtype.
///
/// # Errors
/// - [`Error::MissingTensor`] if the name is absent.
/// - [`Error::ShapeMismatch`] if the tensor type is unrecognised or
///   its payload bytes are unreadable / misaligned.
pub fn resolve_blob<'a>(gguf: &'a GgufFile, name: &str) -> Result<WeightBlob<'a>, Error> {
    let info = lookup(gguf, name)?;
    let bytes = gguf.tensor_bytes(info).ok_or_else(|| Error::ShapeMismatch {
        key: "loader.tensor_bytes",
        expected: "valid byte range".into(),
        actual: format!("{name}: out of bounds in GGUF mmap"),
    })?;
    let dtype = GgmlType::from_u32(info.raw_type).ok_or_else(|| Error::ShapeMismatch {
        key: "loader.dtype",
        expected: "known GgmlType".into(),
        actual: format!("{name}: raw_type={}", info.raw_type),
    })?;
    if dtype == GgmlType::F32 {
        Ok(WeightBlob::F32(reinterpret_as_f32(bytes, name)?))
    } else {
        Ok(WeightBlob::Quant { data: bytes, dtype })
    }
}

/// Resolve an F32-typed tensor to a `&[f32]` slice. Convenience over
/// [`resolve_blob`] for the norm-scale / per-head-weight cases.
///
/// # Errors
/// Same as [`resolve_blob`], plus [`Error::ShapeMismatch`] if the
/// declared dtype is not F32.
pub fn resolve_f32_slice<'a>(gguf: &'a GgufFile, name: &str) -> Result<&'a [f32], Error> {
    match resolve_blob(gguf, name)? {
        WeightBlob::F32(s) => Ok(s),
        WeightBlob::Quant { dtype, .. } => Err(Error::ShapeMismatch {
            key: "loader.expected_f32",
            expected: "F32".into(),
            actual: format!("{name}: {dtype:?}"),
        }),
    }
}

/// Optional lookup — returns `None` if the tensor is absent (instead
/// of erroring). Used for tensors that may or may not be present
/// depending on the layer's regime (e.g. compressor on dense layers).
///
/// Implementation note: maps `MissingTensor` to `Ok(None)` rather
/// than pre-checking with a second linear scan over `gguf.tensors()`.
pub fn resolve_blob_opt<'a>(
    gguf: &'a GgufFile,
    name: &str,
) -> Result<Option<WeightBlob<'a>>, Error> {
    match resolve_blob(gguf, name) {
        Ok(blob) => Ok(Some(blob)),
        Err(Error::MissingTensor(_)) => Ok(None),
        Err(e) => Err(e),
    }
}

/// Reinterpret a byte slice as `&[i32]`. Same shape as
/// [`reinterpret_as_f32`] — used for the hash-router `tid2eid`
/// lookup table (`[N_EXPERT_USED × N_VOCAB]` row-major `i32`).
fn reinterpret_as_i32<'a>(bytes: &'a [u8], name: &str) -> Result<&'a [i32], Error> {
    if !bytes.len().is_multiple_of(4) {
        return Err(Error::ShapeMismatch {
            key: "loader.i32.length",
            expected: "byte length divisible by 4 (I32)".into(),
            actual: format!("{name}: {}", bytes.len()),
        });
    }
    if !(bytes.as_ptr() as usize).is_multiple_of(std::mem::align_of::<i32>()) {
        return Err(Error::ShapeMismatch {
            key: "loader.i32.alignment",
            expected: "4-byte aligned I32 tensor".into(),
            actual: format!(
                "{name}: alignment offset {}",
                bytes.as_ptr() as usize % std::mem::align_of::<i32>()
            ),
        });
    }
    // SAFETY: same reasoning as reinterpret_as_f32. i32 has no
    // invalid bit patterns; every 4-byte sequence is a valid i32.
    let slice = unsafe {
        std::slice::from_raw_parts(bytes.as_ptr() as *const i32, bytes.len() / 4)
    };
    Ok(slice)
}

/// Resolve an I32-typed tensor (GGUF type `I32 = 26`, per
/// `rsllm-gguf::GgmlType`) to a `&[i32]`.
///
/// # Errors
/// - [`Error::MissingTensor`] if the name is absent.
/// - [`Error::ShapeMismatch`] if the declared dtype is not I32 or
///   the byte payload is misaligned.
pub fn resolve_i32_slice<'a>(gguf: &'a GgufFile, name: &str) -> Result<&'a [i32], Error> {
    let info = lookup(gguf, name)?;
    let bytes = gguf.tensor_bytes(info).ok_or_else(|| Error::ShapeMismatch {
        key: "loader.tensor_bytes",
        expected: "valid byte range".into(),
        actual: format!("{name}: out of bounds in GGUF mmap"),
    })?;
    let dtype = GgmlType::from_u32(info.raw_type).ok_or_else(|| Error::ShapeMismatch {
        key: "loader.dtype",
        expected: "known GgmlType".into(),
        actual: format!("{name}: raw_type={}", info.raw_type),
    })?;
    if dtype != GgmlType::I32 {
        return Err(Error::ShapeMismatch {
            key: "loader.expected_i32",
            expected: "I32".into(),
            actual: format!("{name}: {dtype:?}"),
        });
    }
    reinterpret_as_i32(bytes, name)
}

/// Build the per-layer `CompressorWeights` for layer `il`, **if** the
/// layer regime calls for it. Returns `None` for dense layers
/// (`compress_ratio == 0`).
///
/// Loads all four ds4 tensors:
/// - `attn_compressor_kv.weight`    F16 `[N_EMBD × comp_width]`
/// - `attn_compressor_gate.weight`  F16 `[N_EMBD × comp_width]`
/// - `attn_compressor_ape.weight`   F16 `[comp_width × compress_ratio]`
/// - `attn_compressor_norm.weight`  F32 `[N_HEAD_DIM = 512]`
///
/// where `comp_width = (compress_ratio == 4 ? 2 : 1) * N_HEAD_DIM`.
///
/// # Errors
/// [`Error::MissingTensor`] if any of the four tensors is absent
/// (names embed the concrete `blk.N` index in the error message).
/// [`Error::ShapeMismatch`] on a byte-count disagreement.
pub fn load_compressor<'a>(
    gguf: &'a GgufFile,
    il: usize,
) -> Result<Option<CompressorWeights<'a>>, Error> {
    let ratio = layer_compress_ratio(il) as usize;
    if ratio == 0 {
        return Ok(None);
    }
    let coff = if ratio == 4 { 2 } else { 1 };
    let comp_width = coff * DSV4_HEAD_DIM;

    let kv_name = tensor_names::blk(il, tensor_names::ATTN_COMPRESSOR_KV);
    let gate_name = tensor_names::blk(il, tensor_names::ATTN_COMPRESSOR_GATE);
    let ape_name = tensor_names::blk(il, tensor_names::ATTN_COMPRESSOR_APE);
    let norm_name = tensor_names::blk(il, tensor_names::ATTN_COMPRESSOR_NORM);

    let kv = resolve_blob_opt(gguf, &kv_name)?
        .ok_or_else(|| Error::MissingTensor(kv_name.clone()))?;
    kv.check_shape(comp_width, DSV4_N_EMBD, "loader.compressor.kv")?;

    let gate = resolve_blob_opt(gguf, &gate_name)?
        .ok_or_else(|| Error::MissingTensor(gate_name.clone()))?;
    gate.check_shape(comp_width, DSV4_N_EMBD, "loader.compressor.gate")?;

    let ape = resolve_blob_opt(gguf, &ape_name)?
        .ok_or_else(|| Error::MissingTensor(ape_name.clone()))?;
    // APE is [comp_width × compress_ratio]: out_dim = ratio, in_dim = comp_width.
    ape.check_shape(ratio, comp_width, "loader.compressor.ape")?;

    let norm = resolve_f32_slice(gguf, &norm_name)?;
    if norm.len() != DSV4_HEAD_DIM {
        return Err(Error::ShapeMismatch {
            key: "loader.compressor.norm",
            expected: format!("{DSV4_HEAD_DIM}"),
            actual: format!("{norm_name}: {}", norm.len()),
        });
    }

    Ok(Some(CompressorWeights { kv, gate, ape, norm }))
}

/// Build the [`IndexerWeights`] for layer `il`. Returns `None` on
/// non-ratio-4 layers (dense or ratio-128).
///
/// Loads all six ds4.c tensors (ds4.c:2610-2615, commit `ef0a490`):
/// - `indexer.attn_q_b.weight`      F16 `[N_LORA_Q × (N_INDEXER_HEAD × N_INDEXER_HEAD_DIM)]`
/// - `indexer.proj.weight`          F16 `[N_EMBD × N_INDEXER_HEAD]`
/// - `indexer_compressor_ape.weight` F16 `[index_width × ratio]`
/// - `indexer_compressor_kv.weight`  F16 `[N_EMBD × index_width]`
/// - `indexer_compressor_gate.weight` F16 `[N_EMBD × index_width]`
/// - `indexer_compressor_norm.weight` F32 `[N_INDEXER_HEAD_DIM]`
///
/// where `index_width = 2 × N_INDEXER_HEAD_DIM = 256` and `ratio = 4`.
///
/// # Errors
/// [`Error::MissingTensor`] if the layer needs indexer weights but
/// the GGUF is missing any of the six expected names.
/// [`Error::ShapeMismatch`] on a byte-count disagreement or if
/// `comp_norm` is not F32-typed.
pub fn load_indexer<'a>(
    gguf: &'a GgufFile,
    il: usize,
) -> Result<Option<IndexerWeights<'a>>, Error> {
    if !layer_has_indexer(il) {
        return Ok(None);
    }
    // ratio-4 only: index_width = 2 × N_INDEXER_HEAD_DIM
    let index_width = 2 * DSV4_N_INDEXER_HEAD_DIM;
    let ratio = 4usize;

    let attn_q_b_name = tensor_names::blk(il, tensor_names::INDEXER_ATTN_Q_B);
    let attn_q_b = resolve_blob_opt(gguf, &attn_q_b_name)?
        .ok_or_else(|| Error::MissingTensor(attn_q_b_name.clone()))?;
    // Upstream stores `[N_LORA_Q × (N_INDEXER_HEAD * N_INDEXER_HEAD_DIM)]`
    // (`ds4.c:2326`). Our `check_shape(out_dim, in_dim)` is byte-count
    // commutative, and the convention across this loader is to pass the
    // matmul *output* width as `out_dim` — matches the MLA `attn_q_b`
    // call site (line 633). F011 will verify the orientation when it
    // wires the matmul.
    attn_q_b.check_shape(
        DSV4_N_INDEXER_HEAD * DSV4_N_INDEXER_HEAD_DIM,
        DSV4_N_LORA_Q,
        "loader.indexer.attn_q_b",
    )?;

    let proj_name = tensor_names::blk(il, tensor_names::INDEXER_PROJ);
    let proj = resolve_blob_opt(gguf, &proj_name)?
        .ok_or_else(|| Error::MissingTensor(proj_name.clone()))?;
    // [N_EMBD × N_INDEXER_HEAD]
    proj.check_shape(DSV4_N_INDEXER_HEAD, DSV4_N_EMBD, "loader.indexer.proj")?;

    let comp_ape_name = tensor_names::blk(il, tensor_names::INDEXER_COMPRESSOR_APE);
    let comp_ape = resolve_blob_opt(gguf, &comp_ape_name)?
        .ok_or_else(|| Error::MissingTensor(comp_ape_name.clone()))?;
    // Upstream stores `[index_width × ratio]` (`ds4.c:2328`). Matches
    // the main `attn_compressor_ape` convention (line 355) — out_dim
    // here is `ratio`, in_dim is `index_width`.
    comp_ape.check_shape(ratio, index_width, "loader.indexer.comp_ape")?;

    let comp_kv_name = tensor_names::blk(il, tensor_names::INDEXER_COMPRESSOR_KV);
    let comp_kv = resolve_blob_opt(gguf, &comp_kv_name)?
        .ok_or_else(|| Error::MissingTensor(comp_kv_name.clone()))?;
    // [N_EMBD × index_width]
    comp_kv.check_shape(index_width, DSV4_N_EMBD, "loader.indexer.comp_kv")?;

    let comp_gate_name = tensor_names::blk(il, tensor_names::INDEXER_COMPRESSOR_GATE);
    let comp_gate = resolve_blob_opt(gguf, &comp_gate_name)?
        .ok_or_else(|| Error::MissingTensor(comp_gate_name.clone()))?;
    // [N_EMBD × index_width]
    comp_gate.check_shape(index_width, DSV4_N_EMBD, "loader.indexer.comp_gate")?;

    let comp_norm_name = tensor_names::blk(il, tensor_names::INDEXER_COMPRESSOR_NORM);
    let comp_norm = resolve_f32_slice(gguf, &comp_norm_name)?;
    // [N_INDEXER_HEAD_DIM]
    if comp_norm.len() != DSV4_N_INDEXER_HEAD_DIM {
        return Err(Error::ShapeMismatch {
            key: "loader.indexer.comp_norm",
            expected: format!("{DSV4_N_INDEXER_HEAD_DIM}"),
            actual: format!("{comp_norm_name}: {}", comp_norm.len()),
        });
    }

    Ok(Some(IndexerWeights {
        attn_q_b,
        proj,
        comp_ape,
        comp_kv,
        comp_gate,
        comp_norm,
    }))
}

/// Build an [`HcSublayerWeights`] for one sublayer (attention or
/// FFN) on layer `il`. `family` is `"hc_attn"` or `"hc_ffn"`.
///
/// Verified against ds4 commit `ef0a490`:
///
/// - `blk.{il}.{family}_fn.weight`: F16, `[HC_DIM × HC_MIX_DIM]`
///   (`ds4.c:2302 / 2334 tensor_expect_layout`).
/// - `blk.{il}.{family}_scale.weight`: F32, `[3]`.
/// - `blk.{il}.{family}_base.weight`: F32, `[HC_MIX_DIM = 24]`.
///
/// # Errors
/// - [`Error::MissingTensor`] if any of the three tensors is absent.
/// - [`Error::ShapeMismatch`] if a backing storage byte count
///   disagrees with the expected shape.
pub fn load_hc_sublayer_weights<'a>(
    gguf: &'a GgufFile,
    il: usize,
    family: &str,
) -> Result<HcSublayerWeights<'a>, Error> {
    let fn_name = tensor_names::blk(il, &format!("{family}_fn.weight"));
    let scale_name = tensor_names::blk(il, &format!("{family}_scale.weight"));
    let base_name = tensor_names::blk(il, &format!("{family}_base.weight"));

    let mix_fn = resolve_blob(gguf, &fn_name)?;
    // mix_fn is `[HC_DIM × HC_MIX_DIM]` — flat residual in, 24 mix
    // logits out. Matches ds4's matvec_f16(mix_fn, flat) shape.
    mix_fn.check_shape(HC_MIX_DIM, HC_DIM, "loader.hc.mix_fn")?;

    let scale = resolve_f32_slice(gguf, &scale_name)?;
    if scale.len() != 3 {
        return Err(Error::ShapeMismatch {
            key: "loader.hc.scale",
            expected: "3".to_string(),
            actual: format!("{scale_name}: {}", scale.len()),
        });
    }

    let base = resolve_f32_slice(gguf, &base_name)?;
    if base.len() != HC_MIX_DIM {
        return Err(Error::ShapeMismatch {
            key: "loader.hc.base",
            expected: format!("{HC_MIX_DIM}"),
            actual: format!("{base_name}: {}", base.len()),
        });
    }

    Ok(HcSublayerWeights {
        mix_fn,
        scale,
        base,
    })
}

/// Build the routed-expert `MoeExpertWeights` for layer `il`.
///
/// # Errors
/// [`Error::MissingTensor`] or [`Error::ShapeMismatch`] as for the
/// underlying lookup / shape checks.
pub fn load_moe_experts<'a>(gguf: &'a GgufFile, il: usize) -> Result<MoeExpertWeights<'a>, Error> {
    let gate_name = tensor_names::blk(il, tensor_names::FFN_GATE_EXPS);
    let up_name = tensor_names::blk(il, tensor_names::FFN_UP_EXPS);
    let down_name = tensor_names::blk(il, tensor_names::FFN_DOWN_EXPS);
    let gate = resolve_blob(gguf, &gate_name)?;
    let up = resolve_blob(gguf, &up_name)?;
    let down = resolve_blob(gguf, &down_name)?;
    let per_expert = DSV4_N_FF_EXP * DSV4_N_EMBD;
    Ok(MoeExpertWeights {
        gate: StackedExperts {
            blob: gate,
            elements_per_expert: per_expert,
        },
        up: StackedExperts {
            blob: up,
            elements_per_expert: per_expert,
        },
        down: StackedExperts {
            blob: down,
            elements_per_expert: per_expert, // same total element count
        },
    })
}

/// Build the shared-expert weights for layer `il`. Always present
/// — every layer has exactly one shared expert.
///
/// # Errors
/// Bubbles up missing-tensor / shape-mismatch from the underlying
/// lookups.
pub fn load_shared_expert<'a>(
    gguf: &'a GgufFile,
    il: usize,
) -> Result<SharedExpertWeights<'a>, Error> {
    let gate = resolve_blob(gguf, &tensor_names::blk(il, tensor_names::FFN_GATE_SHEXP))?;
    let up = resolve_blob(gguf, &tensor_names::blk(il, tensor_names::FFN_UP_SHEXP))?;
    let down = resolve_blob(gguf, &tensor_names::blk(il, tensor_names::FFN_DOWN_SHEXP))?;
    gate.check_shape(DSV4_N_FF_EXP, DSV4_N_EMBD, "loader.shared.gate")?;
    up.check_shape(DSV4_N_FF_EXP, DSV4_N_EMBD, "loader.shared.up")?;
    down.check_shape(DSV4_N_EMBD, DSV4_N_FF_EXP, "loader.shared.down")?;
    Ok(SharedExpertWeights { gate, up, down })
}

/// Build the router pair (hash for layers `< DSV4_HASH_ROUTE_LAYERS`,
/// top-k otherwise). Returns the populated half + a `None` for the
/// other.
///
/// # Errors
/// - [`Error::MissingTensor`] if any required tensor is absent.
/// - [`Error::ShapeMismatch`] from the underlying lookups.
pub fn load_router<'a>(
    gguf: &'a GgufFile,
    il: usize,
) -> Result<(Option<MoeHashRouter<'a>>, Option<MoeTopkRouter<'a>>), Error> {
    let gate_inp = resolve_blob(gguf, &tensor_names::blk(il, tensor_names::FFN_GATE_INP))?;
    gate_inp.check_shape(DSV4_N_EXPERT, DSV4_N_EMBD, "loader.router.gate_inp")?;
    // Optional gate bias `[N_EXPERT]`. Most checkpoints ship it.
    let gate_bias_name = tensor_names::blk(il, tensor_names::FFN_EXP_PROBS_B);
    let gate_bias = match resolve_blob_opt(gguf, &gate_bias_name)? {
        Some(WeightBlob::F32(s)) => {
            if s.len() != DSV4_N_EXPERT {
                return Err(Error::ShapeMismatch {
                    key: "loader.router.gate_bias",
                    expected: format!("{DSV4_N_EXPERT}"),
                    actual: format!("{gate_bias_name}: {}", s.len()),
                });
            }
            Some(s)
        }
        Some(WeightBlob::Quant { .. }) => {
            return Err(Error::ShapeMismatch {
                key: "loader.router.gate_bias",
                expected: "F32".into(),
                actual: format!("{gate_bias_name}: quantised"),
            });
        }
        None => None,
    };

    if il < DSV4_HASH_ROUTE_LAYERS {
        let tid2eid_name = tensor_names::blk(il, tensor_names::TID2EID);
        let tid2eid = resolve_i32_slice(gguf, &tid2eid_name)?;
        let expected = DSV4_N_EXPERT_USED * DSV4_N_VOCAB;
        if tid2eid.len() != expected {
            return Err(Error::ShapeMismatch {
                key: "loader.router.tid2eid",
                expected: format!("{expected}"),
                actual: format!("{tid2eid_name}: {}", tid2eid.len()),
            });
        }
        Ok((
            Some(MoeHashRouter {
                tid2eid,
                gate_inp,
                gate_bias,
            }),
            None,
        ))
    } else {
        Ok((
            None,
            Some(MoeTopkRouter {
                gate_inp,
                gate_bias,
            }),
        ))
    }
}

/// Assemble the [`MlaWeights`] for layer `il`.
fn load_mla_weights<'a>(gguf: &'a GgufFile, il: usize) -> Result<MlaWeights<'a>, Error> {
    let attn_q_a = resolve_blob(gguf, &tensor_names::blk(il, tensor_names::ATTN_Q_A))?;
    attn_q_a.check_shape(DSV4_N_LORA_Q, DSV4_N_EMBD, "loader.mla.attn_q_a")?;
    let q_a_norm = resolve_f32_slice(gguf, &tensor_names::blk(il, tensor_names::ATTN_Q_A_NORM))?;
    if q_a_norm.len() != DSV4_N_LORA_Q {
        return Err(Error::ShapeMismatch {
            key: "loader.mla.q_a_norm",
            expected: format!("{DSV4_N_LORA_Q}"),
            actual: format!("{}", q_a_norm.len()),
        });
    }
    let attn_q_b = resolve_blob(gguf, &tensor_names::blk(il, tensor_names::ATTN_Q_B))?;
    attn_q_b.check_shape(
        DSV4_N_HEAD * DSV4_HEAD_DIM,
        DSV4_N_LORA_Q,
        "loader.mla.attn_q_b",
    )?;
    let attn_kv_a = resolve_blob(gguf, &tensor_names::blk(il, tensor_names::ATTN_KV_A))?;
    attn_kv_a.check_shape(DSV4_HEAD_DIM, DSV4_N_EMBD, "loader.mla.attn_kv_a")?;
    let kv_a_norm = resolve_f32_slice(gguf, &tensor_names::blk(il, tensor_names::ATTN_KV_A_NORM))?;
    if kv_a_norm.len() != DSV4_HEAD_DIM {
        return Err(Error::ShapeMismatch {
            key: "loader.mla.kv_a_norm",
            expected: format!("{DSV4_HEAD_DIM}"),
            actual: format!("{}", kv_a_norm.len()),
        });
    }
    Ok(MlaWeights {
        attn_q_a,
        q_a_norm,
        attn_q_b,
        attn_kv_a,
        kv_a_norm,
    })
}

/// Assemble one [`DsV4Block`] for layer `il`.
///
/// # Errors
/// Any missing-tensor or shape-mismatch from the underlying lookups.
pub fn load_block<'a>(gguf: &'a GgufFile, il: usize) -> Result<DsV4Block<'a>, Error> {
    let attn_norm = resolve_f32_slice(gguf, &tensor_names::blk(il, tensor_names::ATTN_NORM))?;
    let mla = load_mla_weights(gguf, il)?;
    let attn_sinks = resolve_f32_slice(gguf, &tensor_names::blk(il, tensor_names::ATTN_SINKS))?;
    let attn_output_a = resolve_blob(gguf, &tensor_names::blk(il, tensor_names::ATTN_OUTPUT_A))?;
    let attn_output_b = resolve_blob(gguf, &tensor_names::blk(il, tensor_names::ATTN_OUTPUT_B))?;

    let hc_attn = load_hc_sublayer_weights(gguf, il, "hc_attn")?;
    let hc_ffn = load_hc_sublayer_weights(gguf, il, "hc_ffn")?;

    let ffn_norm = resolve_f32_slice(gguf, &tensor_names::blk(il, tensor_names::FFN_NORM))?;
    let moe_experts = load_moe_experts(gguf, il)?;
    let shared_expert = load_shared_expert(gguf, il)?;
    let (hash_router, topk_router) = load_router(gguf, il)?;

    let compressor = load_compressor(gguf, il)?;
    let indexer = load_indexer(gguf, il)?;

    Ok(DsV4Block {
        attn_norm,
        mla,
        attn_sinks,
        attn_output_a,
        attn_output_b,
        hc_attn,
        ffn_norm,
        moe_experts,
        shared_expert,
        hash_router,
        topk_router,
        hc_ffn,
        compressor,
        indexer,
    })
}

/// Top-level GGUF → model loader. Validates metadata, locates every
/// per-layer tensor, and constructs a [`DeepSeekV4Flash`] borrowing
/// the GGUF mmap for its weight views.
///
/// ## HC scales now come from GGUF
///
/// Per-layer HC `[pre, post, comb]` scales are loaded from the
/// `blk.{il}.hc_{attn,ffn}_scale.weight` tensors (verified against
/// ds4 commit `ef0a490` at `ds4.c:2592, 2618`). The previous
/// `[1.0, 1.0, 1.0]` placeholder is gone; if any of those tensors
/// is missing in a GGUF file, the loader fails with
/// [`Error::MissingTensor`] naming the concrete blk index.
///
/// # Errors
/// - [`Error::MissingMetadata`] / [`Error::ShapeMismatch`] for any
///   GGUF metadata key that disagrees with the DS V4 Flash spec.
/// - [`Error::MissingTensor`] for any required tensor that is
///   absent (the message identifies the concrete `blk.N.foo` name).
/// - [`Error::ShapeMismatch`] when a tensor's byte count disagrees
///   with the expected logical shape.
pub fn load_dsv4_flash(gguf: &GgufFile) -> Result<DeepSeekV4Flash<'_>, Error> {
    // Top-level INFO span so per-block DEBUG events under a single
    // load are easy to group when filtering tracing output.
    let _span = tracing::info_span!(
        target: "rsllm_models::dsv4::loader",
        "load_dsv4_flash",
        n_layer = DSV4_N_LAYER,
    )
    .entered();
    let started = std::time::Instant::now();

    // 1. Metadata sanity. Catches arch / shape constant drift early.
    validate_metadata(gguf.metadata())?;
    tracing::info!(
        target: "rsllm_models::dsv4::loader",
        arch = "deepseek4",
        n_layer = DSV4_N_LAYER,
        n_vocab = DSV4_N_VOCAB,
        n_embd = DSV4_N_EMBD,
        "metadata validated",
    );

    // 2. Global tensors.
    let embed_tokens = resolve_blob(gguf, tensor_names::TOKEN_EMBD)?;
    embed_tokens.check_shape(DSV4_N_VOCAB, DSV4_N_EMBD, "loader.token_embd")?;
    let output_norm = resolve_f32_slice(gguf, tensor_names::OUTPUT_NORM)?;
    if output_norm.len() != DSV4_N_EMBD {
        return Err(Error::ShapeMismatch {
            key: "loader.output_norm",
            expected: format!("{DSV4_N_EMBD}"),
            actual: format!("{}", output_norm.len()),
        });
    }
    // `output.weight` is optional — falls back to the tied
    // `token_embd.weight` matrix when the LM head shares parameters.
    //
    // Note: `WeightBlob` is `Copy` (it holds only a slice reference +
    // `GgmlType` enum), so the `None => embed_tokens` arm copies
    // rather than moves. Both `embed_tokens` and `lm_head` then alias
    // the same mmap region — that's exactly what tied-weight semantics
    // demand. The final `DeepSeekV4Flash::new(embed_tokens, ..., lm_head)`
    // call is therefore well-formed despite the apparent move.
    let (lm_head, tied_lm_head) = match resolve_blob_opt(gguf, tensor_names::OUTPUT)? {
        Some(blob) => {
            blob.check_shape(DSV4_N_VOCAB, DSV4_N_EMBD, "loader.lm_head")?;
            (blob, false)
        }
        None => (embed_tokens, true),
    };

    // Output-head HC merge tensors. Shapes are enforced inside
    // `DeepSeekV4Flash::new`; this resolve step turns
    // MissingTensor into a clear blk-less error message.
    let oh_fn = resolve_blob(gguf, tensor_names::OUTPUT_HC_FN)?;
    let oh_scale = resolve_f32_slice(gguf, tensor_names::OUTPUT_HC_SCALE)?;
    let oh_base = resolve_f32_slice(gguf, tensor_names::OUTPUT_HC_BASE)?;
    let output_hc = crate::deepseek_v4_flash::OutputHcWeights {
        mix_fn: oh_fn,
        scale: oh_scale,
        base: oh_base,
    };

    tracing::info!(
        target: "rsllm_models::dsv4::loader",
        tied_lm_head,
        "global tensors loaded (token_embd, output_norm, lm_head, output_hc)",
    );

    // 3. Per-layer blocks. Per-block DEBUG events are reasonable
    // signal-to-noise (43 lines) for a multi-minute load; the
    // wider INFO heartbeat below keeps the default-level operator
    // informed without flooding.
    let mut blocks = Vec::with_capacity(DSV4_N_LAYER);
    for il in 0..DSV4_N_LAYER {
        let block_started = std::time::Instant::now();
        blocks.push(load_block(gguf, il)?);
        tracing::debug!(
            target: "rsllm_models::dsv4::loader",
            il,
            elapsed_ms = block_started.elapsed().as_millis() as u64,
            compress_ratio = layer_compress_ratio(il),
            has_indexer = layer_has_indexer(il),
            "block loaded",
        );
        // Integer-division midway. For odd N_LAYER this rounds down
        // (`43/2 == 21`), so the heartbeat fires after block 21, i.e.
        // 22/43 — within one block of the true center. Close enough
        // for a progress signal and the math stays obvious.
        if il == DSV4_N_LAYER / 2 {
            tracing::info!(
                target: "rsllm_models::dsv4::loader",
                blocks_loaded = il + 1,
                n_layer = DSV4_N_LAYER,
                elapsed_ms = started.elapsed().as_millis() as u64,
                "midway heartbeat",
            );
        }
    }

    let model = DeepSeekV4Flash::new(embed_tokens, blocks, output_hc, output_norm, lm_head)?;
    tracing::info!(
        target: "rsllm_models::dsv4::loader",
        elapsed_ms = started.elapsed().as_millis() as u64,
        "load complete",
    );
    Ok(model)
}

/// Inspect the metadata table directly. Convenience for callers
/// (e.g. `rsllm inspect`) that want the architecture string without
/// constructing the full model.
#[must_use]
pub fn architecture_name(meta: &Metadata) -> Option<&str> {
    meta.get_str("general.architecture")
}

/// Count of expected GGUF tensors for layer `il`. Used by
/// `rsllm inspect` to surface the per-layer regime distribution
/// — not load-bearing for any test.
///
/// Baseline (every layer):
///   attn_norm                         1
///   MLA: q_a, q_a_norm, q_b, kv_a, kv_a_norm   5
///   attn_sinks                        1
///   attn_output_a, attn_output_b      2
///   HC: 2 sublayers × 3 tensors each (fn + scale + base)   6
///   ffn_norm                          1
///   MoE routed: gate_exps, up_exps, down_exps      3
///   shared expert: gate, up, down     3
///   router: gate_inp                  1
///                                    --
///                                    23
///
/// Plus per-regime additions:
///   compressed (ratio > 0): +4 (attn_compressor_{kv, gate, ape, norm})
///   ratio-4 only:           +6 (indexer.attn_q_b, indexer.proj,
///                              indexer_compressor_{ape, kv, gate, norm})
///                              — verified against ds4 commit `ef0a490`
///                                lines 2326-2331 and 2610-2615 (F010.C)
///   hash-routed (il<3):     +1 (tid2eid)
///   gate_bias (when shipped): +1 (optional, not counted here)
#[must_use]
pub fn expected_layer_tensor_count(il: usize) -> usize {
    // Baseline = 1+5+1+2+6+1+3+3+1 = 23.
    let mut count = 23;
    if layer_compress_ratio(il) > 0 {
        count += 4; // compressor kv/gate/ape/norm
    }
    if layer_has_indexer(il) {
        count += 6; // indexer: attn_q_b, proj, comp_ape/kv/gate/norm
    }
    if il < DSV4_HASH_ROUTE_LAYERS {
        count += 1; // tid2eid
    }
    count
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Inline mini-builder to assemble a minimal GGUF in tests, without
    /// borrowing the rsllm-gguf crate's private builder. We only need
    /// the byte-level constructor for a handful of name-resolution tests.
    fn build_gguf(
        kv: &[(&str, u32, Vec<u8>)],
        tensors: &[(&str, u32, Vec<u64>, Vec<u8>)],
    ) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(b"GGUF");
        out.extend_from_slice(&3u32.to_le_bytes());
        out.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
        out.extend_from_slice(&(kv.len() as u64).to_le_bytes());

        for (key, ttype, value_bytes) in kv {
            out.extend_from_slice(&(key.len() as u64).to_le_bytes());
            out.extend_from_slice(key.as_bytes());
            out.extend_from_slice(&ttype.to_le_bytes());
            out.extend_from_slice(value_bytes);
        }
        let mut payload_offsets = Vec::with_capacity(tensors.len());
        let mut cursor = 0u64;
        for (_, _, _, payload) in tensors {
            payload_offsets.push(cursor);
            cursor += payload.len() as u64;
        }
        for (idx, (name, ttype, dims, _)) in tensors.iter().enumerate() {
            out.extend_from_slice(&(name.len() as u64).to_le_bytes());
            out.extend_from_slice(name.as_bytes());
            out.extend_from_slice(&(dims.len() as u32).to_le_bytes());
            for &d in dims {
                out.extend_from_slice(&d.to_le_bytes());
            }
            out.extend_from_slice(&ttype.to_le_bytes());
            out.extend_from_slice(&payload_offsets[idx].to_le_bytes());
        }
        let pad = (32 - (out.len() as u64 % 32)) % 32;
        out.extend(std::iter::repeat_n(0u8, pad as usize));
        for (_, _, _, payload) in tensors {
            out.extend_from_slice(payload);
        }
        out
    }

    /// Serialize a `&[f32]` to little-endian bytes for the test
    /// builder. GGUF on-disk is little-endian by convention.
    fn f32s_to_bytes(values: &[f32]) -> Vec<u8> {
        let mut out = Vec::with_capacity(values.len() * 4);
        for &v in values {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out
    }

    #[test]
    fn blk_format_matches_llama_cpp() {
        assert_eq!(tensor_names::blk(0, "attn_norm.weight"), "blk.0.attn_norm.weight");
        assert_eq!(tensor_names::blk(42, "ffn_gate_exps.weight"), "blk.42.ffn_gate_exps.weight");
    }

    #[test]
    fn lookup_missing_tensor_is_clear_error() {
        let bytes = build_gguf(&[], &[]);
        let file = GgufFile::from_bytes(bytes).unwrap();
        let err = lookup(&file, "nonexistent").unwrap_err();
        match err {
            Error::MissingTensor(name) => assert_eq!(name, "nonexistent"),
            other => panic!("expected MissingTensor, got {other:?}"),
        }
    }

    #[test]
    fn resolve_blob_for_f32_returns_aligned_slice() {
        let payload = f32s_to_bytes(&[1.0, 2.0, 3.0, 4.0]);
        let bytes = build_gguf(
            &[],
            &[("toy", GgmlType::F32 as u32, vec![4], payload)],
        );
        let file = GgufFile::from_bytes(bytes).unwrap();
        let blob = resolve_blob(&file, "toy").unwrap();
        match blob {
            WeightBlob::F32(s) => {
                assert_eq!(s, &[1.0, 2.0, 3.0, 4.0]);
            }
            other => panic!("expected F32, got {other:?}"),
        }
    }

    #[test]
    fn resolve_f32_slice_rejects_quantised_tensor() {
        let payload = vec![0u8; 144]; // arbitrary Q4_K-sized
        let bytes = build_gguf(
            &[],
            &[("q", GgmlType::Q4_K as u32, vec![256], payload)],
        );
        let file = GgufFile::from_bytes(bytes).unwrap();
        let err = resolve_f32_slice(&file, "q").unwrap_err();
        assert!(matches!(err, Error::ShapeMismatch { .. }));
    }

    #[test]
    fn resolve_blob_opt_returns_none_when_missing() {
        let bytes = build_gguf(&[], &[]);
        let file = GgufFile::from_bytes(bytes).unwrap();
        let result = resolve_blob_opt(&file, "blk.0.attn_compressor.weight").unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn resolve_blob_opt_returns_some_with_correct_values_when_present() {
        let payload = f32s_to_bytes(&[7.5, -3.25, 0.0]);
        let bytes = build_gguf(
            &[],
            &[("present", GgmlType::F32 as u32, vec![3], payload)],
        );
        let file = GgufFile::from_bytes(bytes).unwrap();
        let blob = resolve_blob_opt(&file, "present").unwrap().expect("Some");
        match blob {
            WeightBlob::F32(s) => assert_eq!(s, &[7.5, -3.25, 0.0]),
            other => panic!("expected F32, got {other:?}"),
        }
    }

    #[test]
    fn missing_tensor_error_carries_layer_index() {
        // Regression test for the prior bug where MissingTensor was
        // a &'static template `"blk.N.attn_compressor.weight"` —
        // it now carries the concrete layer index.
        let bytes = build_gguf(&[], &[]);
        let file = GgufFile::from_bytes(bytes).unwrap();
        let err = load_compressor(&file, 37).unwrap_err();
        match err {
            Error::MissingTensor(name) => {
                assert!(name.contains("blk.37"), "got {name}");
                assert!(name.contains("attn_compressor"), "got {name}");
            }
            other => panic!("expected MissingTensor, got {other:?}"),
        }
    }

    #[test]
    fn load_compressor_dense_layer_returns_none() {
        // Layer 0 / 1 are dense — even without the tensor present.
        let bytes = build_gguf(&[], &[]);
        let file = GgufFile::from_bytes(bytes).unwrap();
        assert!(load_compressor(&file, 0).unwrap().is_none());
        assert!(load_compressor(&file, 1).unwrap().is_none());
    }

    #[test]
    fn load_compressor_compressed_layer_requires_tensor() {
        // Layer 3 is ratio-128 — compressor is required.
        let bytes = build_gguf(&[], &[]);
        let file = GgufFile::from_bytes(bytes).unwrap();
        let err = load_compressor(&file, 3).unwrap_err();
        assert!(matches!(err, Error::MissingTensor(_)));
    }

    #[test]
    fn load_compressor_returns_blob_when_present() {
        // F010.B: load all four compressor tensors for a ratio-128 layer
        // (il=3 → coff=1, comp_width = HEAD_DIM = 512). Each tensor
        // must be byte-size correct or the loader rejects.
        let comp_width = DSV4_HEAD_DIM; // ratio-128 ⇒ coff = 1
        let ratio = 128usize;

        let kv_n = comp_width * DSV4_N_EMBD;
        let gate_n = comp_width * DSV4_N_EMBD;
        let ape_n = comp_width * ratio;
        let norm_n = DSV4_HEAD_DIM;

        let bytes = build_gguf(
            &[],
            &[
                (
                    "blk.3.attn_compressor_kv.weight",
                    GgmlType::F32 as u32,
                    vec![DSV4_N_EMBD as u64, comp_width as u64],
                    vec![0u8; kv_n * 4],
                ),
                (
                    "blk.3.attn_compressor_gate.weight",
                    GgmlType::F32 as u32,
                    vec![DSV4_N_EMBD as u64, comp_width as u64],
                    vec![0u8; gate_n * 4],
                ),
                (
                    "blk.3.attn_compressor_ape.weight",
                    GgmlType::F32 as u32,
                    vec![comp_width as u64, ratio as u64],
                    vec![0u8; ape_n * 4],
                ),
                (
                    "blk.3.attn_compressor_norm.weight",
                    GgmlType::F32 as u32,
                    vec![norm_n as u64],
                    vec![0u8; norm_n * 4],
                ),
            ],
        );
        let file = GgufFile::from_bytes(bytes).unwrap();
        let c = load_compressor(&file, 3).unwrap().expect("Some");
        let _ = c.kv.byte_len();
        let _ = c.gate.byte_len();
        let _ = c.ape.byte_len();
        assert_eq!(c.norm.len(), norm_n);
    }

    #[test]
    fn load_indexer_dense_layer_returns_none() {
        let bytes = build_gguf(&[], &[]);
        let file = GgufFile::from_bytes(bytes).unwrap();
        assert!(load_indexer(&file, 0).unwrap().is_none());
        assert!(load_indexer(&file, 3).unwrap().is_none()); // ratio-128, no indexer
    }

    #[test]
    fn load_indexer_ratio4_layer_requires_all_six() {
        // Layer 2 is the first ratio-4 layer; missing any of the six
        // expected indexer tensors must produce a clear error.
        let bytes = build_gguf(&[], &[]);
        let file = GgufFile::from_bytes(bytes).unwrap();
        let err = load_indexer(&file, 2).unwrap_err();
        assert!(matches!(err, Error::MissingTensor(_)));
    }

    #[test]
    fn load_indexer_returns_weights_when_all_six_tensors_present() {
        // Layer 2: ratio-4. Build all six required tensors at the
        // correct shapes; verify load_indexer returns Some(IndexerWeights).
        // Shapes per ds4.c:2326-2331, 2610-2615 (commit ef0a490).
        let index_width = 2 * DSV4_N_INDEXER_HEAD_DIM; // 256
        let ratio = 4usize;

        let attn_q_b_elems = DSV4_N_LORA_Q * DSV4_N_INDEXER_HEAD * DSV4_N_INDEXER_HEAD_DIM;
        let proj_elems = DSV4_N_EMBD * DSV4_N_INDEXER_HEAD;
        let comp_ape_elems = index_width * ratio;
        let comp_kv_elems = DSV4_N_EMBD * index_width;
        let comp_gate_elems = DSV4_N_EMBD * index_width;
        let comp_norm_elems = DSV4_N_INDEXER_HEAD_DIM;

        let bytes = build_gguf(
            &[],
            &[
                (
                    "blk.2.indexer.attn_q_b.weight",
                    GgmlType::F32 as u32,
                    vec![DSV4_N_LORA_Q as u64, (DSV4_N_INDEXER_HEAD * DSV4_N_INDEXER_HEAD_DIM) as u64],
                    vec![0u8; attn_q_b_elems * 4],
                ),
                (
                    "blk.2.indexer.proj.weight",
                    GgmlType::F32 as u32,
                    vec![DSV4_N_EMBD as u64, DSV4_N_INDEXER_HEAD as u64],
                    vec![0u8; proj_elems * 4],
                ),
                (
                    "blk.2.indexer_compressor_ape.weight",
                    GgmlType::F32 as u32,
                    vec![index_width as u64, ratio as u64],
                    vec![0u8; comp_ape_elems * 4],
                ),
                (
                    "blk.2.indexer_compressor_kv.weight",
                    GgmlType::F32 as u32,
                    vec![DSV4_N_EMBD as u64, index_width as u64],
                    vec![0u8; comp_kv_elems * 4],
                ),
                (
                    "blk.2.indexer_compressor_gate.weight",
                    GgmlType::F32 as u32,
                    vec![DSV4_N_EMBD as u64, index_width as u64],
                    vec![0u8; comp_gate_elems * 4],
                ),
                (
                    "blk.2.indexer_compressor_norm.weight",
                    GgmlType::F32 as u32,
                    vec![comp_norm_elems as u64],
                    vec![0u8; comp_norm_elems * 4],
                ),
            ],
        );
        let file = GgufFile::from_bytes(bytes).unwrap();
        let iw = load_indexer(&file, 2).unwrap().expect("Some");
        // Make all shape checks load-bearing.
        assert_eq!(iw.attn_q_b.byte_len(), attn_q_b_elems * 4);
        assert_eq!(iw.proj.byte_len(), proj_elems * 4);
        assert_eq!(iw.comp_ape.byte_len(), comp_ape_elems * 4);
        assert_eq!(iw.comp_kv.byte_len(), comp_kv_elems * 4);
        assert_eq!(iw.comp_gate.byte_len(), comp_gate_elems * 4);
        assert_eq!(iw.comp_norm.len(), comp_norm_elems);
    }

    #[test]
    fn resolve_i32_slice_returns_data_for_i32_tensor() {
        let values: [i32; 4] = [10, -5, 0, i32::MAX];
        let mut payload = Vec::with_capacity(16);
        for v in &values {
            payload.extend_from_slice(&v.to_le_bytes());
        }
        let bytes = build_gguf(
            &[],
            &[("toy_i32", GgmlType::I32 as u32, vec![4], payload)],
        );
        let file = GgufFile::from_bytes(bytes).unwrap();
        let s = resolve_i32_slice(&file, "toy_i32").unwrap();
        assert_eq!(s, &values);
    }

    #[test]
    fn resolve_i32_slice_rejects_f32_tensor() {
        let payload = f32s_to_bytes(&[1.0, 2.0]);
        let bytes = build_gguf(&[], &[("toy", GgmlType::F32 as u32, vec![2], payload)]);
        let file = GgufFile::from_bytes(bytes).unwrap();
        let err = resolve_i32_slice(&file, "toy").unwrap_err();
        assert!(matches!(err, Error::ShapeMismatch { .. }));
    }

    #[test]
    fn load_block_missing_attn_norm_surfaces_layer_index() {
        // Empty GGUF — every per-layer tensor is missing. The first
        // failure should mention "blk.5.attn_norm.weight" precisely.
        let bytes = build_gguf(&[], &[]);
        let file = GgufFile::from_bytes(bytes).unwrap();
        let err = load_block(&file, 5).unwrap_err();
        match err {
            Error::MissingTensor(name) => {
                assert!(name.contains("blk.5"), "got {name}");
                assert!(name.contains("attn_norm"), "got {name}");
            }
            other => panic!("expected MissingTensor, got {other:?}"),
        }
    }

    #[test]
    fn load_dsv4_flash_metadata_failure_surfaces() {
        // Empty GGUF lacks `general.architecture` etc — validate_metadata
        // must reject before any tensor lookup runs.
        let bytes = build_gguf(&[], &[]);
        let file = GgufFile::from_bytes(bytes).unwrap();
        let err = load_dsv4_flash(&file).unwrap_err();
        // validate_metadata yields MissingMetadata; either that or
        // ShapeMismatch is acceptable — both indicate the metadata
        // sanity step fired before any tensor lookup.
        assert!(matches!(
            err,
            Error::MissingMetadata(_) | Error::ShapeMismatch { .. }
        ));
    }

    #[test]
    fn architecture_name_reads_metadata() {
        let mut key_buf = Vec::new();
        // Pack a String value: 8-byte length prefix + bytes.
        let v = "deepseek-v4-flash";
        key_buf.extend_from_slice(&(v.len() as u64).to_le_bytes());
        key_buf.extend_from_slice(v.as_bytes());
        let bytes = build_gguf(
            &[("general.architecture", 8 /* String */, key_buf)],
            &[],
        );
        let file = GgufFile::from_bytes(bytes).unwrap();
        assert_eq!(architecture_name(file.metadata()), Some("deepseek-v4-flash"));
    }

    #[test]
    fn load_router_tid2eid_wrong_length_carries_layer_index() {
        // Build a tid2eid that's correctly-typed (I32) but has the
        // wrong row count. load_router's explicit length check should
        // fire AND the error should mention "blk.1.ffn_hash_tid2eid".
        let bad = vec![0i32; DSV4_N_EXPERT_USED * DSV4_N_VOCAB - 1];
        let mut payload = Vec::with_capacity(bad.len() * 4);
        for v in &bad {
            payload.extend_from_slice(&v.to_le_bytes());
        }
        // The gate_inp also needs to exist before load_router gets to
        // tid2eid (the function resolves gate_inp first). Build a
        // correctly-shaped F32 gate_inp.
        let gate_inp_bytes = vec![0u8; DSV4_N_EXPERT * DSV4_N_EMBD * 4];
        let bytes = build_gguf(
            &[],
            &[
                (
                    "blk.1.ffn_gate_inp.weight",
                    GgmlType::F32 as u32,
                    vec![DSV4_N_EMBD as u64, DSV4_N_EXPERT as u64],
                    gate_inp_bytes,
                ),
                (
                    "blk.1.ffn_hash_tid2eid",
                    GgmlType::I32 as u32,
                    vec![bad.len() as u64],
                    payload,
                ),
            ],
        );
        let file = GgufFile::from_bytes(bytes).unwrap();
        let err = load_router(&file, 1).unwrap_err();
        match err {
            Error::ShapeMismatch { key, actual, .. } => {
                assert_eq!(key, "loader.router.tid2eid");
                assert!(
                    actual.contains("blk.1.ffn_hash_tid2eid"),
                    "expected layer index in error, got {actual}"
                );
            }
            other => panic!("expected ShapeMismatch, got {other:?}"),
        }
    }

    #[test]
    fn expected_layer_tensor_count_grows_with_regime() {
        // Reference table: baseline = 23 per the function's doc
        // (post-F010.A: HC contributes 6 tensors; post-F010.B:
        // compressor contributes 4 tensors per compressed layer;
        // post-F010.C: indexer contributes 6 tensors on ratio-4 layers).
        //   il=0: dense + hash router      → 23 +  0      + 1 = 24
        //   il=2: ratio-4 + hash router    → 23 +  4 +  6 + 1 = 34
        //   il=3: ratio-128 + topk router  → 23 +  4 +  0 + 0 = 27
        //   il=4: ratio-4 + topk router    → 23 +  4 +  6 + 0 = 33
        assert_eq!(expected_layer_tensor_count(0), 24);
        assert_eq!(expected_layer_tensor_count(2), 34);
        assert_eq!(expected_layer_tensor_count(3), 27);
        assert_eq!(expected_layer_tensor_count(4), 33);
    }
}
