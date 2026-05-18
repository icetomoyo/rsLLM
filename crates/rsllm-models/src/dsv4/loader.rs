//! GGUF → [`crate::DeepSeekV4Flash`] loader.
//!
//! Resolves the per-layer tensor lookups, picks the right `WeightBlob`
//! variant from the GGUF declared type, and threads them through the
//! `DsV4Block` slots defined in F005 (+ F008.B compressor/indexer
//! additions).
//!
//! ## Tensor naming
//!
//! Names follow llama.cpp's DeepSeek convention plus the F008.B
//! `attn_compressor` / `attn_indexer` extensions. The full mapping
//! lives in [`tensor_names`] as a single source of truth — if ds4
//! upstream renames anything, this is the only file to patch.
//!
//! The well-known names (`token_embd.weight`, `output_norm.weight`,
//! `blk.N.attn_norm.weight`, the MLA q/kv LoRA family, the MoE
//! `ffn_*_exps.weight` triplet, etc.) are stable across llama.cpp
//! forks. The F008.B additions (`attn_compressor.weight`,
//! `attn_indexer_kv.weight`, `attn_indexer_kv_score.weight`,
//! `attn_indexer_q.weight`, `attn_indexer_head_weight.weight`)
//! and the HC / shared-expert / hash-router fields are best-guess
//! names against ds4 commit `ef0a490` — flagged with `// TODO(ds4)`
//! comments where a checked-against-source confirmation is still
//! pending. The structural surface (this loader's interface, error
//! shape, per-layer regime dispatch) does not depend on the exact
//! strings.
//!
//! Ported by reference from `ds4.c:1846-1853` and the surrounding
//! layer-weight struct (MIT, The ds4.c authors). Line numbers pinned
//! to ds4 commit `ef0a490` (2026-05-17).

use rsllm_gguf::{GgmlType, GgufFile, TensorInfo};
use rsllm_kvcache::dsv4::shape::{layer_compress_ratio, layer_has_indexer};

use crate::Error;
use crate::dsv4::compressor::{CompressorWeights, IndexerReadWeights, IndexerWriteWeights};
use crate::dsv4::shape::{
    DSV4_HEAD_DIM, DSV4_N_EMBD, DSV4_N_INDEXER_HEAD, DSV4_N_INDEXER_HEAD_DIM,
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

    /// HC pre/post weights — TODO(ds4): confirm exact strings.
    pub const HC_PRE_ATTN_W: &str = "hc_pre_attn.weight";
    pub const HC_PRE_ATTN_BASE: &str = "hc_pre_attn.base";
    pub const HC_POST_ATTN_W: &str = "hc_post_attn.weight";
    pub const HC_POST_ATTN_BASE: &str = "hc_post_attn.base";
    pub const HC_PRE_FFN_W: &str = "hc_pre_ffn.weight";
    pub const HC_PRE_FFN_BASE: &str = "hc_pre_ffn.base";
    pub const HC_POST_FFN_W: &str = "hc_post_ffn.weight";
    pub const HC_POST_FFN_BASE: &str = "hc_post_ffn.base";

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

    /// F008.B compressor / indexer LoRAs — TODO(ds4): confirm strings.
    pub const ATTN_COMPRESSOR: &str = "attn_compressor.weight";
    pub const ATTN_INDEXER_KV: &str = "attn_indexer_kv.weight";
    pub const ATTN_INDEXER_KV_SCORE: &str = "attn_indexer_kv_score.weight";
    pub const ATTN_INDEXER_Q: &str = "attn_indexer_q.weight";
    /// TODO(ds4): the `.weight` suffix is conjectural — the head-weight
    /// vector may be stored without it under the canonical name
    /// `attn_indexer_head_weight`. Patch when confirmed against
    /// ds4 upstream.
    pub const ATTN_INDEXER_HEAD_WEIGHT: &str = "attn_indexer_head_weight";

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

/// Build the F008.B `CompressorWeights` for layer `il`, **if** the
/// layer regime calls for it. Returns `None` for dense layers
/// (`compress_ratio == 0`).
///
/// # Errors
/// [`Error::MissingTensor`] if the layer needs a compressor but the
/// GGUF doesn't have one.
pub fn load_compressor<'a>(
    gguf: &'a GgufFile,
    il: usize,
) -> Result<Option<CompressorWeights<'a>>, Error> {
    if layer_compress_ratio(il) == 0 {
        return Ok(None);
    }
    let name = tensor_names::blk(il, tensor_names::ATTN_COMPRESSOR);
    let blob = resolve_blob_opt(gguf, &name)?
        .ok_or_else(|| Error::MissingTensor(name.clone()))?;
    blob.check_shape(
        DSV4_HEAD_DIM,
        DSV4_N_EMBD,
        "loader.compressor.attn_compressor",
    )?;
    Ok(Some(CompressorWeights { attn_compressor: blob }))
}

/// Build the F008.B indexer pair for layer `il`. Returns `None` on
/// non-ratio-4 layers.
///
/// # Errors
/// [`Error::MissingTensor`] if the layer needs indexer weights but
/// the GGUF is missing one of the four expected names.
pub fn load_indexer<'a>(
    gguf: &'a GgufFile,
    il: usize,
) -> Result<Option<(IndexerWriteWeights<'a>, IndexerReadWeights<'a>)>, Error> {
    if !layer_has_indexer(il) {
        return Ok(None);
    }
    let kv_name = tensor_names::blk(il, tensor_names::ATTN_INDEXER_KV);
    let kv_blob = resolve_blob_opt(gguf, &kv_name)?
        .ok_or_else(|| Error::MissingTensor(kv_name.clone()))?;
    let score_name = tensor_names::blk(il, tensor_names::ATTN_INDEXER_KV_SCORE);
    let score_blob = resolve_blob_opt(gguf, &score_name)?
        .ok_or_else(|| Error::MissingTensor(score_name.clone()))?;
    let q_name = tensor_names::blk(il, tensor_names::ATTN_INDEXER_Q);
    let q_blob = resolve_blob_opt(gguf, &q_name)?
        .ok_or_else(|| Error::MissingTensor(q_name.clone()))?;
    let head_w_name = tensor_names::blk(il, tensor_names::ATTN_INDEXER_HEAD_WEIGHT);
    let head_w_blob = resolve_blob_opt(gguf, &head_w_name)?
        .ok_or_else(|| Error::MissingTensor(head_w_name.clone()))?;
    let head_w = match head_w_blob {
        WeightBlob::F32(s) => s,
        WeightBlob::Quant { .. } => {
            return Err(Error::ShapeMismatch {
                key: "loader.indexer.head_weight",
                expected: "F32".into(),
                actual: format!("{head_w_name}: quantised"),
            });
        }
    };

    // Shape checks (mirrors the in-DsV4Block validation in F008.B).
    kv_blob.check_shape(
        DSV4_N_INDEXER_HEAD_DIM,
        DSV4_N_EMBD,
        "loader.indexer.kv",
    )?;
    score_blob.check_shape(
        DSV4_N_INDEXER_HEAD_DIM,
        DSV4_N_EMBD,
        "loader.indexer.kv_score",
    )?;
    q_blob.check_shape(
        DSV4_N_INDEXER_HEAD * DSV4_N_INDEXER_HEAD_DIM,
        DSV4_N_EMBD,
        "loader.indexer.q",
    )?;
    if head_w.len() != DSV4_N_INDEXER_HEAD {
        return Err(Error::ShapeMismatch {
            key: "loader.indexer.head_weight",
            expected: format!("{DSV4_N_INDEXER_HEAD}"),
            actual: format!("{}", head_w.len()),
        });
    }

    Ok(Some((
        IndexerWriteWeights {
            attn_indexer_kv: kv_blob,
            attn_indexer_kv_score: score_blob,
        },
        IndexerReadWeights {
            attn_indexer_q: q_blob,
            attn_indexer_head_weight: head_w,
        },
    )))
}

/// Report the dimensions of all expected MLA weights so a caller
/// can summarise which tensors were located vs missing without
/// actually building a model. Used by `rsllm inspect` to surface
/// the per-layer regime distribution.
#[must_use]
pub fn expected_layer_tensor_count(il: usize) -> usize {
    // 8 baseline (norm, q_a, q_a_norm, q_b, kv_a, kv_a_norm, sinks,
    // output_a, output_b) — that's 9. Plus 4 HC. Plus 4 MoE
    // (gate/up/down + shared triplet + router gate_inp + tid2eid
    // when hash). Numeric value isn't load-bearing for any test —
    // it's an informational helper.
    let mut count = 9 + 4 + 7 + 1; // mla(7)+sinks(1)+out(2)+hc(4)+ffn_norm(1)+moe(6)+router(1) ≈ 21
    if layer_compress_ratio(il) > 0 {
        count += 1; // compressor
    }
    if layer_has_indexer(il) {
        count += 4; // indexer kv/score/q/head_w
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
        // Build a fake compressor with the right shape — HEAD_DIM ×
        // N_EMBD = 512 × 4096 = 2_097_152 f32 elements = 8 MiB.
        // Too big for the inline builder, so use a constant-fill
        // shortcut: byte-fill with zeros.
        let n_elem = DSV4_HEAD_DIM * DSV4_N_EMBD;
        let payload = vec![0u8; n_elem * 4];
        let bytes = build_gguf(
            &[],
            &[(
                "blk.3.attn_compressor.weight",
                GgmlType::F32 as u32,
                vec![DSV4_N_EMBD as u64, DSV4_HEAD_DIM as u64],
                payload,
            )],
        );
        let file = GgufFile::from_bytes(bytes).unwrap();
        let c = load_compressor(&file, 3).unwrap().expect("Some");
        let _ = c.attn_compressor.byte_len();
    }

    #[test]
    fn load_indexer_dense_layer_returns_none() {
        let bytes = build_gguf(&[], &[]);
        let file = GgufFile::from_bytes(bytes).unwrap();
        assert!(load_indexer(&file, 0).unwrap().is_none());
        assert!(load_indexer(&file, 3).unwrap().is_none()); // ratio-128, no indexer
    }

    #[test]
    fn load_indexer_ratio4_layer_requires_all_four() {
        // Layer 2 is the first ratio-4 layer; missing any of the four
        // expected indexer tensors must produce a clear error.
        let bytes = build_gguf(&[], &[]);
        let file = GgufFile::from_bytes(bytes).unwrap();
        let err = load_indexer(&file, 2).unwrap_err();
        assert!(matches!(err, Error::MissingTensor(_)));
    }

    #[test]
    fn load_indexer_returns_pair_when_all_tensors_present() {
        // Layer 2: ratio-4. Build the four required tensors at the
        // correct shapes; verify load_indexer returns Some((write, read)).
        let kv_elems = DSV4_N_INDEXER_HEAD_DIM * DSV4_N_EMBD;
        let q_elems = DSV4_N_INDEXER_HEAD * DSV4_N_INDEXER_HEAD_DIM * DSV4_N_EMBD;
        let head_w_elems = DSV4_N_INDEXER_HEAD;
        let bytes = build_gguf(
            &[],
            &[
                (
                    "blk.2.attn_indexer_kv.weight",
                    GgmlType::F32 as u32,
                    vec![DSV4_N_EMBD as u64, DSV4_N_INDEXER_HEAD_DIM as u64],
                    vec![0u8; kv_elems * 4],
                ),
                (
                    "blk.2.attn_indexer_kv_score.weight",
                    GgmlType::F32 as u32,
                    vec![DSV4_N_EMBD as u64, DSV4_N_INDEXER_HEAD_DIM as u64],
                    vec![0u8; kv_elems * 4],
                ),
                (
                    "blk.2.attn_indexer_q.weight",
                    GgmlType::F32 as u32,
                    vec![
                        DSV4_N_EMBD as u64,
                        (DSV4_N_INDEXER_HEAD * DSV4_N_INDEXER_HEAD_DIM) as u64,
                    ],
                    vec![0u8; q_elems * 4],
                ),
                (
                    "blk.2.attn_indexer_head_weight",
                    GgmlType::F32 as u32,
                    vec![head_w_elems as u64],
                    vec![0u8; head_w_elems * 4],
                ),
            ],
        );
        let file = GgufFile::from_bytes(bytes).unwrap();
        let (write, read) = load_indexer(&file, 2).unwrap().expect("Some");
        assert_eq!(read.attn_indexer_head_weight.len(), DSV4_N_INDEXER_HEAD);
        // Numerical shape assertions — make the shape check load-bearing
        // (a future regression that produces a smaller blob would have
        // been caught silently before this fix).
        assert_eq!(write.attn_indexer_kv.byte_len(), kv_elems * 4);
        assert_eq!(write.attn_indexer_kv_score.byte_len(), kv_elems * 4);
        assert_eq!(read.attn_indexer_q.byte_len(), q_elems * 4);
    }

    #[test]
    fn expected_layer_tensor_count_grows_with_regime() {
        // Dense layer: baseline count.
        let dense = expected_layer_tensor_count(0);
        // Ratio-128 layer: baseline + 1 (compressor).
        let r128 = expected_layer_tensor_count(3);
        // Ratio-4 layer: baseline + 1 (compressor) + 4 (indexer).
        let r4 = expected_layer_tensor_count(2);
        assert_eq!(r128, dense + 1);
        assert_eq!(r4, dense + 5);
    }
}
