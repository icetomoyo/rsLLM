//! Indexer query projection helpers — the per-token side of the
//! ratio-4 indexer that produces a per-head query vector and a
//! per-head soft gate for the [`rsllm_kvcache::dsv4::indexer::IndexerPool`]
//! top-K selection (F011.E.C).
//!
//! Mirrors `ds4.c:6862-6915` `indexer_allowed_decode_one`. The
//! emitted indexer rows themselves (the long-history compressed
//! state) are produced by `compressor_decode_one` via
//! [`IndexerWeights::as_compressor_view`]; this module covers the
//! *query* side of the indexer pipeline.
//!
//! Two helpers:
//!
//! - [`project_indexer_query`] (`ds4.c:6880-6881`): matmul through
//!   `indexer.attn_q_b` + per-head RoPE tail rotation. Produces an
//!   `[N_INDEXER_HEAD × N_INDEXER_HEAD_DIM] = [64 × 128] = [8192]`
//!   query vector ready to be dotted against each pool row.
//!
//! - [`project_indexer_weights`] (`ds4.c:6883-6885`): matvec through
//!   `indexer.proj` + scale by `1/sqrt(head_dim × n_head)`. Produces
//!   the per-head soft gate `[N_INDEXER_HEAD] = [64]` that weighs
//!   each head's contribution to the per-row score.
//!
//! Ported by reference from `ds4.c` (MIT, The ds4.c authors).
//! Line numbers pinned to ds4 commit `ef0a490` (2026-05-17).

use rsllm_backend_cpu::SimdTier;
use rsllm_backend_cpu::ops::rope::{RoPEParams, rope_yarn_tail};

use super::compressor::IndexerWeights;
use super::shape::{
    DSV4_N_EMBD, DSV4_N_INDEXER_HEAD, DSV4_N_INDEXER_HEAD_DIM, DSV4_N_LORA_Q, DSV4_N_ROT,
    DSV4_ROPE_FREQ_BASE, DSV4_ROPE_ORIG_CTX,
};
use super::weight::matmul_weight_f32;
use crate::Error;

/// Project the LoRA-Q-normalised intermediate `qr_norm` through the
/// layer's `indexer.attn_q_b` matrix and apply the standard MLA RoPE
/// tail rotation to each of the `N_INDEXER_HEAD = 64` heads.
///
/// Mirrors `ds4.c:6880-6881`:
/// ```c
/// matvec_any(q, model, layer->indexer_attn_q_b, qr_norm);
/// rope_tail_layer_inplace(q, n_head, head_dim, DS4_N_ROT, pos, il, false);
/// ```
///
/// **TODO(F012): upstream additionally runs**
/// `dsv4_indexer_qat_rows_inplace_cpu(q, n_head, head_dim)` here
/// (`ds4.c:6969`), a per-head Hadamard-128 + FP4 quantisation-aware
/// transform (`ds4.c:1677-1709`). Without it the top-K scoring is
/// numerically off vs the model's reference graph — but the call
/// path stays sound. F012 will land the Hadamard + FP4 kernels and
/// wire them here plus at the indexer compressor pool emission
/// (`ds4.c:6585`). Until then, `project_indexer_query` returns a
/// RoPE-only query; expect a measurable drop in top-K precision.
///
/// # Arguments
/// - `qr_norm` — single token's LoRA-Q post-norm latent, length
///   `DSV4_N_LORA_Q = 1024`. Produced by `mla::mla_projections` and
///   re-used as input to both the MLA attention and the indexer query
///   projection. The MLA path consumes a different up-projection
///   (`mla.attn_q_b`); this is the indexer's parallel one.
/// - `weights` — `IndexerWeights` for the current layer (only
///   `attn_q_b` is read).
/// - `pos` — absolute token sequence position used to seed RoPE.
/// - `il` — layer index. Reserved for future per-layer RoPE caching;
///   not currently read by the kernel.
/// - `q_out` — write-only buffer, length
///   `DSV4_N_INDEXER_HEAD * DSV4_N_INDEXER_HEAD_DIM = 8192`. Receives
///   the rotated per-head query.
/// - `tier` — SIMD tier for the matmul.
///
/// # Errors
/// - [`Error::ShapeMismatch`] on any length disagreement.
/// - Errors bubbled from `matmul_weight_f32` or `rope_yarn_tail`.
#[allow(clippy::too_many_arguments)]
pub fn project_indexer_query(
    qr_norm: &[f32],
    weights: &IndexerWeights<'_>,
    pos: u32,
    _il: u32,
    q_out: &mut [f32],
    tier: SimdTier,
) -> Result<(), Error> {
    if qr_norm.len() != DSV4_N_LORA_Q {
        return Err(Error::ShapeMismatch {
            key: "project_indexer_query.qr_norm",
            expected: format!("{DSV4_N_LORA_Q}"),
            actual: format!("{}", qr_norm.len()),
        });
    }
    let expected_q = DSV4_N_INDEXER_HEAD * DSV4_N_INDEXER_HEAD_DIM;
    if q_out.len() != expected_q {
        return Err(Error::ShapeMismatch {
            key: "project_indexer_query.q_out",
            expected: format!("{expected_q}"),
            actual: format!("{}", q_out.len()),
        });
    }

    // matvec: q_out = attn_q_b * qr_norm.
    // attn_q_b shape: [N_LORA_Q × (N_INDEXER_HEAD × N_INDEXER_HEAD_DIM)]
    // = [1024 × 8192]. n_tok = 1 here (per-token call site).
    matmul_weight_f32(
        q_out,
        &weights.attn_q_b,
        qr_norm,
        1,
        DSV4_N_LORA_Q,
        expected_q,
        tier,
    )?;

    // Per-head RoPE tail rotation. The standard MLA RoPE regime is
    // freq_base = 10000, freq_scale = 1.0, ext_factor = 0 (no YaRN
    // ramp at v0.1.0 base context). Matches `rope_params_at` in
    // mla.rs except for the n_head / head_dim values.
    let params = indexer_query_rope_params(pos);
    rope_yarn_tail(q_out, &params, tier).map_err(|e| Error::ShapeMismatch {
        key: "project_indexer_query.rope",
        expected: "valid RoPE tail rotation".to_string(),
        actual: format!("{e}"),
    })?;
    Ok(())
}

/// Project the per-token residual (post-RMSNorm) through the layer's
/// `indexer.proj` matrix and apply the model's standard per-head
/// scale factor `1 / sqrt(head_dim × n_head)`. Mirrors `ds4.c:6883-6885`:
/// ```c
/// matvec_any(weights, model, layer->indexer_proj, cur);
/// const float scale = 1.0f / sqrtf((float)(head_dim * n_head));
/// for (uint32_t h = 0; h < n_head; h++) weights[h] *= scale;
/// ```
///
/// # Arguments
/// - `cur` — single token's post-norm residual, length `DSV4_N_EMBD = 4096`.
/// - `weights` — `IndexerWeights` (only `proj` is read).
/// - `w_out` — write-only, length `DSV4_N_INDEXER_HEAD = 64`. Receives
///   the per-head scaled soft-gate weights.
/// - `tier` — SIMD tier for the matvec.
///
/// # Errors
/// - [`Error::ShapeMismatch`] on input/output length disagreement.
/// - Errors bubbled from `matmul_weight_f32`.
pub fn project_indexer_weights(
    cur: &[f32],
    weights: &IndexerWeights<'_>,
    w_out: &mut [f32],
    tier: SimdTier,
) -> Result<(), Error> {
    if cur.len() != DSV4_N_EMBD {
        return Err(Error::ShapeMismatch {
            key: "project_indexer_weights.cur",
            expected: format!("{DSV4_N_EMBD}"),
            actual: format!("{}", cur.len()),
        });
    }
    if w_out.len() != DSV4_N_INDEXER_HEAD {
        return Err(Error::ShapeMismatch {
            key: "project_indexer_weights.w_out",
            expected: format!("{DSV4_N_INDEXER_HEAD}"),
            actual: format!("{}", w_out.len()),
        });
    }

    // matvec: w_out = proj * cur.
    // proj shape: [N_EMBD × N_INDEXER_HEAD] = [4096 × 64].
    matmul_weight_f32(
        w_out,
        &weights.proj,
        cur,
        1,
        DSV4_N_EMBD,
        DSV4_N_INDEXER_HEAD,
        tier,
    )?;

    // Per-head scale = 1 / sqrt(head_dim * n_head). Exposed by the
    // kvcache crate's `indexer::scale_factor` const; we recompute it
    // locally to avoid a cross-crate import and stay self-contained.
    // The two definitions must agree — guarded by a debug-only assert
    // at the bottom of `select_top_k_call_site_consistency` test.
    let scale = 1.0_f32 / ((DSV4_N_INDEXER_HEAD * DSV4_N_INDEXER_HEAD_DIM) as f32).sqrt();
    for w in w_out.iter_mut() {
        *w *= scale;
    }
    Ok(())
}

/// RoPE parameter bundle for the indexer query path. Standard MLA
/// regime (freq_base = 10000, no YaRN ramp), per-head rotation over
/// the last `N_ROT = 64` lanes of each of `N_INDEXER_HEAD = 64` heads.
fn indexer_query_rope_params(pos: u32) -> RoPEParams {
    RoPEParams {
        n_head: DSV4_N_INDEXER_HEAD as u32,
        head_dim: DSV4_N_INDEXER_HEAD_DIM as u32,
        n_rot: DSV4_N_ROT as u32,
        pos,
        n_ctx_orig: DSV4_ROPE_ORIG_CTX,
        freq_base: DSV4_ROPE_FREQ_BASE,
        freq_scale: 1.0,
        ext_factor: 0.0,
        attn_factor: 1.0,
        beta_fast: 32.0,
        beta_slow: 1.0,
        inverse: false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::compressor::IndexerWeights;
    use super::super::weight::WeightBlob;

    fn make_indexer_weights<'a>(
        attn_q_b: &'a [f32],
        proj: &'a [f32],
        comp_ape: &'a [f32],
        comp_kv: &'a [f32],
        comp_gate: &'a [f32],
        comp_norm: &'a [f32],
    ) -> IndexerWeights<'a> {
        IndexerWeights {
            attn_q_b: WeightBlob::F32(attn_q_b),
            proj: WeightBlob::F32(proj),
            comp_ape: WeightBlob::F32(comp_ape),
            comp_kv: WeightBlob::F32(comp_kv),
            comp_gate: WeightBlob::F32(comp_gate),
            comp_norm,
        }
    }

    #[test]
    fn project_indexer_query_rejects_wrong_qr_norm_length() {
        let attn_q_b = vec![0.0_f32; DSV4_N_LORA_Q * DSV4_N_INDEXER_HEAD * DSV4_N_INDEXER_HEAD_DIM];
        let proj = vec![0.0_f32; 1];
        let comp_ape = vec![0.0_f32; 1];
        let comp_kv = vec![0.0_f32; 1];
        let comp_gate = vec![0.0_f32; 1];
        let comp_norm = vec![0.0_f32; 1];
        let w = make_indexer_weights(&attn_q_b, &proj, &comp_ape, &comp_kv, &comp_gate, &comp_norm);
        let qr_norm = vec![0.0_f32; DSV4_N_LORA_Q - 1]; // wrong
        let mut q = vec![0.0_f32; DSV4_N_INDEXER_HEAD * DSV4_N_INDEXER_HEAD_DIM];
        let err = project_indexer_query(&qr_norm, &w, 0, 0, &mut q, SimdTier::Scalar)
            .unwrap_err();
        assert!(matches!(err, Error::ShapeMismatch { .. }));
    }

    #[test]
    fn project_indexer_query_rejects_wrong_q_out_length() {
        let attn_q_b = vec![0.0_f32; DSV4_N_LORA_Q * DSV4_N_INDEXER_HEAD * DSV4_N_INDEXER_HEAD_DIM];
        let proj = vec![0.0_f32; 1];
        let comp_ape = vec![0.0_f32; 1];
        let comp_kv = vec![0.0_f32; 1];
        let comp_gate = vec![0.0_f32; 1];
        let comp_norm = vec![0.0_f32; 1];
        let w = make_indexer_weights(&attn_q_b, &proj, &comp_ape, &comp_kv, &comp_gate, &comp_norm);
        let qr_norm = vec![0.0_f32; DSV4_N_LORA_Q];
        let mut q = vec![0.0_f32; 7]; // wrong
        let err = project_indexer_query(&qr_norm, &w, 0, 0, &mut q, SimdTier::Scalar)
            .unwrap_err();
        assert!(matches!(err, Error::ShapeMismatch { .. }));
    }

    /// With an all-zero `attn_q_b` and a non-zero `qr_norm`, the matvec
    /// produces all zeros — RoPE on zeros stays zero. Verifies the
    /// full pipeline executes without panic and writes a deterministic
    /// output. (A non-zero-weight numerical check would need
    /// hand-rolled RoPE expectations; that's covered by F008's MLA
    /// tests which exercise the same `rope_yarn_tail` kernel.)
    #[test]
    fn project_indexer_query_zero_weights_yields_zero_output() {
        let attn_q_b = vec![0.0_f32; DSV4_N_LORA_Q * DSV4_N_INDEXER_HEAD * DSV4_N_INDEXER_HEAD_DIM];
        let proj = vec![0.0_f32; 1];
        let comp_ape = vec![0.0_f32; 1];
        let comp_kv = vec![0.0_f32; 1];
        let comp_gate = vec![0.0_f32; 1];
        let comp_norm = vec![0.0_f32; 1];
        let w = make_indexer_weights(&attn_q_b, &proj, &comp_ape, &comp_kv, &comp_gate, &comp_norm);
        let qr_norm: Vec<f32> = (0..DSV4_N_LORA_Q).map(|i| (i as f32) + 1.0).collect();
        let mut q = vec![99.0_f32; DSV4_N_INDEXER_HEAD * DSV4_N_INDEXER_HEAD_DIM];
        project_indexer_query(&qr_norm, &w, 5, 2, &mut q, SimdTier::Scalar).unwrap();
        assert!(q.iter().all(|&v| v == 0.0), "expected all zeros, got non-zero");
    }

    #[test]
    fn project_indexer_query_identity_weights_replicates_qr_norm_lanes() {
        // Set attn_q_b so output lane `o` reads input lane `o % N_LORA_Q`
        // (identity-truncation with wraparound). The matvec then gives
        // `q[o] = qr_norm[o % N_LORA_Q]`. With `qr_norm[k] = (k+1)*0.1`
        // we get a deterministic gradient across all output lanes that
        // RoPE leaves untouched on lanes `[0, head_dim - N_ROT) = [0, 64)`.
        let out_dim = DSV4_N_INDEXER_HEAD * DSV4_N_INDEXER_HEAD_DIM;
        let mut attn_q_b = vec![0.0_f32; out_dim * DSV4_N_LORA_Q];
        for o in 0..out_dim {
            attn_q_b[o * DSV4_N_LORA_Q + (o % DSV4_N_LORA_Q)] = 1.0;
        }
        let proj = vec![0.0_f32; 1];
        let comp_ape = vec![0.0_f32; 1];
        let comp_kv = vec![0.0_f32; 1];
        let comp_gate = vec![0.0_f32; 1];
        let comp_norm = vec![0.0_f32; 1];
        let w = make_indexer_weights(&attn_q_b, &proj, &comp_ape, &comp_kv, &comp_gate, &comp_norm);
        let qr_norm: Vec<f32> =
            (0..DSV4_N_LORA_Q).map(|i| (i as f32 + 1.0) * 0.1).collect();
        let mut q = vec![0.0_f32; out_dim];
        project_indexer_query(&qr_norm, &w, 0, 0, &mut q, SimdTier::Scalar).unwrap();
        // Check every head's non-rotated prefix lanes
        // (`[0, head_dim - N_ROT) = [0, 64)`); each lane `d` of head `h`
        // should equal `qr_norm[(h * head_dim + d) % N_LORA_Q]`.
        let n_nope = DSV4_N_INDEXER_HEAD_DIM - DSV4_N_ROT;
        for h in 0..DSV4_N_INDEXER_HEAD {
            for d in 0..n_nope {
                let o = h * DSV4_N_INDEXER_HEAD_DIM + d;
                let expected = qr_norm[o % DSV4_N_LORA_Q];
                let v = q[o];
                assert!(
                    (v - expected).abs() < 1e-5,
                    "head {h} lane {d} (out idx {o}): got {v}, expected {expected}"
                );
            }
        }
    }

    #[test]
    fn project_indexer_weights_rejects_wrong_cur_length() {
        let attn_q_b = vec![0.0_f32; 1];
        let proj = vec![0.0_f32; DSV4_N_EMBD * DSV4_N_INDEXER_HEAD];
        let comp_ape = vec![0.0_f32; 1];
        let comp_kv = vec![0.0_f32; 1];
        let comp_gate = vec![0.0_f32; 1];
        let comp_norm = vec![0.0_f32; 1];
        let w = make_indexer_weights(&attn_q_b, &proj, &comp_ape, &comp_kv, &comp_gate, &comp_norm);
        let cur = vec![0.0_f32; DSV4_N_EMBD + 1];
        let mut out = vec![0.0_f32; DSV4_N_INDEXER_HEAD];
        let err = project_indexer_weights(&cur, &w, &mut out, SimdTier::Scalar).unwrap_err();
        assert!(matches!(err, Error::ShapeMismatch { .. }));
    }

    #[test]
    fn project_indexer_weights_applies_scale() {
        // identity-truncating proj: w_out[h] = cur[h] for h < N_INDEXER_HEAD.
        // Then each w_out[h] *= 1/sqrt(head_dim * n_head).
        let mut proj = vec![0.0_f32; DSV4_N_EMBD * DSV4_N_INDEXER_HEAD];
        for h in 0..DSV4_N_INDEXER_HEAD {
            proj[h * DSV4_N_EMBD + h] = 1.0;
        }
        let attn_q_b = vec![0.0_f32; 1];
        let comp_ape = vec![0.0_f32; 1];
        let comp_kv = vec![0.0_f32; 1];
        let comp_gate = vec![0.0_f32; 1];
        let comp_norm = vec![0.0_f32; 1];
        let w = make_indexer_weights(&attn_q_b, &proj, &comp_ape, &comp_kv, &comp_gate, &comp_norm);
        let mut cur = vec![0.0_f32; DSV4_N_EMBD];
        for (h, v) in cur.iter_mut().enumerate().take(DSV4_N_INDEXER_HEAD) {
            *v = (h as f32) + 1.0;
        }
        let mut out = vec![0.0_f32; DSV4_N_INDEXER_HEAD];
        project_indexer_weights(&cur, &w, &mut out, SimdTier::Scalar).unwrap();
        let scale = 1.0_f32
            / ((DSV4_N_INDEXER_HEAD * DSV4_N_INDEXER_HEAD_DIM) as f32).sqrt();
        for (h, &v) in out.iter().enumerate().take(DSV4_N_INDEXER_HEAD) {
            let expected = ((h as f32) + 1.0) * scale;
            assert!(
                (v - expected).abs() < 1e-6,
                "head {h}: got {v}, expected {expected}"
            );
        }
    }

    #[test]
    fn select_top_k_call_site_consistency() {
        // The scale factor we compute locally MUST match what
        // `rsllm_kvcache::dsv4::indexer::scale_factor()` returns — the
        // top-K selection assumes the caller has already applied this
        // factor, so a mismatch would silently bias scoring.
        let ours = 1.0_f32
            / ((DSV4_N_INDEXER_HEAD * DSV4_N_INDEXER_HEAD_DIM) as f32).sqrt();
        let theirs = rsllm_kvcache::dsv4::indexer::scale_factor();
        assert!((ours - theirs).abs() < 1e-9, "scale mismatch: {ours} vs {theirs}");
    }
}
