//! Multi-head Latent Attention Q/KV LoRA projections.
//!
//! DeepSeek V4 Flash compresses the Q and KV projections through a
//! LoRA bottleneck and then expands them back. The pre-attention
//! sequence is:
//!
//! ```text
//! x  (n_tok × N_EMBD = 4096)
//!   ├── attn_q_a:   4096 → 1024     (Q down-projection)
//!   │   └── q_a_norm:  RMSNorm on the 1024-d bottleneck
//!   │       └── attn_q_b: 1024 → N_HEAD × HEAD_DIM = 64 × 512  (Q up + head split)
//!   │           └── rope_yarn_tail: rotate trailing N_ROT = 64 lanes per head
//!   └── attn_kv_a:  4096 → HEAD_DIM = 512  (KV down-projection, 1-head latent)
//!       └── kv_a_norm: RMSNorm on the 512-d latent
//!           └── rope_yarn_tail: rotate trailing N_ROT = 64 lanes (on the single KV head)
//! ```
//!
//! v0.1.0 implements the **projection** part of MLA only. The actual
//! attention dot-product + softmax + KV cache write lives in F006 /
//! F007 once the three-tier cache is in place.
//!
//! Ported by reference from `ds4.c:1846-1853` (MIT, The ds4.c authors).

use rsllm_backend_cpu::SimdTier;
use rsllm_backend_cpu::ops::{RoPEParams, rmsnorm, rope_yarn_tail};

use super::shape::{DSV4_HEAD_DIM, DSV4_N_EMBD, DSV4_N_HEAD, DSV4_N_LORA_Q, DSV4_N_ROT, DSV4_RMS_EPS};
use super::weight::{WeightBlob, matmul_weight_f32};
use crate::Error;

/// Per-layer MLA weights, borrowed views into the mmap'd GGUF.
///
/// Names follow ds4 / llama.cpp naming convention:
/// - `attn_q_a` is the down-projection from the residual stream.
/// - `attn_q_b` is the up-projection that also implicitly splits heads.
/// - `attn_kv_a` is the KV down-projection (1 KV head per ds4).
/// - `*_norm` are the post-down-projection RMSNorm scale vectors.
#[derive(Debug, Clone, Copy)]
pub struct MlaWeights<'a> {
    /// Q LoRA down-projection: `[N_LORA_Q × N_EMBD]` = `[1024 × 4096]`.
    pub attn_q_a: WeightBlob<'a>,
    /// Q LoRA bottleneck RMSNorm scale: `[N_LORA_Q]` = `[1024]`.
    pub q_a_norm: &'a [f32],
    /// Q LoRA up-projection + head split:
    /// `[N_HEAD × HEAD_DIM × N_LORA_Q]` = `[64 × 512 × 1024]` row-major.
    pub attn_q_b: WeightBlob<'a>,
    /// KV down-projection: `[HEAD_DIM × N_EMBD]` = `[512 × 4096]`.
    pub attn_kv_a: WeightBlob<'a>,
    /// KV latent RMSNorm scale: `[HEAD_DIM]` = `[512]`.
    pub kv_a_norm: &'a [f32],
}

/// Buffers needed to run [`mla_projections`] for a batch of `n_tok`
/// tokens. Owned by the caller so we can re-use across layers and
/// avoid per-layer allocation.
#[derive(Debug)]
pub struct MlaScratch {
    /// `[n_tok × N_LORA_Q]` — Q bottleneck activations (pre-norm).
    pub q_lora: Vec<f32>,
    /// `[n_tok × N_LORA_Q]` — Q bottleneck activations (post-norm).
    /// Kept separately so we don't clone `q_lora` per token in the
    /// prefill loop (prefill can have n_tok up to several thousand).
    pub q_lora_normed: Vec<f32>,
    /// `[n_tok × HEAD_DIM]` — KV latent before RMSNorm.
    pub kv_latent: Vec<f32>,
}

impl MlaScratch {
    /// Allocate scratch sized for `n_tok` tokens.
    #[must_use]
    pub fn new(n_tok: usize) -> Self {
        Self {
            q_lora: vec![0.0_f32; n_tok * DSV4_N_LORA_Q],
            q_lora_normed: vec![0.0_f32; n_tok * DSV4_N_LORA_Q],
            kv_latent: vec![0.0_f32; n_tok * DSV4_HEAD_DIM],
        }
    }

    /// Resize scratch in place for a new `n_tok`. Cheap when the new
    /// size is smaller; reuses the existing allocation when it's not.
    pub fn resize(&mut self, n_tok: usize) {
        self.q_lora.resize(n_tok * DSV4_N_LORA_Q, 0.0);
        self.q_lora_normed.resize(n_tok * DSV4_N_LORA_Q, 0.0);
        self.kv_latent.resize(n_tok * DSV4_HEAD_DIM, 0.0);
    }
}

/// MLA projection output for one layer's `n_tok` tokens.
///
/// `q_out` carries the full multi-head Q latent (per-head dim
/// `HEAD_DIM = 512`, last `N_ROT = 64` lanes already RoPE-rotated).
/// `kv_out` carries the 1-head KV latent (same lane treatment).
pub struct MlaOutput<'a> {
    /// `[n_tok × N_HEAD × HEAD_DIM]` row-major.
    pub q: &'a mut [f32],
    /// `[n_tok × HEAD_DIM]` row-major.
    pub kv: &'a mut [f32],
}

/// Run the MLA Q/KV LoRA projection + norms + RoPE for one layer.
///
/// Both `q` and `kv` outputs already have RoPE applied to their
/// trailing `N_ROT` lanes. Callers feed these directly into the
/// attention dot-product stage (F006).
///
/// `tier` selects the SIMD path; pass [`SimdTier::Scalar`] for tests.
/// `position_of` maps token index → absolute sequence position. The
/// caller owns the position table because prefill and decode supply it
/// from different sources.
///
/// # Errors
/// Bubbles up shape errors from the underlying kernels.
pub fn mla_projections(
    weights: &MlaWeights<'_>,
    x: &[f32],
    out: &mut MlaOutput<'_>,
    scratch: &mut MlaScratch,
    n_tok: usize,
    position_of: impl Fn(usize) -> u32,
    tier: SimdTier,
) -> Result<(), Error> {
    let n_embd = DSV4_N_EMBD;
    let n_lora_q = DSV4_N_LORA_Q;
    let head_dim = DSV4_HEAD_DIM;
    let n_head = DSV4_N_HEAD;

    if x.len() != n_tok * n_embd {
        return Err(Error::ShapeMismatch {
            key: "mla.x",
            expected: format!("{}", n_tok * n_embd),
            actual: format!("{}", x.len()),
        });
    }
    if out.q.len() != n_tok * n_head * head_dim {
        return Err(Error::ShapeMismatch {
            key: "mla.q_out",
            expected: format!("{}", n_tok * n_head * head_dim),
            actual: format!("{}", out.q.len()),
        });
    }
    if out.kv.len() != n_tok * head_dim {
        return Err(Error::ShapeMismatch {
            key: "mla.kv_out",
            expected: format!("{}", n_tok * head_dim),
            actual: format!("{}", out.kv.len()),
        });
    }
    if weights.q_a_norm.len() != n_lora_q {
        return Err(Error::ShapeMismatch {
            key: "mla.q_a_norm",
            expected: format!("{n_lora_q}"),
            actual: format!("{}", weights.q_a_norm.len()),
        });
    }
    if weights.kv_a_norm.len() != head_dim {
        return Err(Error::ShapeMismatch {
            key: "mla.kv_a_norm",
            expected: format!("{head_dim}"),
            actual: format!("{}", weights.kv_a_norm.len()),
        });
    }
    scratch.resize(n_tok);

    // === Q path ===
    // 1. attn_q_a: 4096 → 1024
    matmul_weight_f32(
        &mut scratch.q_lora,
        &weights.attn_q_a,
        x,
        n_tok,
        n_embd,
        n_lora_q,
        tier,
    )?;

    // 2. q_a_norm: RMSNorm per-token on the 1024-d bottleneck. Writes
    //    into the dedicated `q_lora_normed` buffer in scratch so we
    //    avoid a per-token heap allocation in the prefill hot loop.
    for t in 0..n_tok {
        let off = t * n_lora_q;
        // Disjoint source / dest slices satisfy `rmsnorm`'s borrow
        // requirements without a clone.
        let (src, dst) = (&scratch.q_lora, &mut scratch.q_lora_normed);
        rmsnorm(
            &mut dst[off..off + n_lora_q],
            &src[off..off + n_lora_q],
            weights.q_a_norm,
            DSV4_RMS_EPS,
            tier,
        )
        .map_err(map_cpu_err("q_a_norm"))?;
    }

    // 3. attn_q_b: 1024 → N_HEAD * HEAD_DIM (= 32768).
    matmul_weight_f32(
        out.q,
        &weights.attn_q_b,
        &scratch.q_lora_normed,
        n_tok,
        n_lora_q,
        n_head * head_dim,
        tier,
    )?;

    // 4. RoPE on the tail N_ROT lanes of every Q head, per token.
    for t in 0..n_tok {
        let pos = position_of(t);
        let q_t = &mut out.q[t * n_head * head_dim..(t + 1) * n_head * head_dim];
        let params = rope_params_at(pos, n_head as u32);
        rope_yarn_tail(q_t, &params, tier).map_err(map_cpu_err("rope.q"))?;
    }

    // === KV path ===
    // 5. attn_kv_a: 4096 → 512.
    matmul_weight_f32(
        &mut scratch.kv_latent,
        &weights.attn_kv_a,
        x,
        n_tok,
        n_embd,
        head_dim,
        tier,
    )?;

    // 6. kv_a_norm: RMSNorm per-token on the 512-d latent. `out.kv`
    //    and `scratch.kv_latent` are disjoint allocations, so we can
    //    norm directly from one to the other without a temp buffer.
    for t in 0..n_tok {
        let src_off = t * head_dim;
        let src = &scratch.kv_latent[src_off..src_off + head_dim];
        let dst = &mut out.kv[src_off..src_off + head_dim];
        rmsnorm(dst, src, weights.kv_a_norm, DSV4_RMS_EPS, tier)
            .map_err(map_cpu_err("kv_a_norm"))?;
    }

    // 7. RoPE on the tail N_ROT lanes of the single KV head, per token.
    for t in 0..n_tok {
        let pos = position_of(t);
        let kv_t = &mut out.kv[t * head_dim..(t + 1) * head_dim];
        let params = rope_params_at(pos, 1);
        rope_yarn_tail(kv_t, &params, tier).map_err(map_cpu_err("rope.kv"))?;
    }

    Ok(())
}

fn rope_params_at(pos: u32, n_head: u32) -> RoPEParams {
    RoPEParams {
        n_head,
        head_dim: DSV4_HEAD_DIM as u32,
        n_rot: DSV4_N_ROT as u32,
        pos,
        // DS V4 Flash's original context length. ds4.c:104 fixes this at
        // 65536; YaRN extrapolation lets us run beyond it.
        n_ctx_orig: 65_536,
        freq_base: super::shape::DSV4_ROPE_FREQ_BASE,
        freq_scale: 1.0,
        // ext_factor = 0 disables the YaRN ramp; v0.1.0 ships the
        // base (non-extrapolated) regime. Extending past 65k context
        // will set this to 1.0 once F008 wires user-supplied n_ctx.
        ext_factor: 0.0,
        attn_factor: 1.0,
        beta_fast: 32.0,
        beta_slow: 1.0,
        inverse: false,
    }
}

fn map_cpu_err(stage: &'static str) -> impl FnOnce(rsllm_backend_cpu::Error) -> Error {
    move |e| Error::ShapeMismatch {
        key: stage,
        expected: "valid kernel shape".to_string(),
        actual: format!("{e}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build an identity-ish F32 weight blob. `out_dim` rows of `in_dim`
    /// each; row `o` has a `1.0` at column `o % in_dim` and zeros
    /// elsewhere. This makes the matmul output deterministic and easy
    /// to hand-check.
    fn identity_weight(out_dim: usize, in_dim: usize) -> Vec<f32> {
        let mut w = vec![0.0_f32; out_dim * in_dim];
        for o in 0..out_dim {
            w[o * in_dim + (o % in_dim)] = 1.0;
        }
        w
    }

    #[test]
    fn rejects_wrong_x_shape() {
        let q_a = vec![0.0_f32; DSV4_N_LORA_Q * DSV4_N_EMBD];
        let q_b = vec![0.0_f32; DSV4_N_HEAD * DSV4_HEAD_DIM * DSV4_N_LORA_Q];
        let kv_a = vec![0.0_f32; DSV4_HEAD_DIM * DSV4_N_EMBD];
        let weights = MlaWeights {
            attn_q_a: WeightBlob::F32(&q_a),
            q_a_norm: &vec![1.0_f32; DSV4_N_LORA_Q],
            attn_q_b: WeightBlob::F32(&q_b),
            attn_kv_a: WeightBlob::F32(&kv_a),
            kv_a_norm: &vec![1.0_f32; DSV4_HEAD_DIM],
        };
        let x = vec![0.0_f32; 10]; // wrong
        let mut q = vec![0.0; DSV4_N_HEAD * DSV4_HEAD_DIM];
        let mut kv = vec![0.0; DSV4_HEAD_DIM];
        let mut scratch = MlaScratch::new(1);
        let mut out = MlaOutput {
            q: &mut q,
            kv: &mut kv,
        };
        let err = mla_projections(
            &weights,
            &x,
            &mut out,
            &mut scratch,
            1,
            |_| 0,
            SimdTier::Scalar,
        )
        .unwrap_err();
        assert!(matches!(err, Error::ShapeMismatch { .. }));
    }

    #[test]
    fn projection_is_finite_and_shape_correct() {
        // Tiny but full-shape smoke test using a deterministic input.
        // We can't easily check exact numbers (the path is 3 matmuls
        // + 2 norms + 2 ropes), but we *can* verify that:
        //   (a) the call completes,
        //   (b) outputs have the right length,
        //   (c) all outputs are finite.
        let n_tok = 2;
        let q_a = identity_weight(DSV4_N_LORA_Q, DSV4_N_EMBD);
        let q_b = identity_weight(DSV4_N_HEAD * DSV4_HEAD_DIM, DSV4_N_LORA_Q);
        let kv_a = identity_weight(DSV4_HEAD_DIM, DSV4_N_EMBD);

        let weights = MlaWeights {
            attn_q_a: WeightBlob::F32(&q_a),
            q_a_norm: &vec![1.0_f32; DSV4_N_LORA_Q],
            attn_q_b: WeightBlob::F32(&q_b),
            attn_kv_a: WeightBlob::F32(&kv_a),
            kv_a_norm: &vec![1.0_f32; DSV4_HEAD_DIM],
        };

        let x: Vec<f32> = (0..n_tok * DSV4_N_EMBD)
            .map(|i| ((i as f32) * 0.001).sin())
            .collect();
        let mut q = vec![0.0_f32; n_tok * DSV4_N_HEAD * DSV4_HEAD_DIM];
        let mut kv = vec![0.0_f32; n_tok * DSV4_HEAD_DIM];
        let mut scratch = MlaScratch::new(n_tok);
        let mut out = MlaOutput {
            q: &mut q,
            kv: &mut kv,
        };

        mla_projections(
            &weights,
            &x,
            &mut out,
            &mut scratch,
            n_tok,
            |t| t as u32,
            SimdTier::Scalar,
        )
        .unwrap();

        assert!(q.iter().all(|v| v.is_finite()), "q has non-finite entries");
        assert!(
            kv.iter().all(|v| v.is_finite()),
            "kv has non-finite entries"
        );
    }
}
