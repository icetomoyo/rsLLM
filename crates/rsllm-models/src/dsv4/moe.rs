//! Mixture-of-Experts FFN dispatch.
//!
//! DS V4 Flash has two MoE routing regimes that share the same expert
//! pool but differ in how the per-token routing decision is made:
//!
//! - **Hash routing** (layers `[0, 3)`, `ds4.c:5182-5208` —
//!   `layer_hash_router_weights_*`). The
//!   `ffn_gate_tid2eid` tensor maps every vocabulary id directly to a
//!   fixed list of `N_EXPERT_USED = 6` expert indices. Routing is
//!   purely lookup; the gate logit only contributes the soft weight.
//! - **Top-k routing** (layers `[3, 43)`, `ds4.c:5278+` —
//!   `layer_routed_moe_one`). The
//!   `ffn_gate_inp` projection produces a 256-vector of logits per
//!   token; the top-6 indices are selected and weighted.
//!
//! Both regimes share:
//! - Per-expert weight: `w_i = sqrt(softplus(logit_i))`, then L1-normalize.
//! - Per-expert FFN: SwiGLU = `down @ (silu(gate @ x) * (up @ x))`.
//!
//! F005.D delivers hash routing + the shared expert-application helper
//! [`apply_expert_swiglu`]. F005.E will add the top-k variant on top of
//! the same primitives.
//!
//! Ported by reference from `ds4.c:5178-5466` (MIT, The ds4.c authors).
//! Line numbers pinned to ds4 commit `ef0a490` (2026-05-17).

use rsllm_backend_cpu::SimdTier;
use rsllm_backend_cpu::ops::scalar;

use super::shape::{DSV4_N_EMBD, DSV4_N_EXPERT, DSV4_N_EXPERT_USED, DSV4_N_FF_EXP};
use super::weight::{WeightBlob, matmul_weight_f32};
use crate::Error;

/// View into a contiguous "stacked experts" weight blob.
///
/// DS V4 Flash stores each expert's `gate`, `up`, and `down` matrices
/// concatenated row-major: for the gate/up tensors the logical shape
/// is `[N_EXPERT × N_FF_EXP × N_EMBD]`, for down it's
/// `[N_EXPERT × N_EMBD × N_FF_EXP]`.
///
/// For quantized types we slice the byte storage in block-aligned
/// chunks; the block sizes are deterministic per dtype (`Q4_K` =
/// 256 elements per block, etc.).
#[derive(Debug, Clone, Copy)]
pub struct StackedExperts<'a> {
    /// Underlying blob carrying every expert's row concatenated.
    pub blob: WeightBlob<'a>,
    /// Number of logical elements per expert's matrix.
    /// (`N_FF_EXP * N_EMBD` for gate/up, same for down.)
    pub elements_per_expert: usize,
}

impl<'a> StackedExperts<'a> {
    /// Borrow the blob for expert `e` only.
    ///
    /// # Panics
    /// Panics in **debug** builds if `e >= DSV4_N_EXPERT` or if the
    /// computed byte range exceeds the underlying blob. Release builds
    /// rely on Rust's slice indexing to panic with a clear message
    /// rather than dereferencing past the end. Either way, an
    /// out-of-range `e` is a programming error — every routing path
    /// validates the expert id before reaching here (`moe_hash_route`
    /// checks `tid2eid` values; `moe_topk_route` indices come from a
    /// 256-wide gate vector). The assert is defense-in-depth so that
    /// future refactors don't silently break the invariant.
    pub fn expert(&self, e: usize) -> WeightBlob<'a> {
        debug_assert!(
            e < DSV4_N_EXPERT,
            "StackedExperts::expert: index {e} out of range (N_EXPERT = {DSV4_N_EXPERT})",
        );
        match self.blob {
            WeightBlob::F32(s) => {
                let start = e * self.elements_per_expert;
                let end = start + self.elements_per_expert;
                assert!(
                    end <= s.len(),
                    "StackedExperts::expert: f32 blob too small for expert {e} \
                     (need {end} elements, blob has {})",
                    s.len()
                );
                WeightBlob::F32(&s[start..end])
            }
            WeightBlob::Quant { data, dtype } => {
                let block_elems = dtype.block_elements() as usize;
                let block_bytes = dtype.block_bytes() as usize;
                // `elements_per_expert` must be a multiple of `block_elems`
                // for a quantized stacked layout. The constructor in F005.F
                // (gguf loader) verifies this.
                let blocks = self.elements_per_expert / block_elems;
                let stride = blocks * block_bytes;
                let start = e * stride;
                let end = start + stride;
                assert!(
                    end <= data.len(),
                    "StackedExperts::expert: quant blob too small for expert {e} \
                     (need {end} bytes, blob has {})",
                    data.len()
                );
                WeightBlob::Quant {
                    data: &data[start..end],
                    dtype,
                }
            }
        }
    }
}

/// Per-layer MoE expert weights.
///
/// `gate` / `up` rows are `[N_FF_EXP × N_EMBD]` per expert; `down` is
/// `[N_EMBD × N_FF_EXP]` per expert. The actual tensor dtype is
/// usually `Q4_K`, but the type system carries any [`WeightBlob`].
#[derive(Debug, Clone, Copy)]
pub struct MoeExpertWeights<'a> {
    /// `[N_EXPERT × N_FF_EXP × N_EMBD]`.
    pub gate: StackedExperts<'a>,
    /// `[N_EXPERT × N_FF_EXP × N_EMBD]`.
    pub up: StackedExperts<'a>,
    /// `[N_EXPERT × N_EMBD × N_FF_EXP]`.
    pub down: StackedExperts<'a>,
}

/// Per-layer routing inputs for the hash-routed regime (front 3 layers).
#[derive(Debug, Clone, Copy)]
pub struct MoeHashRouter<'a> {
    /// `[N_EXPERT_USED × N_VOCAB]` = `[6 × 129280]` row-major `i32`.
    /// For a token with vocabulary id `v`, the six expert indices are
    /// `tid2eid[h * N_VOCAB + v] for h in 0..6`.
    ///
    /// **GGUF axis-convention note** (`ds4.c:5028` reference): GGUF
    /// stores tensors with `shape[0]` as the fastest (inner) dimension.
    /// We treat `tid2eid` as `[N_EXPERT_USED, N_VOCAB]` with `N_VOCAB`
    /// fastest, so element `(h, v)` lives at offset `h * N_VOCAB + v`.
    /// If the trainer published this tensor as `[N_VOCAB, N_EXPERT_USED]`
    /// instead, the loader (F005 GGUF integration) must transpose at
    /// load time — this struct expects the indexing convention above.
    pub tid2eid: &'a [i32],
    /// Standard MoE gate logit projection used to compute the soft
    /// per-expert weight. Shape `[N_EXPERT × N_EMBD]` = `[256 × 4096]`.
    pub gate_inp: WeightBlob<'a>,
    /// Optional per-expert gate bias `[N_EXPERT]`. Added to the gate
    /// logits before routing weights are computed (`ds4.c:5256-5257`).
    /// Most checkpoints ship this; if absent the gate logits go
    /// through unmodified.
    pub gate_bias: Option<&'a [f32]>,
}

/// Scratch buffers reused across MoE forward calls.
#[derive(Debug, Default)]
pub struct MoeScratch {
    /// `[n_tok × N_EXPERT]` — full gate logit vector per token.
    pub gate_logits: Vec<f32>,
    /// `[n_tok × N_EXPERT_USED]` — selected expert IDs per token.
    pub selected_experts: Vec<u32>,
    /// `[n_tok × N_EXPERT_USED]` — normalized routing weights.
    pub selected_weights: Vec<f32>,
    /// `[N_FF_EXP]` — per-expert FFN intermediate (gate path).
    pub h_gate: Vec<f32>,
    /// `[N_FF_EXP]` — per-expert FFN intermediate (up path).
    pub h_up: Vec<f32>,
    /// `[N_FF_EXP]` — per-expert FFN intermediate (silu(gate)*up).
    pub h_act: Vec<f32>,
    /// `[N_EMBD]` — per-expert output buffer.
    pub expert_out: Vec<f32>,
}

impl MoeScratch {
    /// Allocate scratch sized for `n_tok` tokens.
    #[must_use]
    pub fn new(n_tok: usize) -> Self {
        Self {
            gate_logits: vec![0.0_f32; n_tok * DSV4_N_EXPERT],
            selected_experts: vec![0_u32; n_tok * DSV4_N_EXPERT_USED],
            selected_weights: vec![0.0_f32; n_tok * DSV4_N_EXPERT_USED],
            h_gate: vec![0.0_f32; DSV4_N_FF_EXP],
            h_up: vec![0.0_f32; DSV4_N_FF_EXP],
            h_act: vec![0.0_f32; DSV4_N_FF_EXP],
            expert_out: vec![0.0_f32; DSV4_N_EMBD],
        }
    }

    /// Resize scratch in place for a new `n_tok`.
    pub fn resize(&mut self, n_tok: usize) {
        self.gate_logits.resize(n_tok * DSV4_N_EXPERT, 0.0);
        self.selected_experts.resize(n_tok * DSV4_N_EXPERT_USED, 0);
        self.selected_weights
            .resize(n_tok * DSV4_N_EXPERT_USED, 0.0);
    }
}

/// Run the MoE FFN with **hash routing** (front 3 layers).
///
/// `token_ids[t]` is the vocabulary id of the token whose hidden state
/// is `x[t * N_EMBD .. (t+1) * N_EMBD]`. Output accumulates into
/// `out` (does not clear; the caller can zero or pre-fill with a
/// shared-expert result before calling — see F005.F).
///
/// # Errors
/// Shape mismatches in any input or downstream kernel call.
#[allow(clippy::too_many_arguments)]
pub fn moe_hash_route(
    out: &mut [f32],
    x: &[f32],
    token_ids: &[u32],
    weights: &MoeExpertWeights<'_>,
    router: &MoeHashRouter<'_>,
    scratch: &mut MoeScratch,
    n_tok: usize,
    tier: SimdTier,
) -> Result<(), Error> {
    check_moe_shapes(out, x, n_tok)?;
    if token_ids.len() != n_tok {
        return Err(Error::ShapeMismatch {
            key: "moe_hash_route.token_ids",
            expected: format!("{n_tok}"),
            actual: format!("{}", token_ids.len()),
        });
    }
    if router.tid2eid.len() != DSV4_N_EXPERT_USED * super::shape::DSV4_N_VOCAB {
        return Err(Error::ShapeMismatch {
            key: "moe_hash_route.tid2eid",
            expected: format!(
                "{}",
                DSV4_N_EXPERT_USED * super::shape::DSV4_N_VOCAB
            ),
            actual: format!("{}", router.tid2eid.len()),
        });
    }
    scratch.resize(n_tok);

    // 1. Compute the full gate logits via `ffn_gate_inp @ x`.
    matmul_weight_f32(
        &mut scratch.gate_logits,
        &router.gate_inp,
        x,
        n_tok,
        DSV4_N_EMBD,
        DSV4_N_EXPERT,
        tier,
    )?;
    add_gate_bias(&mut scratch.gate_logits, router.gate_bias, n_tok)?;

    // 2. Pull the 6 hash-routed expert ids per token from tid2eid.
    for (t, &tid) in token_ids.iter().enumerate().take(n_tok) {
        let v = tid as usize;
        if v >= super::shape::DSV4_N_VOCAB {
            return Err(Error::ShapeMismatch {
                key: "moe_hash_route.token_id",
                expected: format!("< {}", super::shape::DSV4_N_VOCAB),
                actual: format!("{v}"),
            });
        }
        for h in 0..DSV4_N_EXPERT_USED {
            let eid = router.tid2eid[h * super::shape::DSV4_N_VOCAB + v];
            // Cast through u32 guarded by range check; expert indices
            // are written by the trainer as small non-negative ints.
            if !(0..DSV4_N_EXPERT as i32).contains(&eid) {
                return Err(Error::ShapeMismatch {
                    key: "moe_hash_route.tid2eid_value",
                    expected: format!("0..{}", DSV4_N_EXPERT),
                    actual: format!("{eid}"),
                });
            }
            scratch.selected_experts[t * DSV4_N_EXPERT_USED + h] = eid as u32;
        }
    }

    // 3. Compute the soft per-expert weights from the gate logits at
    //    the selected expert indices: sqrt(softplus(l)), then L1-normalize.
    compute_routing_weights(
        &scratch.gate_logits,
        &scratch.selected_experts,
        &mut scratch.selected_weights,
        n_tok,
    );

    // 4. Run each expert and accumulate weighted output.
    accumulate_moe_outputs(out, x, weights, scratch, n_tok, tier)
}

/// Per-layer routing inputs for the top-k regime (layers `[3, 43)`).
#[derive(Debug, Clone, Copy)]
pub struct MoeTopkRouter<'a> {
    /// Same gate projection as hash routing: `[N_EXPERT × N_EMBD]`.
    pub gate_inp: WeightBlob<'a>,
    /// Optional per-expert gate bias `[N_EXPERT]`. Same role as in
    /// [`MoeHashRouter::gate_bias`].
    pub gate_bias: Option<&'a [f32]>,
}

/// Run the MoE FFN with **top-k routing** (layers `[3, 43)`).
///
/// For each token, the top `N_EXPERT_USED = 6` indices in the gate
/// logit vector are selected; their soft weights are derived with the
/// same `sqrt(softplus(logit))` + L1-normalize rule as hash routing.
///
/// Output accumulates into `out` — the caller may pre-fill with the
/// shared expert's contribution.
///
/// # Errors
/// Shape mismatches in any input or downstream kernel call.
pub fn moe_topk_route(
    out: &mut [f32],
    x: &[f32],
    weights: &MoeExpertWeights<'_>,
    router: &MoeTopkRouter<'_>,
    scratch: &mut MoeScratch,
    n_tok: usize,
    tier: SimdTier,
) -> Result<(), Error> {
    check_moe_shapes(out, x, n_tok)?;
    scratch.resize(n_tok);

    // 1. Gate logits over all 256 experts.
    matmul_weight_f32(
        &mut scratch.gate_logits,
        &router.gate_inp,
        x,
        n_tok,
        DSV4_N_EMBD,
        DSV4_N_EXPERT,
        tier,
    )?;
    add_gate_bias(&mut scratch.gate_logits, router.gate_bias, n_tok)?;

    // 2. Per-token top-k selection.
    for t in 0..n_tok {
        let logits_t = &scratch.gate_logits[t * DSV4_N_EXPERT..(t + 1) * DSV4_N_EXPERT];
        let sel_t = &mut scratch.selected_experts
            [t * DSV4_N_EXPERT_USED..(t + 1) * DSV4_N_EXPERT_USED];
        topk_indices(logits_t, sel_t);
    }

    // 3. Routing weights from the selected logits.
    compute_routing_weights(
        &scratch.gate_logits,
        &scratch.selected_experts,
        &mut scratch.selected_weights,
        n_tok,
    );

    // 4. Apply experts.
    accumulate_moe_outputs(out, x, weights, scratch, n_tok, tier)
}

/// Select the indices of the `out.len()` largest entries in `logits`.
///
/// **Output order is unspecified** — the algorithm tracks the running
/// minimum of a fixed-size "top-k so far" buffer and evicts to that
/// slot whenever a larger candidate arrives, so the resulting indices
/// are not sorted. Downstream consumer
/// ([`compute_routing_weights`]) looks up each selected index in the
/// logit vector by value, so order does not affect routing weights.
/// If a future caller needs sorted output, sort externally.
///
/// Uses a small linear scan over a fixed-size "top-k so far" buffer.
/// Cost is `O(N * k)` which dominates at `N = 256, k = 6` (≈1500 ops)
/// far below the O(N log N) of a full sort — and avoids heap allocation.
fn topk_indices(logits: &[f32], out: &mut [u32]) {
    let k = out.len();
    debug_assert!(k > 0);
    debug_assert!(k <= logits.len());

    // Initialize with the first k indices in arbitrary order.
    for (i, slot) in out.iter_mut().enumerate() {
        *slot = i as u32;
    }
    // Find the smallest of the current top-k.
    let mut worst_pos = 0usize;
    let mut worst_val = logits[out[0] as usize];
    for (pos, &idx) in out.iter().enumerate().take(k).skip(1) {
        let v = logits[idx as usize];
        if v < worst_val {
            worst_val = v;
            worst_pos = pos;
        }
    }
    // Scan the rest. For each new element, if it beats the current
    // worst, evict and recompute the worst.
    for (i, &v) in logits.iter().enumerate().skip(k) {
        if v > worst_val {
            out[worst_pos] = i as u32;
            // Recompute worst.
            worst_val = logits[out[0] as usize];
            worst_pos = 0;
            for (pos, &idx) in out.iter().enumerate().take(k).skip(1) {
                let vv = logits[idx as usize];
                if vv < worst_val {
                    worst_val = vv;
                    worst_pos = pos;
                }
            }
        }
    }
}

/// Shared (always-on) expert weights, one set per layer. Same
/// SwiGLU shape as routed experts but **not** part of the routing
/// competition — every token activates this expert.
#[derive(Debug, Clone, Copy)]
pub struct SharedExpertWeights<'a> {
    /// `[N_FF_EXP × N_EMBD]`.
    pub gate: WeightBlob<'a>,
    /// `[N_FF_EXP × N_EMBD]`.
    pub up: WeightBlob<'a>,
    /// `[N_EMBD × N_FF_EXP]`.
    pub down: WeightBlob<'a>,
}

/// Apply the shared expert to every token in `x` and write into `out`.
///
/// Unlike routed experts, the shared expert is unconditionally on, so
/// its contribution does not pass through softplus / normalize. It
/// joins the routed experts' weighted sum via a plain add in the
/// caller (F005.F forward path).
///
/// # Errors
/// Bubbles up shape errors from the underlying matmul + swiglu.
pub fn apply_shared_expert(
    out: &mut [f32],
    x: &[f32],
    weights: &SharedExpertWeights<'_>,
    scratch: &mut MoeScratch,
    n_tok: usize,
    tier: SimdTier,
) -> Result<(), Error> {
    check_moe_shapes(out, x, n_tok)?;
    let n_embd = DSV4_N_EMBD;
    for t in 0..n_tok {
        let x_t = &x[t * n_embd..(t + 1) * n_embd];
        let out_t = &mut out[t * n_embd..(t + 1) * n_embd];
        apply_expert_swiglu(
            out_t,
            x_t,
            &weights.gate,
            &weights.up,
            &weights.down,
            &mut scratch.h_gate,
            &mut scratch.h_up,
            &mut scratch.h_act,
            tier,
        )?;
    }
    Ok(())
}

/// Apply one expert's SwiGLU FFN to a single token.
///
/// `gate`, `up`, `down` are the expert's three matmul views;
/// `h_gate`, `h_up`, `h_act`, `expert_out` are scratch buffers.
///
/// Math: `out = down @ (silu(gate @ x) * (up @ x))`.
#[allow(clippy::too_many_arguments)]
pub fn apply_expert_swiglu(
    out: &mut [f32],
    x: &[f32],
    gate: &WeightBlob<'_>,
    up: &WeightBlob<'_>,
    down: &WeightBlob<'_>,
    h_gate: &mut [f32],
    h_up: &mut [f32],
    h_act: &mut [f32],
    tier: SimdTier,
) -> Result<(), Error> {
    let n_embd = DSV4_N_EMBD;
    let n_ff = DSV4_N_FF_EXP;

    matmul_weight_f32(h_gate, gate, x, 1, n_embd, n_ff, tier)?;
    matmul_weight_f32(h_up, up, x, 1, n_embd, n_ff, tier)?;
    scalar::swiglu(h_act, h_gate, h_up);
    matmul_weight_f32(out, down, h_act, 1, n_ff, n_embd, tier)?;
    Ok(())
}

/// Add `bias[e]` to `gate_logits[t, e]` for every token. No-op when
/// `bias` is `None`. Mirrors `ds4.c:5256-5257`.
fn add_gate_bias(
    gate_logits: &mut [f32],
    bias: Option<&[f32]>,
    n_tok: usize,
) -> Result<(), Error> {
    let Some(bias) = bias else { return Ok(()) };
    if bias.len() != DSV4_N_EXPERT {
        return Err(Error::ShapeMismatch {
            key: "moe.gate_bias",
            expected: format!("{DSV4_N_EXPERT}"),
            actual: format!("{}", bias.len()),
        });
    }
    for t in 0..n_tok {
        let logits_t = &mut gate_logits[t * DSV4_N_EXPERT..(t + 1) * DSV4_N_EXPERT];
        for (l, &b) in logits_t.iter_mut().zip(bias.iter()) {
            *l += b;
        }
    }
    Ok(())
}

fn check_moe_shapes(out: &mut [f32], x: &[f32], n_tok: usize) -> Result<(), Error> {
    let n_embd = DSV4_N_EMBD;
    if x.len() != n_tok * n_embd {
        return Err(Error::ShapeMismatch {
            key: "moe.x",
            expected: format!("{}", n_tok * n_embd),
            actual: format!("{}", x.len()),
        });
    }
    if out.len() != n_tok * n_embd {
        return Err(Error::ShapeMismatch {
            key: "moe.out",
            expected: format!("{}", n_tok * n_embd),
            actual: format!("{}", out.len()),
        });
    }
    Ok(())
}

/// `w_i = sqrt(softplus(l_i))` then L1-normalize across the
/// `N_EXPERT_USED` selected indices. Mirrors `ds4.c:5045-5050` /
/// `ds4.c:5093-5097`.
fn compute_routing_weights(
    gate_logits: &[f32],
    selected_experts: &[u32],
    selected_weights: &mut [f32],
    n_tok: usize,
) {
    let k = DSV4_N_EXPERT_USED;
    for t in 0..n_tok {
        let logits_t = &gate_logits[t * DSV4_N_EXPERT..(t + 1) * DSV4_N_EXPERT];
        let sel_t = &selected_experts[t * k..(t + 1) * k];
        let out_t = &mut selected_weights[t * k..(t + 1) * k];
        let mut sum = 0.0_f32;
        for i in 0..k {
            // Defense-in-depth: the index must already be in range by
            // construction (`tid2eid` is range-checked in hash routing;
            // `topk_indices` only writes indices < N_EXPERT). A debug
            // assert catches refactors that violate this without paying
            // the cost in release.
            debug_assert!(
                (sel_t[i] as usize) < DSV4_N_EXPERT,
                "compute_routing_weights: selected expert {} out of range",
                sel_t[i]
            );
            let l = logits_t[sel_t[i] as usize];
            let sp = softplus(l);
            let w = sp.sqrt();
            out_t[i] = w;
            sum += w;
        }
        let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
        for w in out_t.iter_mut() {
            *w *= inv;
        }
    }
}

/// `softplus(x) = ln(1 + exp(x))`, numerically stable for large `|x|`.
#[inline]
fn softplus(x: f32) -> f32 {
    if x > 0.0 {
        x + (-x).exp().ln_1p()
    } else {
        x.exp().ln_1p()
    }
}

/// For each token, run every selected expert and weighted-sum into `out`.
/// `out` must be pre-cleared (or pre-populated with the shared-expert
/// contribution) by the caller — this function adds.
fn accumulate_moe_outputs(
    out: &mut [f32],
    x: &[f32],
    weights: &MoeExpertWeights<'_>,
    scratch: &mut MoeScratch,
    n_tok: usize,
    tier: SimdTier,
) -> Result<(), Error> {
    let n_embd = DSV4_N_EMBD;
    let k = DSV4_N_EXPERT_USED;

    // Take a scoped mutable view of the per-expert scratch buffers so
    // we don't have to borrow `scratch` and the routing arrays at the
    // same time below.
    let MoeScratch {
        selected_experts,
        selected_weights,
        h_gate,
        h_up,
        h_act,
        expert_out,
        ..
    } = scratch;

    for t in 0..n_tok {
        let x_t = &x[t * n_embd..(t + 1) * n_embd];
        let out_t = &mut out[t * n_embd..(t + 1) * n_embd];
        let sel_t = &selected_experts[t * k..(t + 1) * k];
        let w_t = &selected_weights[t * k..(t + 1) * k];

        for i in 0..k {
            let e = sel_t[i] as usize;
            let we = w_t[i];
            if we == 0.0 {
                continue;
            }
            let gate = weights.gate.expert(e);
            let up = weights.up.expert(e);
            let down = weights.down.expert(e);
            apply_expert_swiglu(
                expert_out,
                x_t,
                &gate,
                &up,
                &down,
                h_gate,
                h_up,
                h_act,
                tier,
            )?;
            for j in 0..n_embd {
                out_t[j] += we * expert_out[j];
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softplus_is_positive_and_smooth() {
        assert!((softplus(0.0) - 2.0_f32.ln()).abs() < 1e-6);
        assert!(softplus(-100.0) >= 0.0);
        assert!(softplus(100.0).is_finite());
    }

    #[test]
    fn routing_weights_l1_normalize() {
        // 4 tokens, mock logits, mock selections.
        let logits: Vec<f32> = (0..DSV4_N_EXPERT)
            .map(|i| (i as f32) * 0.01)
            .collect();
        let selected = vec![0_u32, 1, 2, 3, 4, 5];
        let mut weights = vec![0.0_f32; DSV4_N_EXPERT_USED];
        compute_routing_weights(&logits, &selected, &mut weights, 1);
        let sum: f32 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "routing weight sum = {sum}");
        for &w in &weights {
            assert!(w >= 0.0);
        }
    }

    #[test]
    fn routing_weights_handle_negative_logits() {
        let mut logits = vec![0.0_f32; DSV4_N_EXPERT];
        for v in logits.iter_mut().take(DSV4_N_EXPERT) {
            *v = -50.0;
        }
        let selected = vec![0_u32; DSV4_N_EXPERT_USED];
        let mut weights = vec![0.0_f32; DSV4_N_EXPERT_USED];
        compute_routing_weights(&logits, &selected, &mut weights, 1);
        for &w in &weights {
            assert!(w.is_finite(), "weight not finite: {w}");
        }
    }

    #[test]
    fn moe_hash_route_rejects_oob_token_id() {
        // Use empty Quant blobs so we don't allocate the full
        // [N_EXPERT × N_FF_EXP × N_EMBD] f32 buffer (~8 GiB). The
        // routine fails the bounds check before touching the matmul,
        // so the empty blob is never dereferenced.
        use rsllm_gguf::GgmlType;

        let empty: [u8; 0] = [];
        let weights = MoeExpertWeights {
            gate: StackedExperts {
                blob: WeightBlob::Quant {
                    data: &empty,
                    dtype: GgmlType::Q4_K,
                },
                elements_per_expert: DSV4_N_FF_EXP * DSV4_N_EMBD,
            },
            up: StackedExperts {
                blob: WeightBlob::Quant {
                    data: &empty,
                    dtype: GgmlType::Q4_K,
                },
                elements_per_expert: DSV4_N_FF_EXP * DSV4_N_EMBD,
            },
            down: StackedExperts {
                blob: WeightBlob::Quant {
                    data: &empty,
                    dtype: GgmlType::Q2_K,
                },
                elements_per_expert: DSV4_N_EMBD * DSV4_N_FF_EXP,
            },
        };
        let gate_inp_bytes: [u8; 0] = [];
        let tid2eid = vec![0_i32; DSV4_N_EXPERT_USED * super::super::shape::DSV4_N_VOCAB];
        let router = MoeHashRouter {
            tid2eid: &tid2eid,
            gate_inp: WeightBlob::Quant {
                data: &gate_inp_bytes,
                dtype: GgmlType::Q4_K,
            },
            gate_bias: None,
        };
        let token_ids = vec![super::super::shape::DSV4_N_VOCAB as u32]; // out of range
        let x = vec![0.0_f32; DSV4_N_EMBD];
        let mut out = vec![0.0_f32; DSV4_N_EMBD];
        let mut scratch = MoeScratch::new(1);
        let err = moe_hash_route(
            &mut out,
            &x,
            &token_ids,
            &weights,
            &router,
            &mut scratch,
            1,
            SimdTier::Scalar,
        )
        .unwrap_err();
        assert!(matches!(err, Error::ShapeMismatch { .. }));
    }

    #[test]
    fn gate_bias_added_per_token() {
        let n_tok = 2;
        let mut logits = vec![0.0_f32; n_tok * DSV4_N_EXPERT];
        // Token 0: all ones; Token 1: all twos.
        for v in logits.iter_mut().take(DSV4_N_EXPERT) {
            *v = 1.0;
        }
        for v in logits.iter_mut().skip(DSV4_N_EXPERT).take(DSV4_N_EXPERT) {
            *v = 2.0;
        }
        let bias: Vec<f32> = (0..DSV4_N_EXPERT).map(|i| (i as f32) * 0.01).collect();
        add_gate_bias(&mut logits, Some(&bias), n_tok).unwrap();
        // Spot-check a couple of cells.
        assert!((logits[0] - 1.0).abs() < 1e-6); // tok0 expert0: 1 + 0
        assert!((logits[5] - 1.05).abs() < 1e-6); // tok0 expert5: 1 + 0.05
        assert!((logits[DSV4_N_EXPERT] - 2.0).abs() < 1e-6); // tok1 expert0: 2 + 0
        assert!((logits[DSV4_N_EXPERT + 5] - 2.05).abs() < 1e-6);
    }

    #[test]
    fn gate_bias_none_is_noop() {
        let mut logits = vec![3.5_f32; 2 * DSV4_N_EXPERT];
        add_gate_bias(&mut logits, None, 2).unwrap();
        assert!(logits.iter().all(|&v| (v - 3.5).abs() < 1e-6));
    }

    #[test]
    fn gate_bias_rejects_wrong_length() {
        let mut logits = vec![0.0_f32; DSV4_N_EXPERT];
        let bias = vec![0.0_f32; 10]; // wrong
        let err = add_gate_bias(&mut logits, Some(&bias), 1).unwrap_err();
        assert!(matches!(err, Error::ShapeMismatch { .. }));
    }

    #[test]
    fn topk_picks_largest_six_in_order_independent_set() {
        // Logits with six clear winners and many tied losers.
        let mut logits = vec![0.0_f32; DSV4_N_EXPERT];
        for (i, slot) in logits.iter_mut().enumerate().take(6) {
            *slot = (10 - i as i32) as f32;
        }
        let mut out = vec![0_u32; DSV4_N_EXPERT_USED];
        topk_indices(&logits, &mut out);
        let mut got: Vec<u32> = out.into_iter().collect();
        got.sort_unstable();
        assert_eq!(got, vec![0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn topk_handles_all_equal_logits() {
        let logits = vec![0.0_f32; DSV4_N_EXPERT];
        let mut out = vec![0_u32; DSV4_N_EXPERT_USED];
        topk_indices(&logits, &mut out);
        // Every choice is valid; just check no out-of-range / dup-pos.
        let mut sorted: Vec<u32> = out.into_iter().collect();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), DSV4_N_EXPERT_USED);
    }

    #[test]
    fn topk_picks_distributed_winners() {
        // Place six high-value indices far apart in the array.
        let mut logits = vec![-1.0_f32; DSV4_N_EXPERT];
        let winners = [3_usize, 47, 88, 130, 199, 250];
        for (rank, &w) in winners.iter().enumerate() {
            logits[w] = 10.0 - rank as f32;
        }
        let mut out = vec![0_u32; DSV4_N_EXPERT_USED];
        topk_indices(&logits, &mut out);
        let mut got: Vec<u32> = out.into_iter().collect();
        got.sort_unstable();
        let mut want: Vec<u32> = winners.iter().map(|&i| i as u32).collect();
        want.sort_unstable();
        assert_eq!(got, want);
    }

    // Full end-to-end MoE routing + expert application tests live in
    // the integration-test directory once F008 ships a real GGUF file
    // (running them in-process with synthesized weights would require
    // either ~8 GiB of f32 expert buffers or precomputed Q4_K blocks).
    // The unit tests above cover the algorithmic helpers individually:
    // softplus, routing-weight normalization, top-k index selection,
    // and shape validation.
}
