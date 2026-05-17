//! Logits-to-token sampling.
//!
//! F007 of v0.1.0. Implements the DS V4 Flash compatible filter chain:
//! `temperature → top-k → top-p → min-p → multinomial`. With
//! `temperature == 0.0` the chain short-circuits to argmax (greedy).
//!
//! Defaults mirror ds4 commit `613e9b2 "Default sampling to min-p
//! filtering"` (2026-05-15):
//!
//! ```text
//! temperature = 0.7   (think mode: 1.0)
//! top_k       = None  (disabled)
//! top_p       = Some(1.0)   (effectively disabled — keeps the full mass)
//! min_p       = Some(0.05)  (DS4_DEFAULT_MIN_P, the active filter)
//! ```
//!
//! Reproducibility: the [`Sampler`] is seeded from a `u64` and uses a
//! pure-Rust xoshiro256\*\* PRNG. Same seed + same logits ⇒ same
//! token, every time.
//!
//! Ported by reference from `ds4.c:14183-14386` (MIT, The ds4.c authors).
//! Line numbers pinned to ds4 commit `ef0a490` (2026-05-17).

/// Default min-p threshold — matches ds4's `DS4_DEFAULT_MIN_P`
/// (`ds4.c:78`).
pub const DEFAULT_MIN_P: f32 = 0.05;

/// Default temperature for non-think mode.
pub const DEFAULT_TEMPERATURE: f32 = 0.7;

/// Default temperature inside a `<think>` block.
pub const DEFAULT_THINK_TEMPERATURE: f32 = 1.0;

/// User-facing sampling configuration. Construct via
/// [`SamplingParams::default`] for the ds4-compatible defaults, or set
/// fields explicitly.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SamplingParams {
    /// Softmax temperature. `0.0` selects greedy (argmax) and skips the
    /// rest of the filter chain.
    pub temperature: f32,

    /// Keep only the top-`k` logits before normalizing. `None` disables
    /// the filter.
    pub top_k: Option<usize>,

    /// Nucleus filter — keep the smallest prefix of the sorted-by-prob
    /// distribution whose cumulative mass ≥ `top_p`. `None` (or `1.0`)
    /// disables the filter.
    pub top_p: Option<f32>,

    /// Relative-probability cutoff — drop tokens whose probability is
    /// below `min_p × max_prob`. `None` disables the filter. This is
    /// the v0.1.0 default active filter (`min_p = 0.05`).
    pub min_p: Option<f32>,

    /// Seed for the multinomial draw. `None` selects a **fixed
    /// fallback seed**, so a sampler constructed with `seed: None`
    /// is still deterministic across process launches. Callers that
    /// want OS entropy must pin a value (e.g. via the `getrandom`
    /// crate at the call site) before constructing.
    pub seed: Option<u64>,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: DEFAULT_TEMPERATURE,
            top_k: None,
            top_p: Some(1.0),
            min_p: Some(DEFAULT_MIN_P),
            seed: None,
        }
    }
}

impl SamplingParams {
    /// Greedy / deterministic configuration (`temperature = 0.0`).
    #[must_use]
    pub fn greedy() -> Self {
        Self {
            temperature: 0.0,
            top_k: None,
            top_p: None,
            min_p: None,
            seed: None,
        }
    }

    /// Sampling preset for inside a `<think>` block — slightly hotter
    /// to encourage exploration. `min_p = 0.05` is preserved per
    /// ds4 commit `613e9b2`.
    #[must_use]
    pub fn think() -> Self {
        Self {
            temperature: DEFAULT_THINK_TEMPERATURE,
            top_k: None,
            top_p: Some(1.0),
            min_p: Some(DEFAULT_MIN_P),
            seed: None,
        }
    }
}

/// `xoshiro256**` PRNG — deterministic, fast, no external dependency.
///
/// Reference: Blackman & Vigna, "Scrambled Linear Pseudorandom Number
/// Generators" (<https://prng.di.unimi.it/>). Used here to seed the
/// multinomial draw — quality is more than sufficient for sampling
/// without dragging in `rand` / `rand_chacha`.
#[derive(Debug, Clone)]
struct Xoshiro256StarStar {
    state: [u64; 4],
}

impl Xoshiro256StarStar {
    /// Build a PRNG from a single `u64` seed. The seed is run through
    /// `splitmix64` four times to fill the 256-bit state — the standard
    /// initialization recipe from Vigna's reference.
    fn from_seed(seed: u64) -> Self {
        let mut s = [0_u64; 4];
        let mut x = seed;
        for slot in &mut s {
            x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = x;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            *slot = z;
        }
        // Reject the all-zero state (xoshiro requirement).
        if s == [0, 0, 0, 0] {
            s[0] = 1;
        }
        Self { state: s }
    }

    /// Produce the next u64.
    fn next_u64(&mut self) -> u64 {
        let result = self.state[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = self.state[1] << 17;

        self.state[2] ^= self.state[0];
        self.state[3] ^= self.state[1];
        self.state[1] ^= self.state[2];
        self.state[0] ^= self.state[3];
        self.state[2] ^= t;
        self.state[3] = self.state[3].rotate_left(45);

        result
    }

    /// Uniform f32 in [0, 1). Uses the top 24 bits of a u64 draw,
    /// matching the standard "high-bits → float" recipe.
    fn next_f32(&mut self) -> f32 {
        // 24-bit precision is enough for f32; using 53 → f32 cast
        // can produce 1.0 due to rounding.
        let bits = (self.next_u64() >> 40) as u32; // top 24 bits
        (bits as f32) * (1.0 / (1_u32 << 24) as f32)
    }
}

/// Logits sampler. Constructed once per session (the PRNG state
/// persists across calls so successive draws don't all share the
/// same seed).
///
/// Note: `SamplingParams` is `Copy` and carries the seed by value, so
/// constructing two samplers from the same `SamplingParams` value
/// gives two PRNGs that start in the **same** state — the sequences
/// are identical, not independent.
#[derive(Debug, Clone)]
pub struct Sampler {
    params: SamplingParams,
    rng: Xoshiro256StarStar,
    /// Reusable scratch — sorted-prob indices for top-k / top-p.
    /// Held on the sampler so a high-rate decode loop doesn't
    /// reallocate per token.
    scratch: Vec<usize>,
    /// Reusable keep-mask scratch — kept on the sampler for the same
    /// reason. At ~150k vocab a fresh `vec![false; vocab]` per token
    /// would burn ~150 KiB of heap traffic per filter pass.
    keep_mask: Vec<bool>,
}

impl Sampler {
    /// Build a sampler from explicit params. If `params.seed` is
    /// `None`, the sampler uses a fixed fallback seed — see the
    /// [`SamplingParams::seed`] doc.
    #[must_use]
    pub fn new(params: SamplingParams) -> Self {
        let seed = params.seed.unwrap_or(0xDEAD_BEEF_CAFE_F00D);
        Self {
            params,
            rng: Xoshiro256StarStar::from_seed(seed),
            scratch: Vec::new(),
            keep_mask: Vec::new(),
        }
    }

    /// Pure-greedy constructor — equivalent to
    /// `Sampler::new(SamplingParams::greedy())`.
    #[must_use]
    pub fn greedy() -> Self {
        Self::new(SamplingParams::greedy())
    }

    /// The configuration this sampler was constructed with.
    #[must_use]
    pub fn params(&self) -> &SamplingParams {
        &self.params
    }

    /// Sample one token id from `logits`.
    ///
    /// The slice is mutated in place during normalization — pass a
    /// scratch buffer rather than the canonical model output.
    ///
    /// # Panics
    /// Panics if `logits.is_empty()`.
    #[must_use]
    pub fn sample(&mut self, logits: &mut [f32]) -> u32 {
        assert!(!logits.is_empty(), "sampler: logits must be non-empty");

        // 1. Greedy short-circuit. Triggers on `temperature == 0`
        //    (canonical greedy) and also on `NaN` / negative
        //    temperature, which would otherwise propagate NaN through
        //    the chain and silently collapse to "always token 0".
        let t = self.params.temperature;
        if !t.is_finite() || t <= 0.0 {
            return argmax(logits) as u32;
        }

        // 2. Temperature divide. NaN/inf-safe: replace non-finite with
        //    -inf so they end up with 0 probability.
        let inv_t = 1.0 / self.params.temperature;
        for l in logits.iter_mut() {
            if l.is_finite() {
                *l *= inv_t;
            } else {
                *l = f32::NEG_INFINITY;
            }
        }

        // 3. Softmax → probabilities.
        softmax_in_place(logits);

        // 4. top-k.
        if let Some(k) = self.params.top_k {
            apply_top_k(logits, k, &mut self.scratch, &mut self.keep_mask);
        }

        // 5. top-p. `top_p = 1.0` is deliberately treated as "no
        //    filter": it skips the sort + truncation step entirely,
        //    so the full distribution (including any long tail) is
        //    preserved unchanged. `top_p < 1.0` runs the nucleus
        //    filter, which always retains the boundary token (the
        //    one that pushes cumulative mass over the threshold).
        if let Some(p) = self.params.top_p
            && p < 1.0
        {
            apply_top_p(logits, p, &mut self.scratch, &mut self.keep_mask);
        }

        // 6. min-p.
        if let Some(mp) = self.params.min_p
            && mp > 0.0
        {
            apply_min_p(logits, mp);
        }

        // 7. Re-normalize after filtering — any filter may have zeroed
        //    out a chunk of the mass.
        let sum: f32 = logits.iter().sum();
        if !sum.is_finite() || sum <= 0.0 {
            // Pathological — fall back to argmax of the pre-filter
            // input. (Argmax over the current zeroed slice would be
            // arbitrary; the original max is the safer fallback.)
            return argmax(logits) as u32;
        }
        let inv_sum = 1.0 / sum;
        for p in logits.iter_mut() {
            *p *= inv_sum;
        }

        // 8. Multinomial draw.
        let u = self.rng.next_f32();
        let mut cumulative = 0.0_f32;
        for (i, &p) in logits.iter().enumerate() {
            cumulative += p;
            if u < cumulative {
                return i as u32;
            }
        }
        // Floating-point slop — `u` (in `[0, 1)`) can land above the
        // final cumulative sum when re-normalization rounds slightly
        // low. Scan backward for the last bucket with non-zero
        // probability so we never return a filter-zeroed token.
        for (i, &p) in logits.iter().enumerate().rev() {
            if p > 0.0 {
                return i as u32;
            }
        }
        // Truly all-zero (would normally be caught by the sum<=0
        // guard above, but reachable if NaNs slipped past); argmax of
        // the original input is the safest pick.
        argmax(logits) as u32
    }
}

/// Argmax over a slice of f32. Non-finite values are skipped. Returns
/// 0 on an all-NaN slice.
fn argmax(logits: &[f32]) -> usize {
    let mut best_idx = 0_usize;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v.is_finite() && v > best_val {
            best_val = v;
            best_idx = i;
        }
    }
    best_idx
}

/// In-place softmax. Numerically stable (subtracts the max before
/// exp).
fn softmax_in_place(x: &mut [f32]) {
    let mut max_v = f32::NEG_INFINITY;
    for &v in x.iter() {
        if v > max_v {
            max_v = v;
        }
    }
    if !max_v.is_finite() {
        // All -inf / NaN — zero out so the caller's renormalization
        // sees `sum == 0` and falls through to the argmax fallback.
        for v in x.iter_mut() {
            *v = 0.0;
        }
        return;
    }
    let mut sum = 0.0_f32;
    for v in x.iter_mut() {
        *v = (*v - max_v).exp();
        sum += *v;
    }
    if sum > 0.0 {
        let inv = 1.0 / sum;
        for v in x.iter_mut() {
            *v *= inv;
        }
    }
}

/// Reset `mask` to length `n`, all `false`, reusing its capacity.
fn reset_mask(mask: &mut Vec<bool>, n: usize) {
    mask.clear();
    mask.resize(n, false);
}

/// Keep only the top `k` probabilities; zero out the rest. No-op when
/// `k >= probs.len()`. `k == 0` zeroes the full slice, which causes
/// the caller to fall back to argmax on the original input.
fn apply_top_k(probs: &mut [f32], k: usize, scratch: &mut Vec<usize>, keep: &mut Vec<bool>) {
    if k == 0 {
        for p in probs.iter_mut() {
            *p = 0.0;
        }
        return;
    }
    if k >= probs.len() {
        return;
    }
    scratch.clear();
    scratch.extend(0..probs.len());
    // `select_nth_unstable_by` is O(n) — we don't need a full sort.
    scratch.select_nth_unstable_by(k - 1, |&a, &b| {
        probs[b].partial_cmp(&probs[a]).unwrap_or(std::cmp::Ordering::Equal)
    });
    reset_mask(keep, probs.len());
    for &idx in &scratch[..k] {
        keep[idx] = true;
    }
    for (i, p) in probs.iter_mut().enumerate() {
        if !keep[i] {
            *p = 0.0;
        }
    }
}

/// Nucleus filter — keep the smallest set of tokens whose cumulative
/// probability ≥ `p`. `p` must already be in `(0.0, 1.0)`; callers
/// short-circuit on `p >= 1.0`. The boundary token (the one that
/// pushes cumulative mass over the threshold) is always retained.
fn apply_top_p(probs: &mut [f32], p: f32, scratch: &mut Vec<usize>, keep: &mut Vec<bool>) {
    scratch.clear();
    scratch.extend(0..probs.len());
    scratch.sort_unstable_by(|&a, &b| {
        probs[b].partial_cmp(&probs[a]).unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut cumulative = 0.0_f32;
    let mut keep_count = 0_usize;
    for (rank, &idx) in scratch.iter().enumerate() {
        cumulative += probs[idx];
        keep_count = rank + 1;
        if cumulative >= p {
            break;
        }
    }
    // Always keep at least 1 token.
    let keep_count = keep_count.max(1);
    reset_mask(keep, probs.len());
    for &idx in &scratch[..keep_count] {
        keep[idx] = true;
    }
    for (i, prob) in probs.iter_mut().enumerate() {
        if !keep[i] {
            *prob = 0.0;
        }
    }
}

/// Relative-probability cutoff — drop tokens with prob below
/// `min_p × max_prob`. The most expressive filter in the chain;
/// matches ds4's `sample_min_p` (`ds4.c:14264-14289`).
fn apply_min_p(probs: &mut [f32], min_p: f32) {
    let mut max_p = 0.0_f32;
    for &p in probs.iter() {
        if p > max_p {
            max_p = p;
        }
    }
    if max_p <= 0.0 {
        return;
    }
    let threshold = max_p * min_p;
    for p in probs.iter_mut() {
        if *p < threshold {
            *p = 0.0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_params_match_ds4_min_p() {
        let p = SamplingParams::default();
        assert_eq!(p.min_p, Some(DEFAULT_MIN_P));
        assert_eq!(p.top_p, Some(1.0));
        assert_eq!(p.top_k, None);
        assert!((p.temperature - DEFAULT_TEMPERATURE).abs() < 1e-9);
    }

    #[test]
    fn greedy_picks_argmax() {
        let mut s = Sampler::greedy();
        let mut logits = vec![1.0_f32, 5.0, 2.0, 3.0];
        let pick = s.sample(&mut logits);
        assert_eq!(pick, 1);
    }

    #[test]
    fn greedy_handles_nan_logits() {
        let mut s = Sampler::greedy();
        let mut logits = vec![1.0_f32, f32::NAN, 3.0, f32::INFINITY];
        // INF is non-finite → skipped; 3.0 wins.
        let pick = s.sample(&mut logits);
        assert_eq!(pick, 2);
    }

    #[test]
    fn temperature_zero_is_greedy() {
        let params = SamplingParams {
            temperature: 0.0,
            ..SamplingParams::default()
        };
        let mut s = Sampler::new(params);
        let mut logits = vec![1.0_f32, 0.5, 4.0, 2.0];
        assert_eq!(s.sample(&mut logits), 2);
    }

    #[test]
    fn same_seed_same_pick() {
        let mut s1 = Sampler::new(SamplingParams {
            temperature: 1.0,
            seed: Some(42),
            ..SamplingParams::default()
        });
        let mut s2 = Sampler::new(SamplingParams {
            temperature: 1.0,
            seed: Some(42),
            ..SamplingParams::default()
        });
        for _ in 0..16 {
            let mut a = vec![0.1_f32, 0.9, 0.3, 0.5, 0.2];
            let mut b = a.clone();
            assert_eq!(s1.sample(&mut a), s2.sample(&mut b));
        }
    }

    #[test]
    fn different_seeds_diverge() {
        // Two samplers with different seeds and a non-degenerate
        // distribution should disagree on at least one of N draws.
        let mut s1 = Sampler::new(SamplingParams {
            temperature: 1.0,
            min_p: None,
            top_p: None,
            seed: Some(1),
            ..SamplingParams::default()
        });
        let mut s2 = Sampler::new(SamplingParams {
            temperature: 1.0,
            min_p: None,
            top_p: None,
            seed: Some(2),
            ..SamplingParams::default()
        });
        let mut disagreed = false;
        for _ in 0..32 {
            let mut a = vec![1.0_f32, 1.0, 1.0, 1.0];
            let mut b = a.clone();
            if s1.sample(&mut a) != s2.sample(&mut b) {
                disagreed = true;
                break;
            }
        }
        assert!(disagreed, "two different seeds should diverge within 32 draws");
    }

    #[test]
    fn min_p_filters_low_relative_prob() {
        // probs after softmax(/T=1): essentially [0.575, 0.212, 0.078, 0.029, 0.011, ...]
        // For min_p=0.5 with max=0.575 → threshold ≈ 0.287 → only idx 0 survives.
        let params = SamplingParams {
            temperature: 1.0,
            min_p: Some(0.5),
            top_p: None,
            top_k: None,
            seed: Some(7),
        };
        let mut s = Sampler::new(params);
        // Strongly peaked: logit 5 dominates idx 0.
        let mut logits = vec![5.0_f32, 4.0, 3.0, 2.0, 1.0];
        // With idx 0 being the only survivor it must be selected every time.
        for _ in 0..16 {
            let mut l = logits.clone();
            assert_eq!(s.sample(&mut l), 0);
        }
        let _ = &mut logits;
    }

    #[test]
    fn top_k_keeps_only_k_tokens() {
        // top_k=1 forces the highest-prob token to win every draw,
        // regardless of temperature.
        let params = SamplingParams {
            temperature: 1.0,
            top_k: Some(1),
            top_p: None,
            min_p: None,
            seed: Some(11),
        };
        let mut s = Sampler::new(params);
        for _ in 0..8 {
            let mut l = vec![0.5_f32, 2.0, 0.7, 1.5, 0.3];
            assert_eq!(s.sample(&mut l), 1);
        }
    }

    #[test]
    fn top_p_at_one_is_noop() {
        // top_p=1.0 keeps the full distribution; min_p disabled, so
        // the multinomial draw should be able to land on any token.
        let params = SamplingParams {
            temperature: 1.0,
            top_p: Some(1.0),
            min_p: None,
            top_k: None,
            seed: Some(13),
        };
        let mut s = Sampler::new(params);
        let mut seen = [false; 5];
        for _ in 0..200 {
            let mut l = vec![0.1_f32; 5];
            let pick = s.sample(&mut l) as usize;
            seen[pick] = true;
        }
        assert!(seen.iter().all(|&b| b), "top_p=1 should let every token surface; got {seen:?}");
    }

    #[test]
    fn pathological_all_neg_inf_falls_back() {
        // All -inf logits → softmax all-zero → renorm fails → argmax fallback.
        // argmax(all zeros) returns 0; verify no panic.
        let mut s = Sampler::new(SamplingParams {
            temperature: 1.0,
            ..SamplingParams::default()
        });
        let mut l = vec![f32::NEG_INFINITY; 4];
        let pick = s.sample(&mut l);
        assert!((pick as usize) < 4);
    }

    #[test]
    #[should_panic(expected = "non-empty")]
    fn empty_logits_panic() {
        let mut s = Sampler::greedy();
        let _ = s.sample(&mut []);
    }

    #[test]
    fn xoshiro_is_deterministic() {
        let mut r1 = Xoshiro256StarStar::from_seed(0);
        let mut r2 = Xoshiro256StarStar::from_seed(0);
        for _ in 0..1000 {
            assert_eq!(r1.next_u64(), r2.next_u64());
        }
    }

    #[test]
    fn xoshiro_next_f32_in_unit_range() {
        let mut r = Xoshiro256StarStar::from_seed(99);
        for _ in 0..10_000 {
            let v = r.next_f32();
            assert!((0.0..1.0).contains(&v), "out of range: {v}");
        }
    }

    #[test]
    fn think_preset_keeps_min_p_active() {
        let p = SamplingParams::think();
        assert_eq!(p.min_p, Some(DEFAULT_MIN_P));
        assert!((p.temperature - DEFAULT_THINK_TEMPERATURE).abs() < 1e-9);
    }

    #[test]
    fn nan_temperature_falls_back_to_greedy() {
        // NaN temperature must not silently propagate through softmax
        // (would collapse to "always token 0"). The `>0.0` guard
        // treats NaN as a greedy short-circuit.
        let params = SamplingParams {
            temperature: f32::NAN,
            ..SamplingParams::default()
        };
        let mut s = Sampler::new(params);
        let mut logits = vec![1.0_f32, 5.0, 2.0, 3.0];
        assert_eq!(s.sample(&mut logits), 1, "argmax of input is index 1");
    }

    #[test]
    fn negative_temperature_falls_back_to_greedy() {
        let params = SamplingParams {
            temperature: -1.0,
            ..SamplingParams::default()
        };
        let mut s = Sampler::new(params);
        let mut logits = vec![1.0_f32, 5.0, 2.0];
        assert_eq!(s.sample(&mut logits), 1);
    }

    #[test]
    fn top_p_boundary_token_included() {
        // Softmax of [ln(0.6), ln(0.4)] gives probs [0.6, 0.4].
        // top_p=0.7: cumulative at rank 0 is 0.6 (< 0.7, continue);
        // rank 1 brings it to 1.0 (>= 0.7, stop) — both tokens kept.
        // The "boundary" token (rank 1, the one that pushed mass
        // over the threshold) must survive.
        let params = SamplingParams {
            temperature: 1.0,
            top_p: Some(0.7),
            min_p: None,
            top_k: None,
            seed: Some(99),
        };
        let mut s = Sampler::new(params);
        let logit_p = 0.6_f32.ln();
        let logit_q = 0.4_f32.ln();
        let mut seen = [false; 2];
        for _ in 0..200 {
            let mut l = vec![logit_p, logit_q];
            let pick = s.sample(&mut l) as usize;
            seen[pick] = true;
        }
        assert!(seen[0] && seen[1], "top_p=0.7 must retain both tokens, saw {seen:?}");
    }

    #[test]
    fn top_p_strict_threshold_drops_tail() {
        // Same probs [0.6, 0.4]; with top_p=0.6 the cumulative is
        // already met at rank 0, so token 1 must NOT appear in 200
        // draws. Pins the semantic that `cumulative >= p` is
        // non-strict (matches HF/PyTorch convention).
        let params = SamplingParams {
            temperature: 1.0,
            top_p: Some(0.6),
            min_p: None,
            top_k: None,
            seed: Some(100),
        };
        let mut s = Sampler::new(params);
        let logit_p = 0.6_f32.ln();
        let logit_q = 0.4_f32.ln();
        for _ in 0..200 {
            let mut l = vec![logit_p, logit_q];
            assert_eq!(s.sample(&mut l), 0);
        }
    }

    #[test]
    fn top_k_and_top_p_chain_correctly() {
        // top_k=2 selects [logit=5, logit=4]; top_p=0.5 over the
        // surviving two-token distribution should keep just the top.
        let params = SamplingParams {
            temperature: 1.0,
            top_k: Some(2),
            top_p: Some(0.5),
            min_p: None,
            seed: Some(17),
        };
        let mut s = Sampler::new(params);
        for _ in 0..16 {
            let mut l = vec![1.0_f32, 2.0, 5.0, 4.0, 3.0];
            // logit=5 is index 2 — must always win after both filters.
            assert_eq!(s.sample(&mut l), 2);
        }
    }

    #[test]
    fn multinomial_fallback_never_returns_zeroed_bucket() {
        // Set up a distribution where top-k=1 keeps only the highest
        // bucket, then verify across 200 draws every result is
        // exactly that bucket (the fallback path must NOT return
        // index len-1 if its mass was zeroed).
        let params = SamplingParams {
            temperature: 1.0,
            top_k: Some(1),
            top_p: None,
            min_p: None,
            seed: Some(31),
        };
        let mut s = Sampler::new(params);
        for _ in 0..200 {
            // The biggest logit is at index 2; the last slot (index 4)
            // would be the buggy fallback target.
            let mut l = vec![0.1_f32, 0.2, 9.0, 0.3, 0.05];
            let pick = s.sample(&mut l);
            assert_eq!(pick, 2);
        }
    }

    #[test]
    fn scratch_buffers_are_reused() {
        // After a sample, scratch + keep_mask must hold capacity so
        // subsequent calls don't reallocate. We can only observe this
        // through `Vec::capacity` — at least non-zero after one call.
        let mut s = Sampler::new(SamplingParams {
            temperature: 1.0,
            top_k: Some(3),
            top_p: Some(0.9),
            min_p: None,
            seed: Some(5),
        });
        let mut l = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0];
        let _ = s.sample(&mut l);
        assert!(s.scratch.capacity() >= 5);
        assert!(s.keep_mask.capacity() >= 5);
    }
}
