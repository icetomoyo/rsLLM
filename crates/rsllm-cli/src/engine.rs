//! F008.C.3.f — wire the CLI's one-shot + REPL paths to
//! [`DsV4FlashEngine`].
//!
//! This module orchestrates: open the GGUF, build the tokenizer,
//! construct the engine, derive [`SamplingParams`] from CLI flags,
//! start a session, prefill the prompt, then drive a decode loop with
//! streaming token output to stdout and optional `--dump-tokens` /
//! `--dump-logprobs` taps.

use std::io::Write;
use std::path::Path;

use rsllm_core::{DecodeStep, Engine, EngineError, SamplingParams, Session};
use rsllm_gguf::GgufFile;
use rsllm_models::DsV4FlashEngine;
use rsllm_models::dsv4::loader::load_dsv4_flash;
use rsllm_tokenizer::{ThinkMode as TokThinkMode, Tokenizer};

use crate::CliError;
use crate::cli::{RunFlags, ThinkMode};
use crate::dump::{LogprobDumper, LogprobEntry, emit_token_line};

/// Default upper bound on decode tokens per turn. Picked to match
/// ds4's `default_max_tokens` and to keep one-shot CLI runs from
/// burning unbounded compute on a runaway loop.
pub const DEFAULT_MAX_DECODE_TOKENS: usize = 512;

/// CLI-side think mode → tokenizer think mode.
fn map_think(mode: ThinkMode) -> TokThinkMode {
    match mode {
        ThinkMode::NoThink => TokThinkMode::None,
        ThinkMode::Think => TokThinkMode::High,
        ThinkMode::ThinkMax => TokThinkMode::Max,
    }
}

/// Derive [`SamplingParams`] from the CLI flags. Each
/// `--temperature` / `--top-k` / `--top-p` / `--min-p` flag, when
/// set, overrides the corresponding field of
/// [`SamplingParams::default()`]; unset flags pass through the
/// default. Validation enforces sane ranges:
///
/// - `temperature ≥ 0.0` and finite (0.0 = greedy short-circuit).
/// - `top_k ≥ 1` (Some(0) is rejected — disabling the filter is
///   expressed by omitting the flag).
/// - `top_p ∈ (0.0, 1.0]`.
/// - `min_p ∈ [0.0, 1.0)`.
///
/// Returns [`CliError::BadCommand`] for any out-of-range value
/// before the engine even constructs a sampler, so the operator
/// gets a clear message instead of a downstream NaN.
pub fn sampling_params_from_flags(flags: &RunFlags) -> Result<SamplingParams, CliError> {
    let mut params = SamplingParams {
        seed: flags.seed,
        ..SamplingParams::default()
    };
    if let Some(t) = flags.temperature {
        if !t.is_finite() || t < 0.0 {
            return Err(CliError::BadCommand(format!(
                "--temperature must be a non-negative finite number, got {t}"
            )));
        }
        params.temperature = t;
    }
    if let Some(k) = flags.top_k {
        if k == 0 {
            return Err(CliError::BadCommand(
                "--top-k must be ≥ 1; omit the flag to disable the filter".into(),
            ));
        }
        params.top_k = Some(k);
    }
    if let Some(p) = flags.top_p {
        if !p.is_finite() || p <= 0.0 || p > 1.0 {
            return Err(CliError::BadCommand(format!(
                "--top-p must lie in (0.0, 1.0], got {p}"
            )));
        }
        params.top_p = Some(p);
    }
    if let Some(p) = flags.min_p {
        if !p.is_finite() || !(0.0..1.0).contains(&p) {
            return Err(CliError::BadCommand(format!(
                "--min-p must lie in [0.0, 1.0), got {p}"
            )));
        }
        params.min_p = Some(p);
    }
    // Debuggability note: when seed is unset but the sampler is
    // stochastic (T > 0), the sampler still uses a *fixed* fallback
    // seed (see SamplingParams docs). All runs of the same prompt
    // are therefore reproducible. Logged at DEBUG so operators
    // diagnosing "why do I get the same output every time" can find
    // it; INFO would be too noisy on every load.
    if params.seed.is_none() && params.temperature > 0.0 {
        tracing::debug!(
            target: "rsllm_cli::engine",
            temperature = params.temperature,
            "no --seed set + temperature > 0: runs are deterministic via the sampler's fixed fallback seed",
        );
    }
    Ok(params)
}

/// One-shot decode: open the GGUF at `model_path`, tokenize `prompt`
/// under the configured think mode, and stream the generated text to
/// stdout until EOS or [`DEFAULT_MAX_DECODE_TOKENS`].
///
/// `system` is an optional system prompt — `None` means "no system
/// message". The REPL passes its `system_prompt`; the one-shot CLI
/// today passes `None` (no `--system` flag yet).
pub fn run_one_shot(
    model_path: &Path,
    prompt: &str,
    system: Option<&str>,
    flags: &RunFlags,
) -> Result<(), CliError> {
    let _span = tracing::info_span!(
        target: "rsllm_cli::engine",
        "cli_one_shot",
        ctx_size = flags.ctx_size,
    )
    .entered();
    tracing::info!(
        target: "rsllm_cli::engine",
        model = %model_path.display(),
        prompt_chars = prompt.len(),
        "loading model",
    );

    // Open the dumper BEFORE the model load so a bad --dump-logprobs
    // path errors immediately instead of after a multi-minute mmap.
    let mut dumper = match &flags.dump_logprobs {
        Some(p) => Some(LogprobDumper::create(p, flags.logprobs_top_k.max(1))?),
        None => None,
    };

    // Validate sampler flags BEFORE the model load so a bad
    // `--top-p 1.5` surfaces immediately, not after a multi-minute
    // mmap.
    let params = sampling_params_from_flags(flags)?;

    let gguf = GgufFile::open(model_path)?;
    let tokenizer = Tokenizer::from_gguf(&gguf).map_err(map_tokenizer_err)?;
    let model = load_dsv4_flash(&gguf).map_err(map_model_err)?;
    let engine = DsV4FlashEngine::new(model);

    let mut session = engine
        .start_session(flags.ctx_size, params)
        .map_err(map_engine_err)?;

    let think = map_think(flags.think);
    let tokens = tokenizer.encode_prompt(system.unwrap_or(""), prompt, think);
    if tokens.is_empty() {
        return Err(CliError::BadCommand(
            "tokenizer produced 0 tokens for the prompt".into(),
        ));
    }
    if tokens.len() > flags.ctx_size {
        return Err(CliError::BadCommand(format!(
            "prompt is {} tokens but --ctx-size = {}; increase --ctx-size or shorten the prompt",
            tokens.len(),
            flags.ctx_size,
        )));
    }
    tracing::debug!(
        target: "rsllm_cli::engine",
        n_prompt = tokens.len(),
        think = ?think,
        "prompt encoded",
    );

    // Prefill the prompt. The returned logits are discarded — the
    // first decode step is driven by `decode_one` seeded with the
    // last prompt token.
    let _ = session.prefill(&tokens).map_err(map_engine_err)?;

    let mut last_token = *tokens.last().expect("non-empty after empty check");
    let mut emitted_buf: Vec<u8> = Vec::with_capacity(64);
    let mut utf8_carry: Vec<u8> = Vec::with_capacity(4);
    let stdout = std::io::stdout();
    let mut out = stdout.lock();

    let decode_started = std::time::Instant::now();
    let mut decoded = 0_u32;
    let mut hit_eos = false;
    for step_idx in 0..(DEFAULT_MAX_DECODE_TOKENS as u32) {
        let step = session.decode_one(last_token).map_err(map_engine_err)?;
        if step.token_id == tokenizer.eos_id() {
            hit_eos = true;
            break;
        }
        write_step_outputs(
            &tokenizer,
            &mut out,
            step_idx,
            &step,
            flags,
            dumper.as_mut(),
            &mut emitted_buf,
            &mut utf8_carry,
        )?;
        last_token = step.token_id;
        decoded += 1;
    }
    // Drop any trailing partial-UTF-8 carry (the model emitted EOS or
    // hit the cap mid-codepoint). Emit as lossy bytes so the user
    // sees what came through, rather than silently swallowing it.
    if !utf8_carry.is_empty() {
        out.write_all(String::from_utf8_lossy(&utf8_carry).as_bytes())
            .map_err(CliError::Io)?;
    }
    out.write_all(b"\n").map_err(CliError::Io)?;
    out.flush().map_err(CliError::Io)?;

    let elapsed = decode_started.elapsed();
    let tps = tok_per_sec(decoded, elapsed);
    // `tok_per_sec` is logged as a raw f64 so a JSON subscriber can
    // index it as a number; consumers that want fixed-precision
    // formatting (`{:.2}`) apply it at render time.
    tracing::info!(
        target: "rsllm_cli::engine",
        n_decoded = decoded,
        elapsed_ms = elapsed.as_millis() as u64,
        tok_per_sec = tps,
        hit_eos,
        cap_hit = !hit_eos && (decoded as usize) == DEFAULT_MAX_DECODE_TOKENS,
        "generation complete",
    );
    Ok(())
}

/// Wall-clock decode throughput. Returns 0 when fewer than 1 token was
/// produced or elapsed is zero (e.g. EOS at step 0).
fn tok_per_sec(decoded: u32, elapsed: std::time::Duration) -> f64 {
    let sec = elapsed.as_secs_f64();
    if decoded == 0 || sec <= 0.0 {
        0.0
    } else {
        f64::from(decoded) / sec
    }
}

/// Persistent engine handle for the REPL — kept across turns so a
/// `/clear` (or end of a turn) doesn't require re-mmaping the GGUF.
/// The `GgufFile` and `Tokenizer` both outlive the engine; the
/// caller owns them on the stack and hands references to this
/// struct.
pub struct CliEngine<'gguf> {
    pub engine: DsV4FlashEngine<'gguf>,
    pub tokenizer: Tokenizer,
}

impl<'gguf> CliEngine<'gguf> {
    /// Load a model + tokenizer from `gguf`. The caller must keep the
    /// `GgufFile` alive for the lifetime of the returned engine.
    pub fn load(gguf: &'gguf GgufFile) -> Result<Self, CliError> {
        let tokenizer = Tokenizer::from_gguf(gguf).map_err(map_tokenizer_err)?;
        let model = load_dsv4_flash(gguf).map_err(map_model_err)?;
        Ok(Self {
            engine: DsV4FlashEngine::new(model),
            tokenizer,
        })
    }

    /// Drive one REPL turn: tokenize `user_msg` under `system` +
    /// `think`, prefill, then decode-stream to stdout until EOS or
    /// [`DEFAULT_MAX_DECODE_TOKENS`]. The session is constructed by
    /// the caller (so `/ctx N` and `/clear` can recreate it). The
    /// `dumper` is also caller-owned so a multi-turn REPL appends
    /// one JSONL stream rather than truncating the file each turn.
    #[allow(clippy::too_many_arguments)]
    pub fn run_turn(
        &self,
        session: &mut <DsV4FlashEngine<'gguf> as Engine>::Session<'_>,
        dumper: Option<&mut LogprobDumper>,
        user_msg: &str,
        system: Option<&str>,
        think: ThinkMode,
        flags: &RunFlags,
    ) -> Result<(), CliError> {
        let _span = tracing::info_span!(
            target: "rsllm_cli::engine",
            "cli_repl_turn",
            position_before = session.position(),
        )
        .entered();
        let tok_think = map_think(think);
        let tokens =
            self.tokenizer
                .encode_prompt(system.unwrap_or(""), user_msg, tok_think);
        if tokens.is_empty() {
            return Err(CliError::BadCommand(
                "tokenizer produced 0 tokens for the message".into(),
            ));
        }
        if session.position() + tokens.len() > session.capacity() {
            return Err(CliError::BadCommand(format!(
                "message is {} tokens, session at {}/{}; try /clear or /ctx N",
                tokens.len(),
                session.position(),
                session.capacity(),
            )));
        }
        tracing::debug!(
            target: "rsllm_cli::engine",
            n_prompt = tokens.len(),
            think = ?tok_think,
            "turn encoded",
        );

        let _ = session.prefill(&tokens).map_err(map_engine_err)?;
        let mut last_token = *tokens.last().expect("non-empty after empty check");
        let mut emitted_buf: Vec<u8> = Vec::with_capacity(64);
        let mut utf8_carry: Vec<u8> = Vec::with_capacity(4);
        let stdout = std::io::stdout();
        let mut out = stdout.lock();
        // Re-borrow `dumper` as Option<&mut> per step rather than
        // moving it — we want the same handle alive across steps and
        // returned implicitly to the caller via the borrow ending.
        let mut dumper_opt = dumper;
        let decode_started = std::time::Instant::now();
        let mut decoded = 0_u32;
        let mut hit_eos = false;
        for step_idx in 0..(DEFAULT_MAX_DECODE_TOKENS as u32) {
            let step = session.decode_one(last_token).map_err(map_engine_err)?;
            if step.token_id == self.tokenizer.eos_id() {
                hit_eos = true;
                break;
            }
            write_step_outputs(
                &self.tokenizer,
                &mut out,
                step_idx,
                &step,
                flags,
                dumper_opt.as_deref_mut(),
                &mut emitted_buf,
                &mut utf8_carry,
            )?;
            last_token = step.token_id;
            decoded += 1;
        }
        if !utf8_carry.is_empty() {
            out.write_all(String::from_utf8_lossy(&utf8_carry).as_bytes())
                .map_err(CliError::Io)?;
        }
        out.write_all(b"\n").map_err(CliError::Io)?;
        out.flush().map_err(CliError::Io)?;

        let elapsed = decode_started.elapsed();
        let tps = tok_per_sec(decoded, elapsed);
        tracing::info!(
            target: "rsllm_cli::engine",
            n_decoded = decoded,
            elapsed_ms = elapsed.as_millis() as u64,
            tok_per_sec = tps,
            hit_eos,
            cap_hit = !hit_eos && (decoded as usize) == DEFAULT_MAX_DECODE_TOKENS,
            "turn complete",
        );
        Ok(())
    }
}

/// Common per-step writer used by both one-shot and REPL paths.
/// Decodes the token to text, streams it to stdout, and feeds the
/// optional `--dump-tokens` / `--dump-logprobs` taps.
///
/// `utf8_carry` is a per-call carry buffer that preserves any
/// trailing partial UTF-8 bytes across decode steps. Byte-level BPE
/// can split a multibyte codepoint across two adjacent tokens; we
/// flush the longest valid UTF-8 prefix and stash the rest for the
/// next call. The Tokenizer crate's own docstring recommends this
/// pattern.
#[allow(clippy::too_many_arguments)]
fn write_step_outputs(
    tokenizer: &Tokenizer,
    out: &mut std::io::StdoutLock<'_>,
    step_idx: u32,
    step: &DecodeStep,
    flags: &RunFlags,
    dumper: Option<&mut LogprobDumper>,
    emitted_buf: &mut Vec<u8>,
    utf8_carry: &mut Vec<u8>,
) -> Result<(), CliError> {
    emitted_buf.clear();
    tokenizer.decode_into(step.token_id, emitted_buf);

    // Combine carry + this step's bytes, then emit only the longest
    // valid UTF-8 prefix. Anything left over is partial and carried
    // forward for the next call.
    utf8_carry.extend_from_slice(emitted_buf);
    let valid_end = longest_valid_utf8_prefix(utf8_carry);
    // Safe: `valid_end` is the byte length of a known-valid UTF-8 prefix.
    let text =
        std::str::from_utf8(&utf8_carry[..valid_end]).expect("valid prefix");
    out.write_all(text.as_bytes()).map_err(CliError::Io)?;
    out.flush().map_err(CliError::Io)?;

    if flags.dump_tokens {
        // For `--dump-tokens` we report the per-token raw bytes
        // (lossy-decoded) regardless of the carry state — so the
        // operator sees one line per *model token*, not per codepoint.
        let token_text = String::from_utf8_lossy(emitted_buf);
        emit_token_line(step_idx, step.token_id, &token_text);
    }
    if let Some(d) = dumper {
        let top = top_k_entries(&step.probs, flags.logprobs_top_k.max(1));
        d.write_step(step.token_id, &top)?;
    }

    // Drop the emitted prefix from the carry; keep the trailing
    // partial bytes for the next call.
    utf8_carry.drain(..valid_end);
    Ok(())
}

/// Byte length of the longest UTF-8 prefix of `buf` that decodes
/// cleanly. `std::str::from_utf8` reports either Ok (the whole buf
/// is valid) or Err with `valid_up_to()` giving the cut point.
fn longest_valid_utf8_prefix(buf: &[u8]) -> usize {
    match std::str::from_utf8(buf) {
        Ok(_) => buf.len(),
        Err(e) => e.valid_up_to(),
    }
}

/// Pick the top-K (token_id, prob) entries from `probs`, sorted by
/// descending prob. `logit` is `ln(prob)` — the post-sampler raw
/// logits aren't exposed by `DecodeStep` today (would require an
/// engine-API extension); the logprob is what every other inference
/// engine reports anyway, and it's directly comparable to the
/// official ds4 vector format.
fn top_k_entries(probs: &[f32], k: usize) -> Vec<LogprobEntry> {
    let mut indexed: Vec<(u32, f32)> = probs
        .iter()
        .enumerate()
        .map(|(i, p)| (i as u32, *p))
        .collect();
    indexed.sort_unstable_by(|a, b| {
        b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
    });
    indexed
        .into_iter()
        .take(k)
        .map(|(token_id, prob)| LogprobEntry {
            token_id,
            logit: prob.ln(),
            prob,
        })
        .collect()
}

fn map_tokenizer_err(e: rsllm_tokenizer::Error) -> CliError {
    CliError::BadCommand(format!("tokenizer: {e}"))
}

fn map_model_err(e: rsllm_models::Error) -> CliError {
    CliError::BadCommand(format!("model load: {e}"))
}

fn map_engine_err(e: EngineError) -> CliError {
    CliError::BadCommand(format!("engine: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn map_think_round_trip() {
        assert_eq!(map_think(ThinkMode::NoThink), TokThinkMode::None);
        assert_eq!(map_think(ThinkMode::Think), TokThinkMode::High);
        assert_eq!(map_think(ThinkMode::ThinkMax), TokThinkMode::Max);
    }

    #[test]
    fn sampling_params_honor_seed_flag() {
        let flags = RunFlags {
            seed: Some(0xC0FFEE),
            ..RunFlags::default()
        };
        let p = sampling_params_from_flags(&flags).unwrap();
        assert_eq!(p.seed, Some(0xC0FFEE));
    }

    #[test]
    fn sampling_params_seed_none_passes_through() {
        let flags = RunFlags::default();
        let p = sampling_params_from_flags(&flags).unwrap();
        assert_eq!(p.seed, None);
    }

    #[test]
    fn sampling_params_honor_all_sampler_flags() {
        let flags = RunFlags {
            temperature: Some(0.3),
            top_k: Some(50),
            top_p: Some(0.9),
            min_p: Some(0.1),
            ..RunFlags::default()
        };
        let p = sampling_params_from_flags(&flags).unwrap();
        assert!((p.temperature - 0.3).abs() < 1e-6);
        assert_eq!(p.top_k, Some(50));
        assert_eq!(p.top_p, Some(0.9));
        assert_eq!(p.min_p, Some(0.1));
    }

    #[test]
    fn sampling_params_reject_negative_temperature() {
        let flags = RunFlags {
            temperature: Some(-1.0),
            ..RunFlags::default()
        };
        let err = sampling_params_from_flags(&flags).unwrap_err();
        assert!(matches!(err, CliError::BadCommand(_)));
    }

    #[test]
    fn sampling_params_reject_nan_temperature() {
        let flags = RunFlags {
            temperature: Some(f32::NAN),
            ..RunFlags::default()
        };
        let err = sampling_params_from_flags(&flags).unwrap_err();
        assert!(matches!(err, CliError::BadCommand(_)));
    }

    #[test]
    fn sampling_params_reject_zero_top_k() {
        let flags = RunFlags {
            top_k: Some(0),
            ..RunFlags::default()
        };
        let err = sampling_params_from_flags(&flags).unwrap_err();
        assert!(matches!(err, CliError::BadCommand(_)));
    }

    #[test]
    fn sampling_params_reject_top_p_out_of_range() {
        for bad in [0.0_f32, -0.1, 1.5, f32::INFINITY] {
            let flags = RunFlags {
                top_p: Some(bad),
                ..RunFlags::default()
            };
            let err = sampling_params_from_flags(&flags).unwrap_err();
            assert!(matches!(err, CliError::BadCommand(_)), "expected reject for top_p = {bad}");
        }
    }

    #[test]
    fn sampling_params_reject_min_p_out_of_range() {
        for bad in [-0.1_f32, 1.0, 1.5, f32::INFINITY] {
            let flags = RunFlags {
                min_p: Some(bad),
                ..RunFlags::default()
            };
            let err = sampling_params_from_flags(&flags).unwrap_err();
            assert!(matches!(err, CliError::BadCommand(_)), "expected reject for min_p = {bad}");
        }
    }

    #[test]
    fn sampling_params_accept_zero_temperature_for_greedy() {
        let flags = RunFlags {
            temperature: Some(0.0),
            ..RunFlags::default()
        };
        let p = sampling_params_from_flags(&flags).unwrap();
        assert_eq!(p.temperature, 0.0);
    }

    #[test]
    fn sampling_params_accept_min_p_zero() {
        // 0.0 is the boundary that effectively disables the filter
        // but is still in range; must not be rejected.
        let flags = RunFlags {
            min_p: Some(0.0),
            ..RunFlags::default()
        };
        let p = sampling_params_from_flags(&flags).unwrap();
        assert_eq!(p.min_p, Some(0.0));
    }

    #[test]
    fn sampling_params_partial_override_preserves_defaults() {
        // Set ONLY temperature — assert the other three fields keep
        // their SamplingParams::default() values. This catches a
        // future refactor that replaces the `..default()` spread
        // with field-by-field copies and accidentally drops one.
        let flags = RunFlags {
            temperature: Some(0.5),
            ..RunFlags::default()
        };
        let p = sampling_params_from_flags(&flags).unwrap();
        let baseline = SamplingParams::default();
        assert!((p.temperature - 0.5).abs() < 1e-6);
        assert_eq!(p.top_k, baseline.top_k);
        assert_eq!(p.top_p, baseline.top_p);
        assert_eq!(p.min_p, baseline.min_p);
        assert_eq!(p.seed, baseline.seed);
    }

    #[test]
    fn top_k_picks_largest_and_sorts() {
        let probs = vec![0.1_f32, 0.4, 0.05, 0.3, 0.15];
        let top = top_k_entries(&probs, 3);
        assert_eq!(top.len(), 3);
        // Sorted descending by prob.
        assert_eq!(top[0].token_id, 1);
        assert!((top[0].prob - 0.4).abs() < 1e-6);
        assert_eq!(top[1].token_id, 3);
        assert_eq!(top[2].token_id, 4);
    }

    #[test]
    fn top_k_caps_at_probs_len() {
        let probs = vec![0.5_f32, 0.5];
        let top = top_k_entries(&probs, 100);
        assert_eq!(top.len(), 2);
    }

    #[test]
    fn top_k_logit_is_ln_of_prob() {
        let probs = vec![0.5_f32];
        let top = top_k_entries(&probs, 1);
        assert!((top[0].logit - 0.5_f32.ln()).abs() < 1e-6);
    }

    #[test]
    fn tok_per_sec_handles_zero_paths() {
        // Empty generation → 0 not NaN.
        assert!(
            (tok_per_sec(0, std::time::Duration::from_secs(1)) - 0.0).abs() < f64::EPSILON
        );
        // Zero elapsed → 0 (would otherwise divide by zero).
        assert!(
            (tok_per_sec(10, std::time::Duration::from_secs(0)) - 0.0).abs() < f64::EPSILON
        );
        // 100 tokens / 2s = 50 tok/s.
        let r = tok_per_sec(100, std::time::Duration::from_secs(2));
        assert!((r - 50.0).abs() < 1e-9);
    }

    #[test]
    fn longest_valid_utf8_prefix_handles_full_and_partial() {
        // Fully valid → whole length.
        assert_eq!(longest_valid_utf8_prefix(b"hello"), 5);
        // Empty → 0.
        assert_eq!(longest_valid_utf8_prefix(b""), 0);
        // Half of a 2-byte sequence (`é` = 0xC3 0xA9) at the end:
        // "abc" + 0xC3 → cut at 3.
        let mut buf = b"abc".to_vec();
        buf.push(0xC3);
        assert_eq!(longest_valid_utf8_prefix(&buf), 3);
        // Two-thirds of a 3-byte sequence at the end ("ab" + 0xE3 + 0x81):
        // cut at 2.
        let mut buf = b"ab".to_vec();
        buf.extend_from_slice(&[0xE3, 0x81]);
        assert_eq!(longest_valid_utf8_prefix(&buf), 2);
        // Full 3-byte CJK codepoint following ASCII: whole length.
        let mut buf = b"ab".to_vec();
        buf.extend_from_slice("中".as_bytes());
        assert_eq!(longest_valid_utf8_prefix(&buf), buf.len());
    }
}
