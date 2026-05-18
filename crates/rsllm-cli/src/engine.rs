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

/// Derive [`SamplingParams`] from the CLI flags. Today the only
/// knob exposed is `--seed`; the rest of the chain stays on the
/// `SamplingParams::default()` (temperature 0.7 + min_p 0.05).
/// Adding `--temperature`, `--top-k`, `--top-p`, `--min-p` flags is a
/// follow-up to F008.C.3.f — kept out of this commit so the surface
/// stays minimal and the test matrix doesn't explode.
#[must_use]
pub fn sampling_params_from_flags(flags: &RunFlags) -> SamplingParams {
    SamplingParams {
        seed: flags.seed,
        ..SamplingParams::default()
    }
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
    let gguf = GgufFile::open(model_path)?;
    let tokenizer = Tokenizer::from_gguf(&gguf).map_err(map_tokenizer_err)?;
    let model = load_dsv4_flash(&gguf).map_err(map_model_err)?;
    let engine = DsV4FlashEngine::new(model);

    let params = sampling_params_from_flags(flags);
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

    let mut dumper = match &flags.dump_logprobs {
        Some(p) => Some(LogprobDumper::create(p, flags.logprobs_top_k.max(1))?),
        None => None,
    };

    // Prefill the prompt. The returned logits are discarded — the
    // first decode step is driven by `decode_one` seeded with the
    // last prompt token.
    let _ = session.prefill(&tokens).map_err(map_engine_err)?;

    let mut last_token = *tokens.last().expect("non-empty after empty check");
    let mut emitted_buf: Vec<u8> = Vec::with_capacity(64);
    let stdout = std::io::stdout();
    let mut out = stdout.lock();

    for step_idx in 0..(DEFAULT_MAX_DECODE_TOKENS as u32) {
        let step = session.decode_one(last_token).map_err(map_engine_err)?;
        if step.token_id == tokenizer.eos_id() {
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
        )?;
        last_token = step.token_id;
    }
    out.write_all(b"\n").map_err(CliError::Io)?;
    out.flush().map_err(CliError::Io)?;
    Ok(())
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
    /// the caller (so `/ctx N` and `/clear` can recreate it).
    pub fn run_turn(
        &self,
        session: &mut <DsV4FlashEngine<'gguf> as Engine>::Session<'_>,
        user_msg: &str,
        system: Option<&str>,
        think: ThinkMode,
        flags: &RunFlags,
    ) -> Result<(), CliError> {
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

        let mut dumper = match &flags.dump_logprobs {
            Some(p) => Some(LogprobDumper::create(p, flags.logprobs_top_k.max(1))?),
            None => None,
        };

        let _ = session.prefill(&tokens).map_err(map_engine_err)?;
        let mut last_token = *tokens.last().expect("non-empty after empty check");
        let mut emitted_buf: Vec<u8> = Vec::with_capacity(64);
        let stdout = std::io::stdout();
        let mut out = stdout.lock();
        for step_idx in 0..(DEFAULT_MAX_DECODE_TOKENS as u32) {
            let step = session.decode_one(last_token).map_err(map_engine_err)?;
            if step.token_id == self.tokenizer.eos_id() {
                break;
            }
            write_step_outputs(
                &self.tokenizer,
                &mut out,
                step_idx,
                &step,
                flags,
                dumper.as_mut(),
                &mut emitted_buf,
            )?;
            last_token = step.token_id;
        }
        out.write_all(b"\n").map_err(CliError::Io)?;
        out.flush().map_err(CliError::Io)?;
        Ok(())
    }
}

/// Common per-step writer used by both one-shot and REPL paths.
/// Decodes the token to text, streams it to stdout, and feeds the
/// optional `--dump-tokens` / `--dump-logprobs` taps.
#[allow(clippy::too_many_arguments)]
fn write_step_outputs(
    tokenizer: &Tokenizer,
    out: &mut std::io::StdoutLock<'_>,
    step_idx: u32,
    step: &DecodeStep,
    flags: &RunFlags,
    dumper: Option<&mut LogprobDumper>,
    emitted_buf: &mut Vec<u8>,
) -> Result<(), CliError> {
    emitted_buf.clear();
    tokenizer.decode_into(step.token_id, emitted_buf);
    // Decoded bytes can be partial UTF-8 (byte-level BPE). Use
    // from_utf8_lossy so a half-multibyte chunk emits the replacement
    // char rather than panicking — the next decode call usually
    // completes the codepoint.
    let text = String::from_utf8_lossy(emitted_buf);

    out.write_all(text.as_bytes()).map_err(CliError::Io)?;
    out.flush().map_err(CliError::Io)?;

    if flags.dump_tokens {
        emit_token_line(step_idx, step.token_id, &text);
    }
    if let Some(d) = dumper {
        let top = top_k_entries(&step.probs, flags.logprobs_top_k.max(1));
        d.write_step(step.token_id, &top)?;
    }
    Ok(())
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
        let p = sampling_params_from_flags(&flags);
        assert_eq!(p.seed, Some(0xC0FFEE));
    }

    #[test]
    fn sampling_params_seed_none_passes_through() {
        let flags = RunFlags::default();
        let p = sampling_params_from_flags(&flags);
        assert_eq!(p.seed, None);
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
}
