//! clap-based argument parsing for the `rsllm` binary.
//!
//! Mirrors ds4 CLI shape (`ds4_cli.c`):
//!
//! ```text
//! rsllm                                          # interactive REPL (default)
//! rsllm -p "TEXT"                                # one-shot run
//! rsllm --prompt-file FILE                       # one-shot run from file
//! rsllm inspect -m MODEL                         # load + dump summary, no inference
//! rsllm info  [-m MODEL] [-c CTX_SIZE]           # system caps + memory estimate
//! ```
//!
//! Diagnostic flags (preserved from the 2026-05-11 F008 design review):
//!
//! ```text
//! --dump-tokens                  per-step token id + text to stderr
//! --dump-logprobs FILE           per-step top-K logits as JSON
//! --logprobs-top-k N             default 20; only meaningful with --dump-logprobs
//! ```

use std::path::PathBuf;

use clap::{Args, Parser, Subcommand, ValueEnum};

/// Top-level CLI entry point. Sub-commands handle the explicit modes;
/// flags on the root cover the implicit one-shot and REPL paths.
#[derive(Debug, Parser)]
#[command(name = "rsllm", version, about, long_about = None)]
pub struct Cli {
    /// Subcommand (`inspect`, `info`). Omit for REPL / one-shot mode.
    #[command(subcommand)]
    pub command: Option<Command>,

    /// Global log level. Overrides nothing if `RUST_LOG` is set;
    /// otherwise sets the tracing subscriber's level filter. Use
    /// `trace` for per-step decode timing, `debug` for per-layer
    /// progress, `info` for the default per-load summary, `warn`
    /// for placeholders / TODOs, `error` to silence non-fatal
    /// messages.
    #[arg(long = "log-level", value_enum, default_value_t = LogLevel::Info, global = true)]
    pub log_level: LogLevel,

    #[command(flatten)]
    pub run: RunFlags,
}

/// Tracing level filter exposed via `--log-level`. Mirrors
/// [`tracing::Level`] but adds an explicit `Off` for users who want
/// to silence everything including warnings.
#[derive(Debug, Clone, Copy, ValueEnum, PartialEq, Eq, Default)]
pub enum LogLevel {
    /// Emit nothing — useful for scripted runs where stderr should
    /// stay clean.
    Off,
    /// Only fatal-class messages (none today; reserved).
    Error,
    /// Placeholders, TODOs, and recoverable anomalies.
    Warn,
    /// Default — one-shot summary, REPL transitions, model load.
    #[default]
    Info,
    /// Per-layer progress, scratch-resize reasons, sampler choices.
    Debug,
    /// Per-step decode timing, per-token logprob, KV-cache state.
    Trace,
}

impl LogLevel {
    /// Map to the equivalent EnvFilter directive string. Returns
    /// `"off"` for [`LogLevel::Off`]; otherwise the lowercase
    /// `tracing::Level` name.
    #[must_use]
    pub fn as_filter_directive(self) -> &'static str {
        match self {
            LogLevel::Off => "off",
            LogLevel::Error => "error",
            LogLevel::Warn => "warn",
            LogLevel::Info => "info",
            LogLevel::Debug => "debug",
            LogLevel::Trace => "trace",
        }
    }
}

/// Explicit subcommands. Anything else goes through the root flags.
#[derive(Debug, Subcommand)]
pub enum Command {
    /// Load a model and dump its tensor / metadata summary. No inference.
    Inspect(InspectArgs),
    /// Print system capabilities + memory estimate. No inference.
    Info(InfoArgs),
}

/// Arguments accepted on the root command — cover REPL launch + one-shot.
#[derive(Debug, Args, Default)]
pub struct RunFlags {
    /// One-shot prompt text. If provided, exit after one generation.
    /// Mutually exclusive with `--prompt-file`.
    #[arg(short = 'p', long = "prompt")]
    pub prompt: Option<String>,

    /// One-shot prompt read from a file. Mutually exclusive with `-p`.
    #[arg(long = "prompt-file", conflicts_with = "prompt")]
    pub prompt_file: Option<PathBuf>,

    /// Model path. If omitted, the REPL will refuse to start.
    #[arg(short = 'm', long = "model")]
    pub model: Option<PathBuf>,

    /// Context size used by the KV cache. Defaults to ds4's 32k.
    #[arg(short = 'c', long = "ctx-size", default_value_t = 32_768)]
    pub ctx_size: usize,

    /// Initial think mode for the REPL / one-shot run.
    #[arg(long = "think", value_enum, default_value_t = ThinkMode::NoThink)]
    pub think: ThinkMode,

    /// Echo each generated token + its text to stderr.
    #[arg(long = "dump-tokens")]
    pub dump_tokens: bool,

    /// Write per-step top-K logprobs to this JSON file. Schema is
    /// ds4-compatible (see `crate::dump`).
    #[arg(long = "dump-logprobs")]
    pub dump_logprobs: Option<PathBuf>,

    /// `K` for `--dump-logprobs`. Default matches ds4 official-API
    /// vectors (top 20).
    #[arg(long = "logprobs-top-k", default_value_t = 20)]
    pub logprobs_top_k: usize,

    /// Optional fixed seed for the multinomial draw. If omitted the
    /// sampler uses its built-in fallback seed.
    #[arg(long = "seed")]
    pub seed: Option<u64>,

    /// Softmax temperature. `0.0` selects greedy / argmax decoding
    /// (no draw). When omitted, falls back to `SamplingParams::default()`
    /// (the v0.1.0 default = 0.7).
    #[arg(long = "temperature")]
    pub temperature: Option<f32>,

    /// Keep only the top-K logits before normalizing. Disabled when
    /// omitted (the default). Must be ≥ 1 when set.
    #[arg(long = "top-k")]
    pub top_k: Option<usize>,

    /// Nucleus (cumulative-probability) filter. Keep the smallest
    /// prefix of the sorted distribution whose cumulative mass ≥
    /// `top_p`. Must lie in `(0.0, 1.0]`. When omitted, falls back
    /// to `SamplingParams::default()` (= 1.0, i.e. effectively off).
    #[arg(long = "top-p")]
    pub top_p: Option<f32>,

    /// Relative-probability cutoff — drop tokens whose probability
    /// is below `min_p × max_prob`. Must lie in `[0.0, 1.0)`. When
    /// omitted, falls back to `SamplingParams::default()` (= 0.05).
    #[arg(long = "min-p")]
    pub min_p: Option<f32>,
}

/// `inspect` arguments. Matches ds4 `model_summary()`.
#[derive(Debug, Args)]
pub struct InspectArgs {
    /// Model path (GGUF).
    #[arg(short = 'm', long = "model")]
    pub model: PathBuf,
}

/// `info` arguments — model is optional. Without a model we still
/// print system info; with one we add a KV memory estimate.
#[derive(Debug, Args)]
pub struct InfoArgs {
    /// Optional model path. If provided, GGUF metadata is parsed for
    /// the KV memory estimate.
    #[arg(short = 'm', long = "model")]
    pub model: Option<PathBuf>,

    /// Context size for the memory estimate. Matches the default
    /// used by the REPL (32 768 tokens).
    #[arg(short = 'c', long = "ctx-size", default_value_t = 32_768)]
    pub ctx_size: usize,
}

/// Inference think-mode flag. Mirrors ds4's three-way mode switch.
/// Implementation lives in F008.C (decode loop) but the CLI surface
/// is settled today so REPL `/think` etc. can target it.
#[derive(Debug, Clone, Copy, ValueEnum, PartialEq, Eq, Default)]
pub enum ThinkMode {
    /// No `<think>` block. Most concise output.
    #[default]
    NoThink,
    /// Standard think mode — model uses internal scratch.
    Think,
    /// Maximum-thought mode — wider exploration.
    ThinkMax,
}

impl ThinkMode {
    /// Human-friendly label used in REPL status output.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            ThinkMode::NoThink => "nothink",
            ThinkMode::Think => "think",
            ThinkMode::ThinkMax => "think-max",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    #[test]
    fn cli_definition_is_valid() {
        // clap's own consistency check — catches argument conflicts at compile time.
        Cli::command().debug_assert();
    }

    #[test]
    fn prompt_and_prompt_file_conflict() {
        let res = Cli::try_parse_from(["rsllm", "-p", "x", "--prompt-file", "/tmp/p.txt"]);
        assert!(res.is_err(), "expected --prompt + --prompt-file to conflict");
    }

    #[test]
    fn info_subcommand_parses_without_model() {
        let cli = Cli::try_parse_from(["rsllm", "info"]).unwrap();
        match cli.command {
            Some(Command::Info(args)) => {
                assert!(args.model.is_none());
                assert_eq!(args.ctx_size, 32_768);
            }
            _ => panic!("expected Info subcommand"),
        }
    }

    #[test]
    fn inspect_requires_model() {
        let res = Cli::try_parse_from(["rsllm", "inspect"]);
        assert!(res.is_err(), "inspect without -m must error");
    }

    #[test]
    fn run_flags_default_to_nothink() {
        let cli = Cli::try_parse_from(["rsllm"]).unwrap();
        assert!(cli.command.is_none());
        assert_eq!(cli.run.think, ThinkMode::NoThink);
        assert!(!cli.run.dump_tokens);
        assert_eq!(cli.run.logprobs_top_k, 20);
    }

    #[test]
    fn think_mode_label() {
        assert_eq!(ThinkMode::NoThink.label(), "nothink");
        assert_eq!(ThinkMode::Think.label(), "think");
        assert_eq!(ThinkMode::ThinkMax.label(), "think-max");
    }
}
