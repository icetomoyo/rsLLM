//! `rsllm` binary entry point.
//!
//! Argument parsing, sub-command dispatch, and the REPL launcher live
//! here. The actual decode loop is gated on F008.C (Engine / Session
//! integration); today the inference paths return a clear "not yet
//! implemented" error so the rest of the CLI surface (info / inspect
//! / parse / REPL editing / slash commands) is independently
//! testable and shippable.

use std::io::IsTerminal;
use std::process::ExitCode;

use clap::Parser;
use rustyline::DefaultEditor;
use rustyline::error::ReadlineError;

use rsllm_cli::cli::{Cli, Command, RunFlags};
use rsllm_cli::engine::CliEngine;
use rsllm_cli::repl::{ReplState, SlashCommand, parse_command};
use rsllm_cli::{CliError, dump, engine as eng, info, inspect};
use rsllm_core::{Engine, SamplingParams};
use rsllm_gguf::GgufFile;

fn main() -> ExitCode {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let cli = Cli::parse();

    match dispatch(cli) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("rsllm: {e}");
            ExitCode::from(2)
        }
    }
}

fn dispatch(cli: Cli) -> Result<(), CliError> {
    match cli.command {
        Some(Command::Info(args)) => info::run(args.model.as_deref(), args.ctx_size),
        Some(Command::Inspect(args)) => inspect::run(&args.model),
        None => dispatch_default(cli.run),
    }
}

/// No subcommand: pick between one-shot inference, REPL, or the
/// fallback banner. Mirrors ds4's "default = REPL" behavior.
fn dispatch_default(run: RunFlags) -> Result<(), CliError> {
    // 1. Explicit one-shot prompt (`-p TEXT` / `--prompt-file FILE`)
    if let Some(text) = run.prompt.clone() {
        return one_shot(&run, text);
    }
    if let Some(path) = run.prompt_file.clone() {
        let text = std::fs::read_to_string(&path)?;
        return one_shot(&run, text);
    }

    // 2. REPL — only if stdin is a tty AND we have something to run
    //    against. Without a model, the REPL would just print errors
    //    so we shortcut to the banner instead.
    if !std::io::stdin().is_terminal() {
        print_banner();
        return Ok(());
    }
    if run.model.is_none() {
        eprintln!(
            "rsLLM v{} — no model supplied; not entering REPL.",
            rsllm_core::version()
        );
        eprintln!("Try `rsllm -m PATH`, or `rsllm info` / `rsllm inspect -m PATH`");
        eprintln!("for read-only inspection without loading the model.");
        print_banner();
        return Ok(());
    }
    run_repl(run)
}

/// One-shot mode: load the model and stream a single response to
/// stdout. EOS or [`eng::DEFAULT_MAX_DECODE_TOKENS`] terminates.
fn one_shot(run: &RunFlags, prompt: String) -> Result<(), CliError> {
    let model_path = run
        .model
        .as_deref()
        .ok_or_else(|| CliError::ModelRequired("one-shot mode (`-p`) needs `-m PATH`".into()))?;

    // Validate `--dump-logprobs` early — opening the file now means
    // a permission error fires before we burn time loading the model.
    if let Some(path) = &run.dump_logprobs {
        let _ = dump::LogprobDumper::create(path, run.logprobs_top_k.max(1))?;
    }

    eprintln!(
        "rsllm one-shot (prompt={} chars, think={}, ctx={}, dump_tokens={}, seed={:?})",
        prompt.len(),
        run.think.label(),
        run.ctx_size,
        run.dump_tokens,
        run.seed,
    );

    eng::run_one_shot(model_path, &prompt, None, run)
}

/// REPL loop. Uses `rustyline` for line editing + history. Loads the
/// model once at startup; sessions are rebuilt on `/ctx N` and
/// `/clear` so the user can roll back without re-mmaping the GGUF.
fn run_repl(run: RunFlags) -> Result<(), CliError> {
    let model_path = run
        .model
        .as_deref()
        .ok_or_else(|| CliError::ModelRequired("REPL needs `-m PATH`".into()))?;

    let mut state = ReplState::new(run.think, run.ctx_size);
    let mut rl = DefaultEditor::new()
        .map_err(|e| CliError::Io(std::io::Error::other(format!("rustyline init: {e}"))))?;

    // Load history if present — file path mirrors ds4's `~/.ds4_history`
    // convention (rsLLM uses its own name to avoid collisions).
    let hist_path = history_path();
    if let Some(p) = &hist_path {
        let _ = rl.load_history(p);
    }

    println!("rsLLM v{} — loading {} ...", rsllm_core::version(), model_path.display());
    let gguf = GgufFile::open(model_path)?;
    let cli_engine = CliEngine::load(&gguf)?;
    let mut session = cli_engine
        .engine
        .start_session(state.ctx_size, sampling_params(&run))
        .map_err(|e| CliError::BadCommand(format!("start_session: {e}")))?;
    println!("ready. type /help for command list, /quit to exit.");

    loop {
        let prompt = format!("[{}]> ", state.think_mode.label());
        let line = match rl.readline(&prompt) {
            Ok(s) => s,
            Err(ReadlineError::Interrupted) => {
                eprintln!("^C — type /quit to exit");
                continue;
            }
            Err(ReadlineError::Eof) => {
                println!("bye.");
                break;
            }
            Err(e) => {
                return Err(CliError::Io(std::io::Error::other(format!(
                    "readline: {e}"
                ))));
            }
        };

        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let _ = rl.add_history_entry(trimmed);

        match parse_command(trimmed)? {
            Some(cmd) => {
                // `/ctx N` and `/clear` need to rebuild the session
                // *after* state mutation, so capture the variant
                // before delegating to `state.apply`.
                let needs_new_session = matches!(
                    cmd,
                    SlashCommand::SetCtx(_) | SlashCommand::Clear
                );
                let outcome = state.apply(cmd);
                if !outcome.message.is_empty() {
                    println!("{}", outcome.message);
                }
                if outcome.exit {
                    break;
                }
                if needs_new_session {
                    session = cli_engine
                        .engine
                        .start_session(state.ctx_size, sampling_params(&run))
                        .map_err(|e| {
                            CliError::BadCommand(format!("start_session: {e}"))
                        })?;
                }
            }
            None => {
                // Regular user message — run a decode turn.
                if let Err(e) = cli_engine.run_turn(
                    &mut session,
                    trimmed,
                    state.system_prompt.as_deref(),
                    state.think_mode,
                    &run,
                ) {
                    eprintln!("rsllm: {e}");
                }
            }
        }
    }

    if let Some(p) = &hist_path {
        let _ = rl.save_history(p);
        // History lines may contain sensitive prompt text. Tighten
        // permissions on Unix; Windows uses ACLs so this is a no-op.
        restrict_history_perms(p);
    }
    Ok(())
}

fn sampling_params(run: &RunFlags) -> SamplingParams {
    rsllm_cli::engine::sampling_params_from_flags(run)
}

#[cfg(unix)]
fn restrict_history_perms(path: &std::path::Path) {
    use std::os::unix::fs::PermissionsExt;
    let _ = std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600));
}

#[cfg(not(unix))]
fn restrict_history_perms(_path: &std::path::Path) {
    // Windows: ACLs from the parent directory cover us.
}

fn print_banner() {
    println!("rsLLM v{}", rsllm_core::version());
    println!("Try `rsllm info` for system info, or `rsllm inspect -m MODEL` for a GGUF summary.");
    println!("`rsllm -m MODEL` enters the REPL; `rsllm -m MODEL -p TEXT` runs a one-shot prompt.");
}

/// `~/.rsllm_history` — the convention ds4 follows for its REPL
/// history, namespaced to avoid collisions.
fn history_path() -> Option<std::path::PathBuf> {
    let home = std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))?;
    Some(std::path::PathBuf::from(home).join(".rsllm_history"))
}
