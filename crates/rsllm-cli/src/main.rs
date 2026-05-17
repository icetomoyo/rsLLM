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
use rsllm_cli::repl::{ReplState, parse_command};
use rsllm_cli::{CliError, dump, info, inspect};

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
        eprintln!("Try `rsllm -m PATH` (F008.C will wire the full decode loop),");
        eprintln!("or `rsllm info` / `rsllm inspect -m PATH` for read-only inspection.");
        print_banner();
        return Ok(());
    }
    run_repl(run)
}

/// One-shot mode: parse args, then bail with `NotImplemented` until
/// F008.C lands the decode loop. We still validate the dump-logprobs
/// path so a user-facing typo is caught immediately.
fn one_shot(run: &RunFlags, prompt: String) -> Result<(), CliError> {
    if run.model.is_none() {
        return Err(CliError::ModelRequired(
            "one-shot mode (`-p`) needs `-m PATH`".into(),
        ));
    }
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
    Err(CliError::NotImplemented(format!(
        "decode loop lands in F008.C (model={:?}, prompt={} chars)",
        run.model.as_deref().map(std::path::Path::display),
        prompt.len(),
    )))
}

/// REPL loop. Uses `rustyline` for line editing + history. The decode
/// path is stubbed exactly like `one_shot` — slash commands work
/// today, model invocations land with F008.C.
fn run_repl(run: RunFlags) -> Result<(), CliError> {
    let mut state = ReplState::new(run.think, run.ctx_size);
    let mut rl = DefaultEditor::new()
        .map_err(|e| CliError::Io(std::io::Error::other(format!("rustyline init: {e}"))))?;

    // Load history if present — file path mirrors ds4's `~/.ds4_history`
    // convention (rsLLM uses its own name to avoid collisions).
    let hist_path = history_path();
    if let Some(p) = &hist_path {
        let _ = rl.load_history(p);
    }

    println!("rsLLM v{}", rsllm_core::version());
    println!("type /help for command list, /quit to exit.");

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
                let outcome = state.apply(cmd);
                if !outcome.message.is_empty() {
                    println!("{}", outcome.message);
                }
                if outcome.exit {
                    break;
                }
            }
            None => {
                // Regular user message — decode lands in F008.C. We
                // surface the same `NotImplemented` as one-shot so
                // future telemetry can grep one string.
                eprintln!(
                    "rsllm: {}",
                    CliError::NotImplemented(format!(
                        "decode loop lands in F008.C (msg={} chars, think={})",
                        trimmed.len(),
                        state.think_mode.label(),
                    ))
                );
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
    println!("Full decode loop lands with F008.C; see docs/features/v0.1.0.md.");
}

/// `~/.rsllm_history` — the convention ds4 follows for its REPL
/// history, namespaced to avoid collisions.
fn history_path() -> Option<std::path::PathBuf> {
    let home = std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))?;
    Some(std::path::PathBuf::from(home).join(".rsllm_history"))
}
