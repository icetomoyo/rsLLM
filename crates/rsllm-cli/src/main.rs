//! rsLLM command-line interface.
//!
//! v0.0.1 — workspace skeleton only. The real subcommands (`chat` / `run` /
//! `info`) are tracked as FEATURE_008 and will be implemented as the other
//! v0.1.0 features land.

use clap::{Parser, Subcommand};

/// rsLLM — Rust-native LLM inference engine.
#[derive(Debug, Parser)]
#[command(name = "rsllm", version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
}

/// Top-level subcommands.
///
/// All variants are stubs in v0.0.1 — they print a "not yet implemented"
/// message and exit. Stubs are kept so that downstream tooling, shell
/// completions, and documentation can be wired up early.
#[derive(Debug, Subcommand)]
enum Command {
    /// Print system capabilities (OS, CPU features, available backends).
    Info,
    /// Run a one-shot inference against a model.
    Run,
    /// Open an interactive chat REPL.
    Chat,
}

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let cli = Cli::parse();

    match cli.command {
        None => print_banner(),
        Some(Command::Info) => {
            println!("rsLLM info (stub)");
            println!("  Library version : {}", rsllm_core::version());
            println!("  Target os       : {}", std::env::consts::OS);
            println!("  Target arch     : {}", std::env::consts::ARCH);
            println!();
            println!("Real capability detection arrives with FEATURE_008.");
        }
        Some(Command::Run | Command::Chat) => {
            eprintln!(
                "rsLLM v{} — this subcommand is not yet implemented.",
                rsllm_core::version()
            );
            eprintln!("See docs/features/v0.1.0.md for the M0 roadmap.");
            std::process::exit(2);
        }
    }
}

fn print_banner() {
    println!("rsLLM v{}", rsllm_core::version());
    println!("Status: pre-M0 (workspace skeleton only)");
    println!();
    println!("Try `rsllm info` for system info, or see docs/00-overview.md for the project plan.");
    println!("Issue tracker: https://github.com/icetomoyo/rsLLM");
}
