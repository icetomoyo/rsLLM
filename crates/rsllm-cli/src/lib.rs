//! Library face of the `rsllm` binary — exposes the CLI parser,
//! sub-command handlers, and REPL state so they can be unit-tested
//! without spawning a child process.
//!
//! Application binary lives in [`main`].

pub mod cli;
pub mod dump;
pub mod engine;
pub mod info;
pub mod inspect;
pub mod repl;

/// Error type for the CLI. Wraps GGUF errors + std::io + custom
/// REPL parse errors. Display impls intentionally stay terse — the
/// CLI surface is for humans.
#[derive(Debug, thiserror::Error)]
pub enum CliError {
    /// Underlying I/O failure.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    /// GGUF parse / load failure.
    #[error("gguf error: {0}")]
    Gguf(#[from] rsllm_gguf::Error),
    /// A slash command in the REPL was malformed.
    #[error("bad command: {0}")]
    BadCommand(String),
    /// A decode-mode invocation requires a model but the user didn't supply one.
    #[error("model required: {0}")]
    ModelRequired(String),
    /// Sub-feature is parked behind a future commit. Owns its message
    /// so call sites can embed paths / token counts / config snippets.
    ///
    /// Currently unused after F008.C.3.f wired the engine — kept so
    /// in-progress features (e.g. `--temperature` / `--top-k` flags
    /// that will land between v0.1.0 and v0.2.0) can resurface it
    /// without a CliError breaking change.
    #[allow(dead_code)]
    #[error("not yet implemented: {0}")]
    NotImplemented(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cli_error_display_carries_message() {
        let e = CliError::BadCommand("oops".into());
        assert!(format!("{e}").contains("oops"));
    }
}

// thiserror's #[from] uses thiserror::Error which we add as a dep below.
// (Declared via the workspace via crates/rsllm-cli/Cargo.toml.)
