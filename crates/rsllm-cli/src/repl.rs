//! REPL skeleton — rustyline + slash command dispatch.
//!
//! v0.1.0 surface: line editing, history, and the full slash-command
//! menu from ds4 (`ds4_cli.c`). The decode loop itself plugs in via
//! the `on_message` callback — F008.C will wire the real Engine /
//! Session there; this file is independently testable today.
//!
//! Slash commands (mirrors ds4):
//!
//! ```text
//! /help                    show command help
//! /think                   switch to think mode
//! /think-max               switch to think-max mode
//! /nothink                 switch to nothink mode
//! /ctx N                   rebuild session with new ctx size
//! /read FILE               read file as next user message
//! /system "..."            set system prompt
//! /clear                   clear session
//! /quit  /exit             quit (Ctrl-D equivalent)
//! ```

use std::path::PathBuf;

use crate::CliError;
use crate::cli::ThinkMode;

/// Parsed slash command. Returned by [`parse_command`] when a line
/// begins with `/`; otherwise the line is a regular user message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SlashCommand {
    Help,
    Think,
    ThinkMax,
    NoThink,
    /// `/ctx N` — rebuild the session with a new context size.
    SetCtx(usize),
    /// `/read FILE` — read a file into the next message.
    ReadFile(PathBuf),
    /// `/system "..."` — set the system prompt.
    SetSystem(String),
    Clear,
    Quit,
    /// Unrecognized `/foo` — caller prints help and stays in the REPL.
    Unknown(String),
}

/// Parse one user-entered line.
///
/// Returns `Ok(Some(cmd))` if the line is a slash command,
/// `Ok(None)` if it is a regular user message (caller should hand it
/// to the model), and `Err(_)` if the slash command is malformed
/// (e.g. `/ctx` with no number).
pub fn parse_command(line: &str) -> Result<Option<SlashCommand>, CliError> {
    let trimmed = line.trim();
    if !trimmed.starts_with('/') {
        return Ok(None);
    }
    let mut iter = trimmed.splitn(2, char::is_whitespace);
    let head = iter.next().unwrap_or("");
    let tail = iter.next().unwrap_or("").trim();
    let cmd = match head {
        "/help" => SlashCommand::Help,
        "/think" => SlashCommand::Think,
        "/think-max" => SlashCommand::ThinkMax,
        "/nothink" => SlashCommand::NoThink,
        "/clear" => SlashCommand::Clear,
        "/quit" | "/exit" => SlashCommand::Quit,
        "/ctx" => {
            let n: usize = tail.parse().map_err(|_| {
                CliError::BadCommand(format!("/ctx expects a positive integer, got `{tail}`"))
            })?;
            if n == 0 {
                return Err(CliError::BadCommand("/ctx requires N > 0".into()));
            }
            SlashCommand::SetCtx(n)
        }
        "/read" => {
            if tail.is_empty() {
                return Err(CliError::BadCommand("/read expects a file path".into()));
            }
            SlashCommand::ReadFile(PathBuf::from(tail))
        }
        "/system" => {
            // Strip optional surrounding quotes for ergonomic
            // `/system "you are helpful"`.
            let prompt = tail.trim_matches('"').to_string();
            SlashCommand::SetSystem(prompt)
        }
        other => SlashCommand::Unknown(other.to_string()),
    };
    Ok(Some(cmd))
}

/// Mutable REPL state that survives across turns. Decode-side state
/// (Engine / Session / Sampler / KV cache) lives in F008.C; this
/// struct just tracks the CLI-visible knobs.
#[derive(Debug, Clone)]
pub struct ReplState {
    /// Active think mode — changed by `/think`, `/think-max`, `/nothink`.
    pub think_mode: ThinkMode,
    /// Configured KV context size. `/ctx N` updates this.
    pub ctx_size: usize,
    /// Optional system prompt. `None` = no system message injected.
    pub system_prompt: Option<String>,
}

impl ReplState {
    /// Initial state derived from CLI arguments.
    #[must_use]
    pub fn new(think_mode: ThinkMode, ctx_size: usize) -> Self {
        Self {
            think_mode,
            ctx_size,
            system_prompt: None,
        }
    }

    /// Apply a slash command. Returns a status line for the REPL to
    /// print and a flag indicating whether the loop should exit.
    pub fn apply(&mut self, cmd: SlashCommand) -> CommandOutcome {
        match cmd {
            SlashCommand::Help => CommandOutcome::stay(HELP_TEXT.to_string()),
            SlashCommand::Think => {
                self.think_mode = ThinkMode::Think;
                CommandOutcome::stay("think mode: think".into())
            }
            SlashCommand::ThinkMax => {
                self.think_mode = ThinkMode::ThinkMax;
                CommandOutcome::stay("think mode: think-max".into())
            }
            SlashCommand::NoThink => {
                self.think_mode = ThinkMode::NoThink;
                CommandOutcome::stay("think mode: nothink".into())
            }
            SlashCommand::SetCtx(n) => {
                self.ctx_size = n;
                CommandOutcome::stay(format!("ctx_size = {n} (session will rebuild)"))
            }
            SlashCommand::ReadFile(path) => CommandOutcome::stay(format!(
                "/read queued: {} (consumed on next message)",
                path.display()
            )),
            SlashCommand::SetSystem(prompt) => {
                if prompt.is_empty() {
                    self.system_prompt = None;
                    CommandOutcome::stay("system prompt cleared".into())
                } else {
                    let len = prompt.len();
                    self.system_prompt = Some(prompt);
                    CommandOutcome::stay(format!("system prompt set ({len} bytes)"))
                }
            }
            SlashCommand::Clear => CommandOutcome::stay("session cleared".into()),
            SlashCommand::Quit => CommandOutcome::exit(),
            SlashCommand::Unknown(name) => CommandOutcome::stay(format!(
                "unknown command `{name}`. Type /help for a list."
            )),
        }
    }
}

/// Result of applying a slash command — message to print + whether
/// to exit the REPL.
#[derive(Debug, Clone)]
pub struct CommandOutcome {
    pub message: String,
    pub exit: bool,
}

impl CommandOutcome {
    fn stay(message: String) -> Self {
        Self {
            message,
            exit: false,
        }
    }
    fn exit() -> Self {
        Self {
            message: "bye.".into(),
            exit: true,
        }
    }
}

/// Static help text printed by `/help`.
const HELP_TEXT: &str = "\
slash commands:
  /help                 show this help
  /think                switch to think mode
  /think-max            switch to think-max mode
  /nothink              switch to nothink mode
  /ctx N                rebuild session with new ctx size
  /read FILE            read file as next user message
  /system \"...\"        set system prompt (empty = clear)
  /clear                clear session
  /quit  /exit          quit (Ctrl-D equivalent)";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn non_slash_is_user_message() {
        let r = parse_command("hello there").unwrap();
        assert!(r.is_none());
    }

    #[test]
    fn parses_simple_commands() {
        assert_eq!(parse_command("/help").unwrap(), Some(SlashCommand::Help));
        assert_eq!(parse_command("/think").unwrap(), Some(SlashCommand::Think));
        assert_eq!(parse_command("/quit").unwrap(), Some(SlashCommand::Quit));
        assert_eq!(parse_command("/exit").unwrap(), Some(SlashCommand::Quit));
        assert_eq!(parse_command("/clear").unwrap(), Some(SlashCommand::Clear));
    }

    #[test]
    fn parses_ctx_with_number() {
        assert_eq!(
            parse_command("/ctx 4096").unwrap(),
            Some(SlashCommand::SetCtx(4096))
        );
    }

    #[test]
    fn ctx_without_number_errors() {
        let err = parse_command("/ctx foo").unwrap_err();
        assert!(matches!(err, CliError::BadCommand(_)));
    }

    #[test]
    fn ctx_zero_rejected() {
        let err = parse_command("/ctx 0").unwrap_err();
        assert!(matches!(err, CliError::BadCommand(_)));
    }

    #[test]
    fn read_requires_path() {
        let err = parse_command("/read").unwrap_err();
        assert!(matches!(err, CliError::BadCommand(_)));
        let ok = parse_command("/read /tmp/a.txt").unwrap();
        assert_eq!(ok, Some(SlashCommand::ReadFile(PathBuf::from("/tmp/a.txt"))));
    }

    #[test]
    fn system_strips_quotes() {
        let ok = parse_command(r#"/system "you are helpful""#).unwrap();
        assert_eq!(ok, Some(SlashCommand::SetSystem("you are helpful".into())));
    }

    #[test]
    fn unknown_command_is_returned_to_caller() {
        let r = parse_command("/foobar").unwrap();
        assert_eq!(r, Some(SlashCommand::Unknown("/foobar".into())));
    }

    #[test]
    fn state_tracks_think_changes() {
        let mut s = ReplState::new(ThinkMode::NoThink, 1024);
        let out = s.apply(SlashCommand::Think);
        assert!(!out.exit);
        assert_eq!(s.think_mode, ThinkMode::Think);
        let _ = s.apply(SlashCommand::ThinkMax);
        assert_eq!(s.think_mode, ThinkMode::ThinkMax);
        let out = s.apply(SlashCommand::NoThink);
        assert_eq!(s.think_mode, ThinkMode::NoThink);
        assert!(!out.exit);
    }

    #[test]
    fn quit_signals_exit() {
        let mut s = ReplState::new(ThinkMode::NoThink, 1024);
        let out = s.apply(SlashCommand::Quit);
        assert!(out.exit);
    }

    #[test]
    fn ctx_change_updates_state() {
        let mut s = ReplState::new(ThinkMode::NoThink, 1024);
        s.apply(SlashCommand::SetCtx(8192));
        assert_eq!(s.ctx_size, 8192);
    }

    #[test]
    fn system_prompt_set_and_clear() {
        let mut s = ReplState::new(ThinkMode::NoThink, 1024);
        s.apply(SlashCommand::SetSystem("hello".into()));
        assert_eq!(s.system_prompt.as_deref(), Some("hello"));
        s.apply(SlashCommand::SetSystem(String::new()));
        assert!(s.system_prompt.is_none());
    }

    #[test]
    fn help_text_lists_every_command() {
        // Sanity — the help block should mention each slash word so a
        // future renamer doesn't forget to update the help.
        for word in &[
            "/help",
            "/think",
            "/think-max",
            "/nothink",
            "/ctx",
            "/read",
            "/system",
            "/clear",
            "/quit",
            "/exit",
        ] {
            assert!(HELP_TEXT.contains(word), "/help missing `{word}`");
        }
    }
}
