//! DeepSeek V4 Flash chat prompt assembly.
//!
//! Wraps role-tagged messages with the special tokens the model was
//! trained on, and prepends the "Reasoning Effort: Absolute maximum…"
//! preface when the caller selects [`ThinkMode::Max`].
//!
//! Ported by reference from `ds4.c:14700-14830` (MIT, The ds4.c authors).
//! Line numbers pinned to ds4 commit `ef0a490` (2026-05-17).

use crate::bpe::encode_text;
use crate::vocab::Vocab;

/// Thinking-budget mode selector for DS V4 Flash chat prompts.
///
/// DS V4 Flash always conditions on either `<think>` (model emits a
/// reasoning trace) or `</think>` (model jumps straight to the answer).
/// [`ThinkMode::Max`] additionally injects a long preface that asks for
/// the deepest reasoning the model can produce — recommended only for
/// contexts ≥ 384 K tokens (see [`THINK_MAX_MIN_CONTEXT`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThinkMode {
    /// `</think>` — model answers without an externalized trace.
    None,
    /// `<think>` — model emits a normal reasoning trace.
    High,
    /// Max preface + `<think>` — recommended only for ≥384 K contexts.
    Max,
}

impl ThinkMode {
    /// `true` for any mode that opens the assistant turn with
    /// `<think>` (i.e. High or Max).
    pub fn is_enabled(self) -> bool {
        matches!(self, Self::High | Self::Max)
    }

    /// Lowercase name suitable for CLI flags and log lines.
    pub fn name(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::High => "high",
            Self::Max => "max",
        }
    }

    /// Downgrade [`ThinkMode::Max`] to [`ThinkMode::High`] when the
    /// allocated context is shorter than [`THINK_MAX_MIN_CONTEXT`].
    pub fn for_context(self, ctx_size: u32) -> Self {
        if matches!(self, Self::Max) && ctx_size < THINK_MAX_MIN_CONTEXT {
            Self::High
        } else {
            self
        }
    }
}

/// Minimum context length (in tokens) at which Think-Max is enabled by
/// default. ds4 hard-codes this to 384 × 1024 (`ds4.c:71`).
pub const THINK_MAX_MIN_CONTEXT: u32 = 393_216;

/// Verbatim Think-Max preface from `ds4.c:63-66`.
pub const THINK_MAX_PREFIX: &str = "\
Reasoning Effort: Absolute maximum with no shortcuts permitted.
You MUST be very thorough in your thinking and comprehensively decompose the problem to resolve the root cause, rigorously stress-testing your logic against all potential paths, edge cases, and adversarial scenarios.
Explicitly write out your entire deliberation process, documenting every intermediate step, considered alternative, and rejected hypothesis to ensure absolutely no assumption is left unchecked.

";

/// Role of a [`Message`] in a chat transcript.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    /// System prompt — emitted verbatim before the user turn.
    System,
    /// Developer prompt — same handling as [`Role::System`] in ds4
    /// (`ds4.c:14051`). Kept as a distinct variant for API clarity.
    Developer,
    /// User message — wrapped with `<｜User｜>`.
    User,
    /// Assistant message — wrapped with `<｜Assistant｜></think>`.
    Assistant,
    /// Tool / function call result — wrapped with `<｜User｜>Tool: …`.
    Tool,
}

/// One role-tagged message in a chat transcript.
#[derive(Debug, Clone)]
pub struct Message<'a> {
    /// Speaker role.
    pub role: Role,
    /// Raw message text. Will be BPE-encoded.
    pub content: &'a str,
}

/// Build the prompt-time token sequence for a single (system, user)
/// turn under `think_mode`. Mirrors `encode_chat_prompt`
/// (`ds4.c:13943-13964`).
///
/// Layout:
///   - `BOS`
///   - `THINK_MAX_PREFIX` (only when `think_mode == Max`)
///   - encoded `system` text (if non-empty)
///   - `<｜User｜>`
///   - encoded `user_prompt`
///   - `<｜Assistant｜>`
///   - `<think>` (if `think_mode.is_enabled()`) else `</think>`
pub(crate) fn encode_prompt(
    vocab: &Vocab,
    system: &str,
    user_prompt: &str,
    think_mode: ThinkMode,
    out: &mut Vec<u32>,
) {
    out.push(vocab.bos_id());
    if matches!(think_mode, ThinkMode::Max) {
        encode_text(vocab, THINK_MAX_PREFIX, out);
    }
    if !system.is_empty() {
        encode_text(vocab, system, out);
    }
    out.push(vocab.user_id());
    encode_text(vocab, user_prompt, out);
    out.push(vocab.assistant_id());
    if think_mode.is_enabled() {
        out.push(vocab.think_start_id());
    } else {
        out.push(vocab.think_end_id());
    }
}

/// Append `message` to `tokens` per ds4's role-dispatch rules
/// (`ds4.c:14046-14066`).
///
/// Behavior per role:
///   - **System** — encode content verbatim. No marker.
///   - **User** — emit `<｜User｜>`, then encode content.
///   - **Tool** — emit `<｜User｜>`, then `"Tool: "`, then content.
///   - **Assistant** — emit `<｜Assistant｜>`. If the content does not
///     start with `<think>` or `</think>`, also emit `</think>` (so the
///     assistant turn doesn't appear to be a reasoning trace). Then
///     encode content.
pub(crate) fn append_message(vocab: &Vocab, msg: &Message<'_>, tokens: &mut Vec<u32>) {
    match msg.role {
        Role::System | Role::Developer => {
            encode_text(vocab, msg.content, tokens);
        }
        Role::User => {
            tokens.push(vocab.user_id());
            encode_text(vocab, msg.content, tokens);
        }
        Role::Tool => {
            tokens.push(vocab.user_id());
            encode_text(vocab, "Tool: ", tokens);
            encode_text(vocab, msg.content, tokens);
        }
        Role::Assistant => {
            tokens.push(vocab.assistant_id());
            if !msg.content.starts_with("<think>") && !msg.content.starts_with("</think>") {
                tokens.push(vocab.think_end_id());
            }
            encode_text(vocab, msg.content, tokens);
        }
    }
}

/// Append the closing assistant marker (`<｜Assistant｜>` + `<think>` or
/// `</think>`) to `tokens`. Used after the last user message to ask the
/// model to start producing its reply.
pub(crate) fn append_assistant_prefix(vocab: &Vocab, think_mode: ThinkMode, tokens: &mut Vec<u32>) {
    tokens.push(vocab.assistant_id());
    tokens.push(if think_mode.is_enabled() {
        vocab.think_start_id()
    } else {
        vocab.think_end_id()
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vocab::tests::fake_vocab;

    #[test]
    fn think_mode_is_enabled() {
        assert!(!ThinkMode::None.is_enabled());
        assert!(ThinkMode::High.is_enabled());
        assert!(ThinkMode::Max.is_enabled());
    }

    #[test]
    fn think_mode_for_context_downgrades_max() {
        assert_eq!(ThinkMode::Max.for_context(0), ThinkMode::High);
        assert_eq!(ThinkMode::Max.for_context(100_000), ThinkMode::High);
        assert_eq!(
            ThinkMode::Max.for_context(THINK_MAX_MIN_CONTEXT),
            ThinkMode::Max
        );
        assert_eq!(
            ThinkMode::Max.for_context(THINK_MAX_MIN_CONTEXT + 1),
            ThinkMode::Max
        );
        // High and None are never downgraded.
        assert_eq!(ThinkMode::High.for_context(0), ThinkMode::High);
        assert_eq!(ThinkMode::None.for_context(0), ThinkMode::None);
    }

    #[test]
    fn think_mode_name() {
        assert_eq!(ThinkMode::None.name(), "none");
        assert_eq!(ThinkMode::High.name(), "high");
        assert_eq!(ThinkMode::Max.name(), "max");
    }

    #[test]
    fn encode_prompt_emits_bos_user_assistant_think() {
        let v = fake_vocab();
        let mut out = Vec::new();
        encode_prompt(&v, "", "", ThinkMode::None, &mut out);
        // BOS, USER, ASSISTANT, </think> — no system, no user content.
        assert_eq!(
            out,
            vec![v.bos_id(), v.user_id(), v.assistant_id(), v.think_end_id()]
        );
    }

    #[test]
    fn encode_prompt_high_emits_think_start() {
        let v = fake_vocab();
        let mut out = Vec::new();
        encode_prompt(&v, "", "", ThinkMode::High, &mut out);
        assert_eq!(out.last().copied(), Some(v.think_start_id()));
    }

    #[test]
    fn encode_prompt_max_prepends_prefix() {
        let v = fake_vocab();
        let mut out_max = Vec::new();
        encode_prompt(&v, "", "", ThinkMode::Max, &mut out_max);
        let mut out_high = Vec::new();
        encode_prompt(&v, "", "", ThinkMode::High, &mut out_high);
        // Max should emit strictly more tokens than High (the prefix).
        assert!(out_max.len() > out_high.len());
        // Both end with <think>.
        assert_eq!(out_max.last().copied(), Some(v.think_start_id()));
        assert_eq!(out_high.last().copied(), Some(v.think_start_id()));
        // Both start with BOS.
        assert_eq!(out_max.first().copied(), Some(v.bos_id()));
    }

    #[test]
    fn append_user_message_emits_user_marker() {
        let v = fake_vocab();
        let mut out = Vec::new();
        append_message(
            &v,
            &Message {
                role: Role::User,
                content: "",
            },
            &mut out,
        );
        assert_eq!(out, vec![v.user_id()]);
    }

    #[test]
    fn append_assistant_message_inserts_think_end() {
        let v = fake_vocab();
        let mut out = Vec::new();
        append_message(
            &v,
            &Message {
                role: Role::Assistant,
                content: "",
            },
            &mut out,
        );
        // ASSISTANT then </think>.
        assert_eq!(out, vec![v.assistant_id(), v.think_end_id()]);
    }

    #[test]
    fn append_assistant_with_think_prefix_omits_think_end() {
        let v = fake_vocab();
        let mut out = Vec::new();
        append_message(
            &v,
            &Message {
                role: Role::Assistant,
                content: "<think>",
            },
            &mut out,
        );
        // ASSISTANT, then the content goes straight into encode_text.
        // No extra think_end_id. content "<think>" passes through
        // encode_text → joyai → bpe; the first token is ASSISTANT.
        assert_eq!(out[0], v.assistant_id());
        // Followed by whatever encode_text produced — but NOT
        // think_end_id as the second token.
        assert_ne!(out.get(1).copied(), Some(v.think_end_id()));
    }

    #[test]
    fn append_tool_message_uses_user_marker_and_prefix() {
        let v = fake_vocab();
        let mut out = Vec::new();
        append_message(
            &v,
            &Message {
                role: Role::Tool,
                content: "",
            },
            &mut out,
        );
        // First token is USER (tool uses user role); the "Tool: " prefix
        // does not appear because fake_vocab lacks ASCII byte tokens and
        // bpe silently skips unknown bytes.
        assert_eq!(out[0], v.user_id());
    }

    #[test]
    fn append_system_message_has_no_marker() {
        let v = fake_vocab();
        let mut out = Vec::new();
        append_message(
            &v,
            &Message {
                role: Role::System,
                content: "",
            },
            &mut out,
        );
        assert!(out.is_empty());
    }

    #[test]
    fn developer_role_behaves_like_system() {
        // ds4.c:14051 — `"developer"` and `"system"` share the same branch.
        let v = fake_vocab();
        let mut sys = Vec::new();
        let mut dev = Vec::new();
        append_message(
            &v,
            &Message {
                role: Role::System,
                content: "x",
            },
            &mut sys,
        );
        append_message(
            &v,
            &Message {
                role: Role::Developer,
                content: "x",
            },
            &mut dev,
        );
        assert_eq!(sys, dev);
    }

    #[test]
    fn assistant_prefix_emits_marker_and_think() {
        let v = fake_vocab();
        let mut out = Vec::new();
        append_assistant_prefix(&v, ThinkMode::High, &mut out);
        assert_eq!(out, vec![v.assistant_id(), v.think_start_id()]);
        out.clear();
        append_assistant_prefix(&v, ThinkMode::None, &mut out);
        assert_eq!(out, vec![v.assistant_id(), v.think_end_id()]);
    }
}
