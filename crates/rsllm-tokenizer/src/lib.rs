//! # rsllm-tokenizer
//!
//! Pure-Rust tokenizer for DeepSeek V4 Flash (rsLLM v0.1.0). Ports the
//! JoyAI state-machine pre-tokenizer + GPT-2 byte-level BPE from
//! [`ds4.c`](https://github.com/icetomoyo/rsLLM/blob/main/NOTICE.md)
//! (MIT). No HuggingFace `tokenizers` crate dependency.
//!
//! ## Usage
//!
//! ```no_run
//! use rsllm_gguf::GgufFile;
//! use rsllm_tokenizer::{Tokenizer, ThinkMode, Message, Role};
//!
//! let gguf = GgufFile::open("model.gguf")?;
//! let tok = Tokenizer::from_gguf(&gguf)?;
//!
//! // Encode plain text.
//! let ids = tok.encode("hello world");
//!
//! // Build a chat prompt.
//! let ids = tok.encode_prompt(
//!     "You are a helpful assistant.",
//!     "Write a haiku about rust.",
//!     ThinkMode::High,
//! );
//!
//! // Or assemble a multi-turn chat from messages.
//! let mut ids = vec![tok.bos_id()];
//! tok.append_message(&Message { role: Role::User, content: "hi" }, &mut ids);
//! tok.append_assistant_prefix(ThinkMode::None, &mut ids);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! See `docs/features/v0.1.0.md` for the full design.

mod bpe;
mod byte_encode;
mod chat;
mod decode;
mod error;
mod joyai;
mod special;
mod vocab;

pub use chat::{Message, Role, THINK_MAX_MIN_CONTEXT, THINK_MAX_PREFIX, ThinkMode};
pub use error::Error;

use rsllm_gguf::GgufFile;

use vocab::Vocab;

/// DeepSeek V4 Flash tokenizer.
///
/// Construct via [`Tokenizer::from_gguf`] (or [`Tokenizer::from_metadata`])
/// and use it to encode plain text, render chat transcripts, or build
/// prompt-time token sequences.
#[derive(Debug, Clone)]
pub struct Tokenizer {
    vocab: Vocab,
}

impl Tokenizer {
    /// Build a tokenizer from a parsed GGUF file. Returns an error if
    /// the file's `tokenizer.ggml.*` metadata is missing, has the
    /// wrong type, declares an unsupported `pre`, or is missing any
    /// of the 7 DS V4 Flash special tokens.
    pub fn from_gguf(file: &GgufFile) -> Result<Self, Error> {
        Self::from_metadata(file.metadata())
    }

    /// Build a tokenizer directly from a [`rsllm_gguf::Metadata`] map.
    /// Useful for tests or for non-mmap GGUF readers.
    pub fn from_metadata(meta: &rsllm_gguf::Metadata) -> Result<Self, Error> {
        let vocab = Vocab::from_metadata(meta)?;
        Ok(Self { vocab })
    }

    /// Total vocabulary size (number of distinct token ids).
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// Look up a token id by exact text.
    pub fn id_of(&self, text: &str) -> Option<u32> {
        self.vocab.id_of(text)
    }

    /// Look up a token text by id.
    pub fn token_of(&self, id: u32) -> Option<&str> {
        self.vocab.token_of(id)
    }

    /// BOS token id (`<｜begin▁of▁sentence｜>`).
    pub fn bos_id(&self) -> u32 {
        self.vocab.bos_id()
    }

    /// EOS token id (`<｜end▁of▁sentence｜>`).
    pub fn eos_id(&self) -> u32 {
        self.vocab.eos_id()
    }

    /// `<｜User｜>` token id.
    pub fn user_id(&self) -> u32 {
        self.vocab.user_id()
    }

    /// `<｜Assistant｜>` token id.
    pub fn assistant_id(&self) -> u32 {
        self.vocab.assistant_id()
    }

    /// `<think>` token id.
    pub fn think_start_id(&self) -> u32 {
        self.vocab.think_start_id()
    }

    /// `</think>` token id.
    pub fn think_end_id(&self) -> u32 {
        self.vocab.think_end_id()
    }

    /// `｜DSML｜` token id (tool-call protocol).
    pub fn dsml_id(&self) -> u32 {
        self.vocab.dsml_id()
    }

    /// Encode plain text into token ids. Does **not** detect special
    /// token literals; use [`encode_rendered`](Self::encode_rendered)
    /// when the input may contain `<｜User｜>` etc.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::new();
        bpe::encode_text(&self.vocab, text, &mut out);
        out
    }

    /// Encode pre-rendered chat text. Special-token literals
    /// (`<｜User｜>`, `<think>`, etc.) are mapped to their exact ids
    /// instead of being run through BPE.
    pub fn encode_rendered(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::new();
        special::encode_with_specials(&self.vocab, text, &mut out);
        out
    }

    /// Build the token sequence for a single-turn `(system, prompt)`
    /// chat under `think_mode`. See [`chat::encode_prompt`] for the
    /// exact layout.
    pub fn encode_prompt(&self, system: &str, prompt: &str, think_mode: ThinkMode) -> Vec<u32> {
        let mut out = Vec::new();
        chat::encode_prompt(&self.vocab, system, prompt, think_mode, &mut out);
        out
    }

    /// Append `message` to a growing chat token stream. Caller is
    /// responsible for pushing [`Self::bos_id`] first and (when ready)
    /// calling [`Self::append_assistant_prefix`] to ask the model to
    /// generate.
    pub fn append_message(&self, message: &Message<'_>, tokens: &mut Vec<u32>) {
        chat::append_message(&self.vocab, message, tokens);
    }

    /// Append the trailing `<｜Assistant｜>` + `<think>` / `</think>`
    /// marker pair that signals the model to start producing its reply.
    pub fn append_assistant_prefix(&self, think_mode: ThinkMode, tokens: &mut Vec<u32>) {
        chat::append_assistant_prefix(&self.vocab, think_mode, tokens);
    }

    /// Decode a single token id, appending its raw byte expansion to
    /// `out`. Streaming-friendly: call once per generated token.
    ///
    /// A "safe flush boundary" is a point at which `out` ends on a
    /// complete UTF-8 codepoint. For special-marker tokens (those
    /// containing `｜`) every emission is itself valid UTF-8, so the
    /// flush is safe immediately after; for normal tokens, callers
    /// should attempt `std::str::from_utf8(&out)` and only emit the
    /// longest valid prefix, retaining any trailing partial bytes.
    pub fn decode_into(&self, id: u32, out: &mut Vec<u8>) {
        decode::decode_one(&self.vocab, id, out);
    }

    /// Decode a slice of token ids into UTF-8 text. Returns
    /// [`Error::DecodePartialUtf8`] if the result is not valid UTF-8
    /// (typically because the slice cuts a multi-byte codepoint).
    pub fn decode(&self, ids: &[u32]) -> Result<String, Error> {
        decode::decode_ids(&self.vocab, ids)
    }
}
