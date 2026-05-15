//! End-to-end integration test for the `Tokenizer` public API.
//!
//! Builds a synthetic GGUF [`Metadata`] table containing the 7 DS V4
//! Flash special tokens plus enough byte tokens to round-trip simple
//! ASCII text. Verifies that:
//!
//!   * `Tokenizer::from_metadata` accepts a valid synthetic vocab;
//!   * `encode` round-trips ASCII text via byte-level BPE byte fallback;
//!   * `encode_rendered` splits on `<｜User｜>` / `<｜Assistant｜>`;
//!   * `encode_prompt` produces a `[BOS, USER, ..., ASSISTANT, think]`
//!     skeleton for all three [`ThinkMode`] values;
//!   * `decode` inverts `encode` for ASCII input;
//!   * missing-metadata / wrong-type / missing-special errors are
//!     surfaced through [`Error`].

use rsllm_gguf::{Array, Metadata, Value};
use rsllm_tokenizer::{Error, Message, Role, ThinkMode, Tokenizer};

/// Build a Metadata containing a minimal but valid DS V4 Flash vocab:
/// 7 specials + 256 single-byte GPT-2 tokens, no merges.
fn synthetic_metadata() -> Metadata {
    // Token list: specials first, then every GPT-2 single-byte codepoint
    // (so byte fallback covers any ASCII input).
    let specials = [
        "<\u{ff5c}begin\u{2581}of\u{2581}sentence\u{ff5c}>",
        "<\u{ff5c}end\u{2581}of\u{2581}sentence\u{ff5c}>",
        "<\u{ff5c}User\u{ff5c}>",
        "<\u{ff5c}Assistant\u{ff5c}>",
        "<think>",
        "</think>",
        "\u{ff5c}DSML\u{ff5c}",
    ];

    let mut tokens: Vec<String> = specials.iter().map(|s| (*s).to_string()).collect();
    // Re-implement byte-encoding inline (the tokenizer crate keeps it
    // private). This mirrors crate::byte_encode::byte_to_codepoint.
    for b in 0..=255u8 {
        let cp = byte_to_codepoint(b);
        let c = char::from_u32(cp).unwrap();
        let mut s = String::new();
        s.push(c);
        // Skip duplicates: very rare for single-byte tokens but be safe.
        if !tokens.iter().any(|t| t == &s) {
            tokens.push(s);
        }
    }

    let mut meta = Metadata::new();
    meta.insert(
        "tokenizer.ggml.pre".to_string(),
        Value::String("joyai-llm".to_string()),
    );
    meta.insert(
        "tokenizer.ggml.tokens".to_string(),
        Value::Array(Array::String(tokens)),
    );
    meta.insert(
        "tokenizer.ggml.merges".to_string(),
        Value::Array(Array::String(Vec::new())),
    );
    meta
}

fn byte_to_codepoint(b: u8) -> u32 {
    if (33..=126).contains(&b) || (161..=172).contains(&b) || b >= 174 {
        return u32::from(b);
    }
    let mut n: u32 = 0;
    for x in 0..=255u32 {
        let xb = x as u8;
        if (33..=126).contains(&xb) || (161..=172).contains(&xb) || xb >= 174 {
            continue;
        }
        if x == u32::from(b) {
            return 256 + n;
        }
        n += 1;
    }
    u32::from(b)
}

#[test]
fn tokenizer_loads_from_synthetic_metadata() {
    let meta = synthetic_metadata();
    let tok = Tokenizer::from_metadata(&meta).expect("must load");
    // 7 specials + ≥ 188 single-byte tokens; we don't lock the exact
    // count, just verify it's well above 7 and the specials resolved.
    assert!(tok.vocab_size() > 100, "got {}", tok.vocab_size());
    assert_eq!(tok.bos_id(), 0);
    assert_eq!(tok.user_id(), 2);
    assert_eq!(tok.assistant_id(), 3);
    assert_eq!(tok.think_start_id(), 4);
    assert_eq!(tok.think_end_id(), 5);
    assert_eq!(tok.dsml_id(), 6);
}

#[test]
fn encode_ascii_then_decode_round_trips() {
    let tok = Tokenizer::from_metadata(&synthetic_metadata()).unwrap();
    let text = "Hello, world!\n";
    let ids = tok.encode(text);
    assert!(!ids.is_empty(), "encode produced no tokens");
    // No merges in synthetic vocab → byte fallback → one token per byte.
    assert_eq!(ids.len(), text.len());
    let decoded = tok.decode(&ids).expect("decode must succeed");
    assert_eq!(decoded, text);
}

#[test]
fn encode_rendered_splits_on_user_marker() {
    let tok = Tokenizer::from_metadata(&synthetic_metadata()).unwrap();
    let rendered = "<\u{ff5c}User\u{ff5c}>hi";
    let ids = tok.encode_rendered(rendered);
    assert_eq!(ids[0], tok.user_id(), "first id must be user marker");
    assert!(ids.len() > 1, "must encode 'hi' after marker");
}

#[test]
fn encode_prompt_layout_for_all_think_modes() {
    let tok = Tokenizer::from_metadata(&synthetic_metadata()).unwrap();

    let none = tok.encode_prompt("", "hi", ThinkMode::None);
    assert_eq!(none[0], tok.bos_id(), "must start with BOS");
    assert!(none.contains(&tok.user_id()), "must contain USER");
    assert!(none.contains(&tok.assistant_id()), "must contain ASSISTANT");
    assert_eq!(
        *none.last().unwrap(),
        tok.think_end_id(),
        "ThinkMode::None must close with </think>"
    );

    let high = tok.encode_prompt("", "hi", ThinkMode::High);
    assert_eq!(
        *high.last().unwrap(),
        tok.think_start_id(),
        "ThinkMode::High must close with <think>"
    );

    let max = tok.encode_prompt("", "hi", ThinkMode::Max);
    assert_eq!(*max.last().unwrap(), tok.think_start_id());
    assert!(
        max.len() > high.len(),
        "ThinkMode::Max must prepend the reasoning prefix"
    );
}

#[test]
fn append_message_builds_multi_turn_chat() {
    let tok = Tokenizer::from_metadata(&synthetic_metadata()).unwrap();
    let mut ids = vec![tok.bos_id()];
    tok.append_message(
        &Message {
            role: Role::User,
            content: "ping",
        },
        &mut ids,
    );
    tok.append_message(
        &Message {
            role: Role::Assistant,
            content: "pong",
        },
        &mut ids,
    );
    tok.append_message(
        &Message {
            role: Role::User,
            content: "again",
        },
        &mut ids,
    );
    tok.append_assistant_prefix(ThinkMode::None, &mut ids);

    // Verify the marker tokens appear in order.
    let positions = |needle: u32| {
        ids.iter()
            .enumerate()
            .filter(|&(_, &id)| id == needle)
            .map(|(i, _)| i)
            .collect::<Vec<_>>()
    };
    let user_positions = positions(tok.user_id());
    let assistant_positions = positions(tok.assistant_id());
    let think_end_positions = positions(tok.think_end_id());

    assert_eq!(user_positions.len(), 2, "two user turns");
    assert_eq!(assistant_positions.len(), 2, "two assistant slots");
    // The assistant message had no </think> prefix → an implicit
    // </think> is inserted right after the marker. Plus the trailing
    // assistant prefix also emits one. So two </think>s total.
    assert_eq!(think_end_positions.len(), 2);
}

#[test]
fn missing_required_key_returns_error() {
    let meta = synthetic_metadata();
    // Rebuild without the `tokenizer.ggml.pre` key.
    let mut stripped = Metadata::new();
    for (k, v) in meta.iter() {
        if k != "tokenizer.ggml.pre" {
            stripped.insert(k.to_string(), v.clone());
        }
    }
    let err = Tokenizer::from_metadata(&stripped).unwrap_err();
    assert!(
        matches!(err, Error::MissingKey("tokenizer.ggml.pre")),
        "got {err:?}"
    );
}

#[test]
fn unsupported_pre_tokenizer_returns_error() {
    let mut meta = synthetic_metadata();
    meta.insert(
        "tokenizer.ggml.pre".to_string(),
        Value::String("llama3".to_string()),
    );
    let err = Tokenizer::from_metadata(&meta).unwrap_err();
    assert!(
        matches!(err, Error::UnsupportedPreTokenizer(_)),
        "got {err:?}"
    );
}

#[test]
fn missing_special_token_returns_error() {
    let mut meta = synthetic_metadata();
    // Drop one of the specials by replacing the token list with one
    // that's missing `<think>`.
    let specials_minus_think = [
        "<\u{ff5c}begin\u{2581}of\u{2581}sentence\u{ff5c}>",
        "<\u{ff5c}end\u{2581}of\u{2581}sentence\u{ff5c}>",
        "<\u{ff5c}User\u{ff5c}>",
        "<\u{ff5c}Assistant\u{ff5c}>",
        // <think> removed
        "</think>",
        "\u{ff5c}DSML\u{ff5c}",
        "filler",
    ];
    let tokens: Vec<String> = specials_minus_think
        .iter()
        .map(|s| (*s).to_string())
        .collect();
    meta.insert(
        "tokenizer.ggml.tokens".to_string(),
        Value::Array(Array::String(tokens)),
    );
    let err = Tokenizer::from_metadata(&meta).unwrap_err();
    assert!(
        matches!(err, Error::MissingSpecialToken("<think>")),
        "got {err:?}"
    );
}
