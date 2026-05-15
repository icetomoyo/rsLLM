//! Special-token aware tokenization.
//!
//! When the input text already contains special-token literals — e.g. a
//! rendered chat transcript with `<｜User｜>` and `<｜Assistant｜>` markers —
//! we must emit the exact special-token id rather than running the marker
//! through BPE. This module scans for the 7 DS V4 Flash specials and
//! splits the input around them; spans of non-special text are forwarded
//! to [`crate::bpe::encode_text`].
//!
//! Ported by reference from `ds4.c:13970-14023` (MIT, The ds4.c authors).

use crate::bpe::encode_text;
use crate::vocab::{SPECIAL_TEXTS, Vocab};

/// If a special token literal starts at byte `pos` in `bytes`, return
/// `Some((token_id, byte_len))`. Otherwise return `None`.
///
/// Scans all 7 specials in [`SPECIAL_TEXTS`] order; ds4 does a linear
/// scan as well (ds4.c:13984-13991).
fn special_token_at(vocab: &Vocab, bytes: &[u8], pos: usize) -> Option<(u32, usize)> {
    for (idx, text) in SPECIAL_TEXTS.iter().enumerate() {
        let text_bytes = text.as_bytes();
        let end = pos.checked_add(text_bytes.len())?;
        if end > bytes.len() {
            continue;
        }
        if &bytes[pos..end] == text_bytes {
            return Some((vocab.special_ids[idx], text_bytes.len()));
        }
    }
    None
}

/// Tokenize `text` while preserving any special-token literals it
/// contains. Equivalent to ds4's `ds4_tokenize_rendered_chat`.
///
/// Behavior (matching ds4.c:14004-14023):
///   * walks `text` byte by byte;
///   * on each step, checks the 7 specials at the current position;
///   * if a special matches, flushes the pending span via plain BPE
///     encoding, emits the special id, and resumes after the special;
///   * else advances one byte.
///
/// Stepping a single byte at a time is safe because every special token
/// starts with either `<` (0x3C) or `｜` (= U+FF5C, encoded as `0xEF 0xBD
/// 0x9C` — lead byte 0xEF). Both 0x3C and 0xEF are *not* in the UTF-8
/// continuation-byte range (0x80..=0xBF), so [`special_token_at`] cannot
/// produce a false positive at a position that sits mid-codepoint. The
/// 1-byte step itself may land on a continuation byte, but on the next
/// iteration the scan simply finds nothing and steps again, until it
/// reaches the next codepoint boundary. The flushed BPE spans, in turn,
/// always sit on codepoint boundaries because they begin at the input
/// start or immediately after a special (codepoint-aligned by
/// construction) and end at a special-token start (also codepoint
/// aligned).
pub(crate) fn encode_with_specials(vocab: &Vocab, text: &str, out: &mut Vec<u32>) {
    let bytes = text.as_bytes();
    let mut span_start = 0;
    let mut p = 0;
    while p < bytes.len() {
        if let Some((token, len)) = special_token_at(vocab, bytes, p) {
            if span_start < p {
                // Span is on UTF-8 boundaries because specials always
                // begin/end on codepoint boundaries.
                encode_text(vocab, &text[span_start..p], out);
            }
            out.push(token);
            p += len;
            span_start = p;
        } else {
            p += 1;
        }
    }
    if span_start < bytes.len() {
        encode_text(vocab, &text[span_start..], out);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vocab::tests::fake_vocab;

    #[test]
    fn detects_user_marker() {
        let v = fake_vocab();
        let bytes = "<\u{ff5c}User\u{ff5c}>foo".as_bytes();
        let hit = special_token_at(&v, bytes, 0).expect("must match");
        assert_eq!(hit.0, v.user_id());
        assert_eq!(hit.1, "<\u{ff5c}User\u{ff5c}>".len());
    }

    #[test]
    fn returns_none_when_no_match() {
        let v = fake_vocab();
        let bytes = b"hello";
        assert!(special_token_at(&v, bytes, 0).is_none());
    }

    #[test]
    fn returns_none_when_buffer_too_short() {
        let v = fake_vocab();
        let bytes = b"<";
        assert!(special_token_at(&v, bytes, 0).is_none());
    }

    #[test]
    fn encode_pure_text_calls_back_into_bpe() {
        // No specials: full string passes through BPE.
        let mut v = fake_vocab();
        // Give "a" a known id (already 7 in fake_vocab).
        let mut out = Vec::new();
        encode_with_specials(&v, "a", &mut out);
        // fake_vocab has "a" at id 7 — but no merges, so byte fallback;
        // for "a" the GPT-2 codepoint is "a" itself, looked up directly.
        assert_eq!(out, vec![v.id_of("a").unwrap()]);
        // Silence unused-mut warning.
        let _ = &mut v;
    }

    #[test]
    fn special_token_emitted_inline() {
        let v = fake_vocab();
        let mut out = Vec::new();
        encode_with_specials(&v, "<\u{ff5c}User\u{ff5c}>", &mut out);
        assert_eq!(out, vec![v.user_id()]);
    }

    #[test]
    fn span_before_and_after_special() {
        // Add "a" lookup to fake_vocab already, so "a<｜User｜>a" → [a, USER, a].
        let v = fake_vocab();
        let mut out = Vec::new();
        encode_with_specials(&v, "a<\u{ff5c}User\u{ff5c}>a", &mut out);
        assert_eq!(out.len(), 3);
        assert_eq!(out[1], v.user_id());
        assert_eq!(out[0], v.id_of("a").unwrap());
        assert_eq!(out[2], v.id_of("a").unwrap());
    }

    #[test]
    fn two_specials_back_to_back() {
        let v = fake_vocab();
        let mut out = Vec::new();
        encode_with_specials(&v, "<\u{ff5c}User\u{ff5c}><think>", &mut out);
        assert_eq!(out, vec![v.user_id(), v.think_start_id()]);
    }

    #[test]
    fn empty_text_emits_nothing() {
        let v = fake_vocab();
        let mut out = Vec::new();
        encode_with_specials(&v, "", &mut out);
        assert!(out.is_empty());
    }
}
