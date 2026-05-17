//! Token-id → text decoding.
//!
//! Inverts the GPT-2 byte-encoding step done by [`crate::byte_encode`].
//! Special-token literals (anything containing `｜`, U+FF5C) are emitted
//! verbatim; everything else is decoded codepoint-by-codepoint back into
//! the original raw byte sequence.
//!
//! Ported by reference from `ds4.c:14911+` (`ds4_token_text`, MIT, The
//! ds4.c authors). Line numbers pinned to ds4 commit `ef0a490` (2026-05-17).

use crate::byte_encode::codepoint_to_byte;
use crate::error::Error;
use crate::vocab::Vocab;

/// `true` if `s` contains the fullwidth vertical bar U+FF5C, which all
/// 7 DS V4 Flash special tokens use as a delimiter. ds4 uses this as a
/// cheap "is literal special" test (`ds4.c:14140-14147`).
///
/// Note: this heuristic does **not** match `<think>` / `</think>` (no
/// U+FF5C). Those happen to decode correctly via the GPT-2 codepoint
/// inversion path because every byte is printable ASCII (`<`, `/`,
/// alpha, `>`), which round-trips through `codepoint_to_byte`
/// unchanged. If a future special token is added that contains
/// non-ASCII bytes but no U+FF5C, it must either route through this
/// check or have its bytes round-tripped explicitly.
fn token_is_literal_special(s: &str) -> bool {
    s.contains('\u{ff5c}')
}

/// Decode one token id into a raw byte sequence appended to `out`.
///
/// Mirrors `ds4_token_text` (`ds4.c:14149-14177`):
///   * out-of-range ids contribute nothing;
///   * special-marker tokens (those containing `｜`) pass through as UTF-8
///     verbatim;
///   * other tokens are mapped codepoint-by-codepoint through
///     [`codepoint_to_byte`]; unrecognized codepoints are silently
///     skipped, matching ds4's behavior.
pub(crate) fn decode_one(vocab: &Vocab, id: u32, out: &mut Vec<u8>) {
    let Some(text) = vocab.token_of(id) else {
        return;
    };
    if token_is_literal_special(text) {
        out.extend_from_slice(text.as_bytes());
        return;
    }
    for ch in text.chars() {
        if let Some(b) = codepoint_to_byte(ch as u32) {
            out.push(b);
        }
    }
}

/// Decode a slice of token ids into a UTF-8 string. Returns
/// [`Error::DecodePartialUtf8`] if the resulting byte sequence is not
/// valid UTF-8 — typically because the slice cuts a multi-byte
/// codepoint in half. Callers doing streaming output should buffer the
/// bytes via [`decode_one`] and only attempt UTF-8 conversion at safe
/// boundaries.
pub(crate) fn decode_ids(vocab: &Vocab, ids: &[u32]) -> Result<String, Error> {
    let mut buf = Vec::with_capacity(ids.len() * 2);
    for &id in ids {
        decode_one(vocab, id, &mut buf);
    }
    String::from_utf8(buf).map_err(|_| Error::DecodePartialUtf8)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::byte_encode::encode_bytes;
    use crate::vocab::tests::fake_vocab;

    #[test]
    fn out_of_range_id_emits_nothing() {
        let v = fake_vocab();
        let mut buf = Vec::new();
        decode_one(&v, 9_999, &mut buf);
        assert!(buf.is_empty());
    }

    #[test]
    fn special_token_passes_through_verbatim() {
        let v = fake_vocab();
        let mut buf = Vec::new();
        decode_one(&v, v.user_id(), &mut buf);
        assert_eq!(buf, "<\u{ff5c}User\u{ff5c}>".as_bytes());
    }

    #[test]
    fn non_special_token_inverts_byte_encoding() {
        // Build a vocab where token id 8 is the GPT-2 encoding of "A".
        let mut v = fake_vocab();
        let encoded_a = encode_bytes(b"A");
        v.id_to_token.push(encoded_a.clone());
        v.token_to_id
            .insert(encoded_a, v.id_to_token.len() as u32 - 1);
        let id = (v.id_to_token.len() - 1) as u32;

        let mut buf = Vec::new();
        decode_one(&v, id, &mut buf);
        assert_eq!(buf, b"A");
    }

    #[test]
    fn decode_ids_returns_utf8() {
        let v = fake_vocab();
        // Two special tokens emit valid UTF-8.
        let s = decode_ids(&v, &[v.user_id(), v.assistant_id()]).unwrap();
        assert!(s.contains("User"));
        assert!(s.contains("Assistant"));
    }

    #[test]
    fn decode_ids_rejects_partial_utf8() {
        // Build a token whose decoded bytes are a lone UTF-8 lead byte
        // (e.g. 0xE4 — the start of a 3-byte codepoint with no
        // continuations).
        let mut v = fake_vocab();
        let encoded = encode_bytes(&[0xE4]);
        v.id_to_token.push(encoded.clone());
        let id = (v.id_to_token.len() - 1) as u32;
        v.token_to_id.insert(encoded, id);

        let err = decode_ids(&v, &[id]).unwrap_err();
        assert!(matches!(err, Error::DecodePartialUtf8));
    }
}
