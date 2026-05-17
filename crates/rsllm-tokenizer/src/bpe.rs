//! GPT-2 byte-level BPE inner loop.
//!
//! Given a pre-tokenized piece (one of [`crate::joyai::pre_tokenize`]'s
//! output strings), this module byte-encodes it and then repeatedly merges
//! the lowest-ranked adjacent symbol pair until no merge is in the
//! vocab's `merge_rank` table. The final symbol list is mapped to token
//! ids and pushed to `out`.
//!
//! Ported by reference from `ds4.c:14381-14470` (MIT, The ds4.c authors).
//! Line numbers pinned to ds4 commit `ef0a490` (2026-05-17).

use crate::byte_encode::encode_bytes;
use crate::vocab::Vocab;

/// Apply byte-level BPE to one pre-tokenized piece and append token ids
/// to `out`.
///
/// Algorithm (matching ds4.c:14398-14470):
///   1. Byte-encode the piece (raw bytes → GPT-2 printable codepoints).
///   2. Split the encoded UTF-8 string into one symbol per codepoint.
///   3. Repeatedly find the adjacent symbol pair `(a, b)` whose merge
///      `"a b"` has the smallest rank in `merge_rank`; replace the pair
///      with the concatenation. Stop when no adjacent pair is in the
///      merge table.
///   4. Look each symbol up in `token_to_id`. If a symbol is not in the
///      vocab, fall back to byte-level lookup (single-byte symbols are
///      guaranteed to be in the vocab for any well-formed GGUF).
///
/// Complexity: O(n²) per piece in the worst case (n = encoded length).
/// Pieces are short enough (one pre-tokenized word) that this is the
/// approach ds4 itself uses; see ds4.c:14416-14444.
pub(crate) fn encode_piece(vocab: &Vocab, piece: &str, out: &mut Vec<u32>) {
    if piece.is_empty() {
        return;
    }

    // Step 1+2: byte-encode and split into per-codepoint symbols.
    let encoded = encode_bytes(piece.as_bytes());
    let mut sym: Vec<String> = encoded.chars().map(|c| c.to_string()).collect();

    // Step 3: BPE merge loop. Scratch buffer reused across iterations.
    let mut key = String::with_capacity(64);
    loop {
        let mut best_i: Option<usize> = None;
        let mut best_rank: u32 = u32::MAX;

        for i in 0..sym.len().saturating_sub(1) {
            key.clear();
            key.push_str(&sym[i]);
            key.push(' ');
            key.push_str(&sym[i + 1]);
            if let Some(&rank) = vocab.merge_rank.get(&key)
                && rank < best_rank
            {
                best_rank = rank;
                best_i = Some(i);
            }
        }

        let Some(i) = best_i else { break };

        // Concatenate sym[i] and sym[i+1] in place at i, then drop i+1.
        let right = sym.remove(i + 1);
        sym[i].push_str(&right);
    }

    // Step 4: map symbols → token ids.
    for s in &sym {
        if let Some(id) = vocab.token_to_id.get(s) {
            out.push(*id);
            continue;
        }
        // Fallback: emit per GPT-2 codepoint. Every single-codepoint
        // string in the GPT-2 alphabet (188 printable bytes that map to
        // themselves + 68 lifted control / whitespace bytes in
        // U+0100..=U+0143) must be a token in a well-formed DS V4 Flash
        // vocab; ds4 relies on the same invariant (ds4.c:14454-14463).
        // Iterating `chars()` works because `encode_bytes` produces
        // exactly one Unicode scalar per source byte.
        for ch in s.chars() {
            let single = ch.to_string();
            if let Some(id) = vocab.token_to_id.get(&single) {
                out.push(*id);
            }
            // Silently skip if absent; ds4 does the same.
        }
    }
}

/// Pre-tokenize + BPE-encode `text` end to end, appending token ids to
/// `out`. Used by both the public `encode` API and the chat assembler.
pub(crate) fn encode_text(vocab: &Vocab, text: &str, out: &mut Vec<u32>) {
    for piece in crate::joyai::pre_tokenize(text) {
        encode_piece(vocab, piece, out);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vocab::tests::fake_vocab;
    use std::collections::HashMap;

    /// Build a Vocab whose tokens are exactly the per-byte GPT-2
    /// codepoints (so any text round-trips via byte-fallback).
    fn byte_only_vocab() -> Vocab {
        let mut v = fake_vocab();
        // Add every single GPT-2 codepoint as its own token, after the
        // 8 existing entries.
        for b in 0..=255u8 {
            let s = crate::byte_encode::encode_bytes(&[b]);
            if !v.token_to_id.contains_key(&s) {
                let id = v.id_to_token.len() as u32;
                v.token_to_id.insert(s.clone(), id);
                v.id_to_token.push(s);
            }
        }
        v
    }

    #[test]
    fn empty_piece_emits_nothing() {
        let v = byte_only_vocab();
        let mut out = Vec::new();
        encode_piece(&v, "", &mut out);
        assert!(out.is_empty());
    }

    #[test]
    fn byte_fallback_emits_one_token_per_byte() {
        let v = byte_only_vocab();
        let mut out = Vec::new();
        encode_piece(&v, "ab", &mut out);
        // Two tokens (one per ASCII byte; 'a' and 'b' both pass through
        // unchanged in GPT-2 byte encoding).
        assert_eq!(out.len(), 2);
        // Verify they're distinct.
        assert_ne!(out[0], out[1]);
    }

    #[test]
    fn bpe_merge_combines_pair() {
        // Build a vocab where "a b" → rank 0, and the merged "ab" is a
        // token. The encoder should produce a single "ab" token.
        let mut v = fake_vocab();
        v.id_to_token.push("a".into());
        v.id_to_token.push("b".into());
        v.id_to_token.push("ab".into());
        v.token_to_id.insert("a".into(), 8);
        v.token_to_id.insert("b".into(), 9);
        v.token_to_id.insert("ab".into(), 10);
        v.merge_rank.insert("a b".into(), 0);

        let mut out = Vec::new();
        encode_piece(&v, "ab", &mut out);
        assert_eq!(out, vec![10]);
    }

    #[test]
    fn bpe_picks_lowest_rank_first() {
        // Tokens: a, b, c, ab, bc, abc. Merges:
        //   "a b" rank 1
        //   "b c" rank 0    ← should win first
        //   "a bc" rank 2   ← then this
        // Final: single "abc" token.
        let mut v = fake_vocab();
        for (t, id) in [
            ("a", 8u32),
            ("b", 9),
            ("c", 10),
            ("ab", 11),
            ("bc", 12),
            ("abc", 13),
        ] {
            v.id_to_token.push(t.into());
            v.token_to_id.insert(t.into(), id);
        }
        let mut mr = HashMap::new();
        mr.insert("a b".into(), 1u32);
        mr.insert("b c".into(), 0);
        mr.insert("a bc".into(), 2);
        v.merge_rank = mr;

        let mut out = Vec::new();
        encode_piece(&v, "abc", &mut out);
        assert_eq!(out, vec![13]);
    }

    #[test]
    fn bpe_stops_when_no_merges_remain() {
        // Only one merge defined ("a b"); "c" stays standalone.
        let mut v = fake_vocab();
        for (t, id) in [("a", 8u32), ("b", 9), ("c", 10), ("ab", 11)] {
            v.id_to_token.push(t.into());
            v.token_to_id.insert(t.into(), id);
        }
        v.merge_rank.insert("a b".into(), 0);

        let mut out = Vec::new();
        encode_piece(&v, "abc", &mut out);
        // Should be "ab" + "c" → [11, 10].
        assert_eq!(out, vec![11, 10]);
    }

    #[test]
    fn encode_text_iterates_over_pieces() {
        // Empty merge table → byte fallback for every byte.
        let v = byte_only_vocab();
        let mut out = Vec::new();
        encode_text(&v, "hi", &mut out);
        assert_eq!(out.len(), 2);
    }
}
