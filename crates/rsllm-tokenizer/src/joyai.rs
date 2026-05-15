//! JoyAI BPE pre-tokenizer (DeepSeek V4 Flash `tokenizer.ggml.pre =
//! "joyai-llm"`).
//!
//! Ported by reference from `ds4.c:13703-13879` (MIT, The ds4.c authors).
//!
//! Splits an input text into byte ranges per the JoyAI rule set:
//!
//!  1. `\p{N}{1,3}`                  — runs of 1-3 ASCII digits
//!  2. CJK / Hiragana / Katakana run — one or more CJK codepoints
//!  3. `[P/S][A-Za-z]+`              — punct/symbol + ASCII-alpha run
//!  4. letter-like run               — ASCII alpha + non-ASCII non-control
//!  5. one-char prefix + letter-like — generic non-letter + letter run
//!  6. ` ` + punct + trailing `\n`   — single space followed by punct
//!  7. punct + trailing `\n`         — punct run keeping trailing newlines
//!  8. whitespace run                — handles leading spaces before words
//!
//! These rules consume **byte ranges** of the original input. They are
//! returned as `&str` borrows; ds4 byte-encodes them before feeding the
//! BPE inner loop, and we do the same in [`crate::bpe`].

#[inline]
fn ascii_alpha(c: u8) -> bool {
    c.is_ascii_alphabetic()
}

#[inline]
fn ascii_digit(c: u8) -> bool {
    c.is_ascii_digit()
}

#[inline]
fn ascii_space(c: u8) -> bool {
    matches!(c, b' ' | b'\t' | b'\n' | b'\r' | 0x0b | 0x0c)
}

#[inline]
fn ascii_newline(c: u8) -> bool {
    c == b'\n' || c == b'\r'
}

#[inline]
fn joyai_ascii_punct_symbol(c: u8) -> bool {
    matches!(c, b'!'..=b'/' | b':'..=b'@' | b'['..=b'`' | b'{'..=b'~')
}

#[inline]
fn utf8_is_cjk_hira_kata(cp: u32) -> bool {
    (0x4e00..=0x9fa5).contains(&cp)
        || (0x3040..=0x309f).contains(&cp)
        || (0x30a0..=0x30ff).contains(&cp)
}

/// Length in bytes of the UTF-8 sequence starting with `c`. Returns 1 for
/// continuation / invalid bytes, matching ds4's defensive behavior.
fn utf8_len_from_first_byte(c: u8) -> usize {
    if c < 0x80 {
        1
    } else if c & 0xe0 == 0xc0 {
        2
    } else if c & 0xf0 == 0xe0 {
        3
    } else if c & 0xf8 == 0xf0 {
        4
    } else {
        1
    }
}

/// Move `pos` forward by the UTF-8 length of the byte at `pos`, clamped so
/// we never step past `len`.
fn next_utf8_char(bytes: &[u8], pos: usize) -> usize {
    let n = utf8_len_from_first_byte(bytes[pos]);
    let step = if pos + n > bytes.len() { 1 } else { n };
    pos + step
}

/// Decode the UTF-8 codepoint at `pos` and return `(codepoint, next_pos)`.
/// Mirrors ds4's `utf8_peek_one`, including its tolerance for short
/// trailing sequences (treated as a 1-byte step).
fn utf8_peek_one(bytes: &[u8], pos: usize) -> (u32, usize) {
    let c0 = bytes[pos];
    let n_decl = utf8_len_from_first_byte(c0);
    let n = if pos + n_decl > bytes.len() {
        1
    } else {
        n_decl
    };

    let cp = match n {
        1 => u32::from(c0),
        2 => (u32::from(c0 & 0x1f) << 6) | u32::from(bytes[pos + 1] & 0x3f),
        3 => {
            (u32::from(c0 & 0x0f) << 12)
                | (u32::from(bytes[pos + 1] & 0x3f) << 6)
                | u32::from(bytes[pos + 2] & 0x3f)
        }
        _ => {
            (u32::from(c0 & 0x07) << 18)
                | (u32::from(bytes[pos + 1] & 0x3f) << 12)
                | (u32::from(bytes[pos + 2] & 0x3f) << 6)
                | u32::from(bytes[pos + 3] & 0x3f)
        }
    };
    (cp, pos + n)
}

/// Is the codepoint at `pos` "letter-like" in JoyAI's collapsed alphabet?
///
/// ds4's rule (ds4.c:13761-13775): ASCII letters always count; any
/// non-ASCII byte that starts a UTF-8 sequence is treated as a letter,
/// because CJK / kana are already separated by rule 2 before this is
/// consulted.
fn joyai_letter_like_at(bytes: &[u8], pos: usize) -> bool {
    let c = bytes[pos];
    if c < 128 {
        return ascii_alpha(c);
    }
    true
}

/// Hard cap on the byte length of any single pre-tokenized piece.
///
/// The downstream BPE merge loop is O(n²) in the number of symbols per
/// piece, so an unbounded letter-like run (rule 4/5) on adversarial
/// input would otherwise be a DoS vector. 4 KB is well above any real
/// JoyAI piece — even a paragraph-long word in a language without
/// spaces (Thai, CJK fragments that escape rule 2) fits comfortably.
///
/// When the cap fires the piece is closed and the next iteration of
/// `pre_tokenize` starts a fresh piece at the next codepoint boundary,
/// so token output is still well-formed (just split at a non-natural
/// boundary, which mirrors ds4's behavior on these out-of-domain inputs
/// only in the upper bound it imposes via available memory).
const MAX_PIECE_BYTES: usize = 4096;

fn joyai_consume_letters(bytes: &[u8], start: usize, mut pos: usize) -> usize {
    while pos < bytes.len() && joyai_letter_like_at(bytes, pos) {
        let next = next_utf8_char(bytes, pos);
        // Predictive cap: only advance if the *next* boundary would still
        // sit within MAX_PIECE_BYTES of `start`. Guarantees piece length
        // ≤ MAX_PIECE_BYTES exactly.
        if next - start > MAX_PIECE_BYTES {
            break;
        }
        pos = next;
    }
    pos
}

fn joyai_cjk_at(bytes: &[u8], pos: usize) -> bool {
    if bytes[pos] < 128 {
        return false;
    }
    let (cp, _) = utf8_peek_one(bytes, pos);
    utf8_is_cjk_hira_kata(cp)
}

/// Split `text` into pre-tokenizer pieces per the JoyAI ruleset.
///
/// Each piece is a `&str` borrow into `text`. The pieces tile `text`
/// exactly: their byte ranges are contiguous and cover the whole input.
///
/// Pre-tokenization is deterministic and does **not** allocate beyond
/// the returned `Vec`.
pub(crate) fn pre_tokenize(text: &str) -> Vec<&str> {
    let bytes = text.as_bytes();
    let len = bytes.len();
    let mut out: Vec<&str> = Vec::new();
    let mut pos = 0;

    while pos < len {
        let start = pos;
        let c = bytes[pos];

        if ascii_digit(c) {
            // Rule 1: up to 3 ASCII digits.
            let mut ndigits = 0;
            while pos < len && ascii_digit(bytes[pos]) && ndigits < 3 {
                pos += 1;
                ndigits += 1;
            }
        } else if joyai_cjk_at(bytes, pos) {
            // Rule 2: CJK / Hira / Kata run.
            //
            // Predictive cap: only advance if the *next* boundary would
            // still sit within MAX_PIECE_BYTES of `start`. Guarantees
            // piece length ≤ MAX_PIECE_BYTES.
            loop {
                let next = next_utf8_char(bytes, pos);
                if next - start > MAX_PIECE_BYTES {
                    break;
                }
                pos = next;
                if pos >= len || !joyai_cjk_at(bytes, pos) {
                    break;
                }
            }
        } else if joyai_ascii_punct_symbol(c) && pos + 1 < len && ascii_alpha(bytes[pos + 1]) {
            // Rule 3: ASCII punct followed by ASCII alpha run.
            pos += 1;
            while pos < len && (pos - start) < MAX_PIECE_BYTES && ascii_alpha(bytes[pos]) {
                pos += 1;
            }
        } else if joyai_letter_like_at(bytes, pos) {
            // Rule 4: letter-like run (ASCII alpha or non-ASCII).
            pos = joyai_consume_letters(bytes, start, pos);
        } else if !ascii_newline(c)
            && !joyai_ascii_punct_symbol(c)
            && pos + 1 < len
            && joyai_letter_like_at(bytes, pos + 1)
        {
            // Rule 5: one-byte prefix + letter-like run.
            pos += 1;
            pos = joyai_consume_letters(bytes, start, pos);
        } else if c == b' ' && pos + 1 < len && joyai_ascii_punct_symbol(bytes[pos + 1]) {
            // Rule 6: single space + punct run + trailing newlines.
            pos += 1;
            while pos < len
                && (pos - start) < MAX_PIECE_BYTES
                && joyai_ascii_punct_symbol(bytes[pos])
            {
                pos += 1;
            }
            while pos < len && (pos - start) < MAX_PIECE_BYTES && ascii_newline(bytes[pos]) {
                pos += 1;
            }
        } else if joyai_ascii_punct_symbol(c) {
            // Rule 7: punct run + trailing newlines (keep newlines!).
            while pos < len
                && (pos - start) < MAX_PIECE_BYTES
                && joyai_ascii_punct_symbol(bytes[pos])
            {
                pos += 1;
            }
            while pos < len && (pos - start) < MAX_PIECE_BYTES && ascii_newline(bytes[pos]) {
                pos += 1;
            }
        } else if ascii_space(c) {
            // Rule 8: whitespace run.
            //
            //   * If a newline exists in the run, cut after the LAST one.
            //   * Else, if more than one space and the next non-space is
            //     letter-like or punct, leave the trailing space to merge
            //     with the next word ("    int" → "   ", " int").
            //   * Else consume the whole whitespace run.
            let mut p = pos;
            let mut last_newline_end: Option<usize> = None;
            while p < len && (p - start) < MAX_PIECE_BYTES && ascii_space(bytes[p]) {
                let sc = bytes[p];
                p += 1;
                if ascii_newline(sc) {
                    last_newline_end = Some(p);
                }
            }
            if let Some(end) = last_newline_end {
                pos = end;
            } else if p < len
                && p > pos + 1
                && (joyai_letter_like_at(bytes, p) || joyai_ascii_punct_symbol(bytes[p]))
            {
                pos = p - 1;
            } else {
                pos = p;
            }
        } else {
            // Fallback: step one UTF-8 codepoint.
            pos = next_utf8_char(bytes, pos);
        }

        // Safety: if no rule advanced, step one codepoint to avoid a loop.
        if pos == start {
            pos = next_utf8_char(bytes, pos);
        }
        // Byte ranges produced by the rules above always sit on UTF-8
        // boundaries (rules step in codepoint units or over ASCII bytes),
        // so `&text[start..pos]` is valid UTF-8.
        out.push(&text[start..pos]);
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pieces(s: &str) -> Vec<&str> {
        pre_tokenize(s)
    }

    #[test]
    fn empty_input_yields_no_pieces() {
        assert!(pieces("").is_empty());
    }

    #[test]
    fn pieces_tile_input_exactly() {
        let inputs = [
            "Hello, world!\n",
            "    int x = 42;\n",
            "你好，世界！",
            "1234567890",
            "fn main() {\n    println!(\"hi\");\n}",
            "Café résumé naïve",
        ];
        for input in inputs {
            let joined: String = pieces(input).concat();
            assert_eq!(joined, input, "tiling broken for {input:?}");
        }
    }

    #[test]
    fn digits_split_in_groups_of_three() {
        // Rule 1: \p{N}{1,3}
        assert_eq!(pieces("1234567890"), vec!["123", "456", "789", "0"]);
        assert_eq!(pieces("42"), vec!["42"]);
        assert_eq!(pieces("9"), vec!["9"]);
    }

    #[test]
    fn cjk_run_stays_together() {
        // Rule 2: CJK forms one piece per consecutive run.
        let p = pieces("你好");
        assert_eq!(p, vec!["你好"]);
    }

    #[test]
    fn ascii_letters_run_together() {
        // Rule 4: "Hello" is one piece.
        assert_eq!(pieces("Hello"), vec!["Hello"]);
    }

    #[test]
    fn punct_keeps_trailing_newline() {
        // Rule 7: ">;\n" stays as one piece — this is critical for code
        // prompts (verbatim quote from ds4.c:13806).
        let p = pieces(">;\n");
        assert_eq!(p, vec![">;\n"]);
    }

    #[test]
    fn space_punct_joins_with_trailing_newline() {
        // Rule 6: " >;\n" — leading space joins the punct+newline run.
        let p = pieces(" >;\n");
        assert_eq!(p, vec![" >;\n"]);
    }

    #[test]
    fn leading_space_joins_following_word() {
        // ds4 comment ds4.c:13863-13867: "    int" → "   " + " int".
        let p = pieces("    int");
        assert_eq!(p, vec!["   ", " int"]);
    }

    #[test]
    fn single_space_before_word_joins_it() {
        // " int" — single leading space + word. Rule 8's "p > pos + 1"
        // branch requires more than one space, so this falls into rule 5
        // (non-letter prefix + letter-like).
        let p = pieces(" int");
        assert_eq!(p, vec![" int"]);
    }

    #[test]
    fn whitespace_run_cuts_after_last_newline() {
        // " \n  " — newline anchors the cut after the newline. The two
        // trailing spaces become rule-8 whitespace pieces (or join the
        // next word — here, end of input → just emitted).
        let p = pieces(" \n  ");
        assert_eq!(p, vec![" \n", "  "]);
    }

    #[test]
    fn non_alpha_punct_prefix_then_letters() {
        // Rule 3: "(foo" → punct + alpha run as ONE piece.
        let p = pieces("(foo");
        assert_eq!(p, vec!["(foo"]);
    }

    #[test]
    fn newline_alone_emits_as_whitespace() {
        let p = pieces("\n");
        assert_eq!(p, vec!["\n"]);
    }

    #[test]
    fn cjk_then_ascii_punct() {
        // "你好。" — CJK punct U+3002 is non-ASCII, so it's letter-like
        // under rule 4 (or in this case, the second piece after CJK).
        let p = pieces("你好。");
        let joined: String = p.concat();
        assert_eq!(joined, "你好。");
    }

    #[test]
    fn realistic_code_snippet() {
        let p = pieces("if x > 0 {\n");
        let joined: String = p.concat();
        assert_eq!(joined, "if x > 0 {\n");
        // Verify the punct+newline rule kept "{\n" together.
        assert!(p.iter().any(|s| s.ends_with("{\n")));
    }

    #[test]
    fn emoji_is_handled_as_letter_like_run() {
        // 🎉 (U+1F389) is a 4-byte sequence with lead 0xF0. It's not
        // ASCII, not CJK, but `joyai_letter_like_at` treats any non-ASCII
        // lead byte as letter-like, so it should fall under rule 4 and
        // emit as one piece. Validates that rules don't land pos on a
        // UTF-8 continuation byte (which would panic on the &str slice).
        let p = pieces("🎉");
        assert_eq!(p, vec!["🎉"]);
    }

    #[test]
    fn space_then_emoji_does_not_panic() {
        // Leading space → rule 5 (one-byte prefix + letter-like). The
        // emoji lead byte is letter-like, so rule 5 fires and consumes
        // the full 4-byte codepoint via `joyai_consume_letters`. The
        // resulting slice must be valid UTF-8.
        let p = pieces(" 🎉");
        assert_eq!(p, vec![" 🎉"]);
    }

    #[test]
    fn accented_latin_run_stays_letterlike() {
        // Italian accents (ds4.c:13770-13772 explicitly motivates this).
        let p = pieces("Café");
        assert_eq!(p, vec!["Café"]);
    }

    #[test]
    fn very_long_letter_like_run_is_capped() {
        // 10 KB of letter-like bytes (rule 4) must not produce a single
        // 10 KB piece — the BPE merge loop is O(n²) and would otherwise
        // be a DoS surface. Each piece must be ≤ MAX_PIECE_BYTES.
        let long: String = "é".repeat(8_000);
        let p = pieces(&long);
        assert!(
            p.len() >= 2,
            "expected piece split, got {} piece(s)",
            p.len()
        );
        for piece in &p {
            assert!(
                piece.len() <= MAX_PIECE_BYTES,
                "piece exceeds cap: {} bytes",
                piece.len()
            );
        }
        // Tile invariant still holds.
        let joined: String = p.concat();
        assert_eq!(joined, long);
    }

    #[test]
    fn very_long_cjk_run_is_capped() {
        let long: String = "汉".repeat(2_000); // 6 KB of CJK
        let p = pieces(&long);
        assert!(p.len() >= 2);
        for piece in &p {
            assert!(piece.len() <= MAX_PIECE_BYTES);
        }
    }
}
