//! GPT-2 byte-level encoding (raw bytes ↔ printable Unicode codepoints).
//!
//! Standard GPT-2 / DeepSeek BPE preprocessing: raw bytes are first lifted
//! into a printable codepoint range so the BPE merge algorithm can work on
//! UTF-8 strings without losing byte identity. 188 of the 256 byte values
//! map to themselves (the "printable" range), and the remaining 68 control /
//! whitespace bytes are mapped to codepoints `256..324`.
//!
//! Ported by reference from `ds4.c:13567-13595` (MIT, The ds4.c authors).

/// Returns whether byte `b` belongs to the "printable" GPT-2 range that
/// maps to itself.
#[inline]
const fn byte_is_printable(b: u8) -> bool {
    matches!(b, 33..=126 | 161..=172 | 174..=255)
}

/// Map a raw byte to the GPT-2 codepoint that represents it.
///
/// Direct port of `gpt2_byte_to_codepoint` (`ds4.c:13567-13581`):
/// printable bytes map to themselves; non-printable bytes (33 control /
/// whitespace bytes and the gaps at 127, 160, 173) map to consecutive
/// codepoints starting at 256.
pub(crate) fn byte_to_codepoint(b: u8) -> u32 {
    if byte_is_printable(b) {
        return u32::from(b);
    }
    let mut n: u32 = 0;
    for x in 0..=255u32 {
        if byte_is_printable(x as u8) {
            continue;
        }
        if x == u32::from(b) {
            return 256 + n;
        }
        n += 1;
    }
    u32::from(b) // unreachable for valid byte input
}

/// Inverse of [`byte_to_codepoint`]: map a GPT-2 codepoint back to the
/// underlying raw byte. Returns `None` if the codepoint is outside the
/// 256-codepoint GPT-2 alphabet.
pub(crate) fn codepoint_to_byte(cp: u32) -> Option<u8> {
    if cp <= 0xff && byte_is_printable(cp as u8) {
        return Some(cp as u8);
    }
    if !(256..256 + 68).contains(&cp) {
        return None;
    }
    let want = cp - 256;
    let mut n: u32 = 0;
    for x in 0..=255u32 {
        if byte_is_printable(x as u8) {
            continue;
        }
        if n == want {
            return Some(x as u8);
        }
        n += 1;
    }
    None
}

/// Encode a UTF-8 byte slice as a GPT-2-style UTF-8 string where every raw
/// byte has been mapped through [`byte_to_codepoint`].
pub(crate) fn encode_bytes(input: &[u8]) -> String {
    let mut out = String::with_capacity(input.len() * 2);
    for &b in input {
        let cp = byte_to_codepoint(b);
        // Safe: byte_to_codepoint never returns a surrogate or value > U+10FFFF.
        if let Some(c) = char::from_u32(cp) {
            out.push(c);
        }
    }
    out
}

/// Decode a GPT-2-encoded UTF-8 string back into raw bytes. Returns `None`
/// if any character is outside the GPT-2 alphabet. Currently used only
/// by tests; production decoding goes through [`crate::decode`] one
/// token at a time.
#[cfg(test)]
pub(crate) fn decode_string(encoded: &str) -> Option<Vec<u8>> {
    let mut out = Vec::with_capacity(encoded.len());
    for c in encoded.chars() {
        let b = codepoint_to_byte(c as u32)?;
        out.push(b);
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn printable_bytes_map_to_themselves() {
        // ASCII letters, digits, common punctuation.
        for b in [b'A', b'Z', b'a', b'z', b'0', b'9', b'!', b'~'] {
            assert_eq!(byte_to_codepoint(b), u32::from(b));
            assert_eq!(codepoint_to_byte(u32::from(b)), Some(b));
        }
    }

    #[test]
    fn non_printable_bytes_map_above_256() {
        // 0x00 (NUL) is the first non-printable byte → codepoint 256.
        assert_eq!(byte_to_codepoint(0x00), 256);
        // 0x20 (space) is non-printable in GPT-2 → some codepoint >= 256.
        let cp_space = byte_to_codepoint(0x20);
        assert!(cp_space >= 256);
    }

    #[test]
    fn round_trip_all_256_bytes() {
        // Every byte should round-trip through codepoint → byte.
        for b in 0..=255u8 {
            let cp = byte_to_codepoint(b);
            let b2 = codepoint_to_byte(cp).expect("must round-trip");
            assert_eq!(b, b2, "byte {b}: cp {cp} -> byte {b2}");
        }
    }

    #[test]
    fn codepoint_to_byte_rejects_unknown() {
        assert!(codepoint_to_byte(0x110000).is_none());
        // Codepoint 256 + 68 (= 324) is one past the last GPT-2 codepoint.
        assert!(codepoint_to_byte(256 + 68).is_none());
    }

    #[test]
    fn encode_decode_round_trip_arbitrary_bytes() {
        let bytes: Vec<u8> = (0..=200u8).collect();
        let encoded = encode_bytes(&bytes);
        let decoded = decode_string(&encoded).expect("must decode");
        assert_eq!(bytes, decoded);
    }

    #[test]
    fn encode_ascii_letters_is_passthrough() {
        let encoded = encode_bytes(b"Hello");
        assert_eq!(encoded, "Hello");
    }

    #[test]
    fn encode_space_is_lifted() {
        // GPT-2 space → U+0120 (the "Ġ" prefix in the GPT-2 vocab).
        let encoded = encode_bytes(b" ");
        assert_ne!(encoded, " ");
        // First non-printable byte is NUL → 256; counting forward, space (0x20)
        // is among the first ~33 non-printable bytes.
        assert!(encoded.chars().next().unwrap() as u32 >= 256);
    }
}
