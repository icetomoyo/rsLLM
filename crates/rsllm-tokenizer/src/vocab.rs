//! Vocabulary loaded from a GGUF file's `tokenizer.ggml.*` metadata.
//!
//! Holds the forward lookup `token_to_id`, the reverse lookup
//! `id_to_token`, and the BPE merge ranking table — plus the IDs of the
//! special tokens that DeepSeek V4 Flash's chat protocol requires.
//!
//! Loading logic ported by reference from `ds4.c:13891-13931` (MIT, The
//! ds4.c authors).

use std::collections::HashMap;

use rsllm_gguf::{Array, Metadata, Value};

use crate::error::Error;

/// Required pre-tokenizer name for v0.1.0.
pub const REQUIRED_PRE_TOKENIZER: &str = "joyai-llm";

/// Special-token text patterns recognized by the DeepSeek V4 Flash chat
/// protocol. Order matters only for the [`crate::special`] longest-match
/// scanner; all 7 must be present in the vocab or [`Vocab::from_metadata`]
/// returns an error.
pub(crate) const SPECIAL_TEXTS: &[&str] = &[
    "<\u{ff5c}begin\u{2581}of\u{2581}sentence\u{ff5c}>",
    "<\u{ff5c}end\u{2581}of\u{2581}sentence\u{ff5c}>",
    "<\u{ff5c}User\u{ff5c}>",
    "<\u{ff5c}Assistant\u{ff5c}>",
    "<think>",
    "</think>",
    "\u{ff5c}DSML\u{ff5c}",
];

/// Index positions of each special token in [`SPECIAL_TEXTS`].
pub(crate) const BOS_TEXT_IDX: usize = 0;
pub(crate) const EOS_TEXT_IDX: usize = 1;
pub(crate) const USER_TEXT_IDX: usize = 2;
pub(crate) const ASSISTANT_TEXT_IDX: usize = 3;
pub(crate) const THINK_START_TEXT_IDX: usize = 4;
pub(crate) const THINK_END_TEXT_IDX: usize = 5;
pub(crate) const DSML_TEXT_IDX: usize = 6;

/// Parsed DeepSeek V4 Flash vocabulary.
#[derive(Debug, Clone)]
pub struct Vocab {
    /// Token id → token text. Indexed by `u32` token id.
    pub(crate) id_to_token: Vec<String>,

    /// Reverse map: token text → token id.
    pub(crate) token_to_id: HashMap<String, u32>,

    /// BPE merge rank: `"a b"` → rank (lower = earlier merge). Direct
    /// mirror of ds4's `merge_rank` table.
    pub(crate) merge_rank: HashMap<String, u32>,

    /// Cached IDs for the 7 special tokens in [`SPECIAL_TEXTS`].
    pub(crate) special_ids: [u32; 7],
}

impl Vocab {
    /// Load a DeepSeek V4 Flash vocabulary from GGUF metadata.
    ///
    /// Validates:
    /// - `tokenizer.ggml.model == "gpt2"` (DS V4 uses byte-level BPE)
    /// - `tokenizer.ggml.pre == "joyai-llm"`
    /// - `tokenizer.ggml.tokens` is a String array (the vocab itself)
    /// - `tokenizer.ggml.merges` is a String array (each entry `"a b"`)
    /// - all 7 [`SPECIAL_TEXTS`] tokens are present in the vocab
    pub fn from_metadata(meta: &Metadata) -> Result<Self, Error> {
        // tokenizer.ggml.pre — must be "joyai-llm" for v0.1.0.
        let pre = meta
            .get_str("tokenizer.ggml.pre")
            .ok_or(Error::MissingKey("tokenizer.ggml.pre"))?;
        if pre != REQUIRED_PRE_TOKENIZER {
            return Err(Error::UnsupportedPreTokenizer(pre.to_owned()));
        }

        // tokenizer.ggml.tokens — required String array.
        let tokens_arr = match meta.get("tokenizer.ggml.tokens") {
            Some(Value::Array(Array::String(s))) => s,
            Some(_) => {
                return Err(Error::WrongMetadataType {
                    key: "tokenizer.ggml.tokens",
                    reason: "expected Array<String>",
                });
            }
            None => return Err(Error::MissingKey("tokenizer.ggml.tokens")),
        };

        let n_vocab = tokens_arr.len();
        let id_to_token = tokens_arr.clone();
        let mut token_to_id = HashMap::with_capacity(n_vocab);
        for (i, t) in id_to_token.iter().enumerate() {
            // Last-write-wins on duplicates; spec doesn't forbid them but
            // they shouldn't occur in real DS V4 GGUFs.
            token_to_id.insert(t.clone(), i as u32);
        }

        // tokenizer.ggml.merges — required String array, each entry `lhs rhs`.
        let merges_arr = match meta.get("tokenizer.ggml.merges") {
            Some(Value::Array(Array::String(s))) => s,
            Some(_) => {
                return Err(Error::WrongMetadataType {
                    key: "tokenizer.ggml.merges",
                    reason: "expected Array<String>",
                });
            }
            None => return Err(Error::MissingKey("tokenizer.ggml.merges")),
        };

        let mut merge_rank = HashMap::with_capacity(merges_arr.len());
        for (i, m) in merges_arr.iter().enumerate() {
            // Validate format: must contain exactly one space.
            if m.split(' ').count() != 2 {
                return Err(Error::MalformedMerge(m.clone()));
            }
            merge_rank.insert(m.clone(), i as u32);
        }

        // Resolve 7 special-token IDs. ds4 fail-fasts on any missing
        // special token, and we do the same — DS V4 Flash chat is broken
        // without them.
        let mut special_ids = [0u32; 7];
        for (idx, text) in SPECIAL_TEXTS.iter().enumerate() {
            let id = token_to_id
                .get(*text)
                .copied()
                .ok_or(Error::MissingSpecialToken(SPECIAL_TEXTS[idx]))?;
            special_ids[idx] = id;
        }

        Ok(Self {
            id_to_token,
            token_to_id,
            merge_rank,
            special_ids,
        })
    }

    /// Total vocabulary size.
    pub fn len(&self) -> usize {
        self.id_to_token.len()
    }

    /// `true` if the vocab is empty (defensive — never happens for a
    /// valid DS V4 GGUF, but required to satisfy the
    /// `len_without_is_empty` clippy lint).
    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.id_to_token.is_empty()
    }

    /// Look up a token id by exact text. Returns `None` if not in vocab.
    pub fn id_of(&self, text: &str) -> Option<u32> {
        self.token_to_id.get(text).copied()
    }

    /// Look up a token text by id. Returns `None` if id is out of range.
    pub fn token_of(&self, id: u32) -> Option<&str> {
        self.id_to_token.get(id as usize).map(String::as_str)
    }

    /// BOS token id.
    pub fn bos_id(&self) -> u32 {
        self.special_ids[BOS_TEXT_IDX]
    }

    /// EOS token id.
    pub fn eos_id(&self) -> u32 {
        self.special_ids[EOS_TEXT_IDX]
    }

    /// `<｜User｜>` token id.
    pub fn user_id(&self) -> u32 {
        self.special_ids[USER_TEXT_IDX]
    }

    /// `<｜Assistant｜>` token id.
    pub fn assistant_id(&self) -> u32 {
        self.special_ids[ASSISTANT_TEXT_IDX]
    }

    /// `<think>` token id.
    pub fn think_start_id(&self) -> u32 {
        self.special_ids[THINK_START_TEXT_IDX]
    }

    /// `</think>` token id.
    pub fn think_end_id(&self) -> u32 {
        self.special_ids[THINK_END_TEXT_IDX]
    }

    /// `｜DSML｜` token id (used by tool-call protocol).
    pub fn dsml_id(&self) -> u32 {
        self.special_ids[DSML_TEXT_IDX]
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// Build a minimal Vocab in-memory (bypassing GGUF). Used by all
    /// downstream module tests.
    pub(crate) fn fake_vocab() -> Vocab {
        // 8 tokens: 7 specials + a placeholder for an ASCII letter so we
        // have at least one regular token.
        let mut id_to_token = SPECIAL_TEXTS
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();
        id_to_token.push("a".to_string());
        let mut token_to_id = HashMap::new();
        for (i, t) in id_to_token.iter().enumerate() {
            token_to_id.insert(t.clone(), i as u32);
        }
        let mut special_ids = [0u32; 7];
        for (i, t) in SPECIAL_TEXTS.iter().enumerate() {
            special_ids[i] = *token_to_id.get(*t).unwrap();
        }
        Vocab {
            id_to_token,
            token_to_id,
            merge_rank: HashMap::new(),
            special_ids,
        }
    }

    #[test]
    fn fake_vocab_has_expected_special_ids() {
        let v = fake_vocab();
        assert_eq!(v.bos_id(), 0);
        assert_eq!(v.eos_id(), 1);
        assert_eq!(v.user_id(), 2);
        assert_eq!(v.assistant_id(), 3);
        assert_eq!(v.think_start_id(), 4);
        assert_eq!(v.think_end_id(), 5);
        assert_eq!(v.dsml_id(), 6);
        assert_eq!(v.len(), 8);
    }

    #[test]
    fn special_texts_match_ds4() {
        // Spot-check that we wrote the special token strings using the
        // correct Unicode (｜ = U+FF5C "fullwidth vertical line", ▁ = U+2581
        // "lower one eighth block").
        assert_eq!(
            SPECIAL_TEXTS[BOS_TEXT_IDX],
            "<\u{ff5c}begin\u{2581}of\u{2581}sentence\u{ff5c}>"
        );
        // Print it just to sanity-check what the user sees:
        let bos = SPECIAL_TEXTS[BOS_TEXT_IDX];
        assert!(bos.contains('｜'));
        assert!(bos.contains('▁'));
        assert_eq!(SPECIAL_TEXTS[THINK_START_TEXT_IDX], "<think>");
        assert_eq!(SPECIAL_TEXTS[THINK_END_TEXT_IDX], "</think>");
    }
}
