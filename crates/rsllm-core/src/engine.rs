//! Engine / Session abstractions.
//!
//! Two-level API:
//!
//! - [`Engine`] holds a loaded model and its associated weights. One
//!   per process / GPU-binding. Constructed from a GGUF path or
//!   equivalent.
//! - [`Session`] is the per-conversation runtime: holds the KV cache,
//!   the sampler, and the position cursor. Multiple sessions can
//!   share one engine.
//!
//! The traits are intentionally backend-agnostic — `rsllm-models`
//! provides the v0.1.0 implementation against DS V4 Flash + CPU
//! (and Metal/CUDA in later releases).
//!
//! F008.C.3 of v0.1.0 lands the concrete `DsV4Flash` engine; this
//! module defines only the surface so the CLI and integration tests
//! can be wired against the trait.

use crate::SamplingParams;

/// Errors raised by the engine + session layer. Wraps lower-level
/// crate errors so the CLI sees one error type. Concrete engine
/// implementations may down-convert their own crate's errors into
/// this enum.
#[derive(Debug, thiserror::Error)]
pub enum EngineError {
    /// I/O failure (file open, mmap, etc.).
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    /// A required tensor or metadata key was missing.
    #[error("missing: {0}")]
    Missing(String),
    /// A weight / metadata shape did not match expectations.
    #[error("shape mismatch: {what} expected {expected}, got {actual}")]
    ShapeMismatch {
        what: String,
        expected: String,
        actual: String,
    },
    /// A KV cache operation failed.
    #[error("kv-cache: {0}")]
    KvCache(String),
    /// The session's context window is full and the caller must
    /// `reset()` before continuing.
    #[error("context full: {filled} of {capacity} tokens")]
    ContextFull { filled: usize, capacity: usize },
    /// A feature requested by the caller has not landed yet.
    #[error("not yet implemented: {0}")]
    NotImplemented(String),
}

/// Loaded model. One per architecture / weight set.
///
/// Implementations own the GGUF mmap and any pre-resolved weight
/// views. They produce [`Session`]s on demand; sessions share the
/// underlying weights via reference.
pub trait Engine {
    /// Concrete session type bound to this engine.
    type Session<'engine>: Session
    where
        Self: 'engine;

    /// Architecture name (`"deepseek-v4-flash"`, etc.) for reporting.
    fn architecture(&self) -> &'static str;

    /// Vocabulary size — required by samplers and dump writers.
    fn vocab_size(&self) -> usize;

    /// Maximum context length supported by the underlying KV cache
    /// for this engine instance. Sessions inherit this cap.
    fn max_ctx_size(&self) -> usize;

    /// Create a new session sized for `ctx_size <= max_ctx_size()`
    /// tokens, with the supplied sampler parameters.
    ///
    /// # Errors
    /// [`EngineError::ShapeMismatch`] if `ctx_size > max_ctx_size()`.
    fn start_session(
        &self,
        ctx_size: usize,
        params: SamplingParams,
    ) -> Result<Self::Session<'_>, EngineError>;
}

/// Per-conversation runtime: owns the KV cache, the sampler, and
/// the position cursor. Caller drives prefill + decode loops on it.
pub trait Session {
    /// Total tokens consumed so far (= sum of `prefill` lengths plus
    /// emitted decode tokens).
    fn position(&self) -> usize;

    /// Maximum tokens this session can hold before
    /// [`EngineError::ContextFull`] starts firing.
    fn capacity(&self) -> usize;

    /// Append a batch of input tokens to the session, running the
    /// forward pass and updating the KV cache. Returns the
    /// next-token logits over the vocabulary for the **last**
    /// token of the batch (the prefill API mirrors HF's `forward`).
    ///
    /// # Errors
    /// - [`EngineError::ContextFull`] if `tokens.len()` exceeds
    ///   the remaining capacity.
    fn prefill(&mut self, tokens: &[u32]) -> Result<Vec<f32>, EngineError>;

    /// Decode one token: feed the last sampled id, run one forward
    /// step, sample the next id. Returns `(token_id, logits)` so
    /// the caller can drive a `--dump-logprobs`-style trace.
    ///
    /// # Errors
    /// [`EngineError::ContextFull`] when the position cursor would
    /// exceed [`Self::capacity`].
    fn decode_one(&mut self, last_token: u32) -> Result<DecodeStep, EngineError>;

    /// Discard all KV state and reset the position cursor. The
    /// sampler's RNG state is preserved (matches ds4 semantics —
    /// `/clear` does not reseed).
    fn reset(&mut self);
}

/// One decode step's output: the sampled token plus the full
/// post-filter probability distribution (already normalized) so
/// `--dump-logprobs` can record a top-K trace without re-running
/// the model.
#[derive(Debug, Clone)]
pub struct DecodeStep {
    /// Sampled token id.
    pub token_id: u32,
    /// Per-vocab probability after the sampling filter chain.
    /// Length = engine.vocab_size().
    pub probs: Vec<f32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    // A trivial in-memory Engine + Session for the trait surface
    // test. Real implementation lands with F008.C.3.b (GGUF loader).
    #[derive(Debug)]
    struct DummyEngine {
        vocab: usize,
        max_ctx: usize,
    }
    #[derive(Debug)]
    struct DummySession<'e> {
        engine: &'e DummyEngine,
        pos: usize,
        cap: usize,
    }

    impl Engine for DummyEngine {
        type Session<'engine> = DummySession<'engine>;

        fn architecture(&self) -> &'static str {
            "dummy"
        }
        fn vocab_size(&self) -> usize {
            self.vocab
        }
        fn max_ctx_size(&self) -> usize {
            self.max_ctx
        }
        fn start_session(
            &self,
            ctx_size: usize,
            _params: SamplingParams,
        ) -> Result<Self::Session<'_>, EngineError> {
            if ctx_size > self.max_ctx {
                return Err(EngineError::ShapeMismatch {
                    what: "session.ctx_size".into(),
                    expected: format!("<= {}", self.max_ctx),
                    actual: format!("{ctx_size}"),
                });
            }
            Ok(DummySession {
                engine: self,
                pos: 0,
                cap: ctx_size,
            })
        }
    }

    impl Session for DummySession<'_> {
        fn position(&self) -> usize {
            self.pos
        }
        fn capacity(&self) -> usize {
            self.cap
        }
        fn prefill(&mut self, tokens: &[u32]) -> Result<Vec<f32>, EngineError> {
            if self.pos + tokens.len() > self.cap {
                return Err(EngineError::ContextFull {
                    filled: self.pos,
                    capacity: self.cap,
                });
            }
            self.pos += tokens.len();
            // Stub logits — uniform over vocab.
            Ok(vec![0.0_f32; self.engine.vocab])
        }
        fn decode_one(&mut self, _last: u32) -> Result<DecodeStep, EngineError> {
            if self.pos >= self.cap {
                return Err(EngineError::ContextFull {
                    filled: self.pos,
                    capacity: self.cap,
                });
            }
            self.pos += 1;
            Ok(DecodeStep {
                token_id: 0,
                probs: vec![1.0 / self.engine.vocab as f32; self.engine.vocab],
            })
        }
        fn reset(&mut self) {
            self.pos = 0;
        }
    }

    #[test]
    fn engine_surface_compiles() {
        let e = DummyEngine {
            vocab: 32,
            max_ctx: 64,
        };
        assert_eq!(e.architecture(), "dummy");
        assert_eq!(e.vocab_size(), 32);
        assert_eq!(e.max_ctx_size(), 64);
    }

    #[test]
    fn start_session_respects_max_ctx() {
        let e = DummyEngine {
            vocab: 32,
            max_ctx: 64,
        };
        let err = e.start_session(128, SamplingParams::default()).unwrap_err();
        assert!(matches!(err, EngineError::ShapeMismatch { .. }));
    }

    #[test]
    fn session_prefill_and_decode_advance_position() {
        let e = DummyEngine {
            vocab: 16,
            max_ctx: 8,
        };
        let mut s = e.start_session(8, SamplingParams::default()).unwrap();
        assert_eq!(s.position(), 0);
        let logits = s.prefill(&[1, 2, 3]).unwrap();
        assert_eq!(logits.len(), 16);
        assert_eq!(s.position(), 3);
        let step = s.decode_one(3).unwrap();
        assert_eq!(step.probs.len(), 16);
        assert_eq!(s.position(), 4);
    }

    #[test]
    fn context_full_fires_when_capacity_exceeded() {
        let e = DummyEngine {
            vocab: 4,
            max_ctx: 4,
        };
        let mut s = e.start_session(4, SamplingParams::default()).unwrap();
        s.prefill(&[1, 2, 3, 4]).unwrap();
        let err = s.decode_one(4).unwrap_err();
        assert!(matches!(err, EngineError::ContextFull { .. }));
    }

    #[test]
    fn reset_returns_position_to_zero() {
        let e = DummyEngine {
            vocab: 4,
            max_ctx: 4,
        };
        let mut s = e.start_session(4, SamplingParams::default()).unwrap();
        s.prefill(&[1, 2]).unwrap();
        s.reset();
        assert_eq!(s.position(), 0);
    }
}
