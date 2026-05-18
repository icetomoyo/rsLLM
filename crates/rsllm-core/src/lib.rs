//! # rsllm-core
//!
//! Core types and traits for [rsLLM](https://github.com/icetomoyo/rsLLM).
//!
//! This crate defines the shared `Engine` / `Session` abstractions, error
//! types, and `SamplingParams`. It has no GPU dependency and is intentionally
//! kept small so that every other crate in the workspace can depend on it.
//!
//! See [`docs/03-HLD.md`](https://github.com/icetomoyo/rsLLM/blob/main/docs/03-HLD.md)
//! for the architecture overview.

pub mod engine;
pub mod sampler;

pub use engine::{DecodeStep, Engine, EngineError, Session};
pub use sampler::{DEFAULT_MIN_P, DEFAULT_TEMPERATURE, DEFAULT_THINK_TEMPERATURE, Sampler, SamplingParams};

/// Returns the version string of the rsLLM library.
///
/// This is a pre-`v0.1.0` scaffolding build. The current version is
/// taken from `CARGO_PKG_VERSION` at compile time.
#[must_use]
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_is_non_empty() {
        assert!(!version().is_empty());
    }
}
