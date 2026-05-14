//! # rsllm-cal
//!
//! Compute Abstraction Layer for rsLLM.
//!
//! Defines the [`Backend`], `Buffer`, and `Stream` traits that every compute
//! backend (CPU, CUDA, Metal, wgpu, ...) implements. Model graph code in
//! `rsllm-models` is written against these traits and is backend-agnostic.
//!
//! See [`docs/03-HLD.md`](https://github.com/icetomoyo/rsLLM/blob/main/docs/03-HLD.md)
//! §3.2 for the trait shape.

/// Identifier for a compute backend, returned by [`Backend::capability`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendKind {
    /// CPU backend (AVX2 / AVX-512 / NEON / SVE depending on target).
    Cpu,
    /// NVIDIA CUDA backend.
    Cuda,
    /// Apple Metal backend.
    Metal,
    /// Cross-platform wgpu backend.
    Wgpu,
}

/// Marker trait for compute backends. Will be expanded with associated
/// types (`Buffer`, `Stream`, `Event`) and methods (`alloc`, `dispatch`,
/// `sync`) when FEATURE_004 lands.
pub trait Backend {
    /// Identifier of this backend variant.
    fn kind(&self) -> BackendKind;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_kinds_distinct() {
        assert_ne!(BackendKind::Cpu, BackendKind::Cuda);
        assert_ne!(BackendKind::Metal, BackendKind::Wgpu);
    }
}
