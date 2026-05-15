//! # rsllm-cal
//!
//! Compute Abstraction Layer for rsLLM.
//!
//! Defines the [`Backend`] trait that every compute backend (CPU,
//! CUDA, Metal, wgpu, ...) implements, plus the [`DType`] and
//! [`BackendCapability`] descriptors that model graph code uses to
//! pick the right kernel variant at runtime.
//!
//! See [`docs/03-HLD.md`](https://github.com/icetomoyo/rsLLM/blob/main/docs/03-HLD.md)
//! §3.2 for the trait shape.

/// Identifier for a compute backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendKind {
    /// CPU backend (AVX2 / AVX-512 / NEON / SVE depending on target).
    Cpu,
    /// NVIDIA CUDA backend.
    Cuda,
    /// Apple Metal backend.
    Metal,
    /// Cross-platform wgpu / Vulkan backend.
    Wgpu,
}

/// Numeric element type carried by a backend buffer.
///
/// Quantized formats are referenced by their GGUF type tag; the
/// backend is responsible for decoding the corresponding block layout
/// at compute time (see `rsllm-gguf::GgmlType`). Variant names follow
/// the GGUF / ggml spec spelling (`Q4_K`, `IQ2_XXS`, …) rather than
/// strict Rust UpperCamelCase, matching `rsllm_gguf::GgmlType`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum DType {
    /// 32-bit IEEE-754 float.
    F32,
    /// 16-bit IEEE-754 half-precision float.
    F16,
    /// 32-bit signed integer (used by index buffers).
    I32,
    /// 8-bit signed integer (raw Q8_0 quant slot).
    I8,
    /// Q8_0 block-quantized (32 elements / 34 bytes).
    Q8_0,
    /// Q4_K block-quantized (256 elements / 144 bytes).
    Q4_K,
    /// Q2_K block-quantized (256 elements / 84 bytes).
    Q2_K,
    /// IQ2_XXS block-quantized (256 elements / 66 bytes).
    IQ2_XXS,
    /// FP8 E4M3 (8-bit float, used by KV cache).
    Fp8E4M3,
}

/// Runtime capability descriptor reported by [`Backend::capability`].
///
/// Used by the model graph to pick the most specialized kernel variant
/// available on the host. A v0.1.0 CPU backend reports one of `"neon"`,
/// `"avx512"`, `"avx2"`, or `"scalar"`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackendCapability {
    /// Stable backend identifier — `"cpu"`, `"metal"`, `"cuda"`, …
    pub backend: &'static str,
    /// SIMD / extension tier — `"neon"`, `"avx512"`, `"avx2"`, `"scalar"`.
    pub arch: &'static str,
    /// Whether the backend supports the `Q4_K × Q8_K` MoE matmul kernel.
    pub q4k_matmul: bool,
    /// Whether the backend supports the `Q2_K × Q8_K` MoE matmul kernel.
    pub q2k_matmul: bool,
    /// Whether the backend supports the `IQ2_XXS × Q8_K` MoE matmul kernel.
    pub iq2xxs_matmul: bool,
    /// Whether the backend exposes the FP8 E4M3 KV cache codec.
    pub fp8_kv_cache: bool,
}

/// Marker trait for compute backends.
///
/// Concrete backends additionally expose kernel methods directly (e.g.
/// [`crate::BackendKind::Cpu`] backends implement matmul / rmsnorm /
/// rope on a `CpuBackend` handle). v0.1.0 deliberately avoids a giant
/// `KernelOp` enum dispatch in favour of typed kernel methods on each
/// backend's handle — see `docs/features/v0.1.0.md` §F004 rationale.
pub trait Backend {
    /// Identifier of this backend variant.
    fn kind(&self) -> BackendKind;

    /// Capability descriptor for the backend.
    fn capability(&self) -> BackendCapability;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_kinds_distinct() {
        assert_ne!(BackendKind::Cpu, BackendKind::Cuda);
        assert_ne!(BackendKind::Metal, BackendKind::Wgpu);
    }

    #[test]
    fn dtype_variants_distinct() {
        assert_ne!(DType::F32, DType::F16);
        assert_ne!(DType::Q4_K, DType::Q2_K);
    }
}
