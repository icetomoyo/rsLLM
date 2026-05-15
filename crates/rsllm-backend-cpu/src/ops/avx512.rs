//! x86_64 AVX-512 + VNNI kernel specializations (x86_64-only).
//!
//! Inspiration: ggml's `ggml-cpu-x86.c` AVX-512 idioms (MIT, the ggml
//! authors). Borrowed by reference — no runtime linkage. Each kernel
//! here is a SIMD-accelerated specialization of the reference in
//! [`crate::ops::scalar`], guarded by the runtime
//! [`crate::SimdTier::Avx512`] dispatch decision.
//!
//! The AVX-512 VNNI tier is the v0.1.0 primary path for AMD Strix Halo
//! (Zen 5 has full AVX-512 + VNNI). `_mm512_dpbusd_epi32` is the
//! semantic 1:1 counterpart to ARM's `vdotq_s32`, which is what makes
//! the Q8_0 batched matmul kernel portable in shape if not in source.
//!
//! Phase A: stubs only — implementations land in phase C/D.
