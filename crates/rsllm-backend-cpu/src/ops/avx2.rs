//! x86_64 AVX2 kernel specializations (x86_64-only).
//!
//! Inspiration: ggml's `ggml-cpu-x86.c` AVX2 idioms (MIT, the ggml
//! authors). Borrowed by reference — no runtime linkage. Each kernel
//! here is a SIMD-accelerated specialization of the reference in
//! [`crate::ops::scalar`], guarded by the runtime
//! [`crate::SimdTier::Avx2`] dispatch decision.
//!
//! The AVX2 tier exists for hosts that lack AVX-512 VNNI (typical
//! mid-tier Intel / Ryzen 3000-5000). It uses 16×16 → 32-bit
//! shift-and-add for Q8_0 dot products in place of `_mm512_dpbusd_epi32`.
//!
//! Phase A: stubs only — implementations land in phase C/D.
