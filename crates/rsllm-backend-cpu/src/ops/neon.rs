//! ARM NEON dotprod kernel specializations (aarch64-only).
//!
//! Inspiration: `ds4.c:42-43, 324-361, 2727-2781, 3277-3297` (MIT, The
//! ds4.c authors). Borrowed by reference — no runtime linkage. Each
//! kernel here is a SIMD-accelerated specialization of the reference
//! in [`crate::ops::scalar`], guarded by the runtime
//! [`crate::SimdTier::Neon`] dispatch decision in [`crate::ops`].
//!
//! Phase A: stubs only — the actual NEON intrinsics for `vdotq_s32`,
//! `vfmaq_f32`, `vmaxvq_f32`, etc. land in phase C/D as the dependent
//! kernels are implemented. Keeping the module exposed at phase A lets
//! the public dispatch layer compile against a stable surface.
