//! DeepSeek V4 Flash model components.
//!
//! v0.1.0's single supported architecture. Submodules:
//!
//! - [`shape`] — fixed shape constants and GGUF metadata validation.
//! - `mla` — MLA Q/KV LoRA projections (F005.B, upcoming).
//! - `hc` — hyper-connection pre/post + Sinkhorn mix (F005.C, upcoming).
//! - `moe` — hash + top-k MoE routing (F005.D / F005.E, upcoming).
//!
//! See `docs/features/v0.1.0.md` §FEATURE_005 for the full design.
//!
//! Ported by reference from `ds4.c` (MIT, The ds4.c authors).

pub mod hc;
pub mod mla;
pub mod moe;
pub mod shape;
pub mod weight;
