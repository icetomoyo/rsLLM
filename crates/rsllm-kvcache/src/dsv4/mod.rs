//! DeepSeek V4 Flash three-tier KV cache.
//!
//! F006 of v0.1.0. Three tiers (`ds4.c:6071-6092` `ds4_layer_cache`):
//!
//! 1. **Raw SWA ring** ([`swa::RawSwaRing`]) — the most recent
//!    `N_SWA = 128` token KV rows are stored verbatim in a per-layer
//!    ring buffer. Decoded cheaply and used for the dominant attention
//!    contributions.
//! 2. **Compressed KV pool** ([`compressed::CompressedKvPool`]) —
//!    older KV rows are softmax-aggregated into compressed slots every
//!    `compress_ratio` tokens (per-layer ratio from
//!    [`shape::layer_compress_ratio`]: 0 dense / 4 fine / 128 coarse).
//! 3. **Ratio-4 indexer** ([`indexer::IndexerPool`]) — for the
//!    21 ratio-4 layers (even layers `il ≥ 2`), a sparse `top-512`
//!    selection over the compressed pool using 64 indexer heads × 128
//!    latent dim.
//!
//! The top-level [`three_tier::ThreeTierKvCache`] wires the three
//! together. The companion attention adapter (in `rsllm-models::dsv4::attention`)
//! bridges this cache into F005's `AttentionFn` callback.
//!
//! Ported by reference from `ds4.c:6068-6371` (MIT, The ds4.c authors).
//! Line numbers pinned to ds4 commit `ef0a490` (2026-05-17).

pub mod compressed;
pub mod indexer;
pub mod shape;
pub mod swa;
pub mod three_tier;
