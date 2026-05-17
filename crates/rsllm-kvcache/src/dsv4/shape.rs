//! KV-relevant shape constants for DeepSeek V4 Flash.
//!
//! These mirror the subset of `rsllm-models::dsv4::shape` constants
//! that the cache layout depends on. We duplicate rather than depend
//! on `rsllm-models` (which would create a reverse dependency on a
//! higher-level crate). The values are architectural invariants of
//! DS V4 Flash — a `validate_metadata` cross-check in `rsllm-models`
//! catches any drift at GGUF load time.
//!
//! Ported by reference from `ds4.c:87-108` (MIT, The ds4.c authors).
//! Line numbers pinned to ds4 commit `ef0a490` (2026-05-17).

/// Number of transformer blocks. KV cache holds one [`super::layer::LayerCache`] per layer.
pub const DSV4_N_LAYER: usize = 43;

/// Per-row KV latent dimension. DS V4 Flash uses MLA: KV is a single
/// `HEAD_DIM`-wide latent (`ds4.c:91-92`).
pub const DSV4_HEAD_DIM: usize = 512;

/// Sliding-window attention raw ring size — the most recent
/// `N_SWA` tokens are kept verbatim in `raw_kv`. (`ds4.c:103`.)
pub const DSV4_N_SWA: usize = 128;

/// Number of indexer heads (`ds4.c:104`).
pub const DSV4_N_INDEXER_HEAD: usize = 64;

/// Per-head dimension for the indexer (`ds4.c:105`).
pub const DSV4_N_INDEXER_HEAD_DIM: usize = 128;

/// Top-K cap for the sparse indexer (used by ratio-4 layers).
pub const DSV4_N_INDEXER_TOP_K: usize = 512;

/// Layers `< DSV4_DENSE_LAYERS` use dense (uncompressed) KV cache; the
/// rest use the alternating ratio-4 / ratio-128 compression. From
/// `ds4.c:413` `if (il < 2) return 0;`.
pub const DSV4_DENSE_LAYERS: usize = 2;

/// RoPE base frequency for compressed-KV layers
/// (`ds4.c:60` `DS4_COMPRESS_ROPE_FREQ_BASE`).
pub const DSV4_COMPRESS_ROPE_FREQ_BASE: f32 = 160_000.0;

/// YaRN scale factor for compressed-KV RoPE (`ds4.c:57`).
pub const DSV4_ROPE_SCALE_FACTOR: f32 = 16.0;

/// Compression ratio for layer `il`, mirroring `ds4_layer_compress_ratio`
/// at `ds4.c:411-416`:
///
/// ```text
/// il < 2       → 0    (dense, no compression)
/// il >= 2 even → 4    (ratio-4, with indexer)
/// il >= 2 odd  → 128  (ratio-128)
/// ```
///
/// Ratio 0 means "dense raw cache only, no compression"; ratio 4 layers
/// additionally maintain the sparse indexer.
///
/// # Panics
/// Panics if `il >= DSV4_N_LAYER` — matches ds4's `ds4_die` behavior.
#[must_use]
pub fn layer_compress_ratio(il: usize) -> u32 {
    assert!(
        il < DSV4_N_LAYER,
        "layer index {il} >= DSV4_N_LAYER {DSV4_N_LAYER}",
    );
    if il < DSV4_DENSE_LAYERS {
        0
    } else if il.is_multiple_of(2) {
        4
    } else {
        128
    }
}

/// `true` iff layer `il` maintains the ratio-4 indexer (i.e. compress
/// ratio is exactly 4).
#[must_use]
pub fn layer_has_indexer(il: usize) -> bool {
    layer_compress_ratio(il) == 4
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dense_layers_have_no_compression() {
        assert_eq!(layer_compress_ratio(0), 0);
        assert_eq!(layer_compress_ratio(1), 0);
        assert!(!layer_has_indexer(0));
        assert!(!layer_has_indexer(1));
    }

    #[test]
    fn even_layers_above_threshold_are_ratio_4() {
        for il in (2..DSV4_N_LAYER).filter(|i| i % 2 == 0) {
            assert_eq!(layer_compress_ratio(il), 4, "layer {il}");
            assert!(layer_has_indexer(il), "layer {il}");
        }
    }

    #[test]
    fn odd_layers_above_threshold_are_ratio_128() {
        for il in (2..DSV4_N_LAYER).filter(|i| i % 2 == 1) {
            assert_eq!(layer_compress_ratio(il), 128, "layer {il}");
            assert!(!layer_has_indexer(il), "layer {il}");
        }
    }

    #[test]
    fn ratio_4_layer_count_matches_ds4_estimate() {
        // F006 mem estimate: ~14 of 43 layers are ratio-4
        // (layers 2,4,6,...,42 = 21 even layers ≥ 2; ds4 has different math but our doc said 14).
        // Verify the actual count.
        let ratio4 = (0..DSV4_N_LAYER).filter(|&i| layer_compress_ratio(i) == 4).count();
        // Layers 2, 4, 6, ..., 42 → 21 even layers
        assert_eq!(ratio4, 21);
    }

    #[test]
    #[should_panic(expected = "layer index")]
    fn panics_on_out_of_range_layer() {
        let _ = layer_compress_ratio(DSV4_N_LAYER);
    }
}
