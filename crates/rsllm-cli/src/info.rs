//! `rsllm info` — system capabilities + optional KV memory estimate.

use std::path::Path;

use rsllm_gguf::GgufFile;
use rsllm_models::dsv4::shape::{
    DSV4_HEAD_DIM, DSV4_N_INDEXER_HEAD, DSV4_N_INDEXER_HEAD_DIM, DSV4_N_LAYER, DSV4_N_SWA,
};

use crate::CliError;

/// Run `info`. `ctx_size` is used for the KV memory estimate when a
/// model is provided; system info is printed unconditionally.
pub fn run(model: Option<&Path>, ctx_size: usize) -> Result<(), CliError> {
    println!("rsLLM v{}", rsllm_core::version());
    println!();
    println!("System");
    println!("  OS              : {}", std::env::consts::OS);
    println!("  Arch            : {}", std::env::consts::ARCH);
    println!("  Endianness      : little (assumed; GGUF is little-endian)");
    println!("  CPU threads     : {}", available_parallelism());
    println!();
    println!("Backends");
    println!("  CPU (NEON/AVX2) : ✅");
    println!("  Metal           : pending FEATURE_025");
    println!("  CUDA            : pending FEATURE_026");
    println!();

    if let Some(path) = model {
        print_model_summary(path, ctx_size)?;
    } else {
        print_default_kv_estimate(ctx_size);
    }
    Ok(())
}

/// Best-effort thread count — never panics. Falls back to 1 on
/// platforms without `available_parallelism`.
fn available_parallelism() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
}

fn print_default_kv_estimate(ctx_size: usize) {
    println!("KV memory estimate (DS V4 Flash, no model loaded)");
    println!("  context size    : {ctx_size} tokens");
    print_kv_breakdown(ctx_size);
}

fn print_model_summary(path: &Path, ctx_size: usize) -> Result<(), CliError> {
    let gguf = GgufFile::open(path)?;
    println!("Model");
    println!("  path            : {}", path.display());
    println!("  size on disk    : {}", format_bytes(gguf.file_size()));
    println!("  GGUF version    : {}", gguf.version());
    println!("  tensors         : {}", gguf.tensors().len());
    println!("  metadata keys   : {}", gguf.metadata().len());
    println!("  fingerprint     : {}", hex(gguf.fingerprint()));
    println!();
    println!("KV memory estimate (DS V4 Flash)");
    println!("  context size    : {ctx_size} tokens");
    print_kv_breakdown(ctx_size);
    Ok(())
}

/// Memory breakdown sourced from `docs/features/v0.1.0.md#feature_006`.
/// Per-layer raw KV is exact; compressed-pool and indexer estimates
/// use the per-layer ratio constants from `rsllm-models::dsv4::shape`.
fn print_kv_breakdown(ctx_size: usize) {
    let raw = DSV4_N_LAYER * DSV4_N_SWA * DSV4_HEAD_DIM * 4;
    // Conservative average ratio across the compressed layers
    // (21 ratio-4 + 20 ratio-128). Mean reciprocal ~ 1/3 for "tail
    // size relative to ctx_size" matching the v0.1.0 docs.
    let compressed = DSV4_N_LAYER * (ctx_size / 3 + 2) * DSV4_HEAD_DIM * 4;
    // 21 ratio-4 layers each hold an indexer pool of ctx/4 rows.
    let ratio4_layers = 21;
    let indexer =
        ratio4_layers * (ctx_size / 4 + 2) * DSV4_N_INDEXER_HEAD_DIM * DSV4_N_INDEXER_HEAD * 4;
    let total = raw + compressed + indexer;
    println!("  raw SWA ring    : {} (43 × 128 × 512 × 4B)", format_bytes(raw as u64));
    println!("  compressed pool : {}", format_bytes(compressed as u64));
    println!("  indexer pool    : {}", format_bytes(indexer as u64));
    println!("  total           : {}", format_bytes(total as u64));
}

/// Hex-encode a 32-byte fingerprint, no allocation by `String::with_capacity`.
fn hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        use std::fmt::Write;
        let _ = write!(&mut s, "{b:02x}");
    }
    s
}

/// Pretty-print a byte count using binary units. ~10 GiB shows as
/// "10.34 GiB"; ~3 MiB shows as "3.21 MiB".
fn format_bytes(bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "KiB", "MiB", "GiB", "TiB"];
    let mut value = bytes as f64;
    let mut i = 0;
    while value >= 1024.0 && i + 1 < UNITS.len() {
        value /= 1024.0;
        i += 1;
    }
    if i == 0 {
        format!("{} {}", bytes, UNITS[0])
    } else {
        format!("{value:.2} {}", UNITS[i])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_bytes_picks_human_units() {
        assert_eq!(format_bytes(0), "0 B");
        assert_eq!(format_bytes(1023), "1023 B");
        assert_eq!(format_bytes(1024), "1.00 KiB");
        assert_eq!(format_bytes(1_048_576), "1.00 MiB");
        assert_eq!(format_bytes(1_073_741_824), "1.00 GiB");
    }

    #[test]
    fn hex_round_trips() {
        let bytes = [0xDE_u8, 0xAD, 0xBE, 0xEF, 0x00, 0xFF];
        assert_eq!(hex(&bytes), "deadbeef00ff");
    }

    #[test]
    fn run_without_model_succeeds() {
        // Should produce output and exit Ok regardless of the platform.
        run(None, 4096).expect("info should succeed without a model");
    }
}
