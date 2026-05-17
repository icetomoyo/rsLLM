//! `rsllm inspect` — load a GGUF and dump its model summary.
//!
//! Mirrors ds4's `model_summary()` (`ds4.c:1229-1300`): tensor list,
//! metadata block, quantization-type distribution, and stable model
//! fingerprint.
//!
//! Performs no inference and no dequant — purely a directory walk
//! over the GGUF.

use std::collections::BTreeMap;
use std::path::Path;

use rsllm_gguf::{GgufFile, TensorInfo};

use crate::CliError;

/// Run `inspect`. Prints to stdout in a human-readable format.
pub fn run(path: &Path) -> Result<(), CliError> {
    let gguf = GgufFile::open(path)?;
    println!("rsLLM inspect");
    println!("  path            : {}", path.display());
    println!("  size on disk    : {}", format_bytes(gguf.file_size()));
    println!("  GGUF version    : {}", gguf.version());
    println!("  alignment       : {} B", gguf.alignment());
    println!("  fingerprint     : {}", hex(gguf.fingerprint()));
    println!();

    // Metadata — print the architecture-relevant keys first if they
    // exist, then the count of remaining keys (a full dump can be
    // 100+ keys long, which buries the useful summary).
    let meta = gguf.metadata();
    println!("Metadata ({} keys)", meta.len());
    for key in &[
        "general.architecture",
        "general.name",
        "general.quantization_version",
        "deepseek.block_count",
        "deepseek.embedding_length",
        "deepseek.feed_forward_length",
        "deepseek.attention.head_count",
        "deepseek.attention.head_count_kv",
        "deepseek.attention.key_length",
        "deepseek.attention.value_length",
        "deepseek.expert_count",
        "deepseek.expert_used_count",
    ] {
        if let Some(v) = meta.get(key) {
            println!("  {key:<40} {v:?}");
        }
    }
    println!();

    // Quantization-type distribution. Each tensor knows its on-disk
    // type and byte size; we accumulate by type.
    print_quant_distribution(gguf.tensors());
    println!();

    // Per-tensor table — sorted alphabetically for diff stability.
    println!("Tensors ({} total)", gguf.tensors().len());
    let mut sorted: Vec<&TensorInfo> = gguf.tensors().iter().collect();
    sorted.sort_by(|a, b| a.name.cmp(&b.name));
    for t in sorted {
        let shape = t
            .shape
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(" × ");
        let bytes = t
            .byte_size
            .map(format_bytes)
            .unwrap_or_else(|| "?".to_string());
        println!("  {:<58} t={:>3} [{}] {}", t.name, t.raw_type, shape, bytes);
    }
    Ok(())
}

/// Group the tensors by `raw_type` and print "{type-id} count={N}
/// bytes=…" rows. We don't map raw_type ↔ name here — that lives
/// in `rsllm-gguf::tensor::GgmlType`, which we don't depend on
/// directly from the CLI to keep the surface narrow.
fn print_quant_distribution(tensors: &[TensorInfo]) {
    let mut by_type: BTreeMap<u32, (usize, u64)> = BTreeMap::new();
    for t in tensors {
        let entry = by_type.entry(t.raw_type).or_insert((0, 0));
        entry.0 += 1;
        entry.1 = entry.1.saturating_add(t.byte_size.unwrap_or(0));
    }
    println!("Quantization distribution");
    for (rt, (count, bytes)) in by_type {
        println!(
            "  type={rt:>3}   count={count:>4}   bytes={}",
            format_bytes(bytes),
        );
    }
}

fn hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        use std::fmt::Write;
        let _ = write!(&mut s, "{b:02x}");
    }
    s
}

fn format_bytes(bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "KiB", "MiB", "GiB", "TiB"];
    let mut value = bytes as f64;
    let mut i = 0;
    while value >= 1024.0 && i + 1 < UNITS.len() {
        value /= 1024.0;
        i += 1;
    }
    if i == 0 {
        format!("{bytes} B")
    } else {
        format!("{value:.2} {}", UNITS[i])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_model_returns_error() {
        let err = run(Path::new("/nonexistent/model.gguf"));
        assert!(err.is_err());
    }
}
