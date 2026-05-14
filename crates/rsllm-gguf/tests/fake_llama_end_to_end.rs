//! End-to-end smoke test for FEATURE_002 (GGUF parser + dequant).
//!
//! Synthesizes a small "fake-Llama" GGUF file in-memory and writes it to a
//! temporary file on disk. Then exercises the **real** mmap-backed
//! [`GgufFile::open`] path (rather than the in-memory `from_bytes` used by
//! the unit tests) and decodes every supported tensor type through
//! [`dequant_to_f32`].
//!
//! The fixture is intentionally tiny (single block per quantization format)
//! so the test runs in milliseconds — its job is to catch wiring breaks, not
//! to validate decode correctness (which is the unit tests' responsibility).

use std::fs;
use std::path::PathBuf;

use half::f16;
use rsllm_gguf::{GgmlType, GgufFile, dequant_to_f32};

/// Compose a unique temp-file path. We deliberately avoid pulling in the
/// `tempfile` crate just for one test.
fn tmp_path(suffix: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    let pid = std::process::id();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.subsec_nanos())
        .unwrap_or(0);
    p.push(format!("rsllm-gguf-{pid}-{nanos}-{suffix}"));
    p
}

/// Minimal GGUF v3 byte-sequence builder used by this integration test.
/// Mirrors the private `GgufBuilder` in `src/file.rs` so the integration
/// test stays insulated from internal test infrastructure.
struct Builder {
    kv: Vec<(String, u32, Vec<u8>)>,
    tensors: Vec<(String, u32, Vec<u64>, Vec<u8>)>,
    alignment: u64,
}

impl Builder {
    fn new() -> Self {
        Self {
            kv: Vec::new(),
            tensors: Vec::new(),
            alignment: 32,
        }
    }

    fn kv_str(mut self, key: &str, value: &str) -> Self {
        let mut bytes = (value.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(value.as_bytes());
        self.kv.push((key.to_owned(), 8, bytes)); // 8 = String
        self
    }

    fn kv_u32(mut self, key: &str, value: u32) -> Self {
        self.kv
            .push((key.to_owned(), 4, value.to_le_bytes().to_vec())); // 4 = U32
        self
    }

    fn kv_bool(mut self, key: &str, value: bool) -> Self {
        self.kv.push((key.to_owned(), 7, vec![u8::from(value)])); // 7 = Bool
        self
    }

    fn tensor(mut self, name: &str, ttype: u32, dims: Vec<u64>, payload: Vec<u8>) -> Self {
        self.tensors.push((name.to_owned(), ttype, dims, payload));
        self
    }

    fn build(self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(b"GGUF");
        out.extend_from_slice(&3u32.to_le_bytes());
        out.extend_from_slice(&(self.tensors.len() as u64).to_le_bytes());
        out.extend_from_slice(&(self.kv.len() as u64).to_le_bytes());

        for (key, ttype, value_bytes) in &self.kv {
            out.extend_from_slice(&(key.len() as u64).to_le_bytes());
            out.extend_from_slice(key.as_bytes());
            out.extend_from_slice(&ttype.to_le_bytes());
            out.extend_from_slice(value_bytes);
        }

        let mut payload_offsets = Vec::with_capacity(self.tensors.len());
        let mut cursor = 0u64;
        for (_, _, _, payload) in &self.tensors {
            payload_offsets.push(cursor);
            cursor += payload.len() as u64;
        }

        for (idx, (name, ttype, dims, _)) in self.tensors.iter().enumerate() {
            out.extend_from_slice(&(name.len() as u64).to_le_bytes());
            out.extend_from_slice(name.as_bytes());
            out.extend_from_slice(&(dims.len() as u32).to_le_bytes());
            for &d in dims {
                out.extend_from_slice(&d.to_le_bytes());
            }
            out.extend_from_slice(&ttype.to_le_bytes());
            out.extend_from_slice(&payload_offsets[idx].to_le_bytes());
        }

        let pad = (self.alignment - (out.len() as u64 % self.alignment)) % self.alignment;
        out.extend(std::iter::repeat_n(0u8, pad as usize));

        for (_, _, _, payload) in &self.tensors {
            out.extend_from_slice(payload);
        }

        out
    }
}

/// One f32-block payload: 4 elements.
fn f32_payload(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// One Q8_0 block (34 bytes): f16 scale + 32 i8 quants.
fn q8_0_block(scale: f32, qs: [i8; 32]) -> Vec<u8> {
    let mut out = f16::from_f32(scale).to_le_bytes().to_vec();
    for q in &qs {
        out.push(*q as u8);
    }
    out
}

/// One Q4_0 block (18 bytes): f16 scale + 16 packed nibbles.
fn q4_0_block(scale: f32, nibbles: [u8; 32]) -> Vec<u8> {
    let mut out = f16::from_f32(scale).to_le_bytes().to_vec();
    for j in 0..16 {
        let lo = nibbles[j] & 0x0F;
        let hi = nibbles[j + 16] & 0x0F;
        out.push((hi << 4) | lo);
    }
    out
}

#[test]
fn fake_llama_open_via_mmap_and_dequant_all_types() {
    // ----- Build the file ------------------------------------------------

    // Two F32 tensors: 4 elements each.
    let embed_payload = f32_payload(&[1.0, -1.0, 2.0, -2.0]);
    let output_norm_payload = f32_payload(&[0.5, 0.25, 0.125, 0.0]);

    // One Q8_0 tensor: 1 block (32 elements). Scale=2, alternating quants.
    let q8_qs: [i8; 32] = std::array::from_fn(|i| if i % 2 == 0 { 1 } else { -1 });
    let q8_payload = q8_0_block(2.0, q8_qs);

    // One Q4_0 tensor: 1 block (32 elements). Scale=1, nibbles cycle 0..16.
    let q4_nibbles: [u8; 32] = std::array::from_fn(|i| (i % 16) as u8);
    let q4_payload = q4_0_block(1.0, q4_nibbles);

    let bytes = Builder::new()
        // Metadata mirroring a real Llama GGUF.
        .kv_str("general.architecture", "llama")
        .kv_str("general.name", "rsllm-fake-llama")
        .kv_u32("llama.context_length", 2048)
        .kv_u32("llama.embedding_length", 4)
        .kv_u32("llama.block_count", 1)
        .kv_bool("tokenizer.ggml.add_bos_token", true)
        // Tensors: two f32 + one Q8_0 + one Q4_0 = 4 tensors covering both
        // simple-float and quantized paths.
        .tensor(
            "token_embd.weight",
            GgmlType::F32 as u32,
            vec![2, 2],
            embed_payload.clone(),
        )
        .tensor(
            "output_norm.weight",
            GgmlType::F32 as u32,
            vec![4],
            output_norm_payload.clone(),
        )
        .tensor(
            "blk.0.attn_q.weight",
            GgmlType::Q8_0 as u32,
            vec![32],
            q8_payload.clone(),
        )
        .tensor(
            "blk.0.ffn_gate.weight",
            GgmlType::Q4_0 as u32,
            vec![32],
            q4_payload.clone(),
        )
        .build();

    // Write to disk so we exercise the real File::open + mmap path.
    let path = tmp_path("fake-llama.gguf");
    fs::write(&path, &bytes).expect("write fake gguf");

    // ----- Open + verify -------------------------------------------------

    let file = GgufFile::open(&path).expect("open mmap");
    assert_eq!(file.version(), 3);
    assert_eq!(file.alignment(), 32);
    assert_eq!(file.tensors().len(), 4);

    // Metadata convenience accessors.
    assert_eq!(
        file.metadata().get_str("general.architecture"),
        Some("llama")
    );
    assert_eq!(file.metadata().get_u32("llama.context_length"), Some(2048));
    assert_eq!(file.metadata().get_u32("llama.block_count"), Some(1));
    assert_eq!(
        file.metadata().get_bool("tokenizer.ggml.add_bos_token"),
        Some(true)
    );

    // Fingerprint should be deterministic — building the same file twice
    // yields the same fingerprint.
    let fp1 = *file.fingerprint();
    drop(file); // release mmap before re-opening on Windows
    let file2 = GgufFile::open(&path).expect("re-open");
    assert_eq!(*file2.fingerprint(), fp1, "fingerprint must be stable");

    // ----- Dequantize each tensor and check values -----------------------

    let embed_info = file2.tensor("token_embd.weight").expect("embed present");
    assert_eq!(embed_info.dtype, Some(GgmlType::F32));
    assert_eq!(embed_info.elements, 4);
    let embed_src = file2.tensor_bytes(embed_info).expect("embed payload");
    let mut embed_dst = vec![0.0f32; 4];
    dequant_to_f32(GgmlType::F32, embed_src, &mut embed_dst).expect("dequant f32");
    assert_eq!(embed_dst, vec![1.0, -1.0, 2.0, -2.0]);

    let q8_info = file2.tensor("blk.0.attn_q.weight").expect("q8 present");
    assert_eq!(q8_info.dtype, Some(GgmlType::Q8_0));
    let q8_src = file2.tensor_bytes(q8_info).expect("q8 payload");
    let mut q8_dst = vec![0.0f32; 32];
    dequant_to_f32(GgmlType::Q8_0, q8_src, &mut q8_dst).expect("dequant q8_0");
    // Scale = 2; quants alternate ±1 → expect ±2 alternating.
    for (i, v) in q8_dst.iter().enumerate() {
        let want = if i % 2 == 0 { 2.0 } else { -2.0 };
        assert!((v - want).abs() < 1e-3, "q8 i={i} got {v} want {want}");
    }

    let q4_info = file2.tensor("blk.0.ffn_gate.weight").expect("q4 present");
    assert_eq!(q4_info.dtype, Some(GgmlType::Q4_0));
    let q4_src = file2.tensor_bytes(q4_info).expect("q4 payload");
    let mut q4_dst = vec![0.0f32; 32];
    dequant_to_f32(GgmlType::Q4_0, q4_src, &mut q4_dst).expect("dequant q4_0");
    // Scale=1, nibbles cycle 0..16 → dst = nibble - 8.
    for (i, v) in q4_dst.iter().enumerate() {
        let want = (i % 16) as f32 - 8.0;
        assert!((v - want).abs() < 1e-3, "q4 i={i} got {v} want {want}");
    }

    // Cleanup. Drop mmap first to release the Windows file handle.
    drop(file2);
    let _ = fs::remove_file(&path);
}

#[test]
fn fake_llama_unknown_tensor_type_does_not_crash_open() {
    // Use a Q5_0 tensor type (recognized but not decodable in v0.1.0). The
    // file must still open cleanly — `inspect`-style tooling depends on this.
    // We avoid emitting bytes for this tensor since byte_size is None for
    // unsupported types; the parser should compute its placeholder payload
    // size as 0 and refuse to dequant it.
    //
    // Actually: byte_size = None propagates upward; the file.rs validation
    // loop already skips byte_size = None entries. So we just include zero
    // bytes for the payload and place the tensor with a trivial shape.
    let q5_payload = vec![0u8; 22]; // one Q5_0 block size
    let bytes = Builder::new()
        .kv_str("general.architecture", "llama")
        .tensor(
            "blk.0.exotic.weight",
            GgmlType::Q5_0 as u32,
            vec![32],
            q5_payload,
        )
        .build();

    let path = tmp_path("exotic.gguf");
    fs::write(&path, &bytes).expect("write exotic gguf");

    let file = GgufFile::open(&path).expect("open should succeed even for unknown decode");
    assert_eq!(file.tensors().len(), 1);
    let info = &file.tensors()[0];
    // The type is recognized at the descriptor level...
    assert_eq!(info.dtype, Some(GgmlType::Q5_0));
    assert!(!info.dtype.unwrap().is_decodable_v0_1_0());
    // ...but dequant must explicitly refuse.
    let src = file.tensor_bytes(info).expect("payload bytes present");
    let mut dst = vec![0.0f32; 32];
    let err = dequant_to_f32(GgmlType::Q5_0, src, &mut dst)
        .expect_err("dequant must reject unsupported type");
    let msg = format!("{err:?}").to_lowercase();
    assert!(
        msg.contains("q5_0") && msg.contains("unsupported"),
        "error must identify the unsupported type: {err:?}"
    );

    drop(file);
    let _ = fs::remove_file(&path);
}
