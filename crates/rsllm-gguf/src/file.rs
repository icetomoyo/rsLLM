//! High-level `GgufFile` wrapper: open + memory-map + parse header,
//! metadata, and tensor directory into owned Rust structures, while keeping
//! tensor payload bytes as zero-copy slices into the original mapping.
//!
//! Parser flow mirrors `model_open()` in `ds4.c:1176-1222` (MIT, The ds4.c
//! authors), with two key adaptations:
//!
//! 1. **macOS uses `MAP_PRIVATE`** to side-step Darwin's VM map-count
//!    accounting bug (see `ds4.c:1188-1200`).
//! 2. **A stable SHA-256 model fingerprint** is computed from the sorted
//!    tensor directory and stored on `GgufFile`. This is required by
//!    FEATURE_022 (disk KV cache) to detect "different model, same prompt"
//!    cache key collisions.

use std::fs::File;
use std::path::Path;

use memmap2::Mmap;
use sha2::{Digest, Sha256};
use tracing::warn;

use crate::error::Error;
use crate::metadata::Metadata;
use crate::reader::Reader;
use crate::tensor::{TensorInfo, parse_tensor_directory};
use crate::{MAGIC, SUPPORTED_VERSION};

/// Default tensor alignment when `general.alignment` is not present.
const DEFAULT_ALIGNMENT: u64 = 32;

/// Upper bound on the tensor alignment we will accept from a file.
///
/// Real GGUF files use 32 or 64. The cap exists purely to make hostile inputs
/// (e.g. `alignment = u64::MAX - 1`) reject cleanly instead of overflowing
/// `align_up`. 64 KiB is plenty of headroom for any plausible hardware
/// alignment requirement.
const MAX_ALIGNMENT: u64 = 65_536;

/// Backing storage for a `GgufFile`. Hidden behind the public API so the
/// distinction between mmap-backed and in-memory files is irrelevant to
/// callers (other than for cleanup semantics).
enum Storage {
    Mmap(Mmap),
    /// In-memory buffer. Primarily used by the test suite, but also useful
    /// for callers that already have the file bytes in a `Vec<u8>`.
    Bytes(Vec<u8>),
}

impl Storage {
    fn as_slice(&self) -> &[u8] {
        match self {
            Self::Mmap(m) => m,
            Self::Bytes(v) => v.as_slice(),
        }
    }

    fn len(&self) -> u64 {
        self.as_slice().len() as u64
    }
}

/// A parsed GGUF file with zero-copy access to tensor payloads.
pub struct GgufFile {
    storage: Storage,
    version: u32,
    alignment: u64,
    tensor_data_offset: u64,
    metadata: Metadata,
    tensors: Vec<TensorInfo>,
    fingerprint: [u8; 32],
}

impl GgufFile {
    /// Open a GGUF file from disk, memory-map it, and parse the header,
    /// metadata, and tensor directory.
    ///
    /// On macOS the mapping uses `MAP_PRIVATE` to avoid a known Darwin VM
    /// accounting bug that can panic the kernel when a large file-backed
    /// shared mapping is paged in incrementally during CPU inference
    /// (`ds4.c:1188-1200`).
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, Error> {
        let file = File::open(path.as_ref())?;
        let file_size = file.metadata()?.len();
        if file_size < 32 {
            return Err(Error::TooSmall(file_size));
        }

        // SAFETY: We do not modify the file while mapped, and we never
        // expose mutable access to the mapping. memmap2 documents that
        // unsafe is required because the OS could in principle let another
        // process truncate the file underneath us; that is an accepted risk.
        #[cfg(target_os = "macos")]
        let mmap = unsafe { memmap2::MmapOptions::new().map_copy_read_only(&file)? };
        #[cfg(not(target_os = "macos"))]
        let mmap = unsafe { Mmap::map(&file)? };

        Self::from_storage(Storage::Mmap(mmap))
    }

    /// Parse a GGUF file from an in-memory byte vector. Primarily used in
    /// tests and for callers that already hold the bytes in memory.
    pub fn from_bytes(bytes: Vec<u8>) -> Result<Self, Error> {
        if (bytes.len() as u64) < 32 {
            return Err(Error::TooSmall(bytes.len() as u64));
        }
        Self::from_storage(Storage::Bytes(bytes))
    }

    fn from_storage(storage: Storage) -> Result<Self, Error> {
        // Run all the parsing against a borrow of `storage`; emit owned
        // results that we can then store next to `storage` on `Self`.
        let parsed = parse_all(storage.as_slice(), storage.len())?;

        let fingerprint = compute_fingerprint(&parsed.tensors);

        Ok(Self {
            storage,
            version: parsed.version,
            alignment: parsed.alignment,
            tensor_data_offset: parsed.tensor_data_offset,
            metadata: parsed.metadata,
            tensors: parsed.tensors,
            fingerprint,
        })
    }

    /// GGUF format version (always [`SUPPORTED_VERSION`] = 3 for this build).
    pub fn version(&self) -> u32 {
        self.version
    }

    /// Tensor data alignment in bytes (default 32, may be overridden by the
    /// `general.alignment` metadata key).
    pub fn alignment(&self) -> u64 {
        self.alignment
    }

    /// Absolute file offset at which the tensor data region begins.
    ///
    /// Add this to a [`TensorInfo::relative_offset`] to get the absolute
    /// offset of that tensor's payload.
    pub fn tensor_data_offset(&self) -> u64 {
        self.tensor_data_offset
    }

    /// File size in bytes.
    pub fn file_size(&self) -> u64 {
        self.storage.len()
    }

    /// The parsed metadata key-value table.
    pub fn metadata(&self) -> &Metadata {
        &self.metadata
    }

    /// All tensor descriptors, in directory order.
    pub fn tensors(&self) -> &[TensorInfo] {
        &self.tensors
    }

    /// Look up a tensor descriptor by name. Linear scan; suitable for the
    /// typical case of a few hundred tensors. Promote to a `HashMap` later
    /// if profiling shows it.
    pub fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.iter().find(|t| t.name == name)
    }

    /// Stable SHA-256 fingerprint of the **tensor layout** (sorted by name,
    /// hashing `{name, raw_type, shape}` for each tensor — but **not** the
    /// payload bytes).
    ///
    /// Two GGUF files with the same architecture and quantization scheme
    /// will produce the same fingerprint. Used by FEATURE_022 (disk KV
    /// cache) as part of the cache key.
    pub fn fingerprint(&self) -> &[u8; 32] {
        &self.fingerprint
    }

    /// Borrow the raw on-disk bytes for a tensor's payload. Returns `None`
    /// if the tensor's type or size is unknown.
    pub fn tensor_bytes(&self, info: &TensorInfo) -> Option<&[u8]> {
        let size = info.byte_size?;
        let start = self.tensor_data_offset.checked_add(info.relative_offset)?;
        let end = start.checked_add(size)?;
        let bytes = self.storage.as_slice();
        bytes.get(start as usize..end as usize)
    }
}

impl std::fmt::Debug for GgufFile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GgufFile")
            .field("version", &self.version)
            .field("alignment", &self.alignment)
            .field("file_size", &self.file_size())
            .field("tensor_data_offset", &self.tensor_data_offset)
            .field("metadata_keys", &self.metadata.len())
            .field("tensors", &self.tensors.len())
            .field("fingerprint", &hex32(&self.fingerprint))
            .finish()
    }
}

fn hex32(bytes: &[u8; 32]) -> String {
    let mut out = String::with_capacity(64);
    for b in bytes {
        use std::fmt::Write as _;
        write!(out, "{b:02x}").expect("writing to String never fails");
    }
    out
}

/// Round `value` up to the next multiple of `alignment`. `alignment` must
/// be non-zero (callers default it to 32 when unset). Returns `None` if the
/// rounding would overflow `u64` — callers should treat this as an
/// [`Error::InvalidAlignment`].
fn align_up(value: u64, alignment: u64) -> Option<u64> {
    debug_assert!(alignment > 0);
    let rem = value % alignment;
    if rem == 0 {
        Some(value)
    } else {
        value.checked_add(alignment - rem)
    }
}

struct Parsed {
    version: u32,
    alignment: u64,
    tensor_data_offset: u64,
    metadata: Metadata,
    tensors: Vec<TensorInfo>,
}

/// Pure parsing routine: takes a byte slice + total size, returns owned
/// parsed structures. Does **not** validate full tensor payload presence
/// (that's done by the caller after this returns, since it requires the
/// file size that this function already has access to via `file_size`).
fn parse_all(bytes: &[u8], file_size: u64) -> Result<Parsed, Error> {
    let mut reader = Reader::new(bytes);

    // Magic.
    let magic_bytes = reader.read_bytes(4)?;
    let mut magic_arr = [0u8; 4];
    magic_arr.copy_from_slice(magic_bytes);
    if magic_arr != MAGIC {
        return Err(Error::BadMagic { found: magic_arr });
    }

    // Version.
    let version = reader.read_u32_le()?;
    if version != SUPPORTED_VERSION {
        return Err(Error::UnsupportedVersion(version));
    }

    // Counts.
    let n_tensors = reader.read_u64_le()?;
    let n_kv = reader.read_u64_le()?;

    // Metadata + tensor directory.
    let metadata = Metadata::parse(&mut reader, n_kv)?;
    let tensors = parse_tensor_directory(&mut reader, n_tensors)?;

    // Alignment: prefer u32 form (per spec), fall back to u64. Default 32.
    let alignment = metadata
        .get_u32("general.alignment")
        .map(u64::from)
        .or_else(|| metadata.get_u64("general.alignment"))
        .unwrap_or(DEFAULT_ALIGNMENT);
    if alignment == 0 || alignment > MAX_ALIGNMENT {
        return Err(Error::InvalidAlignment(alignment));
    }

    let tensor_data_offset =
        align_up(reader.pos(), alignment).ok_or(Error::InvalidAlignment(alignment))?;

    // Validate that every recognized tensor fits inside the file.
    for t in &tensors {
        let Some(byte_size) = t.byte_size else {
            // Unknown type → already produced byte_size = None; warn once.
            warn!(
                tensor = %t.name,
                raw_type = t.raw_type,
                "tensor has unsupported GGUF type {}; rsllm-gguf can describe it but cannot decode it",
                t.raw_type,
            );
            continue;
        };
        let abs_offset =
            tensor_data_offset
                .checked_add(t.relative_offset)
                .ok_or(Error::Truncated {
                    pos: tensor_data_offset,
                    need: t.relative_offset,
                    have: 0,
                })?;
        let end = abs_offset.checked_add(byte_size).ok_or(Error::Truncated {
            pos: abs_offset,
            need: byte_size,
            have: file_size.saturating_sub(abs_offset),
        })?;
        if end > file_size {
            return Err(Error::Truncated {
                pos: abs_offset,
                need: byte_size,
                have: file_size.saturating_sub(abs_offset),
            });
        }
    }

    Ok(Parsed {
        version,
        alignment,
        tensor_data_offset,
        metadata,
        tensors,
    })
}

/// Compute the model fingerprint: SHA-256 over the canonicalized tensor
/// layout (sorted by name, hashing `name || 0x00 || raw_type || ndim ||
/// dims... || 0x00` per tensor).
///
/// The fingerprint deliberately **excludes** payload bytes (too expensive to
/// hash 80GB at every load) and absolute file offsets (which can vary
/// between equivalent reconverts of the same model).
fn compute_fingerprint(tensors: &[TensorInfo]) -> [u8; 32] {
    let mut sorted: Vec<&TensorInfo> = tensors.iter().collect();
    sorted.sort_by(|a, b| a.name.cmp(&b.name));

    let mut hasher = Sha256::new();
    for t in sorted {
        hasher.update(t.name.as_bytes());
        hasher.update([0u8]);
        hasher.update(t.raw_type.to_le_bytes());
        // Hash ndim explicitly so that shapes [4, 8] and [4, 8, 1] don't collide.
        let ndim = u32::try_from(t.shape.len()).unwrap_or(u32::MAX);
        hasher.update(ndim.to_le_bytes());
        for &dim in &t.shape {
            hasher.update(dim.to_le_bytes());
        }
        hasher.update([0u8]);
    }
    let result = hasher.finalize();
    let mut out = [0u8; 32];
    out.copy_from_slice(&result);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metadata::ValueType;

    /// Build a minimal valid GGUF v3 byte sequence with the given metadata
    /// and tensor entries. Layout:
    ///
    /// ```text
    /// magic "GGUF" | version u32 | n_tensors u64 | n_kv u64
    /// kv entries: { key str | type u32 | value bytes }*
    /// tensor entries: { name str | ndim u32 | dims u64* | type u32 | rel_offset u64 }*
    /// alignment padding
    /// tensor payloads
    /// ```
    struct GgufBuilder {
        kv: Vec<(String, u32, Vec<u8>)>, // (key, value_type_id, value_bytes)
        tensors: Vec<(String, u32, Vec<u64>, Vec<u8>)>, // (name, ttype, dims, payload)
        alignment: u64,
    }

    impl GgufBuilder {
        fn new() -> Self {
            Self {
                kv: Vec::new(),
                tensors: Vec::new(),
                alignment: 32,
            }
        }

        fn kv_str(mut self, key: &str, value: &str) -> Self {
            let mut value_bytes = (value.len() as u64).to_le_bytes().to_vec();
            value_bytes.extend_from_slice(value.as_bytes());
            self.kv
                .push((key.to_owned(), ValueType::String as u32, value_bytes));
            self
        }

        fn kv_u32(mut self, key: &str, value: u32) -> Self {
            self.kv.push((
                key.to_owned(),
                ValueType::U32 as u32,
                value.to_le_bytes().to_vec(),
            ));
            self
        }

        fn alignment(mut self, alignment: u64) -> Self {
            self.alignment = alignment;
            self
        }

        fn tensor(mut self, name: &str, ttype: u32, dims: Vec<u64>, payload: Vec<u8>) -> Self {
            self.tensors.push((name.to_owned(), ttype, dims, payload));
            self
        }

        fn build(self) -> Vec<u8> {
            let mut out = Vec::new();
            out.extend_from_slice(b"GGUF");
            out.extend_from_slice(&3u32.to_le_bytes()); // version
            out.extend_from_slice(&(self.tensors.len() as u64).to_le_bytes());
            out.extend_from_slice(&(self.kv.len() as u64).to_le_bytes());

            for (key, ttype, value_bytes) in &self.kv {
                out.extend_from_slice(&(key.len() as u64).to_le_bytes());
                out.extend_from_slice(key.as_bytes());
                out.extend_from_slice(&ttype.to_le_bytes());
                out.extend_from_slice(value_bytes);
            }

            // First pass: emit tensor directory entries with relative offsets
            // computed assuming payloads are concatenated in declaration order.
            let mut payload_offsets = Vec::with_capacity(self.tensors.len());
            let mut cursor = 0u64;
            for (_, _, _, payload) in &self.tensors {
                payload_offsets.push(cursor);
                cursor += payload.len() as u64;
            }

            for (idx, (name, ttype, dims, _payload)) in self.tensors.iter().enumerate() {
                out.extend_from_slice(&(name.len() as u64).to_le_bytes());
                out.extend_from_slice(name.as_bytes());
                out.extend_from_slice(&(dims.len() as u32).to_le_bytes());
                for &d in dims {
                    out.extend_from_slice(&d.to_le_bytes());
                }
                out.extend_from_slice(&ttype.to_le_bytes());
                out.extend_from_slice(&payload_offsets[idx].to_le_bytes());
            }

            // Alignment padding.
            let pad = (self.alignment - (out.len() as u64 % self.alignment)) % self.alignment;
            out.extend(std::iter::repeat_n(0u8, pad as usize));

            // Tensor payloads.
            for (_, _, _, payload) in &self.tensors {
                out.extend_from_slice(payload);
            }

            out
        }
    }

    #[test]
    fn open_minimal_empty_file() {
        let bytes = GgufBuilder::new()
            .kv_str("general.name", "test-model")
            .build();
        let file = GgufFile::from_bytes(bytes).unwrap();
        assert_eq!(file.version(), 3);
        assert_eq!(file.alignment(), 32);
        assert_eq!(file.metadata().len(), 1);
        assert_eq!(file.metadata().get_str("general.name"), Some("test-model"));
        assert_eq!(file.tensors().len(), 0);
    }

    #[test]
    fn open_with_two_tensors_zero_copy_payload() {
        let payload_a = vec![1u8, 2, 3, 4, 5, 6, 7, 8]; // 2 f32 elements
        let payload_b = vec![9u8, 10, 11, 12]; // 1 f32 element
        let bytes = GgufBuilder::new()
            .kv_str("general.name", "two-tensor-test")
            .tensor("t_a", 0 /* F32 */, vec![2], payload_a.clone())
            .tensor("t_b", 0 /* F32 */, vec![1], payload_b.clone())
            .build();

        let file = GgufFile::from_bytes(bytes).unwrap();
        assert_eq!(file.tensors().len(), 2);

        let info_a = file.tensor("t_a").expect("t_a present");
        assert_eq!(info_a.elements, 2);
        let bytes_a = file.tensor_bytes(info_a).expect("payload present");
        assert_eq!(bytes_a, payload_a.as_slice());

        let info_b = file.tensor("t_b").expect("t_b present");
        let bytes_b = file.tensor_bytes(info_b).expect("payload present");
        assert_eq!(bytes_b, payload_b.as_slice());
    }

    #[test]
    fn bad_magic_rejected() {
        let mut bytes = b"XXXX".to_vec();
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes()); // n_tensors
        bytes.extend_from_slice(&0u64.to_le_bytes()); // n_kv
        bytes.resize(64, 0);
        match GgufFile::from_bytes(bytes) {
            Err(Error::BadMagic { found }) => assert_eq!(&found, b"XXXX"),
            other => panic!("expected BadMagic, got {other:?}"),
        }
    }

    #[test]
    fn unsupported_version_rejected() {
        let mut bytes = b"GGUF".to_vec();
        bytes.extend_from_slice(&2u32.to_le_bytes()); // v2, unsupported
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.resize(64, 0);
        match GgufFile::from_bytes(bytes) {
            Err(Error::UnsupportedVersion(2)) => {}
            other => panic!("expected UnsupportedVersion(2), got {other:?}"),
        }
    }

    #[test]
    fn too_small_rejected() {
        let bytes = vec![0u8; 10];
        match GgufFile::from_bytes(bytes) {
            Err(Error::TooSmall(10)) => {}
            other => panic!("expected TooSmall(10), got {other:?}"),
        }
    }

    #[test]
    fn alignment_override_via_metadata() {
        // Force alignment = 64 instead of default 32.
        let bytes = GgufBuilder::new()
            .alignment(64)
            .kv_u32("general.alignment", 64)
            .tensor("t", 0, vec![1], vec![1u8, 2, 3, 4])
            .build();

        let file = GgufFile::from_bytes(bytes).unwrap();
        assert_eq!(file.alignment(), 64);
        // tensor_data_offset must be 64-aligned.
        assert_eq!(file.tensor_data_offset() % 64, 0);
    }

    #[test]
    fn fingerprint_is_stable_and_layout_sensitive() {
        let bytes1 = GgufBuilder::new()
            .tensor("a", 0, vec![4], vec![0; 16])
            .tensor("b", 1, vec![2, 2], vec![0; 8])
            .build();
        let bytes2 = GgufBuilder::new()
            // Different declaration order, same tensors.
            .tensor("b", 1, vec![2, 2], vec![0; 8])
            .tensor("a", 0, vec![4], vec![0; 16])
            .build();
        let bytes3 = GgufBuilder::new()
            // Different shape: should produce a different fingerprint.
            .tensor("a", 0, vec![8], vec![0; 32])
            .tensor("b", 1, vec![2, 2], vec![0; 8])
            .build();

        let f1 = *GgufFile::from_bytes(bytes1).unwrap().fingerprint();
        let f2 = *GgufFile::from_bytes(bytes2).unwrap().fingerprint();
        let f3 = *GgufFile::from_bytes(bytes3).unwrap().fingerprint();

        assert_eq!(f1, f2, "fingerprint must be order-independent");
        assert_ne!(f1, f3, "fingerprint must change when shapes differ");
    }

    #[test]
    fn debug_format_includes_fingerprint() {
        let bytes = GgufBuilder::new()
            .tensor("a", 0, vec![4], vec![0; 16])
            .build();
        let file = GgufFile::from_bytes(bytes).unwrap();
        let dbg = format!("{file:?}");
        assert!(dbg.contains("GgufFile"));
        assert!(dbg.contains("version: 3"));
        assert!(dbg.contains("fingerprint"));
    }

    #[test]
    fn align_up_helper() {
        assert_eq!(align_up(0, 32), Some(0));
        assert_eq!(align_up(1, 32), Some(32));
        assert_eq!(align_up(32, 32), Some(32));
        assert_eq!(align_up(33, 32), Some(64));
        assert_eq!(align_up(100, 64), Some(128));
    }

    #[test]
    fn align_up_overflow_returns_none() {
        // u64::MAX - 1 rounded up to a 32-aligned boundary overflows.
        assert_eq!(align_up(u64::MAX - 1, 32), None);
    }

    #[test]
    fn pathological_alignment_rejected() {
        // alignment > MAX_ALIGNMENT must be rejected. The builder embeds
        // general.alignment in metadata.
        let bytes = GgufBuilder::new()
            .kv_u32("general.alignment", 131_072) // 128 KiB > MAX_ALIGNMENT 64 KiB
            .build();
        match GgufFile::from_bytes(bytes) {
            Err(Error::InvalidAlignment(131_072)) => {}
            other => panic!("expected InvalidAlignment(131072), got {other:?}"),
        }
    }

    #[test]
    fn zero_alignment_rejected() {
        let bytes = GgufBuilder::new().kv_u32("general.alignment", 0).build();
        match GgufFile::from_bytes(bytes) {
            Err(Error::InvalidAlignment(0)) => {}
            other => panic!("expected InvalidAlignment(0), got {other:?}"),
        }
    }
}
