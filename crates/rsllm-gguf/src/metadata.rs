//! GGUF metadata (key-value table) types and parser.
//!
//! Format definitions follow the GGUF v3 spec, cross-referenced with
//! `ds4.c:813-1118` (MIT, The ds4.c authors). The 13 metadata value types
//! match the `GGUF_VALUE_*` enum in `ds4.c:814-828`.

use std::collections::BTreeMap;

use crate::error::Error;
use crate::reader::Reader;

/// Maximum recursion depth for nested metadata arrays.
///
/// Matches the limit in `ds4.c:933`. Protects against pathological GGUF
/// inputs that could otherwise blow the parser stack.
const MAX_ARRAY_DEPTH: u32 = 8;

/// The 13 GGUF metadata value types, in spec order.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(missing_docs)]
pub enum ValueType {
    U8 = 0,
    I8 = 1,
    U16 = 2,
    I16 = 3,
    U32 = 4,
    I32 = 5,
    F32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    U64 = 10,
    I64 = 11,
    F64 = 12,
}

impl ValueType {
    /// Convert a raw `u32` (as stored in the file) to a typed [`ValueType`].
    pub fn from_u32(v: u32) -> Result<Self, Error> {
        match v {
            0 => Ok(Self::U8),
            1 => Ok(Self::I8),
            2 => Ok(Self::U16),
            3 => Ok(Self::I16),
            4 => Ok(Self::U32),
            5 => Ok(Self::I32),
            6 => Ok(Self::F32),
            7 => Ok(Self::Bool),
            8 => Ok(Self::String),
            9 => Ok(Self::Array),
            10 => Ok(Self::U64),
            11 => Ok(Self::I64),
            12 => Ok(Self::F64),
            other => Err(Error::UnknownValueType(other)),
        }
    }

    /// The fixed byte size of a scalar value of this type.
    ///
    /// Returns `None` for `String` and `Array`, which are variable-length.
    pub fn scalar_size(self) -> Option<u64> {
        match self {
            Self::U8 | Self::I8 | Self::Bool => Some(1),
            Self::U16 | Self::I16 => Some(2),
            Self::U32 | Self::I32 | Self::F32 => Some(4),
            Self::U64 | Self::I64 | Self::F64 => Some(8),
            Self::String | Self::Array => None,
        }
    }
}

/// A scalar or array value attached to a GGUF metadata key.
#[derive(Debug, Clone, PartialEq)]
#[allow(missing_docs)]
pub enum Value {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    U64(u64),
    I64(i64),
    F64(f64),
    Array(Array),
}

impl Value {
    /// The type discriminant of this value.
    pub fn ty(&self) -> ValueType {
        match self {
            Self::U8(_) => ValueType::U8,
            Self::I8(_) => ValueType::I8,
            Self::U16(_) => ValueType::U16,
            Self::I16(_) => ValueType::I16,
            Self::U32(_) => ValueType::U32,
            Self::I32(_) => ValueType::I32,
            Self::F32(_) => ValueType::F32,
            Self::Bool(_) => ValueType::Bool,
            Self::String(_) => ValueType::String,
            Self::U64(_) => ValueType::U64,
            Self::I64(_) => ValueType::I64,
            Self::F64(_) => ValueType::F64,
            Self::Array(_) => ValueType::Array,
        }
    }

    /// Borrow this value as a string if it is one.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(s) => Some(s.as_str()),
            _ => None,
        }
    }

    /// Read this value as `u32`, widening narrower unsigned types.
    ///
    /// Returns `None` if the value is signed, float, string, array, or out of range.
    pub fn as_u32(&self) -> Option<u32> {
        match self {
            Self::U8(v) => Some(u32::from(*v)),
            Self::U16(v) => Some(u32::from(*v)),
            Self::U32(v) => Some(*v),
            Self::U64(v) => u32::try_from(*v).ok(),
            _ => None,
        }
    }

    /// Read this value as `u64`, widening narrower unsigned types.
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Self::U8(v) => Some(u64::from(*v)),
            Self::U16(v) => Some(u64::from(*v)),
            Self::U32(v) => Some(u64::from(*v)),
            Self::U64(v) => Some(*v),
            _ => None,
        }
    }

    /// Read this value as `f32` (only exact match — no `f64 -> f32` widening).
    pub fn as_f32(&self) -> Option<f32> {
        if let Self::F32(v) = self {
            Some(*v)
        } else {
            None
        }
    }

    /// Read this value as `bool`.
    pub fn as_bool(&self) -> Option<bool> {
        if let Self::Bool(v) = self {
            Some(*v)
        } else {
            None
        }
    }
}

/// A homogeneously-typed array of values.
///
/// Arrays of arrays are valid per the GGUF spec but are uncommon in practice;
/// they require recursive parsing bounded by [`MAX_ARRAY_DEPTH`].
#[derive(Debug, Clone, PartialEq)]
#[allow(missing_docs)]
pub enum Array {
    U8(Vec<u8>),
    I8(Vec<i8>),
    U16(Vec<u16>),
    I16(Vec<i16>),
    U32(Vec<u32>),
    I32(Vec<i32>),
    F32(Vec<f32>),
    Bool(Vec<bool>),
    String(Vec<String>),
    U64(Vec<u64>),
    I64(Vec<i64>),
    F64(Vec<f64>),
    /// Nested arrays. The outer `Vec` element is one inner array.
    Array(Vec<Array>),
}

impl Array {
    /// The element type of this array.
    pub fn item_type(&self) -> ValueType {
        match self {
            Self::U8(_) => ValueType::U8,
            Self::I8(_) => ValueType::I8,
            Self::U16(_) => ValueType::U16,
            Self::I16(_) => ValueType::I16,
            Self::U32(_) => ValueType::U32,
            Self::I32(_) => ValueType::I32,
            Self::F32(_) => ValueType::F32,
            Self::Bool(_) => ValueType::Bool,
            Self::String(_) => ValueType::String,
            Self::U64(_) => ValueType::U64,
            Self::I64(_) => ValueType::I64,
            Self::F64(_) => ValueType::F64,
            Self::Array(_) => ValueType::Array,
        }
    }

    /// Number of elements in this array.
    pub fn len(&self) -> usize {
        match self {
            Self::U8(v) => v.len(),
            Self::I8(v) => v.len(),
            Self::U16(v) => v.len(),
            Self::I16(v) => v.len(),
            Self::U32(v) => v.len(),
            Self::I32(v) => v.len(),
            Self::F32(v) => v.len(),
            Self::Bool(v) => v.len(),
            Self::String(v) => v.len(),
            Self::U64(v) => v.len(),
            Self::I64(v) => v.len(),
            Self::F64(v) => v.len(),
            Self::Array(v) => v.len(),
        }
    }

    /// Whether this array is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// A parsed GGUF metadata table.
///
/// Keys are stored as owned `String`s. Values are also owned (eager parse);
/// for large arrays (e.g. a 128k-entry tokenizer vocabulary) this trades
/// memory for simpler API. Optimisation paths (lazy array views) are deferred
/// to a later version.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Metadata {
    entries: BTreeMap<String, Value>,
}

impl Metadata {
    /// Construct an empty metadata table.
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of key-value pairs in this table.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the table is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Look up a value by key.
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.entries.get(key)
    }

    /// Iterate over all `(key, value)` pairs in lexicographic order.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &Value)> {
        self.entries.iter().map(|(k, v)| (k.as_str(), v))
    }

    /// Iterate over all keys in lexicographic order.
    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.entries.keys().map(String::as_str)
    }

    /// Convenience: get a value as `&str`.
    pub fn get_str(&self, key: &str) -> Option<&str> {
        self.get(key)?.as_str()
    }

    /// Convenience: get a value as `u32` (with widening from narrower types).
    pub fn get_u32(&self, key: &str) -> Option<u32> {
        self.get(key)?.as_u32()
    }

    /// Convenience: get a value as `u64` (with widening from narrower types).
    pub fn get_u64(&self, key: &str) -> Option<u64> {
        self.get(key)?.as_u64()
    }

    /// Convenience: get a value as `f32`.
    pub fn get_f32(&self, key: &str) -> Option<f32> {
        self.get(key)?.as_f32()
    }

    /// Convenience: get a value as `bool`.
    pub fn get_bool(&self, key: &str) -> Option<bool> {
        self.get(key)?.as_bool()
    }

    /// Parse `n_kv` metadata entries from the current reader position.
    ///
    /// Mirrors `parse_metadata()` in `ds4.c:1093-1118`, but eagerly decodes
    /// values into owned Rust types instead of recording offsets.
    pub(crate) fn parse(reader: &mut Reader<'_>, n_kv: u64) -> Result<Self, Error> {
        let mut entries = BTreeMap::new();
        for _ in 0..n_kv {
            let key = reader.read_str()?.to_owned();
            let ty = ValueType::from_u32(reader.read_u32_le()?)?;
            let value = parse_value(reader, ty)?;
            entries.insert(key, value);
        }
        Ok(Self { entries })
    }
}

/// Parse a single value at the top level of a KV entry (depth 0).
///
/// Arrays start their depth tracking inside [`parse_array`], so this helper
/// does not take a `depth` parameter.
fn parse_value(reader: &mut Reader<'_>, ty: ValueType) -> Result<Value, Error> {
    Ok(match ty {
        ValueType::U8 => Value::U8(reader.read_u8()?),
        ValueType::I8 => Value::I8(reader.read_i8()?),
        ValueType::U16 => Value::U16(reader.read_u16_le()?),
        ValueType::I16 => Value::I16(reader.read_i16_le()?),
        ValueType::U32 => Value::U32(reader.read_u32_le()?),
        ValueType::I32 => Value::I32(reader.read_i32_le()?),
        ValueType::F32 => Value::F32(reader.read_f32_le()?),
        ValueType::Bool => Value::Bool(reader.read_bool()?),
        ValueType::String => Value::String(reader.read_str()?.to_owned()),
        ValueType::U64 => Value::U64(reader.read_u64_le()?),
        ValueType::I64 => Value::I64(reader.read_i64_le()?),
        ValueType::F64 => Value::F64(reader.read_f64_le()?),
        ValueType::Array => Value::Array(parse_array(reader, 1)?),
    })
}

/// Parse an array given that the leading `value_type` byte has already been
/// consumed (and was Array). Reads inner type and length, then the elements.
///
/// `depth` is the array nesting depth (1 = outermost array). Bumps by one
/// for each nested array.
fn parse_array(reader: &mut Reader<'_>, depth: u32) -> Result<Array, Error> {
    if depth > MAX_ARRAY_DEPTH {
        return Err(Error::NestingTooDeep);
    }
    let item_ty = ValueType::from_u32(reader.read_u32_le()?)?;
    let len = reader.read_u64_le()?;

    // Defensive: detect impossibly large arrays before we try to allocate.
    if let Some(item_size) = item_ty.scalar_size() {
        if item_size != 0 && len > u64::MAX / item_size {
            return Err(Error::ArrayTooLarge { len, item_size });
        }
    }

    let len_usize = usize::try_from(len).map_err(|_| Error::ArrayTooLarge {
        len,
        item_size: item_ty.scalar_size().unwrap_or(0),
    })?;

    Ok(match item_ty {
        ValueType::U8 => {
            let mut out = Vec::with_capacity(len_usize);
            for _ in 0..len_usize {
                out.push(reader.read_u8()?);
            }
            Array::U8(out)
        }
        ValueType::I8 => {
            let mut out = Vec::with_capacity(len_usize);
            for _ in 0..len_usize {
                out.push(reader.read_i8()?);
            }
            Array::I8(out)
        }
        ValueType::U16 => {
            let mut out = Vec::with_capacity(len_usize);
            for _ in 0..len_usize {
                out.push(reader.read_u16_le()?);
            }
            Array::U16(out)
        }
        ValueType::I16 => {
            let mut out = Vec::with_capacity(len_usize);
            for _ in 0..len_usize {
                out.push(reader.read_i16_le()?);
            }
            Array::I16(out)
        }
        ValueType::U32 => {
            let mut out = Vec::with_capacity(len_usize);
            for _ in 0..len_usize {
                out.push(reader.read_u32_le()?);
            }
            Array::U32(out)
        }
        ValueType::I32 => {
            let mut out = Vec::with_capacity(len_usize);
            for _ in 0..len_usize {
                out.push(reader.read_i32_le()?);
            }
            Array::I32(out)
        }
        ValueType::F32 => {
            let mut out = Vec::with_capacity(len_usize);
            for _ in 0..len_usize {
                out.push(reader.read_f32_le()?);
            }
            Array::F32(out)
        }
        ValueType::Bool => {
            let mut out = Vec::with_capacity(len_usize);
            for _ in 0..len_usize {
                out.push(reader.read_bool()?);
            }
            Array::Bool(out)
        }
        ValueType::String => {
            let mut out = Vec::with_capacity(len_usize);
            for _ in 0..len_usize {
                out.push(reader.read_str()?.to_owned());
            }
            Array::String(out)
        }
        ValueType::U64 => {
            let mut out = Vec::with_capacity(len_usize);
            for _ in 0..len_usize {
                out.push(reader.read_u64_le()?);
            }
            Array::U64(out)
        }
        ValueType::I64 => {
            let mut out = Vec::with_capacity(len_usize);
            for _ in 0..len_usize {
                out.push(reader.read_i64_le()?);
            }
            Array::I64(out)
        }
        ValueType::F64 => {
            let mut out = Vec::with_capacity(len_usize);
            for _ in 0..len_usize {
                out.push(reader.read_f64_le()?);
            }
            Array::F64(out)
        }
        ValueType::Array => {
            let mut out = Vec::with_capacity(len_usize);
            for _ in 0..len_usize {
                out.push(parse_array(reader, depth + 1)?);
            }
            Array::Array(out)
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a length-prefixed string blob: 8-byte LE length + raw bytes.
    fn pack_str(s: &str) -> Vec<u8> {
        let mut out = (s.len() as u64).to_le_bytes().to_vec();
        out.extend_from_slice(s.as_bytes());
        out
    }

    /// Encode a single KV entry: key string + value type u32 + value bytes.
    fn pack_kv(key: &str, value_type: u32, value_bytes: &[u8]) -> Vec<u8> {
        let mut out = pack_str(key);
        out.extend_from_slice(&value_type.to_le_bytes());
        out.extend_from_slice(value_bytes);
        out
    }

    #[test]
    fn value_type_roundtrip() {
        for raw in 0u32..=12 {
            let ty = ValueType::from_u32(raw).unwrap();
            assert_eq!(ty as u32, raw);
        }
    }

    #[test]
    fn unknown_value_type_errors() {
        assert!(matches!(
            ValueType::from_u32(99),
            Err(Error::UnknownValueType(99))
        ));
    }

    #[test]
    fn scalar_size_lookup() {
        assert_eq!(ValueType::U8.scalar_size(), Some(1));
        assert_eq!(ValueType::Bool.scalar_size(), Some(1));
        assert_eq!(ValueType::U32.scalar_size(), Some(4));
        assert_eq!(ValueType::F32.scalar_size(), Some(4));
        assert_eq!(ValueType::U64.scalar_size(), Some(8));
        assert_eq!(ValueType::F64.scalar_size(), Some(8));
        assert_eq!(ValueType::String.scalar_size(), None);
        assert_eq!(ValueType::Array.scalar_size(), None);
    }

    #[test]
    fn parse_single_u32_kv() {
        // 1 entry: key "answer", type u32 (=4), value 42
        let mut data = pack_kv("answer", 4, &42u32.to_le_bytes());
        // n_kv read separately by caller, so we just exercise Metadata::parse
        // directly. Prepend nothing.
        let mut reader = Reader::new(&data);
        let md = Metadata::parse(&mut reader, 1).unwrap();
        assert_eq!(md.len(), 1);
        assert_eq!(md.get_u32("answer"), Some(42));
        // Force `data` to live long enough for the borrow.
        let _ = &mut data;
    }

    #[test]
    fn parse_string_kv() {
        let mut data = pack_kv("general.name", 8, &pack_str("Llama 3.1"));
        let mut reader = Reader::new(&data);
        let md = Metadata::parse(&mut reader, 1).unwrap();
        assert_eq!(md.get_str("general.name"), Some("Llama 3.1"));
        let _ = &mut data;
    }

    #[test]
    fn parse_bool_kv() {
        let data = pack_kv("tokenizer.add_bos", 7, &[1u8]);
        let mut reader = Reader::new(&data);
        let md = Metadata::parse(&mut reader, 1).unwrap();
        assert_eq!(md.get_bool("tokenizer.add_bos"), Some(true));
    }

    #[test]
    fn parse_f32_kv() {
        let data = pack_kv("rope.freq_base", 6, &500_000.0f32.to_le_bytes());
        let mut reader = Reader::new(&data);
        let md = Metadata::parse(&mut reader, 1).unwrap();
        assert_eq!(md.get_f32("rope.freq_base"), Some(500_000.0));
    }

    #[test]
    fn parse_array_of_u32() {
        // value_type = Array (9); inner_type = U32 (4); len = 3; [1, 2, 3]
        let mut value_bytes = 4u32.to_le_bytes().to_vec(); // inner type U32
        value_bytes.extend_from_slice(&3u64.to_le_bytes()); // length 3
        value_bytes.extend_from_slice(&1u32.to_le_bytes());
        value_bytes.extend_from_slice(&2u32.to_le_bytes());
        value_bytes.extend_from_slice(&3u32.to_le_bytes());
        let data = pack_kv("test.arr", 9, &value_bytes);

        let mut reader = Reader::new(&data);
        let md = Metadata::parse(&mut reader, 1).unwrap();
        match md.get("test.arr").unwrap() {
            Value::Array(Array::U32(v)) => assert_eq!(v, &[1, 2, 3]),
            other => panic!("expected U32 array, got {other:?}"),
        }
    }

    #[test]
    fn parse_array_of_strings_for_tokenizer() {
        // Three vocab entries, like a tokenizer vocabulary fragment.
        // value_type = Array (9); inner_type = String (8); len = 3
        let mut value_bytes = 8u32.to_le_bytes().to_vec();
        value_bytes.extend_from_slice(&3u64.to_le_bytes());
        value_bytes.extend(pack_str("<|begin_of_text|>"));
        value_bytes.extend(pack_str("hello"));
        value_bytes.extend(pack_str("world"));
        let data = pack_kv("tokenizer.ggml.tokens", 9, &value_bytes);

        let mut reader = Reader::new(&data);
        let md = Metadata::parse(&mut reader, 1).unwrap();
        match md.get("tokenizer.ggml.tokens").unwrap() {
            Value::Array(Array::String(v)) => {
                assert_eq!(v.len(), 3);
                assert_eq!(v[0], "<|begin_of_text|>");
                assert_eq!(v[1], "hello");
                assert_eq!(v[2], "world");
            }
            other => panic!("expected String array, got {other:?}"),
        }
    }

    #[test]
    fn unknown_value_type_in_kv_errors() {
        // value_type = 99 (unknown)
        let mut data = pack_str("bad.key");
        data.extend_from_slice(&99u32.to_le_bytes());
        let mut reader = Reader::new(&data);
        match Metadata::parse(&mut reader, 1) {
            Err(Error::UnknownValueType(99)) => {}
            other => panic!("expected UnknownValueType(99), got {other:?}"),
        }
    }

    #[test]
    fn deeply_nested_arrays_rejected() {
        // Build an array nested 10 deep — exceeds MAX_ARRAY_DEPTH = 8.
        // Each level: u32 inner_type=Array (9), u64 length=1.
        let mut value_bytes: Vec<u8> = Vec::new();
        for _ in 0..10 {
            value_bytes.extend_from_slice(&9u32.to_le_bytes()); // inner = Array
            value_bytes.extend_from_slice(&1u64.to_le_bytes()); // length 1
        }
        // Innermost: an empty U32 array
        value_bytes.extend_from_slice(&4u32.to_le_bytes());
        value_bytes.extend_from_slice(&0u64.to_le_bytes());

        let data = pack_kv("deep.array", 9, &value_bytes);
        let mut reader = Reader::new(&data);
        match Metadata::parse(&mut reader, 1) {
            Err(Error::NestingTooDeep) => {}
            other => panic!("expected NestingTooDeep, got {other:?}"),
        }
    }

    #[test]
    fn convenience_getters_widen_unsigned() {
        // Stored as u16, asked as u32 — should widen.
        let data = pack_kv("small.value", 2, &(7u16).to_le_bytes());
        let mut reader = Reader::new(&data);
        let md = Metadata::parse(&mut reader, 1).unwrap();
        assert_eq!(md.get_u32("small.value"), Some(7));
        assert_eq!(md.get_u64("small.value"), Some(7));
    }

    #[test]
    fn missing_key_returns_none() {
        let md = Metadata::new();
        assert!(md.get("nonexistent").is_none());
        assert_eq!(md.get_u32("nonexistent"), None);
    }
}
