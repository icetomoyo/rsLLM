//! Low-level byte-cursor reader over an in-memory slice.
//!
//! Modelled after the `ds4_cursor` API in `ds4.c` (MIT, The ds4.c authors),
//! adapted to Rust borrowing rules. The reader does not allocate; all reads
//! are zero-copy views into the underlying slice.

use crate::error::Error;

/// A position-tracking cursor over a fixed-size byte slice.
///
/// All numeric reads are little-endian (the GGUF format is LE by spec). The
/// reader never panics on out-of-bounds: every read returns
/// [`Error::Truncated`] instead.
pub(crate) struct Reader<'a> {
    data: &'a [u8],
    pos: u64,
}

impl<'a> Reader<'a> {
    /// Create a new reader positioned at the start of `data`.
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    /// Current read position, in bytes from the start of the slice.
    pub fn pos(&self) -> u64 {
        self.pos
    }

    /// Number of bytes remaining from the current position to the end.
    pub fn remaining(&self) -> u64 {
        (self.data.len() as u64).saturating_sub(self.pos)
    }

    /// Move the cursor to absolute offset `pos`.
    ///
    /// Returns [`Error::Truncated`] if `pos` is past the end of the slice.
    #[allow(dead_code)] // used by Phase 4 dequant paths (FEATURE_002)
    pub fn seek(&mut self, pos: u64) -> Result<(), Error> {
        if pos > self.data.len() as u64 {
            return Err(Error::Truncated {
                pos: self.pos,
                need: pos.saturating_sub(self.pos),
                have: self.remaining(),
            });
        }
        self.pos = pos;
        Ok(())
    }

    /// Read `n` bytes and return a zero-copy slice into the underlying data.
    pub fn read_bytes(&mut self, n: u64) -> Result<&'a [u8], Error> {
        let start = self.pos;
        let end = start.checked_add(n).ok_or(Error::Truncated {
            pos: start,
            need: n,
            have: self.remaining(),
        })?;
        if end > self.data.len() as u64 {
            return Err(Error::Truncated {
                pos: start,
                need: n,
                have: self.remaining(),
            });
        }
        self.pos = end;
        Ok(&self.data[start as usize..end as usize])
    }

    /// Advance the cursor by `n` bytes without keeping a reference.
    #[allow(dead_code)] // used by Phase 4 dequant paths (FEATURE_002)
    pub fn skip(&mut self, n: u64) -> Result<(), Error> {
        let _ = self.read_bytes(n)?;
        Ok(())
    }

    /// Read a single unsigned byte.
    pub fn read_u8(&mut self) -> Result<u8, Error> {
        Ok(self.read_bytes(1)?[0])
    }

    /// Read a signed byte.
    pub fn read_i8(&mut self) -> Result<i8, Error> {
        Ok(self.read_u8()? as i8)
    }

    /// Read a little-endian `u16`.
    pub fn read_u16_le(&mut self) -> Result<u16, Error> {
        let bytes = self.read_bytes(2)?;
        Ok(u16::from_le_bytes([bytes[0], bytes[1]]))
    }

    /// Read a little-endian `i16`.
    pub fn read_i16_le(&mut self) -> Result<i16, Error> {
        Ok(self.read_u16_le()? as i16)
    }

    /// Read a little-endian `u32`.
    pub fn read_u32_le(&mut self) -> Result<u32, Error> {
        let bytes = self.read_bytes(4)?;
        Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    /// Read a little-endian `i32`.
    pub fn read_i32_le(&mut self) -> Result<i32, Error> {
        Ok(self.read_u32_le()? as i32)
    }

    /// Read a little-endian IEEE-754 `f32`.
    pub fn read_f32_le(&mut self) -> Result<f32, Error> {
        Ok(f32::from_bits(self.read_u32_le()?))
    }

    /// Read a little-endian `u64`.
    pub fn read_u64_le(&mut self) -> Result<u64, Error> {
        let bytes = self.read_bytes(8)?;
        Ok(u64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]))
    }

    /// Read a little-endian `i64`.
    pub fn read_i64_le(&mut self) -> Result<i64, Error> {
        Ok(self.read_u64_le()? as i64)
    }

    /// Read a little-endian IEEE-754 `f64`.
    pub fn read_f64_le(&mut self) -> Result<f64, Error> {
        Ok(f64::from_bits(self.read_u64_le()?))
    }

    /// Read a GGUF-style bool: a single byte. The GGUF spec encodes a bool
    /// as exactly `0` (false) or `1` (true); any other byte is rejected with
    /// [`Error::InvalidBool`] so a crafted file can't smuggle out-of-band
    /// data through a field that downstream code treats as a normal bool.
    pub fn read_bool(&mut self) -> Result<bool, Error> {
        match self.read_u8()? {
            0 => Ok(false),
            1 => Ok(true),
            raw => Err(Error::InvalidBool(raw)),
        }
    }

    /// Read a GGUF length-prefixed string: `u64 length` followed by `length`
    /// UTF-8 bytes. Returns a borrowed `&str` into the underlying buffer.
    pub fn read_str(&mut self) -> Result<&'a str, Error> {
        let str_start = self.pos;
        let len = self.read_u64_le()?;
        let bytes = self.read_bytes(len)?;
        std::str::from_utf8(bytes).map_err(|_| Error::InvalidUtf8(str_start))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_u32_le_basic() {
        let mut r = Reader::new(&[0x78, 0x56, 0x34, 0x12]);
        assert_eq!(r.read_u32_le().unwrap(), 0x1234_5678);
        assert_eq!(r.pos(), 4);
        assert_eq!(r.remaining(), 0);
    }

    #[test]
    fn read_u64_le_basic() {
        let mut r = Reader::new(&[1, 2, 3, 4, 5, 6, 7, 8]);
        let v = r.read_u64_le().unwrap();
        assert_eq!(v, 0x0807_0605_0403_0201);
    }

    #[test]
    fn read_str_basic() {
        // length = 5, "hello"
        let mut data = vec![5, 0, 0, 0, 0, 0, 0, 0];
        data.extend_from_slice(b"hello");
        let mut r = Reader::new(&data);
        assert_eq!(r.read_str().unwrap(), "hello");
        assert_eq!(r.pos(), 13);
    }

    #[test]
    fn truncated_read_reports_error() {
        let mut r = Reader::new(&[1, 2]);
        match r.read_u32_le() {
            Err(Error::Truncated { pos, need, have }) => {
                assert_eq!(pos, 0);
                assert_eq!(need, 4);
                assert_eq!(have, 2);
            }
            other => panic!("expected Truncated, got {other:?}"),
        }
    }

    #[test]
    fn invalid_utf8_string_reports_error() {
        // length = 2, then invalid UTF-8 bytes
        let mut data = vec![2, 0, 0, 0, 0, 0, 0, 0];
        data.extend_from_slice(&[0xFF, 0xFE]);
        let mut r = Reader::new(&data);
        match r.read_str() {
            Err(Error::InvalidUtf8(off)) => assert_eq!(off, 0),
            other => panic!("expected InvalidUtf8, got {other:?}"),
        }
    }

    #[test]
    fn read_bool_basic() {
        let mut r = Reader::new(&[0, 1]);
        assert!(!r.read_bool().unwrap());
        assert!(r.read_bool().unwrap());
    }

    #[test]
    fn read_bool_rejects_non_canonical_byte() {
        let mut r = Reader::new(&[2]);
        match r.read_bool() {
            Err(Error::InvalidBool(2)) => {}
            other => panic!("expected InvalidBool(2), got {other:?}"),
        }
    }

    #[test]
    fn read_f32_le_basic() {
        // 1.0_f32 in little-endian = 0x3f80_0000
        let mut r = Reader::new(&[0x00, 0x00, 0x80, 0x3f]);
        assert_eq!(r.read_f32_le().unwrap(), 1.0_f32);
    }

    #[test]
    fn seek_and_resume() {
        let mut r = Reader::new(&[0, 0, 0, 0, 0x78, 0x56, 0x34, 0x12]);
        r.seek(4).unwrap();
        assert_eq!(r.read_u32_le().unwrap(), 0x1234_5678);
    }

    #[test]
    fn seek_past_end_errors() {
        let mut r = Reader::new(&[1, 2, 3]);
        assert!(r.seek(99).is_err());
    }
}
