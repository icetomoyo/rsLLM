//! Sliding-window attention raw KV ring buffer.
//!
//! Each layer of DS V4 Flash keeps the most recent `N_SWA = 128`
//! KV-latent rows in a per-layer ring buffer. The structure follows
//! ds4's idiom (`ds4.c:6306-6320`):
//!
//! - While `n_raw < cap_raw`, the ring grows by appending at slot `n_raw`.
//! - Once full, every new write **memmoves** the existing rows down by
//!   one and writes the new row at the last slot. This is an O(cap_raw)
//!   write but keeps the rows in chronological order (slot 0 = oldest,
//!   slot `cap_raw-1` = newest), which makes attention reads trivial
//!   linear sweeps with no index arithmetic.
//!
//! The chronological-shift idiom is intentional — it costs one
//! `memmove` per token at full ring but avoids any modulo / wrap
//! arithmetic in the attention hot loop. For DS V4 Flash's
//! `cap_raw = 128` and `head_dim = 512`, each shift is 127 × 512 × 4 =
//! ~261 KiB — negligible vs the attention compute. ds4 ships exactly
//! this idiom.
//!
//! Ported by reference from `ds4.c:6306-6320` (MIT, The ds4.c authors).
//! Line numbers pinned to ds4 commit `ef0a490` (2026-05-17).

use crate::Error;

use super::shape::{DSV4_HEAD_DIM, DSV4_N_SWA};

/// Per-layer raw KV ring buffer.
///
/// Stores the most recent `cap` KV-latent rows of `head_dim` `f32` each.
/// Rows are kept in chronological order: slot 0 is the oldest row in
/// the window, slot `n_raw - 1` is the newest.
#[derive(Debug)]
pub struct RawSwaRing {
    /// Flat row-major storage `[cap × head_dim]`.
    buf: Vec<f32>,
    /// Number of valid rows in the ring (`0 <= n_raw <= cap`).
    n_raw: usize,
    /// Total capacity in rows.
    cap: usize,
    /// Per-row dimension (kept as a field so the buffer is generic over
    /// any DS V4-compatible MLA head dim — v0.1.0 always uses `DSV4_HEAD_DIM`).
    head_dim: usize,
}

impl RawSwaRing {
    /// Create an empty ring with DS V4 Flash defaults (`cap = N_SWA = 128`,
    /// `head_dim = HEAD_DIM = 512`).
    #[must_use]
    pub fn with_dsv4_defaults() -> Self {
        Self::new(DSV4_N_SWA, DSV4_HEAD_DIM)
    }

    /// Create an empty ring with explicit capacity and row dim. Used in
    /// tests and for tiny-context configurations where `cap < N_SWA`.
    #[must_use]
    pub fn new(cap: usize, head_dim: usize) -> Self {
        debug_assert!(cap > 0);
        debug_assert!(head_dim > 0);
        Self {
            buf: vec![0.0_f32; cap * head_dim],
            n_raw: 0,
            cap,
            head_dim,
        }
    }

    /// Append one KV-latent row to the tail of the ring. Once the ring
    /// is full, the oldest row is shifted out to make room.
    ///
    /// # Errors
    /// Returns [`Error::ShapeMismatch`] if `row.len() != head_dim`.
    pub fn append(&mut self, row: &[f32]) -> Result<(), Error> {
        if row.len() != self.head_dim {
            return Err(Error::ShapeMismatch {
                what: "RawSwaRing::append: row length",
                expected: self.head_dim,
                actual: row.len(),
            });
        }
        let dim = self.head_dim;
        if self.n_raw < self.cap {
            let off = self.n_raw * dim;
            self.buf[off..off + dim].copy_from_slice(row);
            self.n_raw += 1;
        } else {
            // Ring full: shift everything down by one row, write at last.
            // `copy_within` handles the overlap correctly.
            self.buf.copy_within(dim..self.cap * dim, 0);
            let off = (self.cap - 1) * dim;
            self.buf[off..off + dim].copy_from_slice(row);
        }
        Ok(())
    }

    /// Number of valid rows currently in the ring.
    #[must_use]
    pub fn len(&self) -> usize {
        self.n_raw
    }

    /// Total ring capacity.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.cap
    }

    /// Per-row dimension.
    #[must_use]
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Returns `true` if no rows have been written yet.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.n_raw == 0
    }

    /// Returns `true` if the next [`Self::append`] will evict the
    /// oldest row.
    #[must_use]
    pub fn is_full(&self) -> bool {
        self.n_raw == self.cap
    }

    /// Read-only slice of all valid rows in chronological order. Shape
    /// is `[n_raw × head_dim]` row-major. Returns an empty slice if no
    /// rows have been written yet.
    #[must_use]
    pub fn rows(&self) -> &[f32] {
        &self.buf[..self.n_raw * self.head_dim]
    }

    /// Borrow a single row by index (`0 = oldest`).
    ///
    /// # Errors
    /// Returns [`Error::IndexOutOfRange`] if `idx >= len()`.
    pub fn row(&self, idx: usize) -> Result<&[f32], Error> {
        if idx >= self.n_raw {
            return Err(Error::IndexOutOfRange {
                what: "RawSwaRing::row",
                idx,
                len: self.n_raw,
            });
        }
        let off = idx * self.head_dim;
        Ok(&self.buf[off..off + self.head_dim])
    }

    /// Discard all rows. Capacity is preserved.
    pub fn clear(&mut self) {
        self.n_raw = 0;
        // Optional: zero the buffer for hygiene. Skip for v0.1.0 since
        // we never read past `n_raw`. Re-enable if a session leak surfaces.
    }

    /// Trim the ring to keep only the most recent `keep` rows. Used by
    /// `finish_prefill` to align the decode state with the
    /// post-prefill cursor. If `keep >= len()`, the ring is unchanged.
    pub fn truncate_to_recent(&mut self, keep: usize) {
        if keep >= self.n_raw {
            return;
        }
        let drop = self.n_raw - keep;
        let dim = self.head_dim;
        self.buf.copy_within(drop * dim..self.n_raw * dim, 0);
        self.n_raw = keep;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mk_row(seed: f32, dim: usize) -> Vec<f32> {
        (0..dim).map(|i| seed + (i as f32) * 0.01).collect()
    }

    #[test]
    fn new_ring_is_empty() {
        let r = RawSwaRing::new(4, 8);
        assert!(r.is_empty());
        assert_eq!(r.len(), 0);
        assert_eq!(r.capacity(), 4);
        assert!(r.rows().is_empty());
    }

    #[test]
    fn appends_grow_until_capacity() {
        let mut r = RawSwaRing::new(3, 4);
        for s in 0..3 {
            r.append(&mk_row(s as f32, 4)).unwrap();
        }
        assert_eq!(r.len(), 3);
        assert!(r.is_full());
        // Rows preserved in chronological order.
        for s in 0..3 {
            assert_eq!(r.row(s).unwrap()[0], s as f32);
        }
    }

    #[test]
    fn full_ring_shifts_on_append() {
        let mut r = RawSwaRing::new(3, 2);
        for s in 0..5 {
            r.append(&mk_row(s as f32, 2)).unwrap();
        }
        // After 5 appends in a cap=3 ring, rows are seeds 2, 3, 4.
        assert_eq!(r.row(0).unwrap()[0], 2.0);
        assert_eq!(r.row(1).unwrap()[0], 3.0);
        assert_eq!(r.row(2).unwrap()[0], 4.0);
    }

    #[test]
    fn rejects_wrong_row_length() {
        let mut r = RawSwaRing::new(2, 4);
        let err = r.append(&[1.0, 2.0]).unwrap_err();
        assert!(matches!(err, Error::ShapeMismatch { .. }));
    }

    #[test]
    fn row_out_of_range_errors() {
        let mut r = RawSwaRing::new(2, 4);
        r.append(&mk_row(0.0, 4)).unwrap();
        let err = r.row(5).unwrap_err();
        assert!(matches!(err, Error::IndexOutOfRange { .. }));
    }

    #[test]
    fn clear_resets_length() {
        let mut r = RawSwaRing::new(2, 4);
        r.append(&mk_row(0.0, 4)).unwrap();
        r.append(&mk_row(1.0, 4)).unwrap();
        assert_eq!(r.len(), 2);
        r.clear();
        assert_eq!(r.len(), 0);
        assert!(r.is_empty());
    }

    #[test]
    fn truncate_to_recent_keeps_tail() {
        let mut r = RawSwaRing::new(5, 2);
        for s in 0..5 {
            r.append(&mk_row(s as f32, 2)).unwrap();
        }
        r.truncate_to_recent(2);
        assert_eq!(r.len(), 2);
        // Should keep rows 3 and 4.
        assert_eq!(r.row(0).unwrap()[0], 3.0);
        assert_eq!(r.row(1).unwrap()[0], 4.0);
    }

    #[test]
    fn truncate_to_recent_noop_when_keep_geq_len() {
        let mut r = RawSwaRing::new(5, 2);
        for s in 0..3 {
            r.append(&mk_row(s as f32, 2)).unwrap();
        }
        r.truncate_to_recent(5);
        assert_eq!(r.len(), 3);
    }

    #[test]
    fn rows_slice_matches_individual_row_borrows() {
        let mut r = RawSwaRing::new(3, 4);
        for s in 0..3 {
            r.append(&mk_row(s as f32, 4)).unwrap();
        }
        let rows = r.rows();
        for s in 0..3 {
            let row = r.row(s).unwrap();
            assert_eq!(&rows[s * 4..(s + 1) * 4], row);
        }
    }

    #[test]
    fn dsv4_default_ring_has_correct_dimensions() {
        let r = RawSwaRing::with_dsv4_defaults();
        assert_eq!(r.capacity(), DSV4_N_SWA);
        assert_eq!(r.head_dim(), DSV4_HEAD_DIM);
    }
}
