//! Lightweight parallelism wrapper around `rayon`.
//!
//! Matmul-style kernels are parallelized across the output-channel
//! dimension. Centralizing the `rayon` API here keeps the per-kernel
//! code clean and gives one place to swap implementations later
//! (e.g. a custom work-stealing pool tuned for a fixed thread count
//! that mirrors ds4's `pthread_create` loop in `ds4.c:2630-2685`).

use rayon::prelude::*;

/// Run `f(i)` for each `i` in `0..n` in parallel.
///
/// Thread layout: defers to the global `rayon` pool, which defaults to
/// the host's logical CPU count. Override via `RAYON_NUM_THREADS` if a
/// caller wants a specific concurrency level.
///
/// `f` must be `Send + Sync`; it cannot mutate shared state without
/// going through a synchronization primitive. The typical matmul
/// pattern partitions the output buffer into per-row slices using
/// `par_chunks_mut` directly (see [`for_each_row_mut`]).
pub fn parallel_for<F>(n: usize, f: F)
where
    F: Fn(usize) + Send + Sync,
{
    (0..n).into_par_iter().for_each(f);
}

/// Run `f(row_idx, &mut row_slice)` for each row of an `n_rows × row_len`
/// flat row-major output buffer in parallel. Each thread owns its row
/// exclusively, so `f` may write freely without locks.
///
/// This is the kernel-author-facing primitive for matmul / quantize-batch
/// loops where each output row is produced independently.
///
/// # Panics
/// Panics if `out.len()` is not divisible by `row_len`.
pub fn for_each_row_mut<F>(out: &mut [f32], row_len: usize, f: F)
where
    F: Fn(usize, &mut [f32]) + Send + Sync,
{
    // Reject zero row length first — otherwise `is_multiple_of(0)` below
    // would itself panic with "divide by zero" inside the standard
    // library, eating the more useful diagnostic.
    assert!(row_len > 0, "for_each_row_mut: row_len must be non-zero");
    assert!(
        out.len().is_multiple_of(row_len),
        "for_each_row_mut: out len {} not divisible by row_len {}",
        out.len(),
        row_len
    );
    out.par_chunks_mut(row_len)
        .enumerate()
        .for_each(|(i, row)| f(i, row));
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn parallel_for_runs_every_index() {
        let counter = AtomicUsize::new(0);
        parallel_for(1000, |_| {
            counter.fetch_add(1, Ordering::Relaxed);
        });
        assert_eq!(counter.load(Ordering::Relaxed), 1000);
    }

    #[test]
    fn for_each_row_mut_writes_distinct_rows() {
        let mut out = vec![0.0_f32; 4 * 8];
        for_each_row_mut(&mut out, 8, |row_idx, row| {
            for v in row {
                *v = row_idx as f32;
            }
        });
        for r in 0..4 {
            for c in 0..8 {
                assert_eq!(out[r * 8 + c], r as f32);
            }
        }
    }

    #[test]
    #[should_panic]
    fn for_each_row_mut_rejects_misaligned() {
        let mut out = vec![0.0_f32; 7];
        for_each_row_mut(&mut out, 3, |_, _| {});
    }

    #[test]
    #[should_panic(expected = "row_len must be non-zero")]
    fn for_each_row_mut_rejects_zero_row_len() {
        let mut out: Vec<f32> = Vec::new();
        for_each_row_mut(&mut out, 0, |_, _| {});
    }
}
