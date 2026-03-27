//! Lightweight parallel helpers for ParadoxPE hot paths.
//!
//! This module intentionally avoids `rayon` and keeps execution deterministic:
//! - chunked scoped threads over contiguous slices
//! - deterministic merge order for collect/filter-map operations
//! - sequential fallback for small workloads

use std::thread;

/// Return the logical worker count available to this process.
#[inline]
pub fn worker_count() -> usize {
    thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .max(1)
}

#[inline]
fn should_parallelize(total_items: usize, min_items_per_worker: usize) -> bool {
    let workers = worker_count();
    workers > 1 && total_items >= min_items_per_worker.max(1).saturating_mul(2)
}

#[inline]
fn chunk_len(total_items: usize, min_items_per_worker: usize) -> usize {
    let workers = worker_count().max(1);
    total_items
        .div_ceil(workers)
        .max(min_items_per_worker.max(1))
        .max(1)
}

/// Parallel-for over a mutable slice with stable global index.
pub fn for_each_mut_indexed<T, F>(slice: &mut [T], min_items_per_worker: usize, f: F)
where
    T: Send,
    F: Fn(usize, &mut T) + Sync,
{
    if !should_parallelize(slice.len(), min_items_per_worker) {
        for (index, item) in slice.iter_mut().enumerate() {
            f(index, item);
        }
        return;
    }

    let chunk = chunk_len(slice.len(), min_items_per_worker);
    thread::scope(|scope| {
        for (chunk_index, chunk_slice) in slice.chunks_mut(chunk).enumerate() {
            let base = chunk_index * chunk;
            let f_ref = &f;
            scope.spawn(move || {
                for (offset, item) in chunk_slice.iter_mut().enumerate() {
                    f_ref(base + offset, item);
                }
            });
        }
    });
}

/// Parallel-for over `0..len` with deterministic chunk order.
pub fn for_each_index<F>(len: usize, min_items_per_worker: usize, f: F)
where
    F: Fn(usize) + Sync,
{
    if !should_parallelize(len, min_items_per_worker) {
        for index in 0..len {
            f(index);
        }
        return;
    }

    let chunk = chunk_len(len, min_items_per_worker);
    thread::scope(|scope| {
        for start in (0..len).step_by(chunk) {
            let end = (start + chunk).min(len);
            let f_ref = &f;
            scope.spawn(move || {
                for index in start..end {
                    f_ref(index);
                }
            });
        }
    });
}

/// Collect values with a parallel filter-map over a read-only slice.
///
/// Output order is deterministic and follows input order by chunk.
pub fn collect_filter_map<T, U, F>(input: &[T], min_items_per_worker: usize, map: F) -> Vec<U>
where
    T: Sync,
    U: Send,
    F: Fn(&T) -> Option<U> + Sync,
{
    if !should_parallelize(input.len(), min_items_per_worker) {
        let mut out = Vec::with_capacity(input.len() / 2);
        for item in input {
            if let Some(mapped) = map(item) {
                out.push(mapped);
            }
        }
        return out;
    }

    let chunk = chunk_len(input.len(), min_items_per_worker);
    let mut chunk_outputs: Vec<Vec<U>> = Vec::new();
    thread::scope(|scope| {
        let mut handles = Vec::new();
        for chunk_slice in input.chunks(chunk) {
            let map_ref = &map;
            handles.push(scope.spawn(move || {
                let mut local = Vec::with_capacity(chunk_slice.len() / 2);
                for item in chunk_slice {
                    if let Some(mapped) = map_ref(item) {
                        local.push(mapped);
                    }
                }
                local
            }));
        }
        chunk_outputs.reserve(handles.len());
        for handle in handles {
            chunk_outputs.push(
                handle
                    .join()
                    .expect("ParadoxPE parallel collect worker panicked"),
            );
        }
    });

    let total = chunk_outputs.iter().map(Vec::len).sum::<usize>();
    let mut out = Vec::with_capacity(total);
    for mut local in chunk_outputs {
        out.append(&mut local);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn for_each_mut_indexed_visits_every_index_once() {
        let mut data = vec![0usize; 4096];
        for_each_mut_indexed(&mut data, 64, |index, item| {
            *item = index + 1;
        });
        for (index, value) in data.iter().enumerate() {
            assert_eq!(*value, index + 1);
        }
    }

    #[test]
    fn collect_filter_map_keeps_chunk_deterministic_order() {
        let data = (0u32..2048).collect::<Vec<_>>();
        let out = collect_filter_map(
            &data,
            64,
            |value| {
                if value % 3 == 0 {
                    Some(*value)
                } else {
                    None
                }
            },
        );
        let expected = data
            .iter()
            .copied()
            .filter(|value| value % 3 == 0)
            .collect::<Vec<_>>();
        assert_eq!(out, expected);
    }

    #[test]
    fn collect_filter_map_runs_map_on_all_items() {
        let data = (0u32..1024).collect::<Vec<_>>();
        let calls = AtomicUsize::new(0);
        let _ = collect_filter_map(&data, 32, |value| {
            calls.fetch_add(1, Ordering::Relaxed);
            Some(*value)
        });
        assert_eq!(calls.load(Ordering::Relaxed), data.len());
    }
}
