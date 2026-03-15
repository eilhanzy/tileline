//! Parallel broadphase candidate generation for ParadoxPE.
//!
//! The current implementation uses a cache-friendly sweep-and-prune pass over body AABBs. It is
//! designed to run inside Tileline's CPU tasking environment and uses Rayon-style parallel shard
//! scans without allocating in the hot `rebuild_pairs_parallel` loop.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use rayon::prelude::*;

use crate::body::Aabb;
use crate::handle::BodyHandle;
use crate::storage::BodyRegistry;

/// Broadphase configuration tuned for shard-parallel sweep-and-prune.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BroadphaseConfig {
    /// Number of sorted bodies processed by one parallel shard.
    pub chunk_size: usize,
    /// Maximum merged candidate pair count stored without reallocating.
    pub max_candidate_pairs: usize,
    /// Minimum reserved pair slots per shard.
    pub shard_pair_reserve: usize,
}

impl Default for BroadphaseConfig {
    fn default() -> Self {
        Self {
            chunk_size: 128,
            max_candidate_pairs: 16_384,
            shard_pair_reserve: 512,
        }
    }
}

/// Broadphase execution statistics for profiling/diagnostics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct BroadphaseStats {
    pub body_count: usize,
    pub shard_count: usize,
    pub candidate_pairs: usize,
    pub dropped_pairs: usize,
    pub overflowed: bool,
}

/// Reusable broadphase pipeline with preallocated shard-local and merged pair buffers.
#[derive(Debug, Clone)]
pub struct BroadphasePipeline {
    config: BroadphaseConfig,
    sorted_dense: Vec<usize>,
    shard_pairs: Vec<Vec<(BodyHandle, BodyHandle)>>,
    merged_pairs: Vec<(BodyHandle, BodyHandle)>,
    stats: BroadphaseStats,
}

impl BroadphasePipeline {
    pub fn new(config: BroadphaseConfig) -> Self {
        Self {
            config,
            sorted_dense: Vec::new(),
            shard_pairs: Vec::new(),
            merged_pairs: Vec::new(),
            stats: BroadphaseStats::default(),
        }
    }

    pub fn config(&self) -> &BroadphaseConfig {
        &self.config
    }

    pub fn stats(&self) -> BroadphaseStats {
        self.stats
    }

    pub fn candidate_pairs(&self) -> &[(BodyHandle, BodyHandle)] {
        &self.merged_pairs
    }

    pub fn max_candidate_pairs_capacity(&self) -> usize {
        self.merged_pairs.capacity()
    }

    #[inline]
    fn effective_chunk_size(&self, body_count: usize) -> usize {
        let base = self.config.chunk_size.max(16);
        let workers = rayon::current_num_threads().max(1);
        // Keep enough shards for work-stealing, but avoid tiny shard overhead on high-core CPUs.
        let desired_shards = if body_count >= 4_096 {
            workers.saturating_mul(3)
        } else {
            workers.saturating_mul(4)
        };
        let adaptive = body_count.div_ceil(desired_shards).max(16);
        adaptive.min(base)
    }

    /// Pre-size internal buffers for the current body count so `rebuild_pairs_parallel` can remain
    /// allocation-free during the hot step loop.
    pub fn sync_for_body_count(&mut self, body_count: usize) {
        if self.sorted_dense.capacity() < body_count {
            self.sorted_dense
                .reserve(body_count.saturating_sub(self.sorted_dense.capacity()));
        }
        let chunk_size = self.effective_chunk_size(body_count);
        let shard_count = shard_count_for(body_count, chunk_size);
        let target_shard_capacity = self
            .config
            .shard_pair_reserve
            .max(self.config.max_candidate_pairs.div_ceil(shard_count.max(1)));
        while self.shard_pairs.len() < shard_count.max(1) {
            self.shard_pairs
                .push(Vec::with_capacity(target_shard_capacity));
        }
        for shard in &mut self.shard_pairs {
            if shard.capacity() < target_shard_capacity {
                shard.reserve(target_shard_capacity.saturating_sub(shard.capacity()));
            }
        }
        if self.merged_pairs.capacity() < self.config.max_candidate_pairs {
            self.merged_pairs.reserve(
                self.config
                    .max_candidate_pairs
                    .saturating_sub(self.merged_pairs.capacity()),
            );
        }
    }

    /// Rebuilds candidate body pairs in parallel using sorted AABB minima and shard-local output
    /// buffers. The returned slice borrows a reusable internal pair buffer.
    pub fn rebuild_pairs_parallel<'a>(
        &'a mut self,
        bodies: &BodyRegistry,
    ) -> &'a [(BodyHandle, BodyHandle)] {
        let body_count = bodies.len();
        self.stats = BroadphaseStats {
            body_count,
            ..BroadphaseStats::default()
        };
        self.merged_pairs.clear();
        if body_count < 2 {
            return &self.merged_pairs;
        }

        self.sync_for_body_count(body_count);
        self.sorted_dense.clear();
        self.sorted_dense.extend(0..body_count);
        let aabbs = bodies.aabbs();
        let handles = bodies.handles();
        self.sorted_dense
            .sort_unstable_by(|left, right| aabbs[*left].min.x.total_cmp(&aabbs[*right].min.x));

        let chunk_size = self.effective_chunk_size(body_count);
        let shard_count = shard_count_for(body_count, chunk_size);
        self.stats.shard_count = shard_count;
        let dropped_pairs = AtomicUsize::new(0);
        let overflowed = AtomicBool::new(false);
        let sorted_dense = &self.sorted_dense;

        self.shard_pairs[..shard_count]
            .par_iter_mut()
            .enumerate()
            .for_each(|(shard_index, local_pairs)| {
                local_pairs.clear();
                // Cyclic scheduling balances SAP work better than contiguous chunks because
                // lower sorted indices usually have longer overlap scans.
                for sorted_index in (shard_index..body_count).step_by(shard_count.max(1)) {
                    let dense_a = sorted_dense[sorted_index];
                    let handle_a = handles[dense_a];
                    let aabb_a = aabbs[dense_a];
                    let max_x = aabb_a.max.x;
                    for candidate_index in (sorted_index + 1)..body_count {
                        let dense_b = sorted_dense[candidate_index];
                        let aabb_b = aabbs[dense_b];
                        if aabb_b.min.x > max_x {
                            break;
                        }
                        if intersects_full(aabb_a, aabb_b) {
                            if local_pairs.len() < local_pairs.capacity() {
                                local_pairs.push(order_pair(handle_a, handles[dense_b]));
                            } else {
                                overflowed.store(true, Ordering::Relaxed);
                                dropped_pairs.fetch_add(1, Ordering::Relaxed);
                            }
                        }
                    }
                }
            });

        let mut remaining_capacity = self.merged_pairs.capacity();
        for local_pairs in &self.shard_pairs[..shard_count] {
            if remaining_capacity == 0 {
                if !local_pairs.is_empty() {
                    overflowed.store(true, Ordering::Relaxed);
                    dropped_pairs.fetch_add(local_pairs.len(), Ordering::Relaxed);
                }
                continue;
            }
            let to_copy = local_pairs.len().min(remaining_capacity);
            self.merged_pairs.extend_from_slice(&local_pairs[..to_copy]);
            remaining_capacity -= to_copy;
            if to_copy < local_pairs.len() {
                overflowed.store(true, Ordering::Relaxed);
                dropped_pairs.fetch_add(local_pairs.len() - to_copy, Ordering::Relaxed);
            }
        }

        self.stats.candidate_pairs = self.merged_pairs.len();
        self.stats.dropped_pairs = dropped_pairs.load(Ordering::Relaxed);
        self.stats.overflowed = overflowed.load(Ordering::Relaxed);
        &self.merged_pairs
    }
}

fn shard_count_for(body_count: usize, chunk_size: usize) -> usize {
    body_count.div_ceil(chunk_size.max(1)).max(1)
}

#[inline]
fn intersects_full(a: Aabb, b: Aabb) -> bool {
    a.intersects(b)
}

#[inline]
fn order_pair(left: BodyHandle, right: BodyHandle) -> (BodyHandle, BodyHandle) {
    if left.raw() <= right.raw() {
        (left, right)
    } else {
        (right, left)
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::Vector3;

    use super::*;
    use crate::body::{Aabb, BodyDesc};

    #[test]
    fn broadphase_emits_overlapping_body_pairs() {
        let mut bodies = BodyRegistry::new();
        let a = bodies.spawn(BodyDesc {
            position: Vector3::new(0.0, 0.0, 0.0),
            local_bounds: Aabb::from_center_half_extents(Vector3::zeros(), Vector3::repeat(1.0)),
            ..BodyDesc::default()
        });
        let b = bodies.spawn(BodyDesc {
            position: Vector3::new(1.25, 0.0, 0.0),
            local_bounds: Aabb::from_center_half_extents(Vector3::zeros(), Vector3::repeat(1.0)),
            ..BodyDesc::default()
        });
        let _c = bodies.spawn(BodyDesc {
            position: Vector3::new(10.0, 0.0, 0.0),
            ..BodyDesc::default()
        });

        let mut broadphase = BroadphasePipeline::new(BroadphaseConfig::default());
        broadphase.sync_for_body_count(bodies.len());
        let pairs = broadphase.rebuild_pairs_parallel(&bodies);
        assert_eq!(pairs, &[(a, b)]);
        assert!(!broadphase.stats().overflowed);
    }
}
