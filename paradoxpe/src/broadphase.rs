//! Parallel broadphase candidate generation for ParadoxPE.
//!
//! The current implementation uses a cache-friendly sweep-and-prune pass over body AABBs. It is
//! designed to run inside Tileline's CPU tasking environment and uses Rayon-style parallel shard
//! scans without allocating in the hot `rebuild_pairs_parallel` loop.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use rayon::prelude::*;

use crate::body::{Aabb, BodyKind};
use crate::handle::BodyHandle;
use crate::storage::BodyRegistry;

/// Broadphase configuration tuned for shard-parallel sweep-and-prune.
#[derive(Debug, Clone, PartialEq)]
pub struct BroadphaseConfig {
    /// Number of sorted bodies processed by one parallel shard.
    pub chunk_size: usize,
    /// Maximum merged candidate pair count stored without reallocating.
    pub max_candidate_pairs: usize,
    /// Minimum reserved pair slots per shard.
    pub shard_pair_reserve: usize,
    /// Enables swept AABB broadphase for high-speed anti-tunneling candidate capture.
    pub speculative_sweep: bool,
    /// Maximum per-axis sweep distance (world units) used by speculative broadphase.
    pub speculative_max_distance: f32,
}

impl Default for BroadphaseConfig {
    fn default() -> Self {
        Self {
            chunk_size: 128,
            max_candidate_pairs: 16_384,
            shard_pair_reserve: 512,
            speculative_sweep: true,
            speculative_max_distance: 1.0,
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
    predictive_dt: f32,
    sorted_dense: Vec<usize>,
    swept_aabbs: Vec<Aabb>,
    shard_pairs: Vec<Vec<(BodyHandle, BodyHandle)>>,
    merged_pairs: Vec<(BodyHandle, BodyHandle)>,
    stats: BroadphaseStats,
}

impl BroadphasePipeline {
    pub fn new(config: BroadphaseConfig) -> Self {
        Self {
            config,
            predictive_dt: 0.0,
            sorted_dense: Vec::new(),
            swept_aabbs: Vec::new(),
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

    /// Update per-step predictive delta for swept broadphase.
    pub fn set_predictive_dt(&mut self, dt: f32) {
        self.predictive_dt = dt.max(0.0);
    }

    /// Update swept broadphase speculative parameters at runtime.
    pub fn set_speculative_sweep_config(&mut self, enabled: bool, max_distance: f32) {
        self.config.speculative_sweep = enabled;
        self.config.speculative_max_distance = max_distance.max(0.0);
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
        if self.swept_aabbs.capacity() < body_count {
            self.swept_aabbs
                .reserve(body_count.saturating_sub(self.swept_aabbs.capacity()));
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
        let read = bodies.read_domain();
        let aabbs = read.aabbs;
        let handles = read.handles;
        let velocities = read.linear_velocities;
        let kinds = read.kinds;
        self.swept_aabbs.clear();
        self.swept_aabbs.extend(aabbs.iter().copied());
        if self.config.speculative_sweep && self.predictive_dt > 1e-6 {
            let predictive_dt = self.predictive_dt;
            let speculative_max_distance = self.config.speculative_max_distance;
            self.swept_aabbs
                .par_iter_mut()
                .enumerate()
                .for_each(|(dense, swept)| {
                    if matches!(kinds[dense], BodyKind::Dynamic | BodyKind::Kinematic) {
                        *swept = swept_aabb(
                            aabbs[dense],
                            velocities[dense],
                            predictive_dt,
                            speculative_max_distance,
                        );
                    }
                });
        }
        let swept_aabbs = &self.swept_aabbs;
        self.sorted_dense.par_sort_unstable_by(|left, right| {
            swept_aabbs[*left]
                .min
                .x
                .total_cmp(&swept_aabbs[*right].min.x)
        });

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
                    let aabb_a = swept_aabbs[dense_a];
                    let max_x = aabb_a.max.x;
                    for candidate_index in (sorted_index + 1)..body_count {
                        let dense_b = sorted_dense[candidate_index];
                        let aabb_b = swept_aabbs[dense_b];
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

#[inline]
fn swept_aabb(aabb: Aabb, velocity: nalgebra::Vector3<f32>, dt: f32, max_distance: f32) -> Aabb {
    let dt = dt.max(0.0);
    let max_distance = max_distance.max(0.0);
    if dt <= 1e-6 || max_distance <= 1e-6 {
        return aabb;
    }

    let mut travel = velocity * dt;
    travel.x = travel.x.clamp(-max_distance, max_distance);
    travel.y = travel.y.clamp(-max_distance, max_distance);
    travel.z = travel.z.clamp(-max_distance, max_distance);
    if travel.x.abs() <= 1e-6 && travel.y.abs() <= 1e-6 && travel.z.abs() <= 1e-6 {
        return aabb;
    }

    Aabb {
        min: nalgebra::Vector3::new(
            aabb.min.x.min(aabb.min.x + travel.x),
            aabb.min.y.min(aabb.min.y + travel.y),
            aabb.min.z.min(aabb.min.z + travel.z),
        ),
        max: nalgebra::Vector3::new(
            aabb.max.x.max(aabb.max.x + travel.x),
            aabb.max.y.max(aabb.max.y + travel.y),
            aabb.max.z.max(aabb.max.z + travel.z),
        ),
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::Vector3;

    use super::*;
    use crate::body::{Aabb, BodyDesc, BodyKind};

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

    #[test]
    fn broadphase_swept_mode_catches_fast_approach_pair() {
        let mut bodies = BodyRegistry::new();
        let moving = bodies.spawn(BodyDesc {
            kind: BodyKind::Dynamic,
            position: Vector3::new(0.0, 0.0, 0.0),
            linear_velocity: Vector3::new(32.0, 0.0, 0.0),
            local_bounds: Aabb::from_center_half_extents(Vector3::zeros(), Vector3::repeat(0.2)),
            ..BodyDesc::default()
        });
        let wall = bodies.spawn(BodyDesc {
            kind: BodyKind::Static,
            position: Vector3::new(1.1, 0.0, 0.0),
            local_bounds: Aabb::from_center_half_extents(Vector3::zeros(), Vector3::repeat(0.2)),
            ..BodyDesc::default()
        });

        let mut broadphase = BroadphasePipeline::new(BroadphaseConfig::default());
        broadphase.set_predictive_dt(1.0 / 30.0);
        broadphase.sync_for_body_count(bodies.len());
        let pairs = broadphase.rebuild_pairs_parallel(&bodies);
        assert!(pairs.contains(&(moving, wall)));
    }
}
