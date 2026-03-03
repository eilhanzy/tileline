//! Sleep and island management for ParadoxPE.
//!
//! The island manager groups connected bodies through contacts and joints, then updates sleep state
//! using reusable flat buffers so the fixed-step loop does not allocate after capacity is prepared.

use crate::body::ContactManifold;
use crate::handle::BodyHandle;
use crate::joint::DistanceJoint;
use crate::storage::BodyRegistry;

/// Sleep manager tuning.
#[derive(Debug, Clone, PartialEq)]
pub struct SleepConfig {
    pub linear_speed_threshold: f32,
    pub time_to_sleep: f32,
}

impl Default for SleepConfig {
    fn default() -> Self {
        Self {
            linear_speed_threshold: 0.05,
            time_to_sleep: 0.5,
        }
    }
}

/// Sleep/island statistics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SleepStats {
    pub dynamic_bodies: usize,
    pub islands: usize,
    pub sleeping_bodies: usize,
    pub awakened_bodies: usize,
}

/// Reusable island/sleep manager.
#[derive(Debug, Clone)]
pub struct SleepIslandManager {
    config: SleepConfig,
    stats: SleepStats,
    roots: Vec<usize>,
    dynamic_dense: Vec<usize>,
    island_counts: Vec<u32>,
    island_all_slow: Vec<bool>,
}

impl SleepIslandManager {
    pub fn new(config: SleepConfig) -> Self {
        Self {
            config,
            stats: SleepStats::default(),
            roots: Vec::new(),
            dynamic_dense: Vec::new(),
            island_counts: Vec::new(),
            island_all_slow: Vec::new(),
        }
    }

    pub fn config(&self) -> &SleepConfig {
        &self.config
    }

    pub fn stats(&self) -> SleepStats {
        self.stats
    }

    pub fn prepare_for_bodies(&mut self, body_count: usize) {
        if self.roots.len() < body_count {
            self.roots.resize(body_count, 0);
        }
        if self.island_counts.len() < body_count {
            self.island_counts.resize(body_count, 0);
        }
        if self.island_all_slow.len() < body_count {
            self.island_all_slow.resize(body_count, true);
        }
        if self.dynamic_dense.capacity() < body_count {
            self.dynamic_dense
                .reserve(body_count.saturating_sub(self.dynamic_dense.capacity()));
        }
    }

    pub fn update(
        &mut self,
        bodies: &mut BodyRegistry,
        manifolds: &[ContactManifold],
        joints: &[DistanceJoint],
        dt: f32,
    ) {
        self.stats = SleepStats::default();
        self.prepare_for_bodies(bodies.len());
        self.dynamic_dense.clear();
        for dense in 0..bodies.handles.len() {
            if bodies.inverse_masses[dense] > 0.0 {
                self.dynamic_dense.push(dense);
            }
        }
        self.stats.dynamic_bodies = self.dynamic_dense.len();
        if self.dynamic_dense.is_empty() {
            return;
        }

        for &dense in &self.dynamic_dense {
            self.roots[dense] = dense;
        }
        for manifold in manifolds {
            self.union_handles(bodies, manifold.body_a, manifold.body_b);
        }
        for joint in joints {
            self.union_handles(bodies, joint.body_a, joint.body_b);
        }

        self.island_counts.fill(0);
        self.island_all_slow.fill(true);
        let threshold_sq = self.config.linear_speed_threshold * self.config.linear_speed_threshold;
        let mut islands = 0usize;
        for index in 0..self.dynamic_dense.len() {
            let dense = self.dynamic_dense[index];
            let root = self.find(dense);
            if self.island_counts[root] == 0 {
                islands += 1;
            }
            self.island_counts[root] += 1;
            if bodies.linear_velocities[dense].norm_squared() > threshold_sq {
                self.island_all_slow[root] = false;
            }
        }
        self.stats.islands = islands;

        for index in 0..self.dynamic_dense.len() {
            let dense = self.dynamic_dense[index];
            let root = self.find(dense);
            if self.island_all_slow[root] {
                bodies.sleep_timers[dense] += dt;
                if bodies.sleep_timers[dense] >= self.config.time_to_sleep {
                    if bodies.awake[dense] {
                        self.stats.sleeping_bodies = self.stats.sleeping_bodies.saturating_add(1);
                    }
                    bodies.set_sleeping_dense(dense);
                }
            } else {
                let was_awake = bodies.awake[dense];
                bodies.wake_dense(dense);
                if !was_awake {
                    self.stats.awakened_bodies = self.stats.awakened_bodies.saturating_add(1);
                }
            }
        }
    }

    fn union_handles(&mut self, bodies: &BodyRegistry, a: BodyHandle, b: BodyHandle) {
        let Some(a_dense) = bodies.dense_index_of(a) else {
            return;
        };
        let Some(b_dense) = bodies.dense_index_of(b) else {
            return;
        };
        self.union(a_dense, b_dense);
    }

    fn find(&mut self, value: usize) -> usize {
        let parent = self.roots[value];
        if parent == value {
            value
        } else {
            let root = self.find(parent);
            self.roots[value] = root;
            root
        }
    }

    fn union(&mut self, left: usize, right: usize) {
        let left_root = self.find(left);
        let right_root = self.find(right);
        if left_root != right_root {
            self.roots[right_root] = left_root;
        }
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::Vector3;

    use super::*;
    use crate::body::{BodyDesc, ContactId, ContactManifold};
    use crate::handle::{ColliderHandle, JointHandle};
    use crate::joint::{DistanceJoint, DistanceJointDesc};

    #[test]
    fn islands_can_put_slow_bodies_to_sleep() {
        let mut bodies = BodyRegistry::new();
        let a = bodies.spawn(BodyDesc::default());
        let b = bodies.spawn(BodyDesc {
            position: Vector3::new(1.0, 0.0, 0.0),
            ..BodyDesc::default()
        });
        let manifold = ContactManifold {
            contact_id: ContactId::new(7),
            collider_a: ColliderHandle::new(a.index() as u16, a.generation()),
            collider_b: ColliderHandle::new(b.index() as u16, b.generation()),
            body_a: a,
            body_b: b,
            point: Vector3::new(0.5, 0.0, 0.0),
            normal: Vector3::new(1.0, 0.0, 0.0),
            penetration: 0.01,
            persisted_frames: 1,
            restitution: 0.0,
            friction: 0.5,
        };
        let joint =
            DistanceJoint::from_desc(JointHandle::new(0, 1), DistanceJointDesc::new(a, b, 1.0));
        let mut sleep = SleepIslandManager::new(SleepConfig {
            linear_speed_threshold: 0.1,
            time_to_sleep: 0.01,
        });
        sleep.update(&mut bodies, &[manifold], &[joint], 0.02);
        assert!(!bodies.body(a).unwrap().awake);
        assert!(!bodies.body(b).unwrap().awake);
        assert!(sleep.stats().sleeping_bodies >= 2);
    }
}
