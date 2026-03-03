//! Cache-friendly Structure-of-Arrays storage for ParadoxPE bodies.
//!
//! The registry keeps hot simulation components in separate contiguous vectors so broadphase,
//! integration, and `.tlscript @parallel(domain="bodies")` hooks can read/write slices without
//! lock contention or pointer chasing.

use nalgebra::{UnitQuaternion, Vector3};

use crate::body::{Aabb, BodyDesc, BodyKind, RigidBody};
use crate::handle::{BodyHandle, ColliderHandle};

const INVALID_DENSE_INDEX: u32 = u32::MAX;

#[derive(Debug, Clone, Copy)]
struct SparseSlot {
    generation: u16,
    dense_index: u32,
}

impl Default for SparseSlot {
    fn default() -> Self {
        Self {
            generation: 1,
            dense_index: INVALID_DENSE_INDEX,
        }
    }
}

/// Read-only SoA view intended for broadphase and parallel script reads.
#[derive(Debug, Clone, Copy)]
pub struct BodyReadDomain<'a> {
    pub handles: &'a [BodyHandle],
    pub positions: &'a [Vector3<f32>],
    pub rotations: &'a [UnitQuaternion<f32>],
    pub linear_velocities: &'a [Vector3<f32>],
    pub aabbs: &'a [Aabb],
    pub inverse_masses: &'a [f32],
    pub kinds: &'a [BodyKind],
    pub awake: &'a [bool],
}

/// Write-only velocity slice view used by parallel body jobs.
#[derive(Debug)]
pub struct BodyVelocityWriteDomain<'a> {
    pub handles: &'a [BodyHandle],
    pub linear_velocities: &'a mut [Vector3<f32>],
}

/// Dense SoA body registry backed by stable generational handles.
#[derive(Debug, Clone, Default)]
pub struct BodyRegistry {
    sparse: Vec<SparseSlot>,
    free_sparse: Vec<u16>,
    pub(crate) handles: Vec<BodyHandle>,
    pub(crate) kinds: Vec<BodyKind>,
    pub(crate) positions: Vec<Vector3<f32>>,
    pub(crate) rotations: Vec<UnitQuaternion<f32>>,
    pub(crate) linear_velocities: Vec<Vector3<f32>>,
    pub(crate) accumulated_forces: Vec<Vector3<f32>>,
    pub(crate) local_bounds: Vec<Aabb>,
    pub(crate) aabbs: Vec<Aabb>,
    pub(crate) inverse_masses: Vec<f32>,
    pub(crate) linear_dampings: Vec<f32>,
    pub(crate) user_tags: Vec<u32>,
    pub(crate) awake: Vec<bool>,
    pub(crate) sleep_timers: Vec<f32>,
    pub(crate) primary_colliders: Vec<Option<ColliderHandle>>,
}

impl BodyRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.handles.len()
    }

    pub fn is_empty(&self) -> bool {
        self.handles.is_empty()
    }

    pub fn handles(&self) -> &[BodyHandle] {
        &self.handles
    }

    pub fn positions(&self) -> &[Vector3<f32>] {
        &self.positions
    }

    pub fn rotations(&self) -> &[UnitQuaternion<f32>] {
        &self.rotations
    }

    pub fn linear_velocities(&self) -> &[Vector3<f32>] {
        &self.linear_velocities
    }

    pub fn linear_velocities_mut(&mut self) -> &mut [Vector3<f32>] {
        &mut self.linear_velocities
    }

    pub fn aabbs(&self) -> &[Aabb] {
        &self.aabbs
    }

    pub fn read_domain(&self) -> BodyReadDomain<'_> {
        BodyReadDomain {
            handles: &self.handles,
            positions: &self.positions,
            rotations: &self.rotations,
            linear_velocities: &self.linear_velocities,
            aabbs: &self.aabbs,
            inverse_masses: &self.inverse_masses,
            kinds: &self.kinds,
            awake: &self.awake,
        }
    }

    pub fn write_velocity_domain(&mut self) -> BodyVelocityWriteDomain<'_> {
        BodyVelocityWriteDomain {
            handles: &self.handles,
            linear_velocities: &mut self.linear_velocities,
        }
    }

    pub fn contains(&self, handle: BodyHandle) -> bool {
        self.dense_index(handle).is_some()
    }

    pub fn spawn(&mut self, desc: BodyDesc) -> BodyHandle {
        let sparse_index = self.allocate_sparse_index();
        let generation = self.sparse[sparse_index as usize].generation;
        let handle = BodyHandle::new(sparse_index, generation);
        let dense_index = self.handles.len() as u32;
        self.sparse[sparse_index as usize].dense_index = dense_index;

        let inverse_mass = match desc.kind {
            BodyKind::Dynamic if desc.mass > 0.0 => 1.0 / desc.mass,
            _ => 0.0,
        };
        let world_aabb = desc.local_bounds.translated(desc.position);

        self.handles.push(handle);
        self.kinds.push(desc.kind);
        self.positions.push(desc.position);
        self.rotations.push(desc.rotation);
        self.linear_velocities.push(desc.linear_velocity);
        self.accumulated_forces.push(Vector3::zeros());
        self.local_bounds.push(desc.local_bounds);
        self.aabbs.push(world_aabb);
        self.inverse_masses.push(inverse_mass);
        self.linear_dampings
            .push(desc.linear_damping.clamp(0.0, 1.0));
        self.user_tags.push(desc.user_tag);
        self.awake.push(true);
        self.sleep_timers.push(0.0);
        self.primary_colliders.push(None);
        handle
    }

    pub fn remove(&mut self, handle: BodyHandle) -> bool {
        let Some(dense_index) = self.dense_index(handle) else {
            return false;
        };
        self.swap_remove_dense(dense_index);
        let sparse_index = handle.index();
        let sparse = &mut self.sparse[sparse_index];
        sparse.dense_index = INVALID_DENSE_INDEX;
        sparse.generation = sparse.generation.wrapping_add(1).max(1);
        self.free_sparse.push(sparse_index as u16);
        true
    }

    pub fn body(&self, handle: BodyHandle) -> Option<RigidBody> {
        let dense = self.dense_index(handle)?;
        Some(self.snapshot_dense(dense))
    }

    pub fn primary_collider(&self, handle: BodyHandle) -> Option<ColliderHandle> {
        let dense = self.dense_index(handle)?;
        self.primary_colliders[dense]
    }

    pub fn set_primary_collider(
        &mut self,
        handle: BodyHandle,
        collider: Option<ColliderHandle>,
    ) -> bool {
        let Some(dense) = self.dense_index(handle) else {
            return false;
        };
        self.primary_colliders[dense] = collider;
        true
    }

    pub fn clear_primary_collider_if_matches(
        &mut self,
        handle: BodyHandle,
        collider: ColliderHandle,
    ) -> bool {
        let Some(dense) = self.dense_index(handle) else {
            return false;
        };
        if self.primary_colliders[dense] == Some(collider) {
            self.primary_colliders[dense] = None;
            true
        } else {
            false
        }
    }

    pub fn set_local_bounds(&mut self, handle: BodyHandle, bounds: Aabb) -> bool {
        let Some(dense) = self.dense_index(handle) else {
            return false;
        };
        self.local_bounds[dense] = bounds;
        self.recompute_world_aabb(dense);
        true
    }

    pub fn apply_force(&mut self, body: BodyHandle, force: Vector3<f32>) -> bool {
        let Some(dense) = self.dense_index(body) else {
            return false;
        };
        if self.kinds[dense] != BodyKind::Dynamic {
            return false;
        }
        self.accumulated_forces[dense] += force;
        self.wake_dense(dense);
        true
    }

    pub fn set_velocity(&mut self, body: BodyHandle, velocity: Vector3<f32>) -> bool {
        let Some(dense) = self.dense_index(body) else {
            return false;
        };
        if self.kinds[dense] == BodyKind::Static {
            return false;
        }
        self.linear_velocities[dense] = velocity;
        self.wake_dense(dense);
        true
    }

    pub fn integrate(&mut self, dt: f32, gravity: Vector3<f32>) {
        for dense in 0..self.handles.len() {
            match self.kinds[dense] {
                BodyKind::Static => {
                    self.accumulated_forces[dense] = Vector3::zeros();
                    self.recompute_world_aabb(dense);
                }
                BodyKind::Kinematic => {
                    self.positions[dense] += self.linear_velocities[dense] * dt;
                    self.recompute_world_aabb(dense);
                }
                BodyKind::Dynamic => {
                    if !self.awake[dense] {
                        self.accumulated_forces[dense] = Vector3::zeros();
                        continue;
                    }
                    let acceleration =
                        gravity + self.accumulated_forces[dense] * self.inverse_masses[dense];
                    self.linear_velocities[dense] += acceleration * dt;
                    self.linear_velocities[dense] *=
                        1.0 - self.linear_dampings[dense].clamp(0.0, 0.95);
                    self.positions[dense] += self.linear_velocities[dense] * dt;
                    self.accumulated_forces[dense] = Vector3::zeros();
                    self.recompute_world_aabb(dense);
                }
            }
        }
    }

    pub fn aabb_for(&self, handle: BodyHandle) -> Option<Aabb> {
        let dense = self.dense_index(handle)?;
        Some(self.aabbs[dense])
    }

    pub fn position_for(&self, handle: BodyHandle) -> Option<Vector3<f32>> {
        let dense = self.dense_index(handle)?;
        Some(self.positions[dense])
    }

    pub(crate) fn relative_velocity(&self, body_a: BodyHandle, body_b: BodyHandle) -> Vector3<f32> {
        let velocity_a = self
            .dense_index(body_a)
            .map(|dense| self.linear_velocities[dense])
            .unwrap_or_else(Vector3::zeros);
        let velocity_b = self
            .dense_index(body_b)
            .map(|dense| self.linear_velocities[dense])
            .unwrap_or_else(Vector3::zeros);
        velocity_b - velocity_a
    }

    pub fn reserve_dense(&mut self, additional: usize) {
        self.handles.reserve(additional);
        self.kinds.reserve(additional);
        self.positions.reserve(additional);
        self.rotations.reserve(additional);
        self.linear_velocities.reserve(additional);
        self.accumulated_forces.reserve(additional);
        self.local_bounds.reserve(additional);
        self.aabbs.reserve(additional);
        self.inverse_masses.reserve(additional);
        self.linear_dampings.reserve(additional);
        self.user_tags.reserve(additional);
        self.awake.reserve(additional);
        self.sleep_timers.reserve(additional);
        self.primary_colliders.reserve(additional);
    }

    pub(crate) fn dense_index_of(&self, handle: BodyHandle) -> Option<usize> {
        self.dense_index(handle)
    }

    pub(crate) fn recompute_world_aabb_for_dense(&mut self, dense: usize) {
        self.recompute_world_aabb(dense);
    }

    pub(crate) fn wake_dense(&mut self, dense: usize) {
        self.awake[dense] = true;
        self.sleep_timers[dense] = 0.0;
    }

    pub(crate) fn set_sleeping_dense(&mut self, dense: usize) {
        self.awake[dense] = false;
        self.sleep_timers[dense] = 0.0;
        self.linear_velocities[dense] = Vector3::zeros();
        self.accumulated_forces[dense] = Vector3::zeros();
    }

    fn snapshot_dense(&self, dense: usize) -> RigidBody {
        RigidBody {
            handle: self.handles[dense],
            kind: self.kinds[dense],
            position: self.positions[dense],
            rotation: self.rotations[dense].clone(),
            linear_velocity: self.linear_velocities[dense],
            accumulated_force: self.accumulated_forces[dense],
            inverse_mass: self.inverse_masses[dense],
            linear_damping: self.linear_dampings[dense],
            user_tag: self.user_tags[dense],
            awake: self.awake[dense],
            sleep_timer: self.sleep_timers[dense],
            aabb: self.aabbs[dense],
            primary_collider: self.primary_colliders[dense],
        }
    }

    fn recompute_world_aabb(&mut self, dense: usize) {
        self.aabbs[dense] = self.local_bounds[dense].translated(self.positions[dense]);
    }

    fn dense_index(&self, handle: BodyHandle) -> Option<usize> {
        let sparse = self.sparse.get(handle.index())?;
        if sparse.generation != handle.generation() || sparse.dense_index == INVALID_DENSE_INDEX {
            return None;
        }
        Some(sparse.dense_index as usize)
    }

    fn allocate_sparse_index(&mut self) -> u16 {
        if let Some(index) = self.free_sparse.pop() {
            return index;
        }
        let index = self.sparse.len() as u16;
        self.sparse.push(SparseSlot::default());
        index
    }

    fn swap_remove_dense(&mut self, dense_index: usize) {
        let last = self.handles.len() - 1;
        let removed_handle = self.handles[dense_index];

        self.handles.swap_remove(dense_index);
        self.kinds.swap_remove(dense_index);
        self.positions.swap_remove(dense_index);
        self.rotations.swap_remove(dense_index);
        self.linear_velocities.swap_remove(dense_index);
        self.accumulated_forces.swap_remove(dense_index);
        self.local_bounds.swap_remove(dense_index);
        self.aabbs.swap_remove(dense_index);
        self.inverse_masses.swap_remove(dense_index);
        self.linear_dampings.swap_remove(dense_index);
        self.user_tags.swap_remove(dense_index);
        self.awake.swap_remove(dense_index);
        self.sleep_timers.swap_remove(dense_index);
        self.primary_colliders.swap_remove(dense_index);

        if dense_index != last {
            let moved_handle = self.handles[dense_index];
            if let Some(sparse) = self.sparse.get_mut(moved_handle.index()) {
                sparse.dense_index = dense_index as u32;
            }
        }

        if let Some(sparse) = self.sparse.get_mut(removed_handle.index()) {
            sparse.dense_index = INVALID_DENSE_INDEX;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::body::BodyDesc;

    #[test]
    fn swap_remove_keeps_remaining_handles_valid() {
        let mut registry = BodyRegistry::new();
        let a = registry.spawn(BodyDesc::default());
        let b = registry.spawn(BodyDesc {
            position: Vector3::new(5.0, 0.0, 0.0),
            ..BodyDesc::default()
        });
        let c = registry.spawn(BodyDesc {
            position: Vector3::new(9.0, 0.0, 0.0),
            ..BodyDesc::default()
        });

        assert!(registry.remove(b));
        assert!(registry.body(a).is_some());
        assert!(registry.body(c).is_some());
        assert_eq!(registry.body(c).unwrap().position.x, 9.0);
        assert!(!registry.remove(b));
    }

    #[test]
    fn read_and_write_domains_expose_contiguous_slices() {
        let mut registry = BodyRegistry::new();
        let a = registry.spawn(BodyDesc::default());
        let b = registry.spawn(BodyDesc {
            position: Vector3::new(2.0, 0.0, 0.0),
            ..BodyDesc::default()
        });

        let read = registry.read_domain();
        assert_eq!(read.handles, &[a, b]);
        assert_eq!(read.positions.len(), 2);
        assert_eq!(read.aabbs.len(), 2);

        let write = registry.write_velocity_domain();
        assert_eq!(write.handles.len(), 2);
        assert_eq!(write.linear_velocities.len(), 2);
    }
}
