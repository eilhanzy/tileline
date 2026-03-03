//! Fixed-step ParadoxPE world skeleton built on SoA body storage and a parallel broadphase.
//!
//! This is still a foundation layer, not the final solver. The goal is to lock down the data flow:
//! dense generational handles, cache-friendly body storage, allocation-free hot stepping, and a
//! broadphase pipeline that can scale across Tileline's CPU task execution model.

use nalgebra::Vector3;

use crate::abi::ParadoxScriptHostAbi;
use crate::body::{
    Aabb, BodyDesc, BodyKind, ColliderDesc, ColliderShape, ColliderShapeKind, ContactPair,
    ContactSnapshot, RigidBody,
};
use crate::broadphase::{BroadphaseConfig, BroadphasePipeline};
use crate::handle::{
    BodyHandle, ColliderHandle, ContactHandle, HandleKind, JointHandle, PhysicsHandle,
};
use crate::joint::{DistanceJoint, DistanceJointDesc, JointConstraintSolver, JointSolverConfig};
use crate::narrowphase::{NarrowphaseConfig, NarrowphasePipeline};
use crate::sleep::{SleepConfig, SleepIslandManager};
use crate::solver::{ContactSolver, ContactSolverConfig};
use crate::storage::BodyRegistry;

#[derive(Debug, Clone)]
struct Slot<T> {
    generation: u16,
    value: Option<T>,
}

impl<T> Default for Slot<T> {
    fn default() -> Self {
        Self {
            generation: 1,
            value: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
struct ColliderRecord {
    handle: ColliderHandle,
    desc: ColliderDesc,
}

/// Fixed-step scheduler helper for engine frame times.
#[derive(Debug, Clone, PartialEq)]
pub struct FixedStepClock {
    fixed_dt: f32,
    max_substeps: u32,
    accumulator: f32,
    tick: u64,
}

impl FixedStepClock {
    pub fn new(fixed_dt: f32, max_substeps: u32) -> Self {
        Self {
            fixed_dt: fixed_dt.max(1e-4),
            max_substeps: max_substeps.max(1),
            accumulator: 0.0,
            tick: 0,
        }
    }

    pub fn fixed_dt(&self) -> f32 {
        self.fixed_dt
    }

    pub fn tick(&self) -> u64 {
        self.tick
    }

    pub fn accumulate(&mut self, dt: f32) -> u32 {
        self.accumulator =
            (self.accumulator + dt.max(0.0)).min(self.fixed_dt * self.max_substeps as f32);
        let mut steps = 0;
        while self.accumulator + f32::EPSILON >= self.fixed_dt && steps < self.max_substeps {
            self.accumulator -= self.fixed_dt;
            self.tick = self.tick.saturating_add(1);
            steps += 1;
        }
        steps
    }
}

/// World configuration for the ParadoxPE foundation runtime.
#[derive(Debug, Clone, PartialEq)]
pub struct PhysicsWorldConfig {
    pub gravity: Vector3<f32>,
    pub fixed_dt: f32,
    pub max_substeps: u32,
    pub max_contact_snapshots: usize,
    pub broadphase: BroadphaseConfig,
    pub narrowphase: NarrowphaseConfig,
    pub solver: ContactSolverConfig,
    pub joints: JointSolverConfig,
    pub sleep: SleepConfig,
}

impl Default for PhysicsWorldConfig {
    fn default() -> Self {
        Self {
            gravity: Vector3::new(0.0, -9.81, 0.0),
            fixed_dt: 1.0 / 120.0,
            max_substeps: 8,
            max_contact_snapshots: 256,
            broadphase: BroadphaseConfig::default(),
            narrowphase: NarrowphaseConfig::default(),
            solver: ContactSolverConfig::default(),
            joints: JointSolverConfig::default(),
            sleep: SleepConfig::default(),
        }
    }
}

/// Engine-facing ParadoxPE world skeleton.
#[derive(Debug, Clone)]
pub struct PhysicsWorld {
    config: PhysicsWorldConfig,
    clock: FixedStepClock,
    bodies: BodyRegistry,
    broadphase: BroadphasePipeline,
    narrowphase: NarrowphasePipeline,
    solver: ContactSolver,
    joint_solver: JointConstraintSolver,
    sleep_manager: SleepIslandManager,
    colliders: Vec<Slot<ColliderRecord>>,
    free_colliders: Vec<u16>,
    joints: Vec<Slot<DistanceJoint>>,
    free_joints: Vec<u16>,
    active_joints: Vec<DistanceJoint>,
    contact_snapshots: Vec<Slot<ContactSnapshot>>,
    free_contact_snapshots: Vec<u16>,
    contacts: Vec<ContactPair>,
}

impl PhysicsWorld {
    pub fn new(config: PhysicsWorldConfig) -> Self {
        let mut broadphase = BroadphasePipeline::new(config.broadphase.clone());
        broadphase.sync_for_body_count(0);
        let mut narrowphase = NarrowphasePipeline::new(config.narrowphase.clone());
        narrowphase.sync_for_pair_capacity(0);
        let solver = ContactSolver::new(config.solver.clone());
        let joint_solver = JointConstraintSolver::new(config.joints.clone());
        let sleep_manager = SleepIslandManager::new(config.sleep.clone());
        Self {
            clock: FixedStepClock::new(config.fixed_dt, config.max_substeps),
            config,
            bodies: BodyRegistry::new(),
            broadphase,
            narrowphase,
            solver,
            joint_solver,
            sleep_manager,
            colliders: Vec::new(),
            free_colliders: Vec::new(),
            joints: Vec::new(),
            free_joints: Vec::new(),
            active_joints: Vec::new(),
            contact_snapshots: Vec::new(),
            free_contact_snapshots: Vec::new(),
            contacts: Vec::new(),
        }
    }

    pub fn config(&self) -> &PhysicsWorldConfig {
        &self.config
    }

    pub fn fixed_step_clock(&self) -> &FixedStepClock {
        &self.clock
    }

    pub fn body_registry(&self) -> &BodyRegistry {
        &self.bodies
    }

    pub fn body_registry_mut(&mut self) -> &mut BodyRegistry {
        &mut self.bodies
    }

    /// Active body workload size suitable for `.tlscript @parallel(domain="bodies")` planning.
    pub fn active_parallel_body_count(&self) -> usize {
        self.bodies.active_parallel_body_count()
    }

    pub fn broadphase(&self) -> &BroadphasePipeline {
        &self.broadphase
    }

    pub fn narrowphase(&self) -> &NarrowphasePipeline {
        &self.narrowphase
    }

    pub fn solver(&self) -> &ContactSolver {
        &self.solver
    }

    pub fn joint_solver(&self) -> &JointConstraintSolver {
        &self.joint_solver
    }

    pub fn sleep_manager(&self) -> &SleepIslandManager {
        &self.sleep_manager
    }

    pub fn spawn_body(&mut self, desc: BodyDesc) -> BodyHandle {
        let handle = self.bodies.spawn(desc);
        self.sync_hot_loop_buffers();
        handle
    }

    pub fn body(&self, handle: BodyHandle) -> Option<RigidBody> {
        self.bodies.body(handle)
    }

    pub fn destroy_body(&mut self, handle: BodyHandle) -> bool {
        if !self.bodies.contains(handle) {
            return false;
        }
        let colliders_to_remove = self
            .colliders
            .iter()
            .filter_map(|slot| slot.value.as_ref())
            .filter(|record| record.desc.body == Some(handle))
            .map(|record| record.handle)
            .collect::<Vec<_>>();
        for collider in colliders_to_remove {
            let _ = self.destroy_collider(collider);
        }
        let joints_to_remove = self
            .joints
            .iter()
            .filter_map(|slot| slot.value.as_ref())
            .filter(|joint| joint.body_a == handle || joint.body_b == handle)
            .map(|joint| joint.handle)
            .collect::<Vec<_>>();
        for joint in joints_to_remove {
            let _ = self.destroy_joint(joint);
        }
        let removed = self.bodies.remove(handle);
        if removed {
            self.sync_hot_loop_buffers();
        }
        removed
    }

    pub fn spawn_collider(&mut self, desc: ColliderDesc) -> Option<ColliderHandle> {
        if let Some(body) = desc.body {
            self.bodies.body(body)?;
        }
        let (index, generation, slot) = alloc_slot(&mut self.colliders, &mut self.free_colliders);
        let handle = ColliderHandle::new(index, generation);
        *slot = Some(ColliderRecord { handle, desc });
        if let Some(body) = slot.as_ref().and_then(|record| record.desc.body) {
            self.rebuild_body_bounds_from_colliders(body);
        }
        Some(handle)
    }

    pub fn spawn_distance_joint(&mut self, desc: DistanceJointDesc) -> Option<JointHandle> {
        self.bodies.body(desc.body_a)?;
        self.bodies.body(desc.body_b)?;
        let (index, generation, slot) = alloc_slot(&mut self.joints, &mut self.free_joints);
        let handle = JointHandle::new(index, generation);
        *slot = Some(DistanceJoint::from_desc(handle, desc));
        self.rebuild_active_joint_cache();
        self.sync_hot_loop_buffers();
        Some(handle)
    }

    pub fn destroy_collider(&mut self, handle: ColliderHandle) -> bool {
        let body = get_slot(&self.colliders, handle.erased())
            .and_then(|slot| slot.value.as_ref())
            .and_then(|record| record.desc.body);
        let removed = free_slot(
            &mut self.colliders,
            &mut self.free_colliders,
            handle.erased(),
        )
        .is_some();
        if removed {
            if let Some(body) = body {
                self.rebuild_body_bounds_from_colliders(body);
            }
        }
        removed
    }

    pub fn destroy_joint(&mut self, handle: JointHandle) -> bool {
        let removed = free_slot(&mut self.joints, &mut self.free_joints, handle.erased()).is_some();
        if removed {
            self.rebuild_active_joint_cache();
            self.sync_hot_loop_buffers();
        }
        removed
    }

    pub fn apply_force(&mut self, body: BodyHandle, force: Vector3<f32>) -> bool {
        self.bodies.apply_force(body, force)
    }

    pub fn set_velocity(&mut self, body: BodyHandle, velocity: Vector3<f32>) -> bool {
        self.bodies.set_velocity(body, velocity)
    }

    pub fn contacts(&self) -> &[ContactPair] {
        &self.contacts
    }

    pub fn query_contacts(&mut self, body: BodyHandle) -> Option<ContactHandle> {
        self.bodies.body(body)?;
        let contacts = self
            .contacts
            .iter()
            .filter(|contact| contact.body_a == Some(body) || contact.body_b == Some(body))
            .cloned()
            .collect::<Vec<_>>();
        if self.contact_snapshots.len() >= self.config.max_contact_snapshots
            && self.free_contact_snapshots.is_empty()
        {
            return None;
        }
        let (index, generation, slot) = alloc_slot(
            &mut self.contact_snapshots,
            &mut self.free_contact_snapshots,
        );
        let handle = ContactHandle::new(index, generation);
        *slot = Some(ContactSnapshot { body, contacts });
        Some(handle)
    }

    pub fn contact_snapshot(&self, handle: ContactHandle) -> Option<&ContactSnapshot> {
        get_slot(&self.contact_snapshots, handle.erased()).and_then(|slot| slot.value.as_ref())
    }

    pub fn contact_count(&self, handle: ContactHandle) -> u32 {
        self.contact_snapshot(handle)
            .map(|snapshot| snapshot.contacts.len() as u32)
            .unwrap_or(0)
    }

    pub fn release_handle(&mut self, handle: PhysicsHandle) -> bool {
        match handle.kind() {
            Some(HandleKind::Body) => BodyHandle::try_from(handle)
                .ok()
                .map(|h| self.destroy_body(h))
                .unwrap_or(false),
            Some(HandleKind::Collider) => ColliderHandle::try_from(handle)
                .ok()
                .map(|h| self.destroy_collider(h))
                .unwrap_or(false),
            Some(HandleKind::Joint) => JointHandle::try_from(handle)
                .ok()
                .map(|h| self.destroy_joint(h))
                .unwrap_or(false),
            Some(HandleKind::ContactSnapshot) => free_slot(
                &mut self.contact_snapshots,
                &mut self.free_contact_snapshots,
                handle,
            )
            .is_some(),
            None => false,
        }
    }

    pub fn step(&mut self, dt: f32) -> u32 {
        let substeps = self.clock.accumulate(dt);
        let fixed_dt = self.clock.fixed_dt();
        for _ in 0..substeps {
            self.bodies.integrate(fixed_dt, self.config.gravity);
            let bodies = &self.bodies;
            self.broadphase.rebuild_pairs_parallel(bodies);
            let candidate_pairs = self.broadphase.candidate_pairs();
            let colliders = &self.colliders;
            let manifolds = self
                .narrowphase
                .rebuild_manifolds(bodies, candidate_pairs, |body| {
                    primary_shape_for_body(colliders, bodies, body)
                });
            self.solver.solve(&mut self.bodies, manifolds, fixed_dt);
            self.joint_solver
                .solve(&mut self.bodies, &self.active_joints, fixed_dt);
            Self::rebuild_contacts_from_manifolds(&mut self.contacts, manifolds);
            self.sleep_manager
                .update(&mut self.bodies, manifolds, &self.active_joints, fixed_dt);
        }
        substeps
    }

    fn rebuild_contacts_from_manifolds(
        contacts: &mut Vec<ContactPair>,
        manifolds: &[crate::body::ContactManifold],
    ) {
        contacts.clear();
        for manifold in manifolds {
            if contacts.len() < contacts.capacity() {
                contacts.push(ContactPair {
                    contact_id: manifold.contact_id,
                    collider_a: manifold.collider_a,
                    collider_b: manifold.collider_b,
                    body_a: Some(manifold.body_a),
                    body_b: Some(manifold.body_b),
                    point: manifold.point,
                    normal: manifold.normal,
                    penetration: manifold.penetration,
                    persisted_frames: manifold.persisted_frames,
                });
            }
        }
    }

    fn rebuild_body_bounds_from_colliders(&mut self, body: BodyHandle) {
        if !self.bodies.contains(body) {
            return;
        }
        let mut union: Option<Aabb> = None;
        let mut primary: Option<ColliderHandle> = None;
        for slot in &self.colliders {
            let Some(record) = slot.value.as_ref() else {
                continue;
            };
            if record.desc.body != Some(body) {
                continue;
            }
            primary.get_or_insert(record.handle);
            let local = record.desc.shape.local_aabb();
            union = Some(match union {
                Some(current) => current.union(local),
                None => local,
            });
        }
        let bounds = union.unwrap_or_default();
        let _ = self.bodies.set_local_bounds(body, bounds);
        let _ = self.bodies.set_primary_collider(body, primary);
    }

    fn sync_hot_loop_buffers(&mut self) {
        self.broadphase.sync_for_body_count(self.bodies.len());
        let target = self.broadphase.max_candidate_pairs_capacity();
        self.narrowphase.sync_for_pair_capacity(target);
        self.sleep_manager.prepare_for_bodies(self.bodies.len());
        if self.contacts.capacity() < target {
            self.contacts
                .reserve(target.saturating_sub(self.contacts.capacity()));
        }
        if self.active_joints.capacity() < self.joints.len() {
            self.active_joints.reserve(
                self.joints
                    .len()
                    .saturating_sub(self.active_joints.capacity()),
            );
        }
    }

    fn rebuild_active_joint_cache(&mut self) {
        self.active_joints.clear();
        for slot in &self.joints {
            if let Some(joint) = slot.value.as_ref() {
                self.active_joints.push(joint.clone());
            }
        }
    }
}

impl ParadoxScriptHostAbi for PhysicsWorld {
    fn spawn_body(&mut self, kind: u32, x: f32, y: f32, mass: f32) -> u32 {
        let Some(kind) = BodyKind::from_u32(kind) else {
            return 0;
        };
        PhysicsWorld::spawn_body(
            self,
            BodyDesc {
                kind,
                position: Vector3::new(x, y, 0.0),
                mass,
                ..BodyDesc::default()
            },
        )
        .raw()
    }

    fn spawn_collider(&mut self, body: u32, shape: u32, a: f32, b: f32) -> u32 {
        let Ok(body) = BodyHandle::try_from(PhysicsHandle::from(body)) else {
            return 0;
        };
        let Some(shape) = ColliderShapeKind::from_u32(shape) else {
            return 0;
        };
        PhysicsWorld::spawn_collider(
            self,
            ColliderDesc::attached(body, ColliderShape::from_script(shape, a, b)),
        )
        .map(ColliderHandle::raw)
        .unwrap_or(0)
    }

    fn release_handle(&mut self, handle: u32) -> bool {
        PhysicsWorld::release_handle(self, PhysicsHandle::from(handle))
    }

    fn apply_force(&mut self, body: u32, x: f32, y: f32) -> bool {
        let Ok(body) = BodyHandle::try_from(PhysicsHandle::from(body)) else {
            return false;
        };
        PhysicsWorld::apply_force(self, body, Vector3::new(x, y, 0.0))
    }

    fn set_velocity(&mut self, body: u32, x: f32, y: f32) -> bool {
        let Ok(body) = BodyHandle::try_from(PhysicsHandle::from(body)) else {
            return false;
        };
        PhysicsWorld::set_velocity(self, body, Vector3::new(x, y, 0.0))
    }

    fn query_contacts(&mut self, body: u32) -> u32 {
        let Ok(body) = BodyHandle::try_from(PhysicsHandle::from(body)) else {
            return 0;
        };
        PhysicsWorld::query_contacts(self, body)
            .map(ContactHandle::raw)
            .unwrap_or(0)
    }

    fn contact_count(&self, contacts: u32) -> u32 {
        let Ok(contacts) = ContactHandle::try_from(PhysicsHandle::from(contacts)) else {
            return 0;
        };
        PhysicsWorld::contact_count(self, contacts)
    }

    fn step_world(&mut self, dt: f32) -> u32 {
        PhysicsWorld::step(self, dt)
    }
}

fn primary_shape_for_body(
    colliders: &[Slot<ColliderRecord>],
    bodies: &BodyRegistry,
    body: BodyHandle,
) -> Option<(ColliderHandle, ColliderShape)> {
    let collider = bodies.primary_collider(body)?;
    let record = get_slot(colliders, collider.erased())?.value.as_ref()?;
    Some((collider, record.desc.shape.clone()))
}

fn alloc_slot<'a, T>(
    slots: &'a mut Vec<Slot<T>>,
    free_list: &mut Vec<u16>,
) -> (u16, u16, &'a mut Option<T>) {
    if let Some(index) = free_list.pop() {
        let slot = &mut slots[index as usize];
        let generation = slot.generation;
        return (index, generation, &mut slot.value);
    }
    let index = slots.len() as u16;
    slots.push(Slot::default());
    let slot = &mut slots[index as usize];
    (index, slot.generation, &mut slot.value)
}

fn get_slot<T>(slots: &[Slot<T>], handle: PhysicsHandle) -> Option<&Slot<T>> {
    let slot = slots.get(handle.index())?;
    if slot.generation == handle.generation() && slot.value.is_some() {
        Some(slot)
    } else {
        None
    }
}

fn free_slot<T>(
    slots: &mut [Slot<T>],
    free_list: &mut Vec<u16>,
    handle: PhysicsHandle,
) -> Option<T> {
    let slot = slots.get_mut(handle.index())?;
    if slot.generation != handle.generation() {
        return None;
    }
    let value = slot.value.take()?;
    slot.generation = slot.generation.wrapping_add(1).max(1);
    free_list.push(handle.index() as u16);
    Some(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dynamic_body_integrates_with_force() {
        let mut world = PhysicsWorld::new(PhysicsWorldConfig::default());
        let body = world.spawn_body(BodyDesc::default());
        assert!(world.apply_force(body, Vector3::new(10.0, 0.0, 0.0)));
        let steps = world.step(1.0 / 60.0);
        assert!(steps > 0);
        let body = world.body(body).unwrap();
        assert!(body.position.x > 0.0);
    }

    #[test]
    fn overlapping_colliders_produce_contact_snapshot() {
        let mut world = PhysicsWorld::new(PhysicsWorldConfig::default());
        let a = world.spawn_body(BodyDesc {
            kind: BodyKind::Static,
            position: Vector3::new(0.0, 0.0, 0.0),
            ..BodyDesc::default()
        });
        let b = world.spawn_body(BodyDesc {
            kind: BodyKind::Static,
            position: Vector3::new(0.5, 0.0, 0.0),
            ..BodyDesc::default()
        });
        world
            .spawn_collider(ColliderDesc::attached(
                a,
                ColliderShape::Aabb {
                    half_extents: Vector3::new(1.0, 1.0, 1.0),
                },
            ))
            .unwrap();
        world
            .spawn_collider(ColliderDesc::attached(
                b,
                ColliderShape::Aabb {
                    half_extents: Vector3::new(1.0, 1.0, 1.0),
                },
            ))
            .unwrap();
        world.step(1.0 / 60.0);
        let snapshot = world.query_contacts(a).unwrap();
        assert_eq!(world.contact_count(snapshot), 1);
        assert!(world.release_handle(snapshot.erased()));
    }

    #[test]
    fn releasing_body_invalidates_stale_handle() {
        let mut world = PhysicsWorld::new(PhysicsWorldConfig::default());
        let body = world.spawn_body(BodyDesc::default());
        assert!(world.release_handle(body.erased()));
        assert!(world.body(body).is_none());
        assert!(!world.release_handle(body.erased()));
    }

    #[test]
    fn broadphase_hot_loop_reuses_buffers() {
        let mut world = PhysicsWorld::new(PhysicsWorldConfig::default());
        for i in 0..32 {
            let body = world.spawn_body(BodyDesc {
                kind: BodyKind::Static,
                position: Vector3::new(i as f32 * 0.75, 0.0, 0.0),
                ..BodyDesc::default()
            });
            world
                .spawn_collider(ColliderDesc::attached(
                    body,
                    ColliderShape::Aabb {
                        half_extents: Vector3::new(1.0, 1.0, 1.0),
                    },
                ))
                .unwrap();
        }
        let pair_capacity = world.broadphase().max_candidate_pairs_capacity();
        let contact_capacity = world.contacts.capacity();
        let _ = world.step(1.0 / 60.0);
        assert_eq!(
            world.broadphase().max_candidate_pairs_capacity(),
            pair_capacity
        );
        assert_eq!(world.contacts.capacity(), contact_capacity);
        assert!(!world.broadphase().stats().overflowed);
    }
}
