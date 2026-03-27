//! Fixed-step ParadoxPE world skeleton built on SoA body storage and a parallel broadphase.
//!
//! This is still a foundation layer, not the final solver. The goal is to lock down the data flow:
//! dense generational handles, cache-friendly body storage, allocation-free hot stepping, and a
//! broadphase pipeline that can scale across Tileline's CPU task execution model.

use std::time::{Duration, Instant};

use mps::{
    DispatcherDoubleBufferedTransforms, DispatcherTransformSample, DoubleBufferedTransformStorage,
    TransformSample,
};
use nalgebra::{UnitQuaternion, Vector3};

use crate::abi::ParadoxScriptHostAbi;
use crate::body::{
    Aabb, BodyDesc, BodyKind, ColliderDesc, ColliderMaterial, ColliderShape, ColliderShapeKind,
    ContactPair, ContactSnapshot, RigidBody,
};
use crate::broadphase::{BroadphaseConfig, BroadphasePipeline};
use crate::handle::{
    BodyHandle, ColliderHandle, ContactHandle, HandleKind, JointHandle, PhysicsHandle,
};
use crate::joint::{
    DistanceJoint, DistanceJointDesc, FixedJoint, FixedJointDesc, JointConstraint,
    JointConstraintSolver, JointSolverConfig,
};
use crate::narrowphase::{NarrowphaseConfig, NarrowphasePipeline};
use crate::sleep::{SleepConfig, SleepIslandManager};
use crate::snapshot::{BodyStateFrame, PhysicsInterpolationBuffer, PhysicsSnapshot};
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

    pub fn max_substeps(&self) -> u32 {
        self.max_substeps
    }

    pub fn tick(&self) -> u64 {
        self.tick
    }

    /// Fractional interpolation alpha for render-side snapshot blending.
    ///
    /// `0.0` means exactly on the previous fixed-step snapshot,
    /// `1.0` means exactly on the latest fixed-step snapshot.
    pub fn interpolation_alpha(&self) -> f32 {
        if self.fixed_dt <= f32::EPSILON {
            1.0
        } else {
            (self.accumulator / self.fixed_dt).clamp(0.0, 1.0)
        }
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

    pub fn set_timestep(&mut self, fixed_dt: f32, max_substeps: u32) {
        self.fixed_dt = fixed_dt.max(1e-4);
        self.max_substeps = max_substeps.max(1);
        self.accumulator = self
            .accumulator
            .min(self.fixed_dt * self.max_substeps as f32);
    }
}

/// World configuration for the ParadoxPE foundation runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PhysicsSimulationMode {
    #[default]
    Spatial3d,
    Flat2d,
}

impl PhysicsSimulationMode {
    #[inline]
    pub const fn is_flat_2d(self) -> bool {
        matches!(self, Self::Flat2d)
    }
}

/// World configuration for the ParadoxPE foundation runtime.
#[derive(Debug, Clone, PartialEq)]
pub struct PhysicsWorldConfig {
    pub simulation_mode: PhysicsSimulationMode,
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
            simulation_mode: PhysicsSimulationMode::Spatial3d,
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

/// Per-substep timing breakdown for identifying physics pipeline bottlenecks.
/// All durations are summed across all substeps in a single `step()` call.
#[derive(Debug, Clone, Copy, Default)]
pub struct PhysicsStepTimings {
    pub substeps: u32,
    pub integrate_us: u64,
    pub broadphase_us: u64,
    pub narrowphase_us: u64,
    pub solver_us: u64,
    pub sleep_us: u64,
    pub snapshot_us: u64,
}

impl PhysicsStepTimings {
    pub fn total_us(&self) -> u64 {
        self.integrate_us
            + self.broadphase_us
            + self.narrowphase_us
            + self.solver_us
            + self.sleep_us
            + self.snapshot_us
    }
}

/// Precomputed execution plan for one `step()` call.
///
/// This separates cheap scheduling decisions from the hot substep loop so
/// external runtimes can eventually drive world phases through a custom
/// dispatcher without duplicating planning logic.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PhysicsStepExecutionPlan {
    /// Number of fixed substeps to process for this frame.
    pub substeps: u32,
    /// Fixed simulation delta used by all substeps.
    pub fixed_dt: f32,
    /// Sleep graph update cadence under high load.
    pub sleep_update_stride: usize,
}

pub struct PhysicsWorld {
    config: PhysicsWorldConfig,
    flat_2d_plane_z: f32,
    clock: FixedStepClock,
    bodies: BodyRegistry,
    broadphase: BroadphasePipeline,
    narrowphase: NarrowphasePipeline,
    solver: ContactSolver,
    joint_solver: JointConstraintSolver,
    sleep_manager: SleepIslandManager,
    colliders: Vec<Slot<ColliderRecord>>,
    free_colliders: Vec<u16>,
    joints: Vec<Slot<JointConstraint>>,
    free_joints: Vec<u16>,
    active_joints: Vec<JointConstraint>,
    contact_snapshots: Vec<Slot<ContactSnapshot>>,
    free_contact_snapshots: Vec<u16>,
    contacts: Vec<ContactPair>,
    integration_shards: Vec<std::ops::Range<usize>>,
    interpolation: PhysicsInterpolationBuffer,
    pub last_step_timings: PhysicsStepTimings,
}

impl PhysicsWorld {
    pub fn new(mut config: PhysicsWorldConfig) -> Self {
        if config.simulation_mode.is_flat_2d() {
            config.gravity = flatten_xy_vector(config.gravity);
        }
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
            flat_2d_plane_z: 0.0,
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
            integration_shards: Vec::new(),
            interpolation: PhysicsInterpolationBuffer::default(),
            last_step_timings: PhysicsStepTimings::default(),
        }
    }

    pub fn config(&self) -> &PhysicsWorldConfig {
        &self.config
    }

    pub fn fixed_step_clock(&self) -> &FixedStepClock {
        &self.clock
    }

    /// Current dense body count in the world.
    pub fn body_count(&self) -> usize {
        self.bodies.len()
    }

    /// Current world gravity vector applied to dynamic body integration.
    pub fn gravity(&self) -> Vector3<f32> {
        self.config.gravity
    }

    /// Current simulation mode (`Spatial3d` or `Flat2d`).
    pub fn simulation_mode(&self) -> PhysicsSimulationMode {
        self.config.simulation_mode
    }

    /// Current Z plane used by flat-2D simulation mode.
    pub fn flat_2d_plane_z(&self) -> f32 {
        self.flat_2d_plane_z
    }

    /// Set the Z depth plane for flat-2D simulation.
    ///
    /// This keeps XY simulation semantics while allowing runtime scene coordinates to anchor the
    /// 2D world on a deterministic Z slice in 3D space.
    pub fn set_flat_2d_plane_z(&mut self, plane_z: f32) {
        if (self.flat_2d_plane_z - plane_z).abs() <= f32::EPSILON {
            return;
        }
        self.flat_2d_plane_z = plane_z;
        if self.config.simulation_mode.is_flat_2d() {
            let _ = self.enforce_flat_2d_constraints();
        }
    }

    /// Switch simulation mode at runtime.
    ///
    /// `Flat2d` keeps world state on XY plane (Z locked) and preserves only Z-axis rotation.
    pub fn set_simulation_mode(&mut self, mode: PhysicsSimulationMode) {
        if self.config.simulation_mode == mode {
            return;
        }
        self.config.simulation_mode = mode;
        if mode.is_flat_2d() {
            self.config.gravity = flatten_xy_vector(self.config.gravity);
            let _ = self.enforce_flat_2d_constraints();
        }
    }

    /// Update gravity at runtime without rebuilding the world.
    pub fn set_gravity(&mut self, gravity: Vector3<f32>) {
        self.config.gravity = if self.config.simulation_mode.is_flat_2d() {
            flatten_xy_vector(gravity)
        } else {
            gravity
        };
    }

    /// Update fixed-step scheduler parameters at runtime.
    ///
    /// This is useful for adaptive tick controllers that need to react to render pacing and
    /// workload pressure without rebuilding the world.
    pub fn set_timestep(&mut self, fixed_dt: f32, max_substeps: u32) {
        self.config.fixed_dt = fixed_dt.max(1e-4);
        self.config.max_substeps = max_substeps.max(1);
        self.clock
            .set_timestep(self.config.fixed_dt, self.config.max_substeps);
    }

    /// Update contact solver tuning at runtime without rebuilding world state.
    pub fn set_solver_config(&mut self, solver: ContactSolverConfig) {
        self.config.solver = solver.clone();
        self.solver.set_config(solver);
    }

    /// Apply a high-level anti-penetration guard profile.
    ///
    /// `level` range is `[0.0, 1.0]`:
    /// - `0.0`: softer and faster
    /// - `1.0`: tighter separation, less interpenetration
    pub fn set_contact_guard(&mut self, level: f32) -> bool {
        let level = level.clamp(0.0, 1.0);
        let mut solver = self.config.solver.clone();
        solver.iterations = lerp_u32(4, 10, level);
        solver.baumgarte = lerp_f32(0.22, 0.62, level);
        solver.penetration_slop = lerp_f32(0.0035, 0.0005, level);
        solver.max_position_correction_per_iteration = lerp_f32(0.06, 0.18, level);
        solver.warmstart_impulse_decay = lerp_f32(0.86, 0.96, level);
        solver.persistent_contact_boost = lerp_f32(0.12, 0.34, level);
        solver.parallel_contact_push_strength = lerp_f32(0.18, 0.62, level);
        solver.parallel_contact_push_threshold = lerp_usize(640, 96, level);
        solver.hard_position_projection_strength = lerp_f32(0.90, 1.85, level);
        solver.hard_position_projection_threshold = lerp_usize(192, 24, level);
        solver.max_projection_per_contact = lerp_f32(0.12, 0.42, level);

        if solver == self.config.solver {
            return false;
        }
        self.set_solver_config(solver);
        true
    }

    /// Update broadphase swept-AABB speculative settings.
    pub fn set_broadphase_speculative_sweep(&mut self, enabled: bool, max_distance: f32) -> bool {
        let max_distance = max_distance.max(0.0);
        if self.config.broadphase.speculative_sweep == enabled
            && (self.config.broadphase.speculative_max_distance - max_distance).abs()
                <= f32::EPSILON
        {
            return false;
        }
        self.config.broadphase.speculative_sweep = enabled;
        self.config.broadphase.speculative_max_distance = max_distance;
        self.broadphase
            .set_speculative_sweep_config(enabled, max_distance);
        true
    }

    /// Update narrowphase speculative contact settings.
    pub fn set_narrowphase_speculative_contacts(
        &mut self,
        enabled: bool,
        contact_distance: f32,
        max_prediction_distance: f32,
    ) -> bool {
        let contact_distance = contact_distance.max(0.0);
        let max_prediction_distance = max_prediction_distance.max(contact_distance);
        if self.config.narrowphase.speculative_contacts == enabled
            && (self.config.narrowphase.speculative_contact_distance - contact_distance).abs()
                <= f32::EPSILON
            && (self.config.narrowphase.speculative_max_prediction_distance
                - max_prediction_distance)
                .abs()
                <= f32::EPSILON
        {
            return false;
        }
        self.config.narrowphase.speculative_contacts = enabled;
        self.config.narrowphase.speculative_contact_distance = contact_distance;
        self.config.narrowphase.speculative_max_prediction_distance = max_prediction_distance;
        self.narrowphase.set_speculative_contact_config(
            enabled,
            contact_distance,
            max_prediction_distance,
        );
        true
    }

    /// Current render interpolation alpha derived from fixed-step accumulator state.
    pub fn interpolation_alpha(&self) -> f32 {
        self.clock.interpolation_alpha()
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

    pub fn interpolation(&self) -> &PhysicsInterpolationBuffer {
        &self.interpolation
    }

    pub fn spawn_body(&mut self, desc: BodyDesc) -> BodyHandle {
        let desc = if self.config.simulation_mode.is_flat_2d() {
            BodyDesc {
                position: flatten_position_to_plane(desc.position, self.flat_2d_plane_z),
                rotation: flatten_to_z_axis_rotation(desc.rotation),
                linear_velocity: flatten_xy_vector(desc.linear_velocity),
                ..desc
            }
        } else {
            desc
        };
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
            .filter(|joint| {
                let (body_a, body_b) = joint.bodies();
                body_a == handle || body_b == handle
            })
            .map(|joint| joint.handle())
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
        *slot = Some(JointConstraint::Distance(DistanceJoint::from_desc(
            handle, desc,
        )));
        self.rebuild_active_joint_cache();
        self.sync_hot_loop_buffers();
        Some(handle)
    }

    pub fn spawn_fixed_joint(&mut self, desc: FixedJointDesc) -> Option<JointHandle> {
        self.bodies.body(desc.body_a)?;
        self.bodies.body(desc.body_b)?;
        let (index, generation, slot) = alloc_slot(&mut self.joints, &mut self.free_joints);
        let handle = JointHandle::new(index, generation);
        *slot = Some(JointConstraint::Fixed(FixedJoint::from_desc(handle, desc)));
        self.rebuild_active_joint_cache();
        self.sync_hot_loop_buffers();
        Some(handle)
    }

    pub fn lock_bodies_with_fixed_joint(
        &mut self,
        body_a: BodyHandle,
        body_b: BodyHandle,
    ) -> Option<JointHandle> {
        let position_a = self.bodies.position_for(body_a)?;
        let position_b = self.bodies.position_for(body_b)?;
        self.spawn_fixed_joint(FixedJointDesc::with_offset(
            body_a,
            body_b,
            position_b - position_a,
        ))
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

    /// Update one collider's material in-place.
    ///
    /// Returns `true` when the collider exists and the material changed.
    pub fn set_collider_material(
        &mut self,
        handle: ColliderHandle,
        material: ColliderMaterial,
    ) -> bool {
        let Some(slot) = get_slot_mut(&mut self.colliders, handle.erased()) else {
            return false;
        };
        let Some(record) = slot.value.as_mut() else {
            return false;
        };
        let material = ColliderMaterial {
            restitution: material.restitution.max(0.0),
            friction: material.friction.max(0.0),
        };
        if record.desc.material == material {
            return false;
        }
        record.desc.material = material;
        true
    }

    /// Bulk-update collider materials by packed collision filter group (`user_tag` lower 16 bits).
    ///
    /// Returns number of colliders whose material changed.
    pub fn set_collider_material_for_collision_group(
        &mut self,
        group: u16,
        material: ColliderMaterial,
    ) -> usize {
        let group = group as u32;
        let material = ColliderMaterial {
            restitution: material.restitution.max(0.0),
            friction: material.friction.max(0.0),
        };
        let mut updated = 0usize;
        for slot in &mut self.colliders {
            let Some(record) = slot.value.as_mut() else {
                continue;
            };
            if (record.desc.user_tag & 0xFFFF) != group {
                continue;
            }
            if record.desc.material == material {
                continue;
            }
            record.desc.material = material;
            updated = updated.saturating_add(1);
        }
        updated
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
        let force = if self.config.simulation_mode.is_flat_2d() {
            flatten_xy_vector(force)
        } else {
            force
        };
        self.bodies.apply_force(body, force)
    }

    pub fn set_velocity(&mut self, body: BodyHandle, velocity: Vector3<f32>) -> bool {
        let velocity = if self.config.simulation_mode.is_flat_2d() {
            flatten_xy_vector(velocity)
        } else {
            velocity
        };
        self.bodies.set_velocity(body, velocity)
    }

    /// Move one body to a world-space position and rebuild its broadphase bounds.
    pub fn set_position(&mut self, body: BodyHandle, position: Vector3<f32>) -> bool {
        let position = if self.config.simulation_mode.is_flat_2d() {
            flatten_position_to_plane(position, self.flat_2d_plane_z)
        } else {
            position
        };
        self.bodies.set_position(body, position)
    }

    pub fn set_linear_damping(&mut self, body: BodyHandle, damping: f32) -> bool {
        self.bodies.set_linear_damping(body, damping)
    }

    pub fn set_linear_damping_all_dynamic(&mut self, damping: f32) -> usize {
        self.bodies.set_linear_damping_all_dynamic(damping)
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

    /// Build a deterministic execution plan for the next fixed-step run.
    pub fn prepare_step_execution(&mut self, dt: f32) -> Option<PhysicsStepExecutionPlan> {
        let substeps = self.clock.accumulate(dt);
        if substeps == 0 {
            return None;
        }

        let body_count = self.bodies.len();
        let preferred_shards = std::thread::available_parallelism()
            .map(|threads| threads.get())
            .unwrap_or(1);
        self.bodies
            .plan_integration_shards(preferred_shards, 256, &mut self.integration_shards);

        // Sleep graph maintenance is useful but costly in huge fully-active scenes.
        let sleep_update_stride = if body_count >= 6_000 {
            4
        } else if body_count >= 3_500 {
            2
        } else {
            1
        };

        Some(PhysicsStepExecutionPlan {
            substeps,
            fixed_dt: self.clock.fixed_dt(),
            sleep_update_stride,
        })
    }

    /// Execute an already planned fixed-step run.
    pub fn step_with_execution_plan(&mut self, plan: PhysicsStepExecutionPlan) -> u32 {
        if plan.substeps == 0 {
            return 0;
        }

        let mut timings = PhysicsStepTimings {
            substeps: plan.substeps,
            ..PhysicsStepTimings::default()
        };

        self.capture_pre_step_snapshot(&mut timings);

        for step_index in 0..plan.substeps {
            self.execute_substep(&plan, step_index, &mut timings);
        }

        self.capture_post_step_snapshot(&mut timings);
        self.last_step_timings = timings;
        plan.substeps
    }

    /// Run one fixed-step update with deterministic planning + execution.
    pub fn step(&mut self, dt: f32) -> u32 {
        let Some(plan) = self.prepare_step_execution(dt) else {
            return 0;
        };
        self.step_with_execution_plan(plan)
    }

    fn capture_pre_step_snapshot(&mut self, timings: &mut PhysicsStepTimings) {
        // Keep interpolation snapshots bounded to one pre-step capture per render frame.
        let t = Instant::now();
        self.interpolation.push_snapshot(self.capture_snapshot());
        timings.snapshot_us += duration_us(t.elapsed());
    }

    fn capture_post_step_snapshot(&mut self, timings: &mut PhysicsStepTimings) {
        // Keep interpolation snapshots bounded to one post-step capture per render frame.
        let t = Instant::now();
        self.interpolation.push_snapshot(self.capture_snapshot());
        timings.snapshot_us += duration_us(t.elapsed());
    }

    fn execute_substep(
        &mut self,
        plan: &PhysicsStepExecutionPlan,
        step_index: u32,
        timings: &mut PhysicsStepTimings,
    ) {
        let t = Instant::now();
        self.bodies.integrate_with_shards(
            plan.fixed_dt,
            self.config.gravity,
            &self.integration_shards,
        );
        timings.integrate_us += duration_us(t.elapsed());

        let t = Instant::now();
        self.broadphase.set_predictive_dt(plan.fixed_dt);
        self.narrowphase.set_predictive_dt(plan.fixed_dt);
        let bodies = &self.bodies;
        self.broadphase.rebuild_pairs_parallel(bodies);
        timings.broadphase_us += duration_us(t.elapsed());

        let is_last = step_index + 1 == plan.substeps;
        let on_stride = plan.substeps as usize > plan.sleep_update_stride
            && step_index as usize % plan.sleep_update_stride == 0;

        {
            let t = Instant::now();
            let candidate_pairs = self.broadphase.candidate_pairs();
            let colliders = &self.colliders;
            let manifolds = self
                .narrowphase
                .rebuild_manifolds(bodies, candidate_pairs, |body| {
                    primary_shape_for_body(colliders, bodies, body)
                });
            timings.narrowphase_us += duration_us(t.elapsed());

            let t = Instant::now();
            self.solver
                .solve(&mut self.bodies, manifolds, plan.fixed_dt);
            self.joint_solver
                .solve(&mut self.bodies, &self.active_joints, plan.fixed_dt);
            Self::rebuild_contacts_from_manifolds(&mut self.contacts, manifolds);
            timings.solver_us += duration_us(t.elapsed());

            if is_last || on_stride {
                let t = Instant::now();
                self.sleep_manager.update(
                    &mut self.bodies,
                    manifolds,
                    &self.active_joints,
                    plan.fixed_dt,
                );
                timings.sleep_us += duration_us(t.elapsed());
            }
        }

        if self.config.simulation_mode.is_flat_2d() {
            let t = Instant::now();
            let _ = self.enforce_flat_2d_constraints();
            timings.sleep_us += duration_us(t.elapsed());
        }
    }

    pub fn capture_snapshot(&self) -> PhysicsSnapshot {
        let read = self.bodies.read_domain();
        let body_len = read.handles.len();
        let mut bodies_out = Vec::with_capacity(body_len);
        for index in 0..body_len {
            bodies_out.push(BodyStateFrame {
                handle: read.handles[index],
                position: read.positions[index],
                rotation: read.rotations[index],
                linear_velocity: read.linear_velocities[index],
                awake: read.awake[index],
            });
        }
        PhysicsSnapshot {
            tick: self.clock.tick(),
            bodies: bodies_out,
        }
    }

    /// Export the latest body transforms into MPS-owned double-buffered storage.
    pub fn write_render_transforms_to_storage(
        &self,
        storage: &DoubleBufferedTransformStorage,
        slot: usize,
    ) -> usize {
        let read = self.bodies.read_domain();
        let body_len = read.handles.len().min(storage.capacity());
        for index in 0..body_len {
            let position = read.positions[index];
            let rotation = read.rotations[index];
            let q = rotation.quaternion();
            storage.write_transform_to_slot(
                slot,
                index,
                TransformSample {
                    position: [position.x, position.y, position.z],
                    rotation: [q.i, q.j, q.k, q.w],
                },
            );
        }
        body_len
    }

    /// Export the latest body transforms into the bare-metal dispatcher buffers.
    pub fn write_render_transforms_to_dispatcher_storage(
        &self,
        storage: &DispatcherDoubleBufferedTransforms,
        slot: usize,
    ) -> usize {
        let read = self.bodies.read_domain();
        let body_len = read.handles.len().min(storage.capacity());
        for index in 0..body_len {
            let position = read.positions[index];
            let rotation = read.rotations[index];
            let q = rotation.quaternion();
            storage.write_transform_to_slot(
                slot,
                index,
                DispatcherTransformSample {
                    position: [position.x, position.y, position.z],
                    rotation: [q.i, q.j, q.k, q.w],
                },
            );
        }
        body_len
    }

    pub fn restore_snapshot(&mut self, snapshot: &PhysicsSnapshot) -> usize {
        let mut restored = 0usize;
        for frame in &snapshot.bodies {
            let Some(dense) = self.bodies.dense_index_of(frame.handle) else {
                continue;
            };
            let position = if self.config.simulation_mode.is_flat_2d() {
                flatten_position_to_plane(frame.position, self.flat_2d_plane_z)
            } else {
                frame.position
            };
            let rotation = if self.config.simulation_mode.is_flat_2d() {
                flatten_to_z_axis_rotation(frame.rotation)
            } else {
                frame.rotation
            };
            let linear_velocity = if self.config.simulation_mode.is_flat_2d() {
                flatten_xy_vector(frame.linear_velocity)
            } else {
                frame.linear_velocity
            };
            self.bodies.positions[dense] = position;
            self.bodies.rotations[dense] = rotation;
            self.bodies.linear_velocities[dense] = linear_velocity;
            self.bodies.awake[dense] = frame.awake;
            self.bodies.recompute_world_aabb_for_dense(dense);
            restored += 1;
        }
        self.clock.tick = snapshot.tick;
        self.clock.accumulator = 0.0;
        restored
    }

    pub fn push_interpolation_snapshot(&mut self) {
        self.interpolation.push_snapshot(self.capture_snapshot());
    }

    pub fn interpolate_body_pose(
        &self,
        body: BodyHandle,
        alpha: f32,
    ) -> Option<crate::snapshot::InterpolatedBodyPose> {
        self.interpolation.interpolate_body(body, alpha)
    }

    /// Enforce XY-plane constraints for `Flat2d` mode.
    ///
    /// This keeps runtime, script, and physics transforms coherent when side-view 2D mode is
    /// enabled and prevents solver drift from leaking bodies away from Z=0.
    fn enforce_flat_2d_constraints(&mut self) -> usize {
        if !self.config.simulation_mode.is_flat_2d() {
            return 0;
        }
        let mut corrected = 0usize;
        for dense in 0..self.bodies.len() {
            let mut changed = false;

            if (self.bodies.positions[dense].z - self.flat_2d_plane_z).abs() > 1e-6 {
                self.bodies.positions[dense].z = self.flat_2d_plane_z;
                changed = true;
            }
            if self.bodies.linear_velocities[dense].z.abs() > 1e-6 {
                self.bodies.linear_velocities[dense].z = 0.0;
                changed = true;
            }

            let flattened = flatten_to_z_axis_rotation(self.bodies.rotations[dense]);
            if self.bodies.rotations[dense].angle_to(&flattened) > 1e-6 {
                self.bodies.rotations[dense] = flattened;
                changed = true;
            }

            if changed {
                self.bodies.recompute_world_aabb_for_dense(dense);
                corrected = corrected.saturating_add(1);
            }
        }
        corrected
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
) -> Option<(
    ColliderHandle,
    ColliderShape,
    crate::body::ColliderMaterial,
    u32,
)> {
    let collider = bodies.primary_collider(body)?;
    let record = get_slot(colliders, collider.erased())?.value.as_ref()?;
    Some((
        collider,
        record.desc.shape.clone(),
        record.desc.material,
        record.desc.user_tag,
    ))
}

#[inline]
fn flatten_xy_vector(mut value: Vector3<f32>) -> Vector3<f32> {
    value.z = 0.0;
    value
}

#[inline]
fn flatten_position_to_plane(mut value: Vector3<f32>, plane_z: f32) -> Vector3<f32> {
    value.z = plane_z;
    value
}

#[inline]
fn flatten_to_z_axis_rotation(rotation: UnitQuaternion<f32>) -> UnitQuaternion<f32> {
    let (_, _, yaw) = rotation.euler_angles();
    UnitQuaternion::from_axis_angle(&Vector3::z_axis(), yaw)
}

fn duration_us(d: Duration) -> u64 {
    d.as_micros() as u64
}

fn lerp_f32(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t.clamp(0.0, 1.0)
}

#[inline]
fn lerp_usize(a: usize, b: usize, t: f32) -> usize {
    lerp_f32(a as f32, b as f32, t).round() as usize
}

#[inline]
fn lerp_u32(a: u32, b: u32, t: f32) -> u32 {
    lerp_f32(a as f32, b as f32, t).round() as u32
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

fn get_slot_mut<T>(slots: &mut [Slot<T>], handle: PhysicsHandle) -> Option<&mut Slot<T>> {
    let slot = slots.get_mut(handle.index())?;
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
    use crate::joint::FixedJointDesc;

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

    #[test]
    fn fixed_joint_can_be_spawned_from_world() {
        let mut world = PhysicsWorld::new(PhysicsWorldConfig::default());
        let a = world.spawn_body(BodyDesc::default());
        let b = world.spawn_body(BodyDesc {
            position: Vector3::new(2.0, 0.0, 0.0),
            ..BodyDesc::default()
        });
        let joint = world
            .spawn_fixed_joint(FixedJointDesc::with_offset(
                a,
                b,
                Vector3::new(1.0, 0.0, 0.0),
            ))
            .unwrap();
        assert!(world.release_handle(joint.erased()));
    }

    #[test]
    fn world_can_restore_and_interpolate_snapshots() {
        let mut world = PhysicsWorld::new(PhysicsWorldConfig::default());
        let body = world.spawn_body(BodyDesc::default());
        let first = world.capture_snapshot();
        world.set_velocity(body, Vector3::new(10.0, 0.0, 0.0));
        let _ = world.step(1.0 / 60.0);
        let second = world.capture_snapshot();
        world.restore_snapshot(&first);
        assert!(world.body(body).unwrap().position.x.abs() < 1e-5);

        world.interpolation.push_snapshot(first.clone());
        world.interpolation.push_snapshot(second.clone());
        let pose = world.interpolate_body_pose(body, 0.5).unwrap();
        assert!(pose.position.x >= 0.0);
    }

    #[test]
    fn world_timestep_can_be_updated_at_runtime() {
        let mut world = PhysicsWorld::new(PhysicsWorldConfig::default());
        world.set_timestep(1.0 / 480.0, 24);
        assert!((world.config().fixed_dt - (1.0 / 480.0)).abs() < 1e-6);
        assert_eq!(world.config().max_substeps, 24);
        assert!((world.fixed_step_clock().fixed_dt() - (1.0 / 480.0)).abs() < 1e-6);
        assert_eq!(world.fixed_step_clock().max_substeps(), 24);
    }

    #[test]
    fn flat_2d_mode_flattens_spawn_and_runtime_updates() {
        let mut world = PhysicsWorld::new(PhysicsWorldConfig {
            simulation_mode: PhysicsSimulationMode::Flat2d,
            gravity: Vector3::new(0.0, -9.81, 3.0),
            ..PhysicsWorldConfig::default()
        });
        assert_eq!(world.gravity().z, 0.0);

        let body = world.spawn_body(BodyDesc {
            position: Vector3::new(1.0, 2.0, 6.0),
            linear_velocity: Vector3::new(0.5, -0.3, 4.0),
            rotation: UnitQuaternion::from_euler_angles(0.6, -0.4, 0.3),
            ..BodyDesc::default()
        });
        let snapshot = world.body(body).expect("body");
        assert!(snapshot.position.z.abs() <= 1e-6);
        assert!(snapshot.linear_velocity.z.abs() <= 1e-6);
        let (roll, pitch, _) = snapshot.rotation.euler_angles();
        assert!(roll.abs() <= 1e-5);
        assert!(pitch.abs() <= 1e-5);

        assert!(world.set_position(body, Vector3::new(3.0, 4.0, -9.0)));
        assert!(world.set_velocity(body, Vector3::new(0.0, 1.0, -7.0)));
        let updated = world.body(body).expect("body");
        assert!(updated.position.z.abs() <= 1e-6);
        assert!(updated.linear_velocity.z.abs() <= 1e-6);
    }

    #[test]
    fn enabling_flat_2d_mode_runtime_flattens_existing_bodies() {
        let mut world = PhysicsWorld::new(PhysicsWorldConfig::default());
        let body = world.spawn_body(BodyDesc {
            position: Vector3::new(0.0, 0.0, 3.0),
            linear_velocity: Vector3::new(0.0, 0.0, 2.0),
            rotation: UnitQuaternion::from_euler_angles(0.25, 0.15, -0.4),
            ..BodyDesc::default()
        });

        world.set_simulation_mode(PhysicsSimulationMode::Flat2d);
        assert_eq!(world.simulation_mode(), PhysicsSimulationMode::Flat2d);

        let flattened = world.body(body).expect("body");
        assert!(flattened.position.z.abs() <= 1e-6);
        assert!(flattened.linear_velocity.z.abs() <= 1e-6);
        let (roll, pitch, _) = flattened.rotation.euler_angles();
        assert!(roll.abs() <= 1e-5);
        assert!(pitch.abs() <= 1e-5);

        let _ = world.step(1.0 / 60.0);
        let after_step = world.body(body).expect("body");
        assert!(after_step.position.z.abs() <= 1e-6);
        assert!(after_step.linear_velocity.z.abs() <= 1e-6);
    }

    #[test]
    fn flat_2d_plane_z_can_be_configured() {
        let mut world = PhysicsWorld::new(PhysicsWorldConfig {
            simulation_mode: PhysicsSimulationMode::Flat2d,
            ..PhysicsWorldConfig::default()
        });
        world.set_flat_2d_plane_z(2.75);
        assert!((world.flat_2d_plane_z() - 2.75).abs() < 1e-6);

        let body = world.spawn_body(BodyDesc {
            position: Vector3::new(0.0, 0.0, -5.0),
            linear_velocity: Vector3::new(0.0, 1.0, 9.0),
            ..BodyDesc::default()
        });
        let snapshot = world.body(body).expect("body");
        assert!((snapshot.position.z - 2.75).abs() < 1e-6);
        assert!(snapshot.linear_velocity.z.abs() <= 1e-6);

        world.set_flat_2d_plane_z(-1.25);
        let shifted = world.body(body).expect("body");
        assert!((shifted.position.z + 1.25).abs() < 1e-6);
    }

    #[test]
    fn collider_material_can_be_retuned_by_collision_group() {
        let mut world = PhysicsWorld::new(PhysicsWorldConfig::default());
        let body = world.spawn_body(BodyDesc::default());
        let collider = world
            .spawn_collider(ColliderDesc {
                body: Some(body),
                shape: ColliderShape::Sphere { radius: 0.5 },
                material: crate::body::ColliderMaterial {
                    restitution: 0.2,
                    friction: 0.4,
                },
                is_sensor: false,
                user_tag: (3u32) | ((0xFFFFu32) << 16),
            })
            .expect("collider");
        let changed = world.set_collider_material_for_collision_group(
            3,
            crate::body::ColliderMaterial {
                restitution: 0.85,
                friction: 0.12,
            },
        );
        assert_eq!(changed, 1);
        let slot = get_slot(&world.colliders, collider.erased())
            .and_then(|slot| slot.value.as_ref())
            .expect("slot");
        assert!((slot.desc.material.restitution - 0.85).abs() < 1e-6);
        assert!((slot.desc.material.friction - 0.12).abs() < 1e-6);
    }
}
