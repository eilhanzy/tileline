//! First-pass sequential impulse contact solver for ParadoxPE.
//!
//! This revision adds:
//! - normal impulse accumulation
//! - tangent/friction impulse accumulation
//! - reusable warm-start cache keyed by collider pairs
//! - allocation-aware scratch buffers sized ahead of the hot loop

use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};

use nalgebra::Vector3;
use rayon::prelude::*;

use crate::body::{BodyKind, ContactId, ContactManifold};
use crate::storage::BodyRegistry;

#[derive(Debug, Clone, Copy, PartialEq)]
struct CachedContactImpulse {
    contact_id: ContactId,
    normal_impulse: f32,
    tangent_impulse: f32,
}

/// Solver tuning configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct ContactSolverConfig {
    pub iterations: u32,
    pub baumgarte: f32,
    pub penetration_slop: f32,
    /// Maximum per-contact position correction applied in one sequential impulse iteration.
    pub max_position_correction_per_iteration: f32,
    pub restitution_velocity_threshold: f32,
    pub warm_starting: bool,
    /// Damping factor for cached warm-start impulses.
    pub warmstart_impulse_decay: f32,
    /// Persistent-contact multiplier used to slightly boost penetration resolution.
    pub persistent_contact_boost: f32,
    /// Extra parallel repulsion pass applied on deep overlaps.
    pub parallel_contact_push_strength: f32,
    /// Minimum manifold count before enabling the parallel repulsion pass.
    pub parallel_contact_push_threshold: usize,
    /// Extra post-iteration position projection for deep overlaps.
    pub hard_position_projection_strength: f32,
    /// Minimum manifold count before enabling hard position projection.
    pub hard_position_projection_threshold: usize,
    /// Clamp on per-manifold projection depth to avoid explosive corrections.
    pub max_projection_per_contact: f32,
    /// Tangential speed threshold used to blend static-like to kinetic-like friction.
    ///
    /// `0` disables speed-aware friction scaling and keeps legacy behavior.
    pub friction_transition_speed: f32,
    /// Friction multiplier near zero tangential speed (static-like region).
    pub friction_static_boost: f32,
    /// Friction multiplier at/above transition speed (kinetic-like region).
    pub friction_kinetic_scale: f32,
}

impl Default for ContactSolverConfig {
    fn default() -> Self {
        Self {
            iterations: 4,
            baumgarte: 0.2,
            penetration_slop: 0.005,
            max_position_correction_per_iteration: 0.08,
            restitution_velocity_threshold: 1.0,
            warm_starting: true,
            warmstart_impulse_decay: 0.92,
            persistent_contact_boost: 0.20,
            parallel_contact_push_strength: 0.24,
            parallel_contact_push_threshold: 384,
            hard_position_projection_strength: 0.90,
            hard_position_projection_threshold: 192,
            max_projection_per_contact: 0.12,
            friction_transition_speed: 1.2,
            friction_static_boost: 1.10,
            friction_kinetic_scale: 0.92,
        }
    }
}

/// Solver execution statistics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ContactSolverStats {
    pub manifolds: usize,
    pub iterations: u32,
    pub positional_corrections: u32,
    pub normal_impulses_applied: u32,
    pub friction_impulses_applied: u32,
    pub warmstart_hits: u32,
}

/// Reusable solver state holder.
#[derive(Debug)]
pub struct ContactSolver {
    config: ContactSolverConfig,
    stats: ContactSolverStats,
    cached_impulses: Vec<CachedContactImpulse>,
    next_cached_impulses: Vec<CachedContactImpulse>,
    cached_lookup: HashMap<ContactId, CachedContactImpulse>,
    normal_impulses: Vec<f32>,
    tangent_impulses: Vec<f32>,
    touched_body_flags: Vec<u8>,
    touched_body_indices: Vec<usize>,
    push_delta_x: Vec<AtomicU32>,
    push_delta_y: Vec<AtomicU32>,
    push_delta_z: Vec<AtomicU32>,
}

impl Clone for ContactSolver {
    fn clone(&self) -> Self {
        let clone_atomic_vec = |src: &[AtomicU32]| {
            src.iter()
                .map(|value| AtomicU32::new(value.load(Ordering::Relaxed)))
                .collect::<Vec<_>>()
        };

        Self {
            config: self.config.clone(),
            stats: self.stats,
            cached_impulses: self.cached_impulses.clone(),
            next_cached_impulses: self.next_cached_impulses.clone(),
            cached_lookup: self.cached_lookup.clone(),
            normal_impulses: self.normal_impulses.clone(),
            tangent_impulses: self.tangent_impulses.clone(),
            touched_body_flags: self.touched_body_flags.clone(),
            touched_body_indices: self.touched_body_indices.clone(),
            push_delta_x: clone_atomic_vec(&self.push_delta_x),
            push_delta_y: clone_atomic_vec(&self.push_delta_y),
            push_delta_z: clone_atomic_vec(&self.push_delta_z),
        }
    }
}

impl ContactSolver {
    pub fn new(config: ContactSolverConfig) -> Self {
        Self {
            config,
            stats: ContactSolverStats::default(),
            cached_impulses: Vec::new(),
            next_cached_impulses: Vec::new(),
            cached_lookup: HashMap::new(),
            normal_impulses: Vec::new(),
            tangent_impulses: Vec::new(),
            touched_body_flags: Vec::new(),
            touched_body_indices: Vec::new(),
            push_delta_x: Vec::new(),
            push_delta_y: Vec::new(),
            push_delta_z: Vec::new(),
        }
    }

    pub fn config(&self) -> &ContactSolverConfig {
        &self.config
    }

    pub fn set_config(&mut self, config: ContactSolverConfig) {
        self.config = config;
    }

    pub fn stats(&self) -> ContactSolverStats {
        self.stats
    }

    pub fn solve(&mut self, bodies: &mut BodyRegistry, manifolds: &[ContactManifold], _dt: f32) {
        self.stats = ContactSolverStats {
            manifolds: manifolds.len(),
            iterations: self.config.iterations,
            ..ContactSolverStats::default()
        };
        self.clear_touched_state();
        if manifolds.is_empty() {
            self.cached_impulses.clear();
            self.next_cached_impulses.clear();
            self.cached_lookup.clear();
            self.normal_impulses.clear();
            self.tangent_impulses.clear();
            return;
        }

        self.sync_buffers(manifolds.len(), bodies.len());
        self.next_cached_impulses.clear();
        self.normal_impulses.resize(manifolds.len(), 0.0);
        self.tangent_impulses.resize(manifolds.len(), 0.0);
        self.normal_impulses.fill(0.0);
        self.tangent_impulses.fill(0.0);
        self.rebuild_cached_lookup();

        if self.config.warm_starting {
            self.apply_warmstart(bodies, manifolds);
        }

        self.apply_parallel_contact_push(bodies, manifolds);

        let iterations = self.effective_iteration_budget(manifolds.len());
        self.stats.iterations = iterations;
        for _ in 0..iterations {
            for (index, manifold) in manifolds.iter().enumerate() {
                self.solve_manifold(bodies, manifold, index);
            }
        }
        self.apply_hard_position_projection(bodies, manifolds);
        self.flush_touched_positions(bodies);

        for (index, manifold) in manifolds.iter().enumerate() {
            let normal_impulse = self.normal_impulses[index];
            let tangent_impulse = self.tangent_impulses[index];
            if normal_impulse <= f32::EPSILON && tangent_impulse.abs() <= f32::EPSILON {
                continue;
            }
            if self.next_cached_impulses.len() < self.next_cached_impulses.capacity() {
                self.next_cached_impulses.push(CachedContactImpulse {
                    contact_id: manifold.contact_id,
                    normal_impulse,
                    tangent_impulse,
                });
            }
        }
        std::mem::swap(&mut self.cached_impulses, &mut self.next_cached_impulses);
    }

    fn sync_buffers(&mut self, manifold_count: usize, body_count: usize) {
        if self.normal_impulses.capacity() < manifold_count {
            self.normal_impulses
                .reserve(manifold_count.saturating_sub(self.normal_impulses.capacity()));
        }
        if self.tangent_impulses.capacity() < manifold_count {
            self.tangent_impulses
                .reserve(manifold_count.saturating_sub(self.tangent_impulses.capacity()));
        }
        if self.cached_impulses.capacity() < manifold_count {
            self.cached_impulses
                .reserve(manifold_count.saturating_sub(self.cached_impulses.capacity()));
        }
        if self.next_cached_impulses.capacity() < manifold_count {
            self.next_cached_impulses
                .reserve(manifold_count.saturating_sub(self.next_cached_impulses.capacity()));
        }
        if self.cached_lookup.capacity() < manifold_count {
            self.cached_lookup
                .reserve(manifold_count.saturating_sub(self.cached_lookup.capacity()));
        }
        if self.touched_body_flags.len() < body_count {
            self.touched_body_flags.resize(body_count, 0);
        }
        if self.touched_body_indices.capacity() < body_count {
            self.touched_body_indices
                .reserve(body_count.saturating_sub(self.touched_body_indices.capacity()));
        }
        if self.push_delta_x.len() < body_count {
            self.push_delta_x
                .resize_with(body_count, || AtomicU32::new(0.0f32.to_bits()));
            self.push_delta_y
                .resize_with(body_count, || AtomicU32::new(0.0f32.to_bits()));
            self.push_delta_z
                .resize_with(body_count, || AtomicU32::new(0.0f32.to_bits()));
        }
    }

    fn rebuild_cached_lookup(&mut self) {
        self.cached_lookup.clear();
        for cached in &self.cached_impulses {
            self.cached_lookup.insert(cached.contact_id, *cached);
        }
    }

    fn clear_touched_state(&mut self) {
        for dense in self.touched_body_indices.drain(..) {
            if let Some(flag) = self.touched_body_flags.get_mut(dense) {
                *flag = 0;
            }
        }
    }

    fn mark_touched_body(&mut self, dense: usize) {
        if let Some(flag) = self.touched_body_flags.get_mut(dense) {
            if *flag == 0 {
                *flag = 1;
                self.touched_body_indices.push(dense);
            }
        }
    }

    fn flush_touched_positions(&mut self, bodies: &mut BodyRegistry) {
        for &dense in &self.touched_body_indices {
            bodies.recompute_world_aabb_for_dense(dense);
            if let Some(flag) = self.touched_body_flags.get_mut(dense) {
                *flag = 0;
            }
        }
        self.touched_body_indices.clear();
    }

    fn effective_iteration_budget(&self, manifold_count: usize) -> u32 {
        let mut iterations = self.config.iterations.max(1);
        if iterations > 2
            && self.config.parallel_contact_push_strength > 0.0
            && manifold_count >= self.config.parallel_contact_push_threshold
        {
            iterations = iterations.saturating_sub(1);
        }
        iterations
    }

    fn clear_push_deltas(&mut self, body_count: usize) {
        for dense in 0..body_count {
            self.push_delta_x[dense].store(0.0f32.to_bits(), Ordering::Relaxed);
            self.push_delta_y[dense].store(0.0f32.to_bits(), Ordering::Relaxed);
            self.push_delta_z[dense].store(0.0f32.to_bits(), Ordering::Relaxed);
        }
    }

    fn apply_parallel_contact_push(
        &mut self,
        bodies: &mut BodyRegistry,
        manifolds: &[ContactManifold],
    ) {
        if self.config.parallel_contact_push_strength <= 0.0
            || manifolds.len() < self.config.parallel_contact_push_threshold
            || rayon::current_num_threads() <= 1
        {
            return;
        }

        let body_count = bodies.len();
        self.clear_push_deltas(body_count);

        let bodies_ref: &BodyRegistry = bodies;
        let push_strength = self.config.parallel_contact_push_strength;
        let penetration_slop = self.config.penetration_slop;
        let push_delta_x = &self.push_delta_x;
        let push_delta_y = &self.push_delta_y;
        let push_delta_z = &self.push_delta_z;

        manifolds.par_iter().for_each(|manifold| {
            let Some(dense_a) = bodies_ref.dense_index_of(manifold.body_a) else {
                return;
            };
            let Some(dense_b) = bodies_ref.dense_index_of(manifold.body_b) else {
                return;
            };
            let inv_mass_a = bodies_ref.inverse_masses[dense_a];
            let inv_mass_b = bodies_ref.inverse_masses[dense_b];
            let total_inv_mass = inv_mass_a + inv_mass_b;
            if total_inv_mass <= f32::EPSILON {
                return;
            }

            let overlap = (manifold.penetration - penetration_slop).max(0.0);
            if overlap <= f32::EPSILON {
                return;
            }
            let normal = safe_normal(manifold.normal);
            let impulse_mag = overlap * push_strength / total_inv_mass;
            let impulse = normal * impulse_mag;

            if inv_mass_a > 0.0 && bodies_ref.kinds[dense_a] == BodyKind::Dynamic {
                let delta = -impulse * inv_mass_a;
                atomic_add_f32(&push_delta_x[dense_a], delta.x);
                atomic_add_f32(&push_delta_y[dense_a], delta.y);
                atomic_add_f32(&push_delta_z[dense_a], delta.z);
            }
            if inv_mass_b > 0.0 && bodies_ref.kinds[dense_b] == BodyKind::Dynamic {
                let delta = impulse * inv_mass_b;
                atomic_add_f32(&push_delta_x[dense_b], delta.x);
                atomic_add_f32(&push_delta_y[dense_b], delta.y);
                atomic_add_f32(&push_delta_z[dense_b], delta.z);
            }
        });

        for dense in 0..body_count {
            if bodies.kinds[dense] != BodyKind::Dynamic {
                continue;
            }
            let delta = Vector3::new(
                f32::from_bits(self.push_delta_x[dense].load(Ordering::Relaxed)),
                f32::from_bits(self.push_delta_y[dense].load(Ordering::Relaxed)),
                f32::from_bits(self.push_delta_z[dense].load(Ordering::Relaxed)),
            );
            if delta.norm_squared() <= 1e-10 {
                continue;
            }
            bodies.linear_velocities[dense] += delta;
            bodies.awake[dense] = true;
        }
    }

    fn apply_hard_position_projection(
        &mut self,
        bodies: &mut BodyRegistry,
        manifolds: &[ContactManifold],
    ) {
        if self.config.hard_position_projection_strength <= 0.0 || manifolds.is_empty() {
            return;
        }

        let body_count = bodies.len();
        self.clear_push_deltas(body_count);

        let bodies_ref: &BodyRegistry = bodies;
        let strength = self.config.hard_position_projection_strength;
        let penetration_slop = self.config.penetration_slop;
        let max_projection = self.config.max_projection_per_contact.max(0.001);
        let push_delta_x = &self.push_delta_x;
        let push_delta_y = &self.push_delta_y;
        let push_delta_z = &self.push_delta_z;
        let use_parallel = manifolds.len() >= self.config.hard_position_projection_threshold
            && rayon::current_num_threads() > 1;

        if use_parallel {
            manifolds.par_iter().for_each(|manifold| {
                accumulate_projection_delta(
                    bodies_ref,
                    manifold,
                    strength,
                    penetration_slop,
                    max_projection,
                    push_delta_x,
                    push_delta_y,
                    push_delta_z,
                );
            });
        } else {
            for manifold in manifolds {
                accumulate_projection_delta(
                    bodies_ref,
                    manifold,
                    strength,
                    penetration_slop,
                    max_projection,
                    push_delta_x,
                    push_delta_y,
                    push_delta_z,
                );
            }
        }

        for dense in 0..body_count {
            if bodies.kinds[dense] != BodyKind::Dynamic {
                continue;
            }
            let delta = Vector3::new(
                f32::from_bits(self.push_delta_x[dense].load(Ordering::Relaxed)),
                f32::from_bits(self.push_delta_y[dense].load(Ordering::Relaxed)),
                f32::from_bits(self.push_delta_z[dense].load(Ordering::Relaxed)),
            );
            if delta.norm_squared() <= 1e-10 {
                continue;
            }
            bodies.positions[dense] += delta;
            bodies.awake[dense] = true;
            self.mark_touched_body(dense);
        }
    }

    fn apply_warmstart(&mut self, bodies: &mut BodyRegistry, manifolds: &[ContactManifold]) {
        for (index, manifold) in manifolds.iter().enumerate() {
            let Some(cached) = self.cached_lookup.get(&manifold.contact_id).copied() else {
                continue;
            };
            let persistence = manifold.persisted_frames.saturating_sub(1).min(8) as f32;
            let persistence_scale =
                1.0 + persistence * (self.config.persistent_contact_boost * 0.04);
            let decay = self.config.warmstart_impulse_decay.clamp(0.0, 1.0);
            let normal_impulse = (cached.normal_impulse * decay * persistence_scale).max(0.0);
            let relative_velocity = bodies.relative_velocity(manifold.body_a, manifold.body_b);
            let normal = safe_normal(manifold.normal);
            let tangent = tangent_basis(relative_velocity, normal);
            let tangent_speed = relative_velocity.dot(&tangent).abs();
            let max_tangent =
                self.effective_friction(manifold.friction, tangent_speed) * normal_impulse;
            let tangent_impulse = (cached.tangent_impulse * decay).clamp(-max_tangent, max_tangent);
            if normal_impulse <= f32::EPSILON && tangent_impulse.abs() <= f32::EPSILON {
                continue;
            }
            let impulse = normal * normal_impulse + tangent * tangent_impulse;
            self.apply_impulse_pair(bodies, manifold, impulse);
            self.normal_impulses[index] = normal_impulse;
            self.tangent_impulses[index] = tangent_impulse;
            self.stats.warmstart_hits = self.stats.warmstart_hits.saturating_add(1);
        }
    }

    fn solve_manifold(
        &mut self,
        bodies: &mut BodyRegistry,
        manifold: &ContactManifold,
        index: usize,
    ) {
        let Some(dense_a) = bodies.dense_index_of(manifold.body_a) else {
            return;
        };
        let Some(dense_b) = bodies.dense_index_of(manifold.body_b) else {
            return;
        };
        let inv_mass_a = bodies.inverse_masses[dense_a];
        let inv_mass_b = bodies.inverse_masses[dense_b];
        let total_inv_mass = inv_mass_a + inv_mass_b;
        if total_inv_mass <= f32::EPSILON {
            return;
        }

        let normal = safe_normal(manifold.normal);
        let persistence = manifold.persisted_frames.saturating_sub(1).min(8) as f32;
        let correction_boost = 1.0 + persistence * (self.config.persistent_contact_boost * 0.08);
        let correction_mag = (((manifold.penetration - self.config.penetration_slop).max(0.0)
            * self.config.baumgarte)
            / total_inv_mass)
            * correction_boost;
        let correction_mag =
            correction_mag.min(self.config.max_position_correction_per_iteration.max(0.001));
        if correction_mag > 0.0 {
            let correction = normal * correction_mag;
            if inv_mass_a > 0.0 && bodies.kinds[dense_a] == BodyKind::Dynamic {
                bodies.positions[dense_a] -= correction * inv_mass_a;
                self.mark_touched_body(dense_a);
            }
            if inv_mass_b > 0.0 && bodies.kinds[dense_b] == BodyKind::Dynamic {
                bodies.positions[dense_b] += correction * inv_mass_b;
                self.mark_touched_body(dense_b);
            }
            self.stats.positional_corrections = self.stats.positional_corrections.saturating_add(1);
        }

        let relative_velocity =
            bodies.linear_velocities[dense_b] - bodies.linear_velocities[dense_a];
        let contact_velocity = relative_velocity.dot(&normal);
        // Restitution only when bodies move toward each other across the contact normal.
        let restitution = if contact_velocity < -self.config.restitution_velocity_threshold {
            manifold.restitution.max(0.0)
        } else {
            0.0
        };
        let raw_normal_impulse = -(1.0 + restitution) * contact_velocity / total_inv_mass;
        let prev_normal = self.normal_impulses[index];
        let next_normal = (prev_normal + raw_normal_impulse).max(0.0);
        let delta_normal = next_normal - prev_normal;
        if delta_normal > f32::EPSILON {
            self.apply_impulse_pair(bodies, manifold, normal * delta_normal);
            self.normal_impulses[index] = next_normal;
            self.stats.normal_impulses_applied =
                self.stats.normal_impulses_applied.saturating_add(1);
        }

        let relative_velocity =
            bodies.linear_velocities[dense_b] - bodies.linear_velocities[dense_a];
        let tangent = tangent_basis(relative_velocity, normal);
        if tangent.norm_squared() <= 1e-6 {
            return;
        }
        let tangent_speed = relative_velocity.dot(&tangent);
        let raw_tangent_impulse = -tangent_speed / total_inv_mass;
        let max_friction = self.effective_friction(manifold.friction, tangent_speed.abs())
            * self.normal_impulses[index];
        let prev_tangent = self.tangent_impulses[index];
        let next_tangent = (prev_tangent + raw_tangent_impulse).clamp(-max_friction, max_friction);
        let delta_tangent = next_tangent - prev_tangent;
        if delta_tangent.abs() > f32::EPSILON {
            self.apply_impulse_pair(bodies, manifold, tangent * delta_tangent);
            self.tangent_impulses[index] = next_tangent;
            self.stats.friction_impulses_applied =
                self.stats.friction_impulses_applied.saturating_add(1);
        }
    }

    fn apply_impulse_pair(
        &self,
        bodies: &mut BodyRegistry,
        manifold: &ContactManifold,
        impulse: Vector3<f32>,
    ) {
        let Some(dense_a) = bodies.dense_index_of(manifold.body_a) else {
            return;
        };
        let Some(dense_b) = bodies.dense_index_of(manifold.body_b) else {
            return;
        };
        let inv_mass_a = bodies.inverse_masses[dense_a];
        let inv_mass_b = bodies.inverse_masses[dense_b];
        if inv_mass_a > 0.0 && bodies.kinds[dense_a] == BodyKind::Dynamic {
            bodies.linear_velocities[dense_a] -= impulse * inv_mass_a;
            bodies.awake[dense_a] = true;
        }
        if inv_mass_b > 0.0 && bodies.kinds[dense_b] == BodyKind::Dynamic {
            bodies.linear_velocities[dense_b] += impulse * inv_mass_b;
            bodies.awake[dense_b] = true;
        }
    }

    fn effective_friction(&self, base: f32, tangent_speed_abs: f32) -> f32 {
        let base = base.max(0.0);
        let transition = self.config.friction_transition_speed.max(0.0);
        if transition <= f32::EPSILON {
            return base;
        }
        let static_boost = self.config.friction_static_boost.max(0.0);
        let kinetic_scale = self.config.friction_kinetic_scale.max(0.0);
        let t = (tangent_speed_abs / transition).clamp(0.0, 1.0);
        let speed_scale = static_boost + (kinetic_scale - static_boost) * t;
        base * speed_scale.max(0.0)
    }
}

#[inline]
fn sphere_like_radius(bodies: &BodyRegistry, dense: usize) -> Option<f32> {
    let half = bodies.local_bounds[dense].half_extents();
    let min_extent = half.x.min(half.y.min(half.z));
    let max_extent = half.x.max(half.y.max(half.z));
    if max_extent <= 1e-6 {
        return None;
    }
    // Treat nearly isotropic local bounds as sphere-like for tighter overlap projection.
    if (max_extent - min_extent) <= max_extent * 0.12 {
        Some((half.x + half.y + half.z) / 3.0)
    } else {
        None
    }
}

#[allow(clippy::too_many_arguments)]
fn accumulate_projection_delta(
    bodies: &BodyRegistry,
    manifold: &ContactManifold,
    strength: f32,
    penetration_slop: f32,
    max_projection: f32,
    push_delta_x: &[AtomicU32],
    push_delta_y: &[AtomicU32],
    push_delta_z: &[AtomicU32],
) {
    let Some(dense_a) = bodies.dense_index_of(manifold.body_a) else {
        return;
    };
    let Some(dense_b) = bodies.dense_index_of(manifold.body_b) else {
        return;
    };
    let inv_mass_a = bodies.inverse_masses[dense_a];
    let inv_mass_b = bodies.inverse_masses[dense_b];
    let total_inv_mass = inv_mass_a + inv_mass_b;
    if total_inv_mass <= f32::EPSILON {
        return;
    }

    let dynamic_a = inv_mass_a > 0.0 && bodies.kinds[dense_a] == BodyKind::Dynamic;
    let dynamic_b = inv_mass_b > 0.0 && bodies.kinds[dense_b] == BodyKind::Dynamic;
    if !dynamic_a && !dynamic_b {
        return;
    }

    let mut overlap = (manifold.penetration - penetration_slop).max(0.0);
    let mut normal = safe_normal(manifold.normal);
    let mut sphere_pair = false;

    // For dynamic sphere pairs, derive overlap from current centers so penetration gets resolved
    // immediately instead of waiting for one more narrowphase refresh.
    if dynamic_a && dynamic_b {
        if let (Some(radius_a), Some(radius_b)) = (
            sphere_like_radius(bodies, dense_a),
            sphere_like_radius(bodies, dense_b),
        ) {
            let delta = bodies.positions[dense_b] - bodies.positions[dense_a];
            let target_distance = (radius_a + radius_b).max(0.0);
            if target_distance > f32::EPSILON {
                let distance_sq = delta.norm_squared();
                if distance_sq > 1e-10 {
                    let distance = distance_sq.sqrt();
                    if distance < target_distance {
                        normal = delta / distance;
                        overlap = overlap.max(target_distance - distance);
                        sphere_pair = true;
                    }
                } else {
                    overlap = overlap.max(target_distance * 0.5);
                    sphere_pair = true;
                }
            }
        }
    }

    if overlap <= f32::EPSILON {
        return;
    }
    let projection_cap = if sphere_pair {
        max_projection * 4.0
    } else {
        max_projection
    };
    let correction_depth = overlap.min(projection_cap.max(0.001));
    let correction_mag = correction_depth * strength / total_inv_mass;
    let correction = normal * correction_mag;

    if dynamic_a {
        let delta = -correction * inv_mass_a;
        atomic_add_f32(&push_delta_x[dense_a], delta.x);
        atomic_add_f32(&push_delta_y[dense_a], delta.y);
        atomic_add_f32(&push_delta_z[dense_a], delta.z);
    }
    if dynamic_b {
        let delta = correction * inv_mass_b;
        atomic_add_f32(&push_delta_x[dense_b], delta.x);
        atomic_add_f32(&push_delta_y[dense_b], delta.y);
        atomic_add_f32(&push_delta_z[dense_b], delta.z);
    }
}

fn safe_normal(normal: Vector3<f32>) -> Vector3<f32> {
    let len_sq = normal.norm_squared();
    if len_sq > 1e-6 {
        normal / len_sq.sqrt()
    } else {
        Vector3::new(1.0, 0.0, 0.0)
    }
}

#[inline]
fn atomic_add_f32(target: &AtomicU32, delta: f32) {
    if delta.abs() <= f32::EPSILON {
        return;
    }
    let mut current = target.load(Ordering::Relaxed);
    loop {
        let current_value = f32::from_bits(current);
        let next = (current_value + delta).to_bits();
        match target.compare_exchange_weak(current, next, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => break,
            Err(observed) => current = observed,
        }
    }
}

fn tangent_basis(relative_velocity: Vector3<f32>, normal: Vector3<f32>) -> Vector3<f32> {
    let tangent = relative_velocity - normal * relative_velocity.dot(&normal);
    let len_sq = tangent.norm_squared();
    if len_sq > 1e-6 {
        tangent / len_sq.sqrt()
    } else if normal.x.abs() < 0.9 {
        normal.cross(&Vector3::new(1.0, 0.0, 0.0)).normalize()
    } else {
        normal.cross(&Vector3::new(0.0, 1.0, 0.0)).normalize()
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::Vector3;

    use super::*;
    use crate::body::{Aabb, BodyDesc, ContactId, ContactManifold};
    use crate::handle::ColliderHandle;

    #[test]
    fn solver_separates_penetrating_dynamic_bodies() {
        let mut bodies = BodyRegistry::new();
        let a = bodies.spawn(BodyDesc {
            position: Vector3::new(0.0, 0.0, 0.0),
            linear_velocity: Vector3::new(1.0, 0.0, 0.0),
            local_bounds: Aabb::from_center_half_extents(Vector3::zeros(), Vector3::repeat(1.0)),
            ..BodyDesc::default()
        });
        let b = bodies.spawn(BodyDesc {
            position: Vector3::new(0.5, 0.0, 0.0),
            local_bounds: Aabb::from_center_half_extents(Vector3::zeros(), Vector3::repeat(1.0)),
            ..BodyDesc::default()
        });
        let manifolds = vec![ContactManifold {
            contact_id: ContactId::new(1),
            collider_a: ColliderHandle::new(a.index() as u16, a.generation()),
            collider_b: ColliderHandle::new(b.index() as u16, b.generation()),
            body_a: a,
            body_b: b,
            point: Vector3::new(0.25, 0.0, 0.0),
            normal: Vector3::new(1.0, 0.0, 0.0),
            penetration: 1.5,
            persisted_frames: 1,
            restitution: 0.0,
            friction: 0.5,
        }];
        let mut solver = ContactSolver::new(ContactSolverConfig::default());
        solver.solve(&mut bodies, &manifolds, 1.0 / 60.0);
        let pos_a = bodies.position_for(a).unwrap();
        let pos_b = bodies.position_for(b).unwrap();
        assert!(pos_b.x - pos_a.x > 0.5);
        assert!(solver.stats().positional_corrections > 0);
        assert!(solver.stats().normal_impulses_applied > 0);
    }

    #[test]
    fn solver_applies_friction_and_warmstarting() {
        let mut bodies = BodyRegistry::new();
        let a = bodies.spawn(BodyDesc {
            position: Vector3::new(0.0, 0.0, 0.0),
            ..BodyDesc::default()
        });
        let b = bodies.spawn(BodyDesc {
            position: Vector3::new(0.0, 0.0, 0.0),
            linear_velocity: Vector3::new(2.0, -1.0, 0.0),
            ..BodyDesc::default()
        });
        let manifolds = vec![ContactManifold {
            contact_id: ContactId::new(2),
            collider_a: ColliderHandle::new(a.index() as u16, a.generation()),
            collider_b: ColliderHandle::new(b.index() as u16, b.generation()),
            body_a: a,
            body_b: b,
            point: Vector3::zeros(),
            normal: Vector3::new(0.0, 1.0, 0.0),
            penetration: 0.2,
            persisted_frames: 1,
            restitution: 0.0,
            friction: 1.0,
        }];
        let mut solver = ContactSolver::new(ContactSolverConfig::default());
        solver.solve(&mut bodies, &manifolds, 1.0 / 60.0);
        let vx_after_first = bodies.body(b).unwrap().linear_velocity.x;
        assert!(vx_after_first < 2.0);
        assert!(solver.stats().friction_impulses_applied > 0);

        solver.solve(&mut bodies, &manifolds, 1.0 / 60.0);
        assert!(solver.stats().warmstart_hits > 0);
    }

    #[test]
    fn solver_does_not_apply_restitution_while_separating() {
        let mut bodies = BodyRegistry::new();
        let a = bodies.spawn(BodyDesc {
            position: Vector3::new(0.0, 0.0, 0.0),
            linear_velocity: Vector3::new(0.0, 0.0, 0.0),
            ..BodyDesc::default()
        });
        let b = bodies.spawn(BodyDesc {
            position: Vector3::new(0.0, 0.6, 0.0),
            // Positive Y with normal +Y means separating contact.
            linear_velocity: Vector3::new(0.0, 2.0, 0.0),
            ..BodyDesc::default()
        });
        let manifolds = vec![ContactManifold {
            contact_id: ContactId::new(3),
            collider_a: ColliderHandle::new(a.index() as u16, a.generation()),
            collider_b: ColliderHandle::new(b.index() as u16, b.generation()),
            body_a: a,
            body_b: b,
            point: Vector3::new(0.0, 0.3, 0.0),
            normal: Vector3::new(0.0, 1.0, 0.0),
            penetration: 0.1,
            persisted_frames: 1,
            restitution: 0.95,
            friction: 0.0,
        }];

        let mut solver = ContactSolver::new(ContactSolverConfig::default());
        solver.solve(&mut bodies, &manifolds, 1.0 / 60.0);

        let vy_after = bodies.body(b).unwrap().linear_velocity.y;
        // Restitution should not add energy while bodies are separating.
        assert!(vy_after <= 2.0 + 1e-4);
    }

    #[test]
    fn solver_clamps_position_correction_per_iteration() {
        let mut bodies = BodyRegistry::new();
        let a = bodies.spawn(BodyDesc {
            position: Vector3::new(0.0, 0.0, 0.0),
            local_bounds: Aabb::from_center_half_extents(Vector3::zeros(), Vector3::repeat(1.0)),
            ..BodyDesc::default()
        });
        let b = bodies.spawn(BodyDesc {
            position: Vector3::new(0.1, 0.0, 0.0),
            local_bounds: Aabb::from_center_half_extents(Vector3::zeros(), Vector3::repeat(1.0)),
            ..BodyDesc::default()
        });
        let manifolds = vec![ContactManifold {
            contact_id: ContactId::new(4),
            collider_a: ColliderHandle::new(a.index() as u16, a.generation()),
            collider_b: ColliderHandle::new(b.index() as u16, b.generation()),
            body_a: a,
            body_b: b,
            point: Vector3::new(0.05, 0.0, 0.0),
            normal: Vector3::new(1.0, 0.0, 0.0),
            penetration: 2.5,
            persisted_frames: 4,
            restitution: 0.0,
            friction: 0.2,
        }];

        let mut cfg = ContactSolverConfig::default();
        cfg.iterations = 1;
        cfg.max_position_correction_per_iteration = 0.02;
        cfg.parallel_contact_push_strength = 0.0;
        cfg.hard_position_projection_strength = 0.0;
        let mut solver = ContactSolver::new(cfg);
        solver.solve(&mut bodies, &manifolds, 1.0 / 60.0);

        let pos_a = bodies.position_for(a).unwrap();
        let pos_b = bodies.position_for(b).unwrap();
        let moved = (pos_a.x.abs() + (pos_b.x - 0.1).abs()).max(0.0);
        assert!(moved <= 0.05);
    }

    #[test]
    fn speed_aware_friction_profile_blends_static_to_kinetic() {
        let mut cfg = ContactSolverConfig::default();
        cfg.friction_transition_speed = 2.0;
        cfg.friction_static_boost = 1.4;
        cfg.friction_kinetic_scale = 0.6;
        let solver = ContactSolver::new(cfg);

        let base = 0.5;
        let low = solver.effective_friction(base, 0.1);
        let high = solver.effective_friction(base, 4.0);
        assert!(low > high);
        assert!((high - base * 0.6).abs() < 1e-6);

        let mut disabled_cfg = ContactSolverConfig::default();
        disabled_cfg.friction_transition_speed = 0.0;
        let disabled_solver = ContactSolver::new(disabled_cfg);
        assert!((disabled_solver.effective_friction(base, 999.0) - base).abs() < 1e-6);
    }
}
