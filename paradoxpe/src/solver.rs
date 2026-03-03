//! First-pass sequential impulse contact solver for ParadoxPE.
//!
//! This revision adds:
//! - normal impulse accumulation
//! - tangent/friction impulse accumulation
//! - reusable warm-start cache keyed by collider pairs
//! - allocation-aware scratch buffers sized ahead of the hot loop

use nalgebra::Vector3;

use crate::body::{BodyKind, ContactManifold};
use crate::handle::ColliderHandle;
use crate::storage::BodyRegistry;

#[derive(Debug, Clone, Copy, PartialEq)]
struct CachedContactImpulse {
    collider_a: ColliderHandle,
    collider_b: ColliderHandle,
    normal_impulse: f32,
    tangent_impulse: f32,
}

/// Solver tuning configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct ContactSolverConfig {
    pub iterations: u32,
    pub baumgarte: f32,
    pub penetration_slop: f32,
    pub restitution_velocity_threshold: f32,
    pub warm_starting: bool,
}

impl Default for ContactSolverConfig {
    fn default() -> Self {
        Self {
            iterations: 4,
            baumgarte: 0.2,
            penetration_slop: 0.005,
            restitution_velocity_threshold: 1.0,
            warm_starting: true,
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
#[derive(Debug, Clone)]
pub struct ContactSolver {
    config: ContactSolverConfig,
    stats: ContactSolverStats,
    cached_impulses: Vec<CachedContactImpulse>,
    next_cached_impulses: Vec<CachedContactImpulse>,
    normal_impulses: Vec<f32>,
    tangent_impulses: Vec<f32>,
}

impl ContactSolver {
    pub fn new(config: ContactSolverConfig) -> Self {
        Self {
            config,
            stats: ContactSolverStats::default(),
            cached_impulses: Vec::new(),
            next_cached_impulses: Vec::new(),
            normal_impulses: Vec::new(),
            tangent_impulses: Vec::new(),
        }
    }

    pub fn config(&self) -> &ContactSolverConfig {
        &self.config
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
        if manifolds.is_empty() {
            self.cached_impulses.clear();
            self.next_cached_impulses.clear();
            self.normal_impulses.clear();
            self.tangent_impulses.clear();
            return;
        }

        self.sync_buffers(manifolds.len());
        self.next_cached_impulses.clear();
        self.normal_impulses.resize(manifolds.len(), 0.0);
        self.tangent_impulses.resize(manifolds.len(), 0.0);
        self.normal_impulses.fill(0.0);
        self.tangent_impulses.fill(0.0);

        if self.config.warm_starting {
            self.apply_warmstart(bodies, manifolds);
        }

        for _ in 0..self.config.iterations {
            for (index, manifold) in manifolds.iter().enumerate() {
                self.solve_manifold(bodies, manifold, index);
            }
        }

        for (index, manifold) in manifolds.iter().enumerate() {
            let normal_impulse = self.normal_impulses[index];
            let tangent_impulse = self.tangent_impulses[index];
            if normal_impulse <= f32::EPSILON && tangent_impulse.abs() <= f32::EPSILON {
                continue;
            }
            if self.next_cached_impulses.len() < self.next_cached_impulses.capacity() {
                self.next_cached_impulses.push(CachedContactImpulse {
                    collider_a: manifold.collider_a,
                    collider_b: manifold.collider_b,
                    normal_impulse,
                    tangent_impulse,
                });
            }
        }
        std::mem::swap(&mut self.cached_impulses, &mut self.next_cached_impulses);
    }

    fn sync_buffers(&mut self, manifold_count: usize) {
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
    }

    fn apply_warmstart(&mut self, bodies: &mut BodyRegistry, manifolds: &[ContactManifold]) {
        for (index, manifold) in manifolds.iter().enumerate() {
            let Some(cached) = self.lookup_cached(manifold.collider_a, manifold.collider_b) else {
                continue;
            };
            let normal = safe_normal(manifold.normal);
            let tangent = tangent_basis(
                bodies.relative_velocity(manifold.body_a, manifold.body_b),
                normal,
            );
            let impulse = normal * cached.normal_impulse + tangent * cached.tangent_impulse;
            self.apply_impulse_pair(bodies, manifold, impulse);
            self.normal_impulses[index] = cached.normal_impulse;
            self.tangent_impulses[index] = cached.tangent_impulse;
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
        let correction_mag = ((manifold.penetration - self.config.penetration_slop).max(0.0)
            * self.config.baumgarte)
            / total_inv_mass;
        if correction_mag > 0.0 {
            let correction = normal * correction_mag;
            if inv_mass_a > 0.0 && bodies.kinds[dense_a] == BodyKind::Dynamic {
                bodies.positions[dense_a] -= correction * inv_mass_a;
                bodies.recompute_world_aabb_for_dense(dense_a);
            }
            if inv_mass_b > 0.0 && bodies.kinds[dense_b] == BodyKind::Dynamic {
                bodies.positions[dense_b] += correction * inv_mass_b;
                bodies.recompute_world_aabb_for_dense(dense_b);
            }
            self.stats.positional_corrections = self.stats.positional_corrections.saturating_add(1);
        }

        let relative_velocity =
            bodies.linear_velocities[dense_b] - bodies.linear_velocities[dense_a];
        let contact_velocity = relative_velocity.dot(&normal);
        let restitution = if contact_velocity.abs() > self.config.restitution_velocity_threshold {
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
        let max_friction = manifold.friction.max(0.0) * self.normal_impulses[index];
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

    fn lookup_cached(
        &self,
        collider_a: ColliderHandle,
        collider_b: ColliderHandle,
    ) -> Option<CachedContactImpulse> {
        self.cached_impulses
            .iter()
            .find(|cached| cached.collider_a == collider_a && cached.collider_b == collider_b)
            .copied()
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
    use crate::body::{Aabb, BodyDesc, ContactManifold};
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
            collider_a: ColliderHandle::new(a.index() as u16, a.generation()),
            collider_b: ColliderHandle::new(b.index() as u16, b.generation()),
            body_a: a,
            body_b: b,
            point: Vector3::new(0.25, 0.0, 0.0),
            normal: Vector3::new(1.0, 0.0, 0.0),
            penetration: 1.5,
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
            collider_a: ColliderHandle::new(a.index() as u16, a.generation()),
            collider_b: ColliderHandle::new(b.index() as u16, b.generation()),
            body_a: a,
            body_b: b,
            point: Vector3::zeros(),
            normal: Vector3::new(0.0, 1.0, 0.0),
            penetration: 0.2,
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
}
