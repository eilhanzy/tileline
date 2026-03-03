//! Joint and constraint foundations for ParadoxPE.
//!
//! The first joint type is a distance constraint designed to slot into the fixed-step solver loop
//! without runtime allocations.

use nalgebra::Vector3;

use crate::body::BodyKind;
use crate::handle::{BodyHandle, JointHandle};
use crate::storage::BodyRegistry;

/// Public joint kind tag.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum JointKind {
    Distance,
}

/// Distance joint descriptor.
#[derive(Debug, Clone, PartialEq)]
pub struct DistanceJointDesc {
    pub body_a: BodyHandle,
    pub body_b: BodyHandle,
    pub local_anchor_a: Vector3<f32>,
    pub local_anchor_b: Vector3<f32>,
    pub rest_length: f32,
    pub stiffness: f32,
    pub damping: f32,
}

impl DistanceJointDesc {
    pub fn new(body_a: BodyHandle, body_b: BodyHandle, rest_length: f32) -> Self {
        Self {
            body_a,
            body_b,
            local_anchor_a: Vector3::zeros(),
            local_anchor_b: Vector3::zeros(),
            rest_length: rest_length.max(0.0),
            stiffness: 1.0,
            damping: 0.1,
        }
    }
}

/// Runtime joint record stored by the world.
#[derive(Debug, Clone, PartialEq)]
pub struct DistanceJoint {
    pub handle: JointHandle,
    pub body_a: BodyHandle,
    pub body_b: BodyHandle,
    pub local_anchor_a: Vector3<f32>,
    pub local_anchor_b: Vector3<f32>,
    pub rest_length: f32,
    pub stiffness: f32,
    pub damping: f32,
}

impl DistanceJoint {
    pub fn from_desc(handle: JointHandle, desc: DistanceJointDesc) -> Self {
        Self {
            handle,
            body_a: desc.body_a,
            body_b: desc.body_b,
            local_anchor_a: desc.local_anchor_a,
            local_anchor_b: desc.local_anchor_b,
            rest_length: desc.rest_length.max(0.0),
            stiffness: desc.stiffness.clamp(0.0, 1.0),
            damping: desc.damping.max(0.0),
        }
    }

    pub fn kind(&self) -> JointKind {
        JointKind::Distance
    }
}

/// Joint solver tuning.
#[derive(Debug, Clone, PartialEq)]
pub struct JointSolverConfig {
    pub iterations: u32,
    pub positional_bias: f32,
}

impl Default for JointSolverConfig {
    fn default() -> Self {
        Self {
            iterations: 2,
            positional_bias: 0.15,
        }
    }
}

/// Joint solver statistics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct JointSolverStats {
    pub joints: usize,
    pub iterations: u32,
    pub positional_corrections: u32,
    pub velocity_impulses: u32,
}

/// Reusable distance-joint solver.
#[derive(Debug, Clone)]
pub struct JointConstraintSolver {
    config: JointSolverConfig,
    stats: JointSolverStats,
}

impl JointConstraintSolver {
    pub fn new(config: JointSolverConfig) -> Self {
        Self {
            config,
            stats: JointSolverStats::default(),
        }
    }

    pub fn config(&self) -> &JointSolverConfig {
        &self.config
    }

    pub fn stats(&self) -> JointSolverStats {
        self.stats
    }

    pub fn solve(&mut self, bodies: &mut BodyRegistry, joints: &[DistanceJoint], dt: f32) {
        self.stats = JointSolverStats {
            joints: joints.len(),
            iterations: self.config.iterations,
            ..JointSolverStats::default()
        };
        if joints.is_empty() {
            return;
        }
        for _ in 0..self.config.iterations {
            for joint in joints {
                self.solve_distance_joint(bodies, joint, dt);
            }
        }
    }

    fn solve_distance_joint(&mut self, bodies: &mut BodyRegistry, joint: &DistanceJoint, dt: f32) {
        let Some(dense_a) = bodies.dense_index_of(joint.body_a) else {
            return;
        };
        let Some(dense_b) = bodies.dense_index_of(joint.body_b) else {
            return;
        };
        let inv_mass_a = bodies.inverse_masses[dense_a];
        let inv_mass_b = bodies.inverse_masses[dense_b];
        let total_inv_mass = inv_mass_a + inv_mass_b;
        if total_inv_mass <= f32::EPSILON {
            return;
        }

        let anchor_a = bodies.positions[dense_a] + joint.local_anchor_a;
        let anchor_b = bodies.positions[dense_b] + joint.local_anchor_b;
        let delta = anchor_b - anchor_a;
        let distance = delta.norm();
        let normal = if distance > 1e-6 {
            delta / distance
        } else {
            Vector3::new(1.0, 0.0, 0.0)
        };
        let error = distance - joint.rest_length;
        let correction_mag =
            (error * joint.stiffness * self.config.positional_bias) / total_inv_mass;
        if correction_mag.abs() > f32::EPSILON {
            let correction = normal * correction_mag;
            if inv_mass_a > 0.0 && bodies.kinds[dense_a] != BodyKind::Static {
                bodies.positions[dense_a] += correction * inv_mass_a;
                bodies.recompute_world_aabb_for_dense(dense_a);
                bodies.wake_dense(dense_a);
            }
            if inv_mass_b > 0.0 && bodies.kinds[dense_b] != BodyKind::Static {
                bodies.positions[dense_b] -= correction * inv_mass_b;
                bodies.recompute_world_aabb_for_dense(dense_b);
                bodies.wake_dense(dense_b);
            }
            self.stats.positional_corrections = self.stats.positional_corrections.saturating_add(1);
        }

        let relative_velocity =
            bodies.linear_velocities[dense_b] - bodies.linear_velocities[dense_a];
        let constraint_velocity = relative_velocity.dot(&normal);
        let impulse_mag =
            -(constraint_velocity + error * joint.damping / dt.max(1e-4)) / total_inv_mass;
        if impulse_mag.abs() <= f32::EPSILON {
            return;
        }
        let impulse = normal * impulse_mag;
        if inv_mass_a > 0.0 && bodies.kinds[dense_a] == BodyKind::Dynamic {
            bodies.linear_velocities[dense_a] -= impulse * inv_mass_a;
            bodies.wake_dense(dense_a);
        }
        if inv_mass_b > 0.0 && bodies.kinds[dense_b] == BodyKind::Dynamic {
            bodies.linear_velocities[dense_b] += impulse * inv_mass_b;
            bodies.wake_dense(dense_b);
        }
        self.stats.velocity_impulses = self.stats.velocity_impulses.saturating_add(1);
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::Vector3;

    use super::*;
    use crate::body::BodyDesc;

    #[test]
    fn distance_joint_pulls_bodies_toward_rest_length() {
        let mut bodies = BodyRegistry::new();
        let a = bodies.spawn(BodyDesc::default());
        let b = bodies.spawn(BodyDesc {
            position: Vector3::new(4.0, 0.0, 0.0),
            ..BodyDesc::default()
        });
        let joint =
            DistanceJoint::from_desc(JointHandle::new(0, 1), DistanceJointDesc::new(a, b, 2.0));
        let mut solver = JointConstraintSolver::new(JointSolverConfig::default());
        solver.solve(&mut bodies, &[joint], 1.0 / 60.0);
        let separation = (bodies.position_for(b).unwrap() - bodies.position_for(a).unwrap()).norm();
        assert!(separation < 4.0);
    }
}
