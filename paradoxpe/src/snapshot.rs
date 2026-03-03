//! Rollback/interpolation snapshot foundations for ParadoxPE.
//!
//! This module keeps snapshot data plain and handle-based so it can be reused by:
//! - local interpolation
//! - rollback/rewind support
//! - NPS transform snapshot export

use std::collections::VecDeque;

use nalgebra::{UnitQuaternion, Vector3};

use crate::handle::BodyHandle;

/// One body's state inside a fixed-step snapshot.
#[derive(Debug, Clone, PartialEq)]
pub struct BodyStateFrame {
    pub handle: BodyHandle,
    pub position: Vector3<f32>,
    pub rotation: UnitQuaternion<f32>,
    pub linear_velocity: Vector3<f32>,
    pub awake: bool,
}

/// Full world snapshot used for rollback/interpolation/network export.
#[derive(Debug, Clone, PartialEq)]
pub struct PhysicsSnapshot {
    pub tick: u64,
    pub bodies: Vec<BodyStateFrame>,
}

/// Interpolated body pose produced from two snapshots.
#[derive(Debug, Clone, PartialEq)]
pub struct InterpolatedBodyPose {
    pub handle: BodyHandle,
    pub position: Vector3<f32>,
    pub rotation: UnitQuaternion<f32>,
    pub linear_velocity: Vector3<f32>,
    pub awake: bool,
}

/// Small rolling snapshot buffer for interpolation and future rollback.
#[derive(Debug, Clone, PartialEq)]
pub struct PhysicsInterpolationBuffer {
    capacity: usize,
    snapshots: VecDeque<PhysicsSnapshot>,
}

impl Default for PhysicsInterpolationBuffer {
    fn default() -> Self {
        Self::with_capacity(8)
    }
}

impl PhysicsInterpolationBuffer {
    pub fn with_capacity(capacity: usize) -> Self {
        let capacity = capacity.max(2);
        Self {
            capacity,
            snapshots: VecDeque::with_capacity(capacity),
        }
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn len(&self) -> usize {
        self.snapshots.len()
    }

    pub fn is_empty(&self) -> bool {
        self.snapshots.is_empty()
    }

    pub fn latest(&self) -> Option<&PhysicsSnapshot> {
        self.snapshots.back()
    }

    pub fn previous(&self) -> Option<&PhysicsSnapshot> {
        if self.snapshots.len() < 2 {
            None
        } else {
            self.snapshots.get(self.snapshots.len() - 2)
        }
    }

    pub fn push_snapshot(&mut self, snapshot: PhysicsSnapshot) {
        if self.snapshots.len() == self.capacity {
            let _ = self.snapshots.pop_front();
        }
        self.snapshots.push_back(snapshot);
    }

    pub fn snapshot_for_tick(&self, tick: u64) -> Option<&PhysicsSnapshot> {
        self.snapshots.iter().find(|snapshot| snapshot.tick == tick)
    }

    pub fn interpolate_body(&self, handle: BodyHandle, alpha: f32) -> Option<InterpolatedBodyPose> {
        let previous = self.previous()?;
        let latest = self.latest()?;
        let prev = previous.bodies.iter().find(|body| body.handle == handle)?;
        let next = latest.bodies.iter().find(|body| body.handle == handle)?;
        Some(interpolate_body_frames(prev, next, alpha))
    }
}

fn interpolate_body_frames(
    previous: &BodyStateFrame,
    next: &BodyStateFrame,
    alpha: f32,
) -> InterpolatedBodyPose {
    let alpha = alpha.clamp(0.0, 1.0);
    InterpolatedBodyPose {
        handle: next.handle,
        position: previous.position + (next.position - previous.position) * alpha,
        rotation: previous.rotation.nlerp(&next.rotation, alpha),
        linear_velocity: previous.linear_velocity
            + (next.linear_velocity - previous.linear_velocity) * alpha,
        awake: if alpha < 0.5 {
            previous.awake
        } else {
            next.awake
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interpolation_buffer_lerps_between_recent_snapshots() {
        let handle = BodyHandle::new(1, 1);
        let mut buffer = PhysicsInterpolationBuffer::with_capacity(4);
        buffer.push_snapshot(PhysicsSnapshot {
            tick: 1,
            bodies: vec![BodyStateFrame {
                handle,
                position: Vector3::new(0.0, 0.0, 0.0),
                rotation: UnitQuaternion::identity(),
                linear_velocity: Vector3::new(1.0, 0.0, 0.0),
                awake: true,
            }],
        });
        buffer.push_snapshot(PhysicsSnapshot {
            tick: 2,
            bodies: vec![BodyStateFrame {
                handle,
                position: Vector3::new(10.0, 0.0, 0.0),
                rotation: UnitQuaternion::identity(),
                linear_velocity: Vector3::new(3.0, 0.0, 0.0),
                awake: true,
            }],
        });

        let interpolated = buffer.interpolate_body(handle, 0.25).unwrap();
        assert!((interpolated.position.x - 2.5).abs() < 1e-4);
        assert!((interpolated.linear_velocity.x - 1.5).abs() < 1e-4);
    }
}
