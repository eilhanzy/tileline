//! Core ParadoxPE body, collider, and contact data types.

use nalgebra::Vector2;

use crate::handle::{BodyHandle, ColliderHandle};

/// Rigid body motion class.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum BodyKind {
    Static = 0,
    Kinematic = 1,
    Dynamic = 2,
}

impl BodyKind {
    pub fn from_u32(value: u32) -> Option<Self> {
        match value {
            0 => Some(Self::Static),
            1 => Some(Self::Kinematic),
            2 => Some(Self::Dynamic),
            _ => None,
        }
    }
}

/// Script-facing body-kind wrapper used by the host ABI.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum ScriptBodyKind {
    Static = 0,
    Kinematic = 1,
    Dynamic = 2,
}

impl From<ScriptBodyKind> for BodyKind {
    fn from(value: ScriptBodyKind) -> Self {
        match value {
            ScriptBodyKind::Static => Self::Static,
            ScriptBodyKind::Kinematic => Self::Kinematic,
            ScriptBodyKind::Dynamic => Self::Dynamic,
        }
    }
}

/// Collider primitive selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum ColliderShapeKind {
    Circle = 0,
    Aabb = 1,
}

impl ColliderShapeKind {
    pub fn from_u32(value: u32) -> Option<Self> {
        match value {
            0 => Some(Self::Circle),
            1 => Some(Self::Aabb),
            _ => None,
        }
    }
}

/// Script-facing collider-shape wrapper used by the host ABI.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum ScriptColliderShapeKind {
    Circle = 0,
    Aabb = 1,
}

impl From<ScriptColliderShapeKind> for ColliderShapeKind {
    fn from(value: ScriptColliderShapeKind) -> Self {
        match value {
            ScriptColliderShapeKind::Circle => Self::Circle,
            ScriptColliderShapeKind::Aabb => Self::Aabb,
        }
    }
}

/// Collider shape data used by the broadphase/narrowphase starter implementation.
#[derive(Debug, Clone, PartialEq)]
pub enum ColliderShape {
    Circle { radius: f32 },
    Aabb { half_extents: Vector2<f32> },
}

impl ColliderShape {
    pub fn from_script(kind: ColliderShapeKind, a: f32, b: f32) -> Self {
        match kind {
            ColliderShapeKind::Circle => Self::Circle { radius: a.max(0.0) },
            ColliderShapeKind::Aabb => Self::Aabb {
                half_extents: Vector2::new(a.max(0.0), b.max(0.0)),
            },
        }
    }

    pub fn half_extents(&self) -> Vector2<f32> {
        match self {
            Self::Circle { radius } => Vector2::new(*radius, *radius),
            Self::Aabb { half_extents } => *half_extents,
        }
    }
}

/// Rigid body descriptor.
#[derive(Debug, Clone, PartialEq)]
pub struct BodyDesc {
    pub kind: BodyKind,
    pub position: Vector2<f32>,
    pub velocity: Vector2<f32>,
    pub mass: f32,
    pub linear_damping: f32,
    pub user_tag: u32,
}

impl Default for BodyDesc {
    fn default() -> Self {
        Self {
            kind: BodyKind::Dynamic,
            position: Vector2::zeros(),
            velocity: Vector2::zeros(),
            mass: 1.0,
            linear_damping: 0.02,
            user_tag: 0,
        }
    }
}

/// Runtime rigid body record.
#[derive(Debug, Clone, PartialEq)]
pub struct RigidBody {
    pub handle: BodyHandle,
    pub kind: BodyKind,
    pub position: Vector2<f32>,
    pub velocity: Vector2<f32>,
    pub accumulated_force: Vector2<f32>,
    pub inverse_mass: f32,
    pub linear_damping: f32,
    pub user_tag: u32,
    pub awake: bool,
}

impl RigidBody {
    pub fn from_desc(handle: BodyHandle, desc: BodyDesc) -> Self {
        let inverse_mass = match desc.kind {
            BodyKind::Dynamic if desc.mass > 0.0 => 1.0 / desc.mass,
            _ => 0.0,
        };
        Self {
            handle,
            kind: desc.kind,
            position: desc.position,
            velocity: desc.velocity,
            accumulated_force: Vector2::zeros(),
            inverse_mass,
            linear_damping: desc.linear_damping.clamp(0.0, 1.0),
            user_tag: desc.user_tag,
            awake: true,
        }
    }
}

/// Collider descriptor.
#[derive(Debug, Clone, PartialEq)]
pub struct ColliderDesc {
    pub body: Option<BodyHandle>,
    pub shape: ColliderShape,
    pub is_sensor: bool,
    pub user_tag: u32,
}

impl ColliderDesc {
    pub fn attached(body: BodyHandle, shape: ColliderShape) -> Self {
        Self {
            body: Some(body),
            shape,
            is_sensor: false,
            user_tag: 0,
        }
    }
}

/// Physics contact pair produced by the starter collision pipeline.
#[derive(Debug, Clone, PartialEq)]
pub struct ContactPair {
    pub collider_a: ColliderHandle,
    pub collider_b: ColliderHandle,
    pub body_a: Option<BodyHandle>,
    pub body_b: Option<BodyHandle>,
    pub normal: Vector2<f32>,
    pub penetration: f32,
}

/// Immutable body-scoped contact snapshot referenced by an opaque handle.
#[derive(Debug, Clone, PartialEq)]
pub struct ContactSnapshot {
    pub body: BodyHandle,
    pub contacts: Vec<ContactPair>,
}
