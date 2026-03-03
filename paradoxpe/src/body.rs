//! Core ParadoxPE body, collider, and contact data types.

use nalgebra::{UnitQuaternion, Vector3};

use crate::handle::{BodyHandle, ColliderHandle};

/// Axis-aligned bounding box used by the SoA storage and broadphase.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Aabb {
    pub min: Vector3<f32>,
    pub max: Vector3<f32>,
}

impl Aabb {
    pub fn new(min: Vector3<f32>, max: Vector3<f32>) -> Self {
        Self { min, max }
    }

    pub fn from_center_half_extents(center: Vector3<f32>, half_extents: Vector3<f32>) -> Self {
        let extents = half_extents.map(|value| value.max(0.0));
        Self {
            min: center - extents,
            max: center + extents,
        }
    }

    pub fn center(self) -> Vector3<f32> {
        (self.min + self.max) * 0.5
    }

    pub fn half_extents(self) -> Vector3<f32> {
        (self.max - self.min) * 0.5
    }

    pub fn translated(self, offset: Vector3<f32>) -> Self {
        Self {
            min: self.min + offset,
            max: self.max + offset,
        }
    }

    pub fn union(self, other: Self) -> Self {
        Self {
            min: Vector3::new(
                self.min.x.min(other.min.x),
                self.min.y.min(other.min.y),
                self.min.z.min(other.min.z),
            ),
            max: Vector3::new(
                self.max.x.max(other.max.x),
                self.max.y.max(other.max.y),
                self.max.z.max(other.max.z),
            ),
        }
    }

    pub fn intersects(self, other: Self) -> bool {
        self.min.x <= other.max.x
            && self.max.x >= other.min.x
            && self.min.y <= other.max.y
            && self.max.y >= other.min.y
            && self.min.z <= other.max.z
            && self.max.z >= other.min.z
    }

    pub fn overlap(self, other: Self) -> Vector3<f32> {
        Vector3::new(
            (self.max.x.min(other.max.x) - self.min.x.max(other.min.x)).max(0.0),
            (self.max.y.min(other.max.y) - self.min.y.max(other.min.y)).max(0.0),
            (self.max.z.min(other.max.z) - self.min.z.max(other.min.z)).max(0.0),
        )
    }
}

impl Default for Aabb {
    fn default() -> Self {
        Self::from_center_half_extents(Vector3::zeros(), Vector3::repeat(0.5))
    }
}

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
    Sphere = 0,
    Aabb = 1,
}

impl ColliderShapeKind {
    pub fn from_u32(value: u32) -> Option<Self> {
        match value {
            0 => Some(Self::Sphere),
            1 => Some(Self::Aabb),
            _ => None,
        }
    }
}

/// Script-facing collider-shape wrapper used by the host ABI.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum ScriptColliderShapeKind {
    Sphere = 0,
    Aabb = 1,
}

impl From<ScriptColliderShapeKind> for ColliderShapeKind {
    fn from(value: ScriptColliderShapeKind) -> Self {
        match value {
            ScriptColliderShapeKind::Sphere => Self::Sphere,
            ScriptColliderShapeKind::Aabb => Self::Aabb,
        }
    }
}

/// Collider shape data used by the broadphase and future narrowphase.
#[derive(Debug, Clone, PartialEq)]
pub enum ColliderShape {
    Sphere { radius: f32 },
    Aabb { half_extents: Vector3<f32> },
}

impl ColliderShape {
    pub fn from_script(kind: ColliderShapeKind, a: f32, b: f32) -> Self {
        match kind {
            ColliderShapeKind::Sphere => Self::Sphere { radius: a.max(0.0) },
            ColliderShapeKind::Aabb => Self::Aabb {
                half_extents: Vector3::new(a.max(0.0), b.max(0.0), a.max(b).max(0.0)),
            },
        }
    }

    pub fn local_aabb(&self) -> Aabb {
        match self {
            Self::Sphere { radius } => Aabb::from_center_half_extents(
                Vector3::zeros(),
                Vector3::repeat((*radius).max(0.0)),
            ),
            Self::Aabb { half_extents } => {
                Aabb::from_center_half_extents(Vector3::zeros(), half_extents.map(|v| v.max(0.0)))
            }
        }
    }
}

/// Rigid body descriptor consumed by the SoA body registry.
#[derive(Debug, Clone, PartialEq)]
pub struct BodyDesc {
    pub kind: BodyKind,
    pub position: Vector3<f32>,
    pub rotation: UnitQuaternion<f32>,
    pub linear_velocity: Vector3<f32>,
    pub mass: f32,
    pub linear_damping: f32,
    pub user_tag: u32,
    pub local_bounds: Aabb,
}

impl Default for BodyDesc {
    fn default() -> Self {
        Self {
            kind: BodyKind::Dynamic,
            position: Vector3::zeros(),
            rotation: UnitQuaternion::identity(),
            linear_velocity: Vector3::zeros(),
            mass: 1.0,
            linear_damping: 0.02,
            user_tag: 0,
            local_bounds: Aabb::default(),
        }
    }
}

/// Snapshot reconstructed from the SoA body registry for API consumers and tests.
#[derive(Debug, Clone, PartialEq)]
pub struct RigidBody {
    pub handle: BodyHandle,
    pub kind: BodyKind,
    pub position: Vector3<f32>,
    pub rotation: UnitQuaternion<f32>,
    pub linear_velocity: Vector3<f32>,
    pub accumulated_force: Vector3<f32>,
    pub inverse_mass: f32,
    pub linear_damping: f32,
    pub user_tag: u32,
    pub awake: bool,
    pub sleep_timer: f32,
    pub aabb: Aabb,
    pub primary_collider: Option<ColliderHandle>,
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

/// Physics contact pair produced by broadphase/contact candidate generation.
#[derive(Debug, Clone, PartialEq)]
pub struct ContactPair {
    pub collider_a: ColliderHandle,
    pub collider_b: ColliderHandle,
    pub body_a: Option<BodyHandle>,
    pub body_b: Option<BodyHandle>,
    pub point: Vector3<f32>,
    pub normal: Vector3<f32>,
    pub penetration: f32,
}

/// Narrowphase manifold emitted before the solver stage.
#[derive(Debug, Clone, PartialEq)]
pub struct ContactManifold {
    pub collider_a: ColliderHandle,
    pub collider_b: ColliderHandle,
    pub body_a: BodyHandle,
    pub body_b: BodyHandle,
    pub point: Vector3<f32>,
    pub normal: Vector3<f32>,
    pub penetration: f32,
    pub restitution: f32,
    pub friction: f32,
}

/// Immutable body-scoped contact snapshot referenced by an opaque handle.
#[derive(Debug, Clone, PartialEq)]
pub struct ContactSnapshot {
    pub body: BodyHandle,
    pub contacts: Vec<ContactPair>,
}
