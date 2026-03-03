//! Narrowphase manifold generation for ParadoxPE.
//!
//! This stage consumes broadphase body pairs, queries the attached primary collider shapes, and
//! emits a reusable manifold buffer for the solver.

use nalgebra::Vector3;

use crate::body::{
    Aabb, ColliderMaterial, ColliderShape, ContactId, ContactManifold, MaterialCombineRule,
};
use crate::handle::BodyHandle;
use crate::handle::ColliderHandle;
use crate::storage::BodyRegistry;

/// Narrowphase configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct NarrowphaseConfig {
    /// Maximum manifold count stored without reallocating.
    pub max_manifolds: usize,
    /// Global restitution combine rule.
    pub restitution_combine_rule: MaterialCombineRule,
    /// Global friction combine rule.
    pub friction_combine_rule: MaterialCombineRule,
}

impl Default for NarrowphaseConfig {
    fn default() -> Self {
        Self {
            max_manifolds: 16_384,
            restitution_combine_rule: MaterialCombineRule::Max,
            friction_combine_rule: MaterialCombineRule::Average,
        }
    }
}

/// Narrowphase execution statistics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct NarrowphaseStats {
    pub candidate_pairs: usize,
    pub manifolds: usize,
    pub persistent_manifolds: usize,
    pub culled_pairs: usize,
    pub overflowed: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PersistentManifoldState {
    contact_id: ContactId,
    persisted_frames: u16,
}

/// Reusable manifold builder.
#[derive(Debug, Clone)]
pub struct NarrowphasePipeline {
    config: NarrowphaseConfig,
    manifolds: Vec<ContactManifold>,
    previous_states: Vec<PersistentManifoldState>,
    next_states: Vec<PersistentManifoldState>,
    stats: NarrowphaseStats,
}

impl NarrowphasePipeline {
    pub fn new(config: NarrowphaseConfig) -> Self {
        Self {
            manifolds: Vec::new(),
            previous_states: Vec::new(),
            next_states: Vec::new(),
            config,
            stats: NarrowphaseStats::default(),
        }
    }

    pub fn config(&self) -> &NarrowphaseConfig {
        &self.config
    }

    pub fn stats(&self) -> NarrowphaseStats {
        self.stats
    }

    pub fn manifolds(&self) -> &[ContactManifold] {
        &self.manifolds
    }

    pub fn max_manifolds_capacity(&self) -> usize {
        self.manifolds.capacity()
    }

    pub fn sync_for_pair_capacity(&mut self, pair_capacity: usize) {
        let target = self.config.max_manifolds.max(pair_capacity);
        if self.manifolds.capacity() < target {
            self.manifolds
                .reserve(target.saturating_sub(self.manifolds.capacity()));
        }
        if self.previous_states.capacity() < target {
            self.previous_states
                .reserve(target.saturating_sub(self.previous_states.capacity()));
        }
        if self.next_states.capacity() < target {
            self.next_states
                .reserve(target.saturating_sub(self.next_states.capacity()));
        }
    }

    pub fn rebuild_manifolds<F>(
        &mut self,
        bodies: &BodyRegistry,
        candidate_pairs: &[(BodyHandle, BodyHandle)],
        mut collider_lookup: F,
    ) -> &[ContactManifold]
    where
        F: FnMut(
            BodyHandle,
        ) -> Option<(
            crate::handle::ColliderHandle,
            ColliderShape,
            ColliderMaterial,
        )>,
    {
        self.stats = NarrowphaseStats {
            candidate_pairs: candidate_pairs.len(),
            ..NarrowphaseStats::default()
        };
        self.manifolds.clear();
        self.next_states.clear();
        self.sync_for_pair_capacity(candidate_pairs.len());

        for &(body_a, body_b) in candidate_pairs {
            let Some((collider_a, shape_a, material_a)) = collider_lookup(body_a) else {
                self.stats.culled_pairs += 1;
                continue;
            };
            let Some((collider_b, shape_b, material_b)) = collider_lookup(body_b) else {
                self.stats.culled_pairs += 1;
                continue;
            };
            let Some(center_a) = bodies.position_for(body_a) else {
                self.stats.culled_pairs += 1;
                continue;
            };
            let Some(center_b) = bodies.position_for(body_b) else {
                self.stats.culled_pairs += 1;
                continue;
            };
            let Some(aabb_a) = bodies.aabb_for(body_a) else {
                self.stats.culled_pairs += 1;
                continue;
            };
            let Some(aabb_b) = bodies.aabb_for(body_b) else {
                self.stats.culled_pairs += 1;
                continue;
            };
            let Some((point, normal, penetration)) =
                collide_shapes(center_a, &shape_a, aabb_a, center_b, &shape_b, aabb_b)
            else {
                self.stats.culled_pairs += 1;
                continue;
            };
            let feature_tag = feature_tag_for_contact(&shape_a, &shape_b, normal);
            let contact_id = build_contact_id(collider_a, collider_b, feature_tag);
            let persisted_frames = self
                .lookup_persistent_state(contact_id)
                .map(|state| state.persisted_frames.saturating_add(1))
                .unwrap_or(1);
            let restitution = self
                .config
                .restitution_combine_rule
                .combine(material_a.restitution, material_b.restitution);
            let friction = self
                .config
                .friction_combine_rule
                .combine(material_a.friction, material_b.friction);
            if self.manifolds.len() < self.manifolds.capacity() {
                self.manifolds.push(ContactManifold {
                    contact_id,
                    collider_a,
                    collider_b,
                    body_a,
                    body_b,
                    point,
                    normal,
                    penetration,
                    persisted_frames,
                    restitution,
                    friction,
                });
                if persisted_frames > 1 {
                    self.stats.persistent_manifolds += 1;
                }
                if self.next_states.len() < self.next_states.capacity() {
                    self.next_states.push(PersistentManifoldState {
                        contact_id,
                        persisted_frames,
                    });
                } else {
                    self.stats.overflowed = true;
                }
            } else {
                self.stats.overflowed = true;
                self.stats.culled_pairs += 1;
            }
        }

        self.stats.manifolds = self.manifolds.len();
        std::mem::swap(&mut self.previous_states, &mut self.next_states);
        &self.manifolds
    }

    fn lookup_persistent_state(&self, contact_id: ContactId) -> Option<PersistentManifoldState> {
        self.previous_states
            .iter()
            .find(|state| state.contact_id == contact_id)
            .copied()
    }
}

fn collide_shapes(
    center_a: Vector3<f32>,
    shape_a: &ColliderShape,
    aabb_a: Aabb,
    center_b: Vector3<f32>,
    shape_b: &ColliderShape,
    aabb_b: Aabb,
) -> Option<(Vector3<f32>, Vector3<f32>, f32)> {
    match (shape_a, shape_b) {
        (ColliderShape::Sphere { radius: ra }, ColliderShape::Sphere { radius: rb }) => {
            sphere_sphere(center_a, *ra, center_b, *rb)
        }
        (ColliderShape::Sphere { radius }, ColliderShape::Aabb { .. }) => {
            sphere_aabb(center_a, *radius, aabb_b)
        }
        (ColliderShape::Aabb { .. }, ColliderShape::Sphere { radius }) => {
            let (point, normal, penetration) = sphere_aabb(center_b, *radius, aabb_a)?;
            Some((point, -normal, penetration))
        }
        (ColliderShape::Aabb { .. }, ColliderShape::Aabb { .. }) => aabb_aabb(aabb_a, aabb_b),
    }
}

fn sphere_sphere(
    center_a: Vector3<f32>,
    radius_a: f32,
    center_b: Vector3<f32>,
    radius_b: f32,
) -> Option<(Vector3<f32>, Vector3<f32>, f32)> {
    let delta = center_b - center_a;
    let distance_sq = delta.norm_squared();
    let radius_sum = radius_a.max(0.0) + radius_b.max(0.0);
    if distance_sq >= radius_sum * radius_sum {
        return None;
    }
    let distance = distance_sq.sqrt();
    let normal = if distance > 1e-6 {
        delta / distance
    } else {
        Vector3::new(1.0, 0.0, 0.0)
    };
    let penetration = radius_sum - distance;
    let point = center_a + normal * (radius_a.max(0.0) - penetration * 0.5);
    Some((point, normal, penetration))
}

fn sphere_aabb(
    sphere_center: Vector3<f32>,
    radius: f32,
    box_aabb: Aabb,
) -> Option<(Vector3<f32>, Vector3<f32>, f32)> {
    let clamped = Vector3::new(
        sphere_center.x.clamp(box_aabb.min.x, box_aabb.max.x),
        sphere_center.y.clamp(box_aabb.min.y, box_aabb.max.y),
        sphere_center.z.clamp(box_aabb.min.z, box_aabb.max.z),
    );
    let delta = sphere_center - clamped;
    let distance_sq = delta.norm_squared();
    let radius = radius.max(0.0);
    if distance_sq > radius * radius {
        return None;
    }
    let distance = distance_sq.sqrt();
    let normal = if distance > 1e-6 {
        delta / distance
    } else {
        let box_center = box_aabb.center();
        let dir = sphere_center - box_center;
        if dir.norm_squared() > 1e-6 {
            dir.normalize()
        } else {
            Vector3::new(1.0, 0.0, 0.0)
        }
    };
    let penetration = radius - distance;
    let point = clamped;
    Some((point, normal, penetration))
}

fn aabb_aabb(a: Aabb, b: Aabb) -> Option<(Vector3<f32>, Vector3<f32>, f32)> {
    if !a.intersects(b) {
        return None;
    }
    let overlap = a.overlap(b);
    let delta = b.center() - a.center();
    let (penetration, normal) = if overlap.x <= overlap.y && overlap.x <= overlap.z {
        (
            overlap.x,
            Vector3::new(if delta.x >= 0.0 { 1.0 } else { -1.0 }, 0.0, 0.0),
        )
    } else if overlap.y <= overlap.z {
        (
            overlap.y,
            Vector3::new(0.0, if delta.y >= 0.0 { 1.0 } else { -1.0 }, 0.0),
        )
    } else {
        (
            overlap.z,
            Vector3::new(0.0, 0.0, if delta.z >= 0.0 { 1.0 } else { -1.0 }),
        )
    };
    let point = (a.center() + b.center()) * 0.5;
    Some((point, normal, penetration))
}

fn build_contact_id(
    collider_a: ColliderHandle,
    collider_b: ColliderHandle,
    feature_tag: u8,
) -> ContactId {
    let low = collider_a.raw().min(collider_b.raw()) as u64;
    let high = collider_a.raw().max(collider_b.raw()) as u64;
    ContactId::new(((low << 32) | high) ^ ((feature_tag as u64) << 56))
}

fn feature_tag_for_contact(
    shape_a: &ColliderShape,
    shape_b: &ColliderShape,
    normal: Vector3<f32>,
) -> u8 {
    let axis_tag = axis_feature_tag(normal);
    match (shape_a, shape_b) {
        (ColliderShape::Sphere { .. }, ColliderShape::Sphere { .. }) => 0x01,
        (ColliderShape::Sphere { .. }, ColliderShape::Aabb { .. }) => 0x10 | axis_tag,
        (ColliderShape::Aabb { .. }, ColliderShape::Sphere { .. }) => 0x20 | axis_tag,
        (ColliderShape::Aabb { .. }, ColliderShape::Aabb { .. }) => 0x30 | axis_tag,
    }
}

fn axis_feature_tag(normal: Vector3<f32>) -> u8 {
    let abs = normal.map(f32::abs);
    if abs.x >= abs.y && abs.x >= abs.z {
        if normal.x >= 0.0 {
            0x1
        } else {
            0x2
        }
    } else if abs.y >= abs.z {
        if normal.y >= 0.0 {
            0x3
        } else {
            0x4
        }
    } else if normal.z >= 0.0 {
        0x5
    } else {
        0x6
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::Vector3;

    use super::*;
    use crate::body::{BodyDesc, ColliderMaterial, ColliderShape};
    use crate::handle::ColliderHandle;

    #[test]
    fn narrowphase_emits_manifold_for_overlapping_aabbs() {
        let mut bodies = BodyRegistry::new();
        let a = bodies.spawn(BodyDesc {
            position: Vector3::new(0.0, 0.0, 0.0),
            ..BodyDesc::default()
        });
        let b = bodies.spawn(BodyDesc {
            position: Vector3::new(0.5, 0.0, 0.0),
            ..BodyDesc::default()
        });
        let candidate_pairs = vec![(a, b)];
        let mut narrowphase = NarrowphasePipeline::new(NarrowphaseConfig::default());
        let manifolds = narrowphase.rebuild_manifolds(&bodies, &candidate_pairs, |body| {
            Some((
                ColliderHandle::new(body.index() as u16, body.generation()),
                ColliderShape::Aabb {
                    half_extents: Vector3::new(0.5, 0.5, 0.5),
                },
                ColliderMaterial::default(),
            ))
        });
        assert_eq!(manifolds.len(), 1);
        assert!(manifolds[0].penetration > 0.0);
        assert_eq!(manifolds[0].persisted_frames, 1);
    }

    #[test]
    fn narrowphase_preserves_contact_ids_across_frames() {
        let mut bodies = BodyRegistry::new();
        let a = bodies.spawn(BodyDesc {
            position: Vector3::new(0.0, 0.0, 0.0),
            ..BodyDesc::default()
        });
        let b = bodies.spawn(BodyDesc {
            position: Vector3::new(0.5, 0.0, 0.0),
            ..BodyDesc::default()
        });
        let candidate_pairs = vec![(a, b)];
        let mut narrowphase = NarrowphasePipeline::new(NarrowphaseConfig::default());

        let first_id = {
            let manifolds = narrowphase.rebuild_manifolds(&bodies, &candidate_pairs, |body| {
                Some((
                    ColliderHandle::new(body.index() as u16, body.generation()),
                    ColliderShape::Aabb {
                        half_extents: Vector3::new(0.5, 0.5, 0.5),
                    },
                    ColliderMaterial::default(),
                ))
            });
            assert_eq!(manifolds[0].persisted_frames, 1);
            manifolds[0].contact_id
        };

        let manifolds = narrowphase.rebuild_manifolds(&bodies, &candidate_pairs, |body| {
            Some((
                ColliderHandle::new(body.index() as u16, body.generation()),
                ColliderShape::Aabb {
                    half_extents: Vector3::new(0.5, 0.5, 0.5),
                },
                ColliderMaterial::default(),
            ))
        });
        assert_eq!(manifolds[0].contact_id, first_id);
        assert_eq!(manifolds[0].persisted_frames, 2);
        assert_eq!(narrowphase.stats().persistent_manifolds, 1);
    }

    #[test]
    fn narrowphase_combines_material_coefficients() {
        let mut bodies = BodyRegistry::new();
        let a = bodies.spawn(BodyDesc::default());
        let b = bodies.spawn(BodyDesc {
            position: Vector3::new(0.5, 0.0, 0.0),
            ..BodyDesc::default()
        });
        let candidate_pairs = vec![(a, b)];
        let mut narrowphase = NarrowphasePipeline::new(NarrowphaseConfig {
            restitution_combine_rule: MaterialCombineRule::Max,
            friction_combine_rule: MaterialCombineRule::Multiply,
            ..NarrowphaseConfig::default()
        });
        let manifolds = narrowphase.rebuild_manifolds(&bodies, &candidate_pairs, |body| {
            let material = if body == a {
                ColliderMaterial {
                    restitution: 0.1,
                    friction: 0.5,
                }
            } else {
                ColliderMaterial {
                    restitution: 0.8,
                    friction: 0.4,
                }
            };
            Some((
                ColliderHandle::new(body.index() as u16, body.generation()),
                ColliderShape::Aabb {
                    half_extents: Vector3::new(0.5, 0.5, 0.5),
                },
                material,
            ))
        });
        assert!((manifolds[0].restitution - 0.8).abs() < 1e-6);
        assert!((manifolds[0].friction - 0.2).abs() < 1e-6);
    }
}
