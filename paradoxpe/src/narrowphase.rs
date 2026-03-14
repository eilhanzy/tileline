//! Narrowphase manifold generation for ParadoxPE.
//!
//! This stage consumes broadphase body pairs, queries the attached primary collider shapes, and
//! emits a reusable manifold buffer for the solver.

use std::collections::HashMap;

use nalgebra::Vector3;
use rayon::prelude::*;

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
    persisted_lookup: HashMap<ContactId, u16>,
    stats: NarrowphaseStats,
}

impl NarrowphasePipeline {
    pub fn new(config: NarrowphaseConfig) -> Self {
        Self {
            manifolds: Vec::new(),
            previous_states: Vec::new(),
            next_states: Vec::new(),
            persisted_lookup: HashMap::new(),
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
        if self.persisted_lookup.capacity() < target {
            self.persisted_lookup
                .reserve(target.saturating_sub(self.persisted_lookup.capacity()));
        }
    }

    pub fn rebuild_manifolds<F>(
        &mut self,
        bodies: &BodyRegistry,
        candidate_pairs: &[(BodyHandle, BodyHandle)],
        collider_lookup: F,
    ) -> &[ContactManifold]
    where
        F: Fn(
                BodyHandle,
            ) -> Option<(
                crate::handle::ColliderHandle,
                ColliderShape,
                ColliderMaterial,
                u32,
            )> + Sync,
    {
        self.stats = NarrowphaseStats {
            candidate_pairs: candidate_pairs.len(),
            ..NarrowphaseStats::default()
        };
        self.manifolds.clear();
        self.next_states.clear();
        self.sync_for_pair_capacity(candidate_pairs.len());
        self.persisted_lookup.clear();
        for state in &self.previous_states {
            self.persisted_lookup
                .insert(state.contact_id, state.persisted_frames);
        }
        let persisted_lookup = &self.persisted_lookup;

        let produced = candidate_pairs
            .par_iter()
            .filter_map(|&(body_a, body_b)| {
                let (collider_a, shape_a, material_a, filter_a) = collider_lookup(body_a)?;
                let (collider_b, shape_b, material_b, filter_b) = collider_lookup(body_b)?;
                if !collision_filter_allows(filter_a, filter_b) {
                    return None;
                }
                let center_a = bodies.position_for(body_a)?;
                let center_b = bodies.position_for(body_b)?;
                let aabb_a = bodies.aabb_for(body_a)?;
                let aabb_b = bodies.aabb_for(body_b)?;
                let (point, normal, penetration) =
                    collide_shapes(center_a, &shape_a, aabb_a, center_b, &shape_b, aabb_b)?;

                let feature_tag = feature_tag_for_contact(&shape_a, &shape_b, normal);
                let contact_id = build_contact_id(collider_a, collider_b, feature_tag);
                let persisted_frames = persisted_lookup
                    .get(&contact_id)
                    .copied()
                    .unwrap_or(0)
                    .saturating_add(1);
                let restitution = self
                    .config
                    .restitution_combine_rule
                    .combine(material_a.restitution, material_b.restitution);
                let friction = self
                    .config
                    .friction_combine_rule
                    .combine(material_a.friction, material_b.friction);

                Some(ContactManifold {
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
                })
            })
            .collect::<Vec<_>>();

        let produced_count = produced.len();
        let capacity = self.manifolds.capacity();
        let keep_count = produced_count.min(capacity);
        self.manifolds.extend(produced.into_iter().take(keep_count));

        self.stats.culled_pairs = candidate_pairs.len().saturating_sub(produced_count);
        if produced_count > capacity {
            self.stats.overflowed = true;
            self.stats.culled_pairs = self
                .stats
                .culled_pairs
                .saturating_add(produced_count.saturating_sub(capacity));
        }

        for manifold in &self.manifolds {
            if manifold.persisted_frames > 1 {
                self.stats.persistent_manifolds += 1;
            }
            if self.next_states.len() < self.next_states.capacity() {
                self.next_states.push(PersistentManifoldState {
                    contact_id: manifold.contact_id,
                    persisted_frames: manifold.persisted_frames,
                });
            } else {
                self.stats.overflowed = true;
                break;
            }
        }

        self.stats.manifolds = keep_count;
        std::mem::swap(&mut self.previous_states, &mut self.next_states);
        &self.manifolds
    }
}

#[inline]
fn collision_filter_allows(tag_a: u32, tag_b: u32) -> bool {
    let (group_a, mask_a) = decode_collision_filter(tag_a);
    let (group_b, mask_b) = decode_collision_filter(tag_b);
    (mask_a & group_b) != 0 && (mask_b & group_a) != 0
}

#[inline]
fn decode_collision_filter(tag: u32) -> (u16, u16) {
    // Backward-compatible fallback: tag=0 means "default group, collide with everything".
    if tag == 0 {
        return (0x0001, u16::MAX);
    }
    let group = (tag & 0xFFFF) as u16;
    let mask = ((tag >> 16) & 0xFFFF) as u16;
    let group = if group == 0 { 0x0001 } else { group };
    let mask = if mask == 0 { u16::MAX } else { mask };
    (group, mask)
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
    if distance_sq > 1e-12 {
        let distance = distance_sq.sqrt();
        if distance > radius {
            return None;
        }
        // Normal points from sphere -> box (A -> B convention for solver).
        let normal = (clamped - sphere_center) / distance;
        let penetration = radius - distance;
        let point = clamped;
        return Some((point, normal, penetration));
    }

    // Sphere center is inside the box volume; resolve against the closest face.
    let dist_to_min = sphere_center - box_aabb.min;
    let dist_to_max = box_aabb.max - sphere_center;

    let mut axis = 0usize;
    let mut use_max_face = dist_to_max.x < dist_to_min.x;
    let mut min_face_distance = if use_max_face {
        dist_to_max.x
    } else {
        dist_to_min.x
    };

    let y_face_uses_max = dist_to_max.y < dist_to_min.y;
    let y_face_distance = if y_face_uses_max {
        dist_to_max.y
    } else {
        dist_to_min.y
    };
    if y_face_distance < min_face_distance {
        axis = 1;
        use_max_face = y_face_uses_max;
        min_face_distance = y_face_distance;
    }

    let z_face_uses_max = dist_to_max.z < dist_to_min.z;
    let z_face_distance = if z_face_uses_max {
        dist_to_max.z
    } else {
        dist_to_min.z
    };
    if z_face_distance < min_face_distance {
        axis = 2;
        use_max_face = z_face_uses_max;
        min_face_distance = z_face_distance;
    }

    let mut normal = Vector3::zeros();
    normal[axis] = if use_max_face { 1.0 } else { -1.0 };

    let mut point = sphere_center;
    point[axis] = if use_max_face {
        box_aabb.max[axis]
    } else {
        box_aabb.min[axis]
    };

    // Distance to closest face + radius required to exit overlap.
    let penetration = radius + min_face_distance.max(0.0);
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
    use crate::body::{Aabb, BodyDesc, ColliderMaterial, ColliderShape};
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
                0,
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
                    0,
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
                0,
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
                0,
            ))
        });
        assert!((manifolds[0].restitution - 0.8).abs() < 1e-6);
        assert!((manifolds[0].friction - 0.2).abs() < 1e-6);
    }

    #[test]
    fn collision_layer_filter_can_reject_pair() {
        let mut bodies = BodyRegistry::new();
        let a = bodies.spawn(BodyDesc::default());
        let b = bodies.spawn(BodyDesc {
            position: Vector3::new(0.2, 0.0, 0.0),
            ..BodyDesc::default()
        });
        let candidate_pairs = vec![(a, b)];
        let mut narrowphase = NarrowphasePipeline::new(NarrowphaseConfig::default());

        let manifolds = narrowphase.rebuild_manifolds(&bodies, &candidate_pairs, |body| {
            // group=1 mask=1 for a, group=2 mask=2 for b -> no overlap in masks.
            let filter = if body == a {
                (0x0001_u32) | ((0x0001_u32) << 16)
            } else {
                (0x0002_u32) | ((0x0002_u32) << 16)
            };
            Some((
                ColliderHandle::new(body.index() as u16, body.generation()),
                ColliderShape::Aabb {
                    half_extents: Vector3::new(0.5, 0.5, 0.5),
                },
                ColliderMaterial::default(),
                filter,
            ))
        });
        assert!(manifolds.is_empty());
    }

    #[test]
    fn sphere_aabb_normal_points_from_sphere_to_box_for_outside_contact() {
        let sphere_center = Vector3::new(1.15, 0.0, 0.0);
        let radius = 0.25;
        let box_aabb = Aabb::from_center_half_extents(Vector3::zeros(), Vector3::repeat(1.0));
        let (_point, normal, penetration) =
            sphere_aabb(sphere_center, radius, box_aabb).expect("contact");

        assert!(penetration > 0.0);
        // Sphere is on +X side of the box, so sphere -> box normal should be -X.
        assert!(normal.x < -0.9);
    }

    #[test]
    fn sphere_aabb_inside_chooses_nearest_face_and_positive_penetration() {
        let sphere_center = Vector3::new(0.92, 0.1, 0.0);
        let radius = 0.20;
        let box_aabb = Aabb::from_center_half_extents(Vector3::zeros(), Vector3::repeat(1.0));
        let (_point, normal, penetration) =
            sphere_aabb(sphere_center, radius, box_aabb).expect("contact");

        // Nearest face is +X, so sphere -> box normal should be +X.
        assert!(normal.x > 0.9);
        assert!(penetration > radius);
    }
}
