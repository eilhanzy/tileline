//! Runtime draw-path compiler from `SceneFrameInstances`.
//!
//! This module converts scene payloads into deterministic draw batches that a backend renderer can
//! consume directly. It is intentionally renderer-agnostic and focuses on stable ordering and
//! compact per-instance payloads.

use std::collections::BTreeMap;

use nalgebra::{Matrix4, Quaternion, Translation3, UnitQuaternion, Vector3};

use crate::scene::{
    SceneFrameInstances, SceneInstance3d, ScenePrimitive3d, ShadingModel, SpriteInstance,
};

/// Batched draw lane.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DrawLane {
    Opaque,
    Transparent,
}

/// Stable sort/group key for one 3D draw batch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct DrawBatchKey {
    pub primitive_code: u8,
    pub shading_code: u8,
    pub shadow_flags: u8,
}

impl DrawBatchKey {
    pub fn from_instance(instance: &SceneInstance3d) -> Self {
        let primitive_code = match instance.primitive {
            ScenePrimitive3d::Sphere => 0,
            ScenePrimitive3d::Box => 1,
        };
        let shading_code = match instance.material.shading {
            ShadingModel::LitPbr => 0,
            ShadingModel::Unlit => 1,
        };
        let shadow_flags =
            u8::from(instance.casts_shadow) | (u8::from(instance.receives_shadow) << 1);
        Self {
            primitive_code,
            shading_code,
            shadow_flags,
        }
    }
}

/// Backend-friendly instance payload for one 3D draw.
#[derive(Debug, Clone, PartialEq)]
pub struct DrawInstance3d {
    pub instance_id: u64,
    pub model_cols: [[f32; 4]; 4],
    pub base_color_rgba: [f32; 4],
    pub material_params: [f32; 4], // roughness, metallic, emissive_strength, reserved
    pub emissive_rgb: [f32; 3],
}

/// One 3D batch ready for backend draw encoding.
#[derive(Debug, Clone, PartialEq)]
pub struct DrawBatch3d {
    pub lane: DrawLane,
    pub key: DrawBatchKey,
    pub instances: Vec<DrawInstance3d>,
}

/// Draw-path statistics for telemetry.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DrawFrameStats {
    pub opaque_instances: usize,
    pub transparent_instances: usize,
    pub sprite_instances: usize,
    pub opaque_batches: usize,
    pub transparent_batches: usize,
    pub total_draw_calls: usize,
}

/// Render-ready frame payload.
#[derive(Debug, Clone, PartialEq)]
pub struct RuntimeDrawFrame {
    pub opaque_batches: Vec<DrawBatch3d>,
    pub transparent_batches: Vec<DrawBatch3d>,
    pub sprites: Vec<SpriteInstance>,
    pub stats: DrawFrameStats,
}

/// Reusable compiler that converts `SceneFrameInstances` into a backend draw frame.
#[derive(Default)]
pub struct DrawPathCompiler {
    opaque_map: BTreeMap<DrawBatchKey, Vec<DrawInstance3d>>,
    transparent_map: BTreeMap<DrawBatchKey, Vec<DrawInstance3d>>,
    sprite_scratch: Vec<SpriteInstance>,
}

impl DrawPathCompiler {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn compile(&mut self, frame: &SceneFrameInstances) -> RuntimeDrawFrame {
        self.opaque_map.clear();
        self.transparent_map.clear();
        self.sprite_scratch.clear();

        for instance in &frame.opaque_3d {
            let key = DrawBatchKey::from_instance(instance);
            self.opaque_map
                .entry(key)
                .or_default()
                .push(pack_draw_instance(instance));
        }
        for instance in &frame.transparent_3d {
            let key = DrawBatchKey::from_instance(instance);
            self.transparent_map
                .entry(key)
                .or_default()
                .push(pack_draw_instance(instance));
        }

        let mut opaque_batches = Vec::with_capacity(self.opaque_map.len());
        for (key, mut instances) in std::mem::take(&mut self.opaque_map) {
            instances.sort_by_key(|i| i.instance_id);
            opaque_batches.push(DrawBatch3d {
                lane: DrawLane::Opaque,
                key,
                instances,
            });
        }

        let mut transparent_batches = Vec::with_capacity(self.transparent_map.len());
        for (key, mut instances) in std::mem::take(&mut self.transparent_map) {
            instances.sort_by_key(|i| i.instance_id);
            transparent_batches.push(DrawBatch3d {
                lane: DrawLane::Transparent,
                key,
                instances,
            });
        }

        self.sprite_scratch.extend(frame.sprites.iter().cloned());
        self.sprite_scratch
            .sort_by_key(|sprite| (sprite.layer, sprite.sprite_id));

        let stats = DrawFrameStats {
            opaque_instances: frame.opaque_3d.len(),
            transparent_instances: frame.transparent_3d.len(),
            sprite_instances: self.sprite_scratch.len(),
            opaque_batches: opaque_batches.len(),
            transparent_batches: transparent_batches.len(),
            total_draw_calls: opaque_batches.len()
                + transparent_batches.len()
                + self.sprite_scratch.len(),
        };

        RuntimeDrawFrame {
            opaque_batches,
            transparent_batches,
            sprites: std::mem::take(&mut self.sprite_scratch),
            stats,
        }
    }
}

fn pack_draw_instance(instance: &SceneInstance3d) -> DrawInstance3d {
    let model = compose_model_matrix(instance);
    let emissive = instance.material.emissive_rgb;
    let emissive_strength = emissive[0].max(emissive[1]).max(emissive[2]);
    DrawInstance3d {
        instance_id: instance.instance_id,
        model_cols: matrix_to_cols(&model),
        base_color_rgba: instance.material.base_color_rgba,
        material_params: [
            instance.material.roughness,
            instance.material.metallic,
            emissive_strength,
            0.0,
        ],
        emissive_rgb: emissive,
    }
}

fn compose_model_matrix(instance: &SceneInstance3d) -> Matrix4<f32> {
    let t = instance.transform.translation;
    let r = instance.transform.rotation_xyzw;
    let s = instance.transform.scale;

    let translation = Translation3::new(t[0], t[1], t[2]).to_homogeneous();
    let rotation =
        UnitQuaternion::from_quaternion(Quaternion::new(r[3], r[0], r[1], r[2])).to_homogeneous();
    let scale = Matrix4::new_nonuniform_scaling(&Vector3::new(s[0], s[1], s[2]));
    translation * rotation * scale
}

fn matrix_to_cols(m: &Matrix4<f32>) -> [[f32; 4]; 4] {
    [
        [m[(0, 0)], m[(1, 0)], m[(2, 0)], m[(3, 0)]],
        [m[(0, 1)], m[(1, 1)], m[(2, 1)], m[(3, 1)]],
        [m[(0, 2)], m[(1, 2)], m[(2, 2)], m[(3, 2)]],
        [m[(0, 3)], m[(1, 3)], m[(2, 3)], m[(3, 3)]],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::{SceneMaterial, SceneTransform3d};

    #[test]
    fn compile_groups_instances_into_deterministic_batches() {
        let mut frame = SceneFrameInstances::default();
        frame.opaque_3d.push(SceneInstance3d {
            instance_id: 5,
            primitive: ScenePrimitive3d::Sphere,
            transform: SceneTransform3d::default(),
            material: SceneMaterial::default(),
            casts_shadow: true,
            receives_shadow: true,
        });
        frame.opaque_3d.push(SceneInstance3d {
            instance_id: 2,
            primitive: ScenePrimitive3d::Sphere,
            transform: SceneTransform3d::default(),
            material: SceneMaterial::default(),
            casts_shadow: true,
            receives_shadow: true,
        });
        frame.transparent_3d.push(SceneInstance3d {
            instance_id: 1,
            primitive: ScenePrimitive3d::Box,
            transform: SceneTransform3d::default(),
            material: SceneMaterial::default(),
            casts_shadow: false,
            receives_shadow: false,
        });
        frame.sprites.push(SpriteInstance {
            sprite_id: 9,
            position: [0.0, 0.0, 0.0],
            size: [0.1, 0.1],
            rotation_rad: 0.0,
            color_rgba: [1.0, 1.0, 1.0, 1.0],
            texture_slot: 0,
            layer: 20,
        });
        frame.sprites.push(SpriteInstance {
            sprite_id: 1,
            position: [0.0, 0.0, 0.0],
            size: [0.1, 0.1],
            rotation_rad: 0.0,
            color_rgba: [1.0, 1.0, 1.0, 1.0],
            texture_slot: 0,
            layer: 10,
        });

        let mut compiler = DrawPathCompiler::new();
        let out = compiler.compile(&frame);
        assert_eq!(out.stats.opaque_batches, 1);
        assert_eq!(out.stats.transparent_batches, 1);
        assert_eq!(out.opaque_batches[0].instances[0].instance_id, 2);
        assert_eq!(out.opaque_batches[0].instances[1].instance_id, 5);
        assert_eq!(out.sprites[0].layer, 10);
        assert_eq!(out.stats.total_draw_calls, 4);
    }

    #[test]
    fn model_matrix_contains_translation() {
        let instance = SceneInstance3d {
            instance_id: 10,
            primitive: ScenePrimitive3d::Sphere,
            transform: SceneTransform3d {
                translation: [2.0, -3.0, 4.0],
                rotation_xyzw: [0.0, 0.0, 0.0, 1.0],
                scale: [1.0, 1.0, 1.0],
            },
            material: SceneMaterial::default(),
            casts_shadow: true,
            receives_shadow: true,
        };
        let packed = pack_draw_instance(&instance);
        assert!((packed.model_cols[3][0] - 2.0).abs() < 1e-6);
        assert!((packed.model_cols[3][1] + 3.0).abs() < 1e-6);
        assert!((packed.model_cols[3][2] - 4.0).abs() < 1e-6);
    }
}
