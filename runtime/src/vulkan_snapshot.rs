//! Runtime draw-frame -> raw Vulkan snapshot translation.
//!
//! This module is the canonical handoff point between runtime scene batching and the
//! `tl_core::VulkanBackend` persistently mapped snapshot ring. It keeps the transform/material
//! payload compact and deterministic so `Render N` can consume a published snapshot while
//! `Physics N+1` continues filling the next slot.

use std::collections::BTreeMap;

use tl_core::{
    FrameInstanceTransform, FrameLightRecord, FrameMaterialRecord, FrameTextureRecord,
    RenderStateSnapshot,
};

use crate::draw_path::{DrawBatch3d, DrawLane, RuntimeDrawFrame};
use crate::scene::{SceneLight, SceneLightKind};

const FLAG_TRANSPARENT: u32 = 1 << 0;
const FLAG_MESH: u32 = 1 << 1;
const MATERIAL_FLAG_UNLIT: u32 = 1 << 0;

/// Summary of one runtime-to-Vulkan snapshot build.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct VulkanSnapshotBuildStats {
    pub total_instances: usize,
    pub opaque_instances: usize,
    pub transparent_instances: usize,
    pub mesh_instances: usize,
    pub material_records: usize,
    pub texture_records: usize,
    pub light_records: usize,
}

/// Flatten a runtime draw frame into the compact per-instance snapshot consumed by
/// `tl_core::VulkanBackend`.
pub fn build_vulkan_render_snapshot<'a>(
    frame_id: u64,
    camera_view_proj: [[f32; 4]; 4],
    draw: &RuntimeDrawFrame,
    transform_scratch: &'a mut Vec<FrameInstanceTransform>,
    material_scratch: &'a mut Vec<FrameMaterialRecord>,
    texture_scratch: &'a mut Vec<FrameTextureRecord>,
    light_scratch: &'a mut Vec<FrameLightRecord>,
) -> (RenderStateSnapshot<'a>, VulkanSnapshotBuildStats) {
    transform_scratch.clear();
    material_scratch.clear();
    texture_scratch.clear();
    light_scratch.clear();
    transform_scratch.reserve(draw.stats.opaque_instances + draw.stats.transparent_instances);
    material_scratch.reserve(draw.stats.opaque_batches + draw.stats.transparent_batches);
    texture_scratch.reserve(draw.stats.opaque_batches + draw.stats.transparent_batches);
    light_scratch.reserve(draw.stats.light_instances);

    let mut stats = VulkanSnapshotBuildStats::default();
    let mut material_map = BTreeMap::new();
    let mut texture_map = BTreeMap::new();

    for batch in &draw.opaque_batches {
        append_batch_instances(
            batch,
            transform_scratch,
            material_scratch,
            texture_scratch,
            &mut material_map,
            &mut texture_map,
            &mut stats,
        );
    }
    for batch in &draw.transparent_batches {
        append_batch_instances(
            batch,
            transform_scratch,
            material_scratch,
            texture_scratch,
            &mut material_map,
            &mut texture_map,
            &mut stats,
        );
    }
    for light in &draw.lights {
        light_scratch.push(pack_light(light));
    }

    stats.total_instances = transform_scratch.len();
    stats.material_records = material_scratch.len();
    stats.texture_records = texture_scratch.len();
    stats.light_records = light_scratch.len();
    (
        RenderStateSnapshot {
            frame_id,
            camera_view_proj,
            transforms: transform_scratch.as_slice(),
            materials: material_scratch.as_slice(),
            textures: texture_scratch.as_slice(),
            lights: light_scratch.as_slice(),
        },
        stats,
    )
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct MaterialDedupKey {
    material_params_bits: [u32; 4],
    emissive_rgb_bits: [u32; 3],
    shading_code: u32,
    texture_index: u32,
    flags: u32,
}

fn append_batch_instances(
    batch: &DrawBatch3d,
    transform_scratch: &mut Vec<FrameInstanceTransform>,
    material_scratch: &mut Vec<FrameMaterialRecord>,
    texture_scratch: &mut Vec<FrameTextureRecord>,
    material_map: &mut BTreeMap<MaterialDedupKey, u32>,
    texture_map: &mut BTreeMap<u32, u32>,
    stats: &mut VulkanSnapshotBuildStats,
) {
    for instance in &batch.instances {
        let texture_index = *texture_map
            .entry(instance.texture_index)
            .or_insert_with(|| {
                let next_index = texture_scratch.len() as u32;
                texture_scratch.push(FrameTextureRecord {
                    texture_slot: instance.texture_index,
                    sampler_code: 0,
                    flags: 0,
                    _padding: 0,
                });
                next_index
            });
        let material_key = instance_material_key(batch, instance);
        let material_index = *material_map.entry(material_key).or_insert_with(|| {
            let next_index = material_scratch.len() as u32;
            material_scratch.push(instance_material_record(batch, instance, texture_index));
            next_index
        });

        let mut flags = 0_u32;
        if matches!(batch.lane, DrawLane::Transparent) {
            flags |= FLAG_TRANSPARENT;
            stats.transparent_instances += 1;
        } else {
            stats.opaque_instances += 1;
        }
        if batch.key.primitive_code >= 2 {
            flags |= FLAG_MESH;
            stats.mesh_instances += 1;
        }

        transform_scratch.push(FrameInstanceTransform {
            model: instance.model_cols,
            color_rgba: instance.base_color_rgba,
            material_index,
            texture_index,
            flags,
            _padding: 0,
        });
    }
}

fn instance_material_key(
    batch: &DrawBatch3d,
    instance: &crate::draw_path::DrawInstance3d,
) -> MaterialDedupKey {
    let flags = if batch.key.shading_code == 1 {
        MATERIAL_FLAG_UNLIT
    } else {
        0
    };
    MaterialDedupKey {
        material_params_bits: instance.material_params.map(f32::to_bits),
        emissive_rgb_bits: instance.emissive_rgb.map(f32::to_bits),
        shading_code: batch.key.shading_code as u32,
        texture_index: instance.texture_index,
        flags,
    }
}

fn instance_material_record(
    batch: &DrawBatch3d,
    instance: &crate::draw_path::DrawInstance3d,
    texture_index: u32,
) -> FrameMaterialRecord {
    FrameMaterialRecord {
        material_params: instance.material_params,
        emissive_rgb: instance.emissive_rgb,
        shading_code: batch.key.shading_code as u32,
        texture_index,
        flags: if batch.key.shading_code == 1 {
            MATERIAL_FLAG_UNLIT
        } else {
            0
        },
        _padding: [0; 2],
    }
}

fn pack_light(light: &SceneLight) -> FrameLightRecord {
    let (position_kind_w, inner_cos, outer_cos) = match light.kind {
        SceneLightKind::Point => (0.0, 1.0, 0.0),
        SceneLightKind::Spot => (
            1.0,
            light.inner_cone_deg.to_radians().cos(),
            light.outer_cone_deg.to_radians().cos(),
        ),
    };
    FrameLightRecord {
        position_kind: [
            light.position[0],
            light.position[1],
            light.position[2],
            position_kind_w,
        ],
        direction_inner: [
            light.direction[0],
            light.direction[1],
            light.direction[2],
            inner_cos,
        ],
        color_intensity: [
            light.color[0],
            light.color[1],
            light.color[2],
            light.intensity,
        ],
        params: [
            light.range,
            outer_cos,
            light.softness,
            light.specular_strength,
        ],
        shadow: [if light.casts_shadow { 1.0 } else { 0.0 }, -1.0, 0.0, 0.0],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::draw_path::{DrawBatch3d, DrawBatchKey, DrawInstance3d};
    use crate::scene::RuntimeSceneMode;

    #[test]
    fn build_snapshot_preserves_order_and_flags() {
        let draw = RuntimeDrawFrame {
            mode: RuntimeSceneMode::Spatial3d,
            view_2d: None,
            opaque_batches: vec![DrawBatch3d {
                lane: DrawLane::Opaque,
                key: DrawBatchKey {
                    primitive_code: 0,
                    shading_code: 0,
                    shadow_flags: 0,
                },
                instances: vec![DrawInstance3d {
                    instance_id: 1,
                    model_cols: [[1.0, 0.0, 0.0, 0.0]; 4],
                    base_color_rgba: [1.0, 0.0, 0.0, 1.0],
                    material_params: [0.0; 4],
                    emissive_rgb: [0.0; 3],
                    texture_index: 2,
                }],
            }],
            transparent_batches: vec![DrawBatch3d {
                lane: DrawLane::Transparent,
                key: DrawBatchKey {
                    primitive_code: 3,
                    shading_code: 0,
                    shadow_flags: 0,
                },
                instances: vec![DrawInstance3d {
                    instance_id: 2,
                    model_cols: [[2.0, 0.0, 0.0, 0.0]; 4],
                    base_color_rgba: [0.0, 1.0, 0.0, 0.5],
                    material_params: [0.0; 4],
                    emissive_rgb: [0.0; 3],
                    texture_index: 2,
                }],
            }],
            sprites: Vec::new(),
            lights: Vec::new(),
            stats: crate::draw_path::DrawFrameStats {
                opaque_instances: 1,
                transparent_instances: 1,
                sprite_instances: 0,
                light_instances: 0,
                opaque_batches: 1,
                transparent_batches: 1,
                total_draw_calls: 2,
            },
        };

        let mut transform_scratch = Vec::new();
        let mut material_scratch = Vec::new();
        let mut texture_scratch = Vec::new();
        let mut light_scratch = Vec::new();
        let (snapshot, stats) = build_vulkan_render_snapshot(
            42,
            [[1.0, 0.0, 0.0, 0.0]; 4],
            &draw,
            &mut transform_scratch,
            &mut material_scratch,
            &mut texture_scratch,
            &mut light_scratch,
        );
        assert_eq!(snapshot.frame_id, 42);
        assert_eq!(snapshot.camera_view_proj[0], [1.0, 0.0, 0.0, 0.0]);
        assert_eq!(snapshot.transforms.len(), 2);
        assert_eq!(snapshot.materials.len(), 1);
        assert_eq!(snapshot.textures.len(), 1);
        assert_eq!(snapshot.lights.len(), 0);
        assert_eq!(stats.total_instances, 2);
        assert_eq!(stats.opaque_instances, 1);
        assert_eq!(stats.transparent_instances, 1);
        assert_eq!(stats.mesh_instances, 1);
        assert_eq!(stats.material_records, 1);
        assert_eq!(stats.texture_records, 1);
        assert_eq!(stats.light_records, 0);
        assert_eq!(snapshot.transforms[0].material_index, 0);
        assert_eq!(snapshot.transforms[1].material_index, 0);
        assert_eq!(snapshot.transforms[0].texture_index, 0);
        assert_eq!(snapshot.materials[0].texture_index, 0);
        assert_eq!(snapshot.textures[0].texture_slot, 2);
        assert_ne!(snapshot.transforms[1].flags & FLAG_TRANSPARENT, 0);
        assert_ne!(snapshot.transforms[1].flags & FLAG_MESH, 0);
    }

    #[test]
    fn keeps_distinct_material_records_for_distinct_texture_slots() {
        let draw = RuntimeDrawFrame {
            mode: RuntimeSceneMode::Spatial3d,
            view_2d: None,
            opaque_batches: vec![DrawBatch3d {
                lane: DrawLane::Opaque,
                key: DrawBatchKey {
                    primitive_code: 0,
                    shading_code: 0,
                    shadow_flags: 0,
                },
                instances: vec![
                    DrawInstance3d {
                        instance_id: 1,
                        model_cols: [[1.0, 0.0, 0.0, 0.0]; 4],
                        base_color_rgba: [1.0, 1.0, 1.0, 1.0],
                        material_params: [0.5, 0.2, 0.0, 0.0],
                        emissive_rgb: [0.0; 3],
                        texture_index: 1,
                    },
                    DrawInstance3d {
                        instance_id: 2,
                        model_cols: [[1.0, 0.0, 0.0, 0.0]; 4],
                        base_color_rgba: [1.0, 1.0, 1.0, 1.0],
                        material_params: [0.5, 0.2, 0.0, 0.0],
                        emissive_rgb: [0.0; 3],
                        texture_index: 3,
                    },
                ],
            }],
            transparent_batches: Vec::new(),
            sprites: Vec::new(),
            lights: Vec::new(),
            stats: crate::draw_path::DrawFrameStats {
                opaque_instances: 2,
                transparent_instances: 0,
                sprite_instances: 0,
                light_instances: 0,
                opaque_batches: 1,
                transparent_batches: 0,
                total_draw_calls: 1,
            },
        };

        let mut transform_scratch = Vec::new();
        let mut material_scratch = Vec::new();
        let mut texture_scratch = Vec::new();
        let mut light_scratch = Vec::new();
        let (snapshot, stats) = build_vulkan_render_snapshot(
            7,
            [[1.0, 0.0, 0.0, 0.0]; 4],
            &draw,
            &mut transform_scratch,
            &mut material_scratch,
            &mut texture_scratch,
            &mut light_scratch,
        );

        assert_eq!(stats.material_records, 2);
        assert_eq!(stats.texture_records, 2);
        assert_eq!(snapshot.materials.len(), 2);
        assert_eq!(snapshot.textures.len(), 2);
        assert_eq!(snapshot.transforms[0].material_index, 0);
        assert_eq!(snapshot.transforms[1].material_index, 1);
        assert_eq!(snapshot.materials[0].texture_index, 0);
        assert_eq!(snapshot.materials[1].texture_index, 1);
        assert_eq!(snapshot.textures[0].texture_slot, 1);
        assert_eq!(snapshot.textures[1].texture_slot, 3);
    }
}
