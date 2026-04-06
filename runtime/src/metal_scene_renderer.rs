//! Runtime-owned adapter that feeds `RuntimeDrawFrame` into `tl_core::MetalBackend`.
//!
//! The v0.5.0 MVP keeps this path intentionally lean:
//! - build a compact `RenderStateSnapshot` from `RuntimeDrawFrame`
//! - submit `Render N` through `tl_core::MetalBackend`
//! - expose deterministic telemetry + fail-soft status surfaces used by TLApp

#![cfg(target_os = "macos")]

use std::error::Error;
use std::fmt::{Display, Formatter};
use std::mem::size_of;
use std::path::Path;
use std::sync::Arc;

use nalgebra::{Isometry3, Matrix4, Perspective3, Point3, Vector3};
use tl_core::{
    FrameInstanceTransform, FrameLightRecord, FrameMaterialRecord, FrameTextureRecord,
    MetalBackend, MetalBackendConfig, MetalBackendError, MetalFrameExecutionTelemetry,
    RenderStateSnapshot,
};
use wgpu::Backend;
use winit::window::Window;

use crate::draw_path::RuntimeDrawFrame;
use crate::scene::RayTracingMode;
use crate::upscaler::{resolve_fsr_status, FsrConfig, FsrStatus};
use crate::vulkan_snapshot::{build_vulkan_render_snapshot, VulkanSnapshotBuildStats};
use crate::wgpu_scene_renderer::{SceneRayTracingStatus, WgpuSceneRendererUploadStats};

const RT_DYNAMIC_CAP: u32 = 16_384;

/// Runtime-facing configuration for the Metal scene renderer adapter.
#[derive(Debug, Clone)]
pub struct MetalSceneRendererConfig {
    /// Low-level raw Metal backend configuration.
    pub backend: MetalBackendConfig,
}

impl Default for MetalSceneRendererConfig {
    fn default() -> Self {
        Self {
            backend: MetalBackendConfig::default(),
        }
    }
}

/// Summary of one runtime draw-frame submission through the raw Metal backend.
#[derive(Debug, Clone)]
pub struct MetalSceneRendererFrameResult {
    pub snapshot: VulkanSnapshotBuildStats,
    pub execution: MetalFrameExecutionTelemetry,
    pub estimated_snapshot_bytes: u64,
}

/// Errors produced by the runtime-side Metal scene renderer adapter.
#[derive(Debug)]
pub enum MetalSceneRendererError {
    Backend(MetalBackendError),
}

impl Display for MetalSceneRendererError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Backend(err) => write!(f, "{err}"),
        }
    }
}

impl Error for MetalSceneRendererError {}

impl From<MetalBackendError> for MetalSceneRendererError {
    fn from(value: MetalBackendError) -> Self {
        Self::Backend(value)
    }
}

/// Runtime-owned raw Metal scene renderer.
pub struct MetalSceneRenderer {
    backend: MetalBackend,
    transform_snapshot_scratch: Vec<FrameInstanceTransform>,
    material_snapshot_scratch: Vec<FrameMaterialRecord>,
    texture_snapshot_scratch: Vec<FrameTextureRecord>,
    light_snapshot_scratch: Vec<FrameLightRecord>,
    camera_eye: [f32; 3],
    camera_target: [f32; 3],
    surface_width: u32,
    surface_height: u32,
    force_full_fbx_sphere: bool,
    msaa_sample_count: u32,
    fsr_config: FsrConfig,
    fsr_status: FsrStatus,
    ray_tracing_status: SceneRayTracingStatus,
    last_upload_stats: WgpuSceneRendererUploadStats,
    last_frame_result: Option<MetalSceneRendererFrameResult>,
}

impl MetalSceneRenderer {
    /// Create the runtime Metal scene renderer around the canonical `tl_core::MetalBackend`.
    pub fn new(
        window: Arc<Window>,
        config: MetalSceneRendererConfig,
    ) -> Result<Self, MetalSceneRendererError> {
        let scratch_capacity = config.backend.max_instances.max(1);
        let backend = MetalBackend::new(window, config.backend)?;
        Ok(Self {
            backend,
            transform_snapshot_scratch: Vec::with_capacity(scratch_capacity),
            material_snapshot_scratch: Vec::with_capacity(scratch_capacity.max(256)),
            texture_snapshot_scratch: Vec::with_capacity(scratch_capacity.max(128)),
            light_snapshot_scratch: Vec::with_capacity(32),
            camera_eye: [0.0, 12.0, 36.0],
            camera_target: [0.0, 0.0, 0.0],
            surface_width: 1,
            surface_height: 1,
            force_full_fbx_sphere: false,
            msaa_sample_count: 1,
            fsr_config: FsrConfig::default(),
            fsr_status: resolve_fsr_status(FsrConfig::default(), Backend::Metal),
            ray_tracing_status: resolve_rt_status(RayTracingMode::Auto, false),
            last_upload_stats: WgpuSceneRendererUploadStats::default(),
            last_frame_result: None,
        })
    }

    /// Resize backend-dependent resources.
    pub fn resize(
        &mut self,
        new_size: winit::dpi::PhysicalSize<u32>,
    ) -> Result<(), MetalSceneRendererError> {
        self.surface_width = new_size.width.max(1);
        self.surface_height = new_size.height.max(1);
        self.backend.resize(new_size)?;
        Ok(())
    }

    /// Access the underlying low-level backend.
    pub fn backend(&self) -> &MetalBackend {
        &self.backend
    }

    /// Most recent runtime Metal submission result.
    pub fn last_frame_result(&self) -> Option<&MetalSceneRendererFrameResult> {
        self.last_frame_result.as_ref()
    }

    /// Most recent upload-like stats projected into current runtime telemetry shape.
    pub fn last_upload_stats(&self) -> WgpuSceneRendererUploadStats {
        self.last_upload_stats
    }

    /// Track active camera view for helper projections and HUD/light placement.
    pub fn set_camera_view(&mut self, width: u32, height: u32, eye: [f32; 3], target: [f32; 3]) {
        self.surface_width = width.max(1);
        self.surface_height = height.max(1);
        self.camera_eye = eye;
        self.camera_target = target;
    }

    /// Current camera eye in world space.
    pub fn camera_eye(&self) -> [f32; 3] {
        self.camera_eye
    }

    /// Project a world-space point to NDC using cached runtime camera.
    pub fn world_to_ndc(&self, world_pos: [f32; 3]) -> Option<[f32; 3]> {
        let aspect = (self.surface_width.max(1) as f32) / (self.surface_height.max(1) as f32);
        let proj = Perspective3::new(aspect.max(0.1), 60f32.to_radians(), 0.1, 500.0);
        let view = Isometry3::look_at_rh(
            &Point3::new(self.camera_eye[0], self.camera_eye[1], self.camera_eye[2]),
            &Point3::new(
                self.camera_target[0],
                self.camera_target[1],
                self.camera_target[2],
            ),
            &Vector3::new(0.0, 1.0, 0.0),
        );
        let view_proj: Matrix4<f32> = proj.to_homogeneous() * view.to_homogeneous();
        let clip =
            view_proj * Point3::new(world_pos[0], world_pos[1], world_pos[2]).to_homogeneous();
        if clip.w <= 0.0 {
            return None;
        }
        Some([clip.x / clip.w, clip.y / clip.w, clip.z / clip.w])
    }

    /// Convert a world-space radius into perspective-correct NDC half-size.
    pub fn world_radius_to_ndc_half_size(&self, world_radius: f32, clip_w: f32) -> f32 {
        let fov_y = 60f32.to_radians();
        let focal = 1.0 / (fov_y * 0.5).tan();
        (world_radius / clip_w.max(0.01)) * focal
    }

    /// Runtime-side FSR policy update.
    pub fn set_fsr_config(&mut self, config: FsrConfig) {
        self.fsr_config = config;
        self.fsr_status = resolve_fsr_status(config, Backend::Metal);
    }

    /// Current effective FSR status.
    pub fn fsr_status(&self) -> FsrStatus {
        self.fsr_status.clone()
    }

    /// Runtime-side RT policy update.
    pub fn set_ray_tracing_mode(&mut self, mode: RayTracingMode) {
        self.ray_tracing_status = resolve_rt_status(mode, false);
    }

    /// Current effective RT status.
    pub fn ray_tracing_status(&self) -> SceneRayTracingStatus {
        self.ray_tracing_status.clone()
    }

    /// Current MSAA sample count reported to TLApp controls.
    pub fn msaa_sample_count(&self) -> u32 {
        self.msaa_sample_count
    }

    /// Runtime-side MSAA request tracking.
    pub fn set_msaa_sample_count(&mut self, count: u32) {
        self.msaa_sample_count = match count {
            0 | 1 => 1,
            2 | 3 => 2,
            _ => 4,
        };
    }

    /// Track FBX-forcing intent from `.tlsprite` / runtime patching.
    pub fn set_force_full_fbx_sphere(&mut self, force: bool) {
        self.force_full_fbx_sphere = force;
    }

    /// Current FBX force flag used by runtime path.
    pub fn force_full_fbx_sphere(&self) -> bool {
        self.force_full_fbx_sphere
    }

    /// Placeholder FBX mesh binding hook for the Metal migration path.
    pub fn bind_fbx_mesh_slot_from_path(&mut self, _slot: u8, _path: &Path) -> Result<(), String> {
        Ok(())
    }

    /// Placeholder sprite texture binding hook for the Metal migration path.
    pub fn bind_sprite_texture_slot_from_path(
        &mut self,
        _slot: u16,
        _path: &Path,
    ) -> Result<(), String> {
        Ok(())
    }

    /// Placeholder procedural sphere binding for runtime compatibility.
    pub fn bind_builtin_sphere_mesh_slot(&mut self, _slot: u8, _high_quality: bool) {}

    /// Estimate snapshot payload bytes for telemetry and planning.
    pub fn estimate_snapshot_bytes(draw: &RuntimeDrawFrame) -> u64 {
        let transform_bytes = (draw.stats.opaque_instances + draw.stats.transparent_instances)
            as u64
            * size_of::<FrameInstanceTransform>() as u64;
        let material_bytes = (draw.stats.opaque_batches + draw.stats.transparent_batches) as u64
            * size_of::<FrameMaterialRecord>() as u64;
        let texture_bytes = (draw.stats.opaque_batches + draw.stats.transparent_batches) as u64
            * size_of::<FrameTextureRecord>() as u64;
        let light_bytes = draw.stats.light_instances as u64 * size_of::<FrameLightRecord>() as u64;
        let sprite_bytes = draw.stats.sprite_instances as u64 * 96;
        transform_bytes + material_bytes + texture_bytes + light_bytes + sprite_bytes
    }

    /// Convert runtime draw-frame into a snapshot and submit through the raw backend.
    pub fn render_draw_frame(
        &mut self,
        frame_id: u64,
        draw: &RuntimeDrawFrame,
    ) -> Result<&MetalSceneRendererFrameResult, MetalSceneRendererError> {
        let estimated_snapshot_bytes = Self::estimate_snapshot_bytes(draw);
        let camera_view_proj = self.camera_view_proj();
        let (
            backend,
            transform_snapshot_scratch,
            material_snapshot_scratch,
            texture_snapshot_scratch,
            light_snapshot_scratch,
        ) = (
            &mut self.backend,
            &mut self.transform_snapshot_scratch,
            &mut self.material_snapshot_scratch,
            &mut self.texture_snapshot_scratch,
            &mut self.light_snapshot_scratch,
        );
        let (snapshot, snapshot_stats): (RenderStateSnapshot<'_>, VulkanSnapshotBuildStats) =
            build_vulkan_render_snapshot(
                frame_id,
                camera_view_proj,
                draw,
                transform_snapshot_scratch,
                material_snapshot_scratch,
                texture_snapshot_scratch,
                light_snapshot_scratch,
            );
        let execution = backend.render_n(snapshot)?;

        self.last_upload_stats = WgpuSceneRendererUploadStats {
            opaque_draw_calls: draw.stats.opaque_batches,
            transparent_draw_calls: draw.stats.transparent_batches,
            sprite_draw_calls: usize::from(draw.stats.sprite_instances > 0),
            total_draw_calls: draw.stats.opaque_batches
                + draw.stats.transparent_batches
                + usize::from(draw.stats.sprite_instances > 0),
            instance_3d_count: snapshot_stats.total_instances,
            sprite_count: draw.stats.sprite_instances,
            light_count: draw.stats.light_instances,
            rt_active: self.ray_tracing_status.active,
            rt_dynamic_count: self.ray_tracing_status.rt_dynamic_count,
            fsr_active: self.fsr_status.active,
            fsr_scale: self.fsr_status.render_scale,
        };

        self.last_frame_result = Some(MetalSceneRendererFrameResult {
            snapshot: snapshot_stats,
            execution,
            estimated_snapshot_bytes,
        });

        Ok(self
            .last_frame_result
            .as_ref()
            .expect("frame result should exist after render submission"))
    }

    fn camera_view_proj(&self) -> [[f32; 4]; 4] {
        let aspect = (self.surface_width.max(1) as f32) / (self.surface_height.max(1) as f32);
        let proj = Perspective3::new(aspect.max(0.1), 60f32.to_radians(), 0.1, 500.0);
        let view = Isometry3::look_at_rh(
            &Point3::new(self.camera_eye[0], self.camera_eye[1], self.camera_eye[2]),
            &Point3::new(
                self.camera_target[0],
                self.camera_target[1],
                self.camera_target[2],
            ),
            &Vector3::new(0.0, 1.0, 0.0),
        );
        let matrix: Matrix4<f32> = proj.to_homogeneous() * view.to_homogeneous();
        [
            [
                matrix[(0, 0)],
                matrix[(1, 0)],
                matrix[(2, 0)],
                matrix[(3, 0)],
            ],
            [
                matrix[(0, 1)],
                matrix[(1, 1)],
                matrix[(2, 1)],
                matrix[(3, 1)],
            ],
            [
                matrix[(0, 2)],
                matrix[(1, 2)],
                matrix[(2, 2)],
                matrix[(3, 2)],
            ],
            [
                matrix[(0, 3)],
                matrix[(1, 3)],
                matrix[(2, 3)],
                matrix[(3, 3)],
            ],
        ]
    }
}

fn resolve_rt_status(mode: RayTracingMode, supports_ray_query: bool) -> SceneRayTracingStatus {
    match mode {
        RayTracingMode::Off => SceneRayTracingStatus {
            mode,
            active: false,
            fallback_reason: "rt disabled by mode=off".to_string(),
            rt_dynamic_count: 0,
            rt_dynamic_cap: RT_DYNAMIC_CAP,
            supports_ray_query,
        },
        RayTracingMode::Auto => SceneRayTracingStatus {
            mode,
            active: supports_ray_query,
            fallback_reason: if supports_ray_query {
                String::new()
            } else {
                "ray query unsupported on current Metal runtime path; auto fallback to forward path"
                    .to_string()
            },
            rt_dynamic_count: 0,
            rt_dynamic_cap: RT_DYNAMIC_CAP,
            supports_ray_query,
        },
        RayTracingMode::On => SceneRayTracingStatus {
            mode,
            active: supports_ray_query,
            fallback_reason: if supports_ray_query {
                String::new()
            } else {
                "mode=on requested but Metal runtime RT path is not wired yet; fail-soft fallback to forward"
                    .to_string()
            },
            rt_dynamic_count: 0,
            rt_dynamic_cap: RT_DYNAMIC_CAP,
            supports_ray_query,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::draw_path::{DrawBatch3d, DrawBatchKey, DrawFrameStats, DrawInstance3d, DrawLane};
    use crate::scene::RuntimeSceneMode;

    fn sample_draw_frame(
        opaque_instances: usize,
        transparent_instances: usize,
        lights: usize,
        sprites: usize,
    ) -> RuntimeDrawFrame {
        let opaque_batch = DrawBatch3d {
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
                    material_params: [0.0; 4],
                    emissive_rgb: [0.0; 3],
                    texture_index: 0,
                };
                opaque_instances
            ],
        };

        let transparent_batch = DrawBatch3d {
            lane: DrawLane::Transparent,
            key: DrawBatchKey {
                primitive_code: 0,
                shading_code: 0,
                shadow_flags: 0,
            },
            instances: vec![
                DrawInstance3d {
                    instance_id: 2,
                    model_cols: [[1.0, 0.0, 0.0, 0.0]; 4],
                    base_color_rgba: [1.0, 1.0, 1.0, 0.5],
                    material_params: [0.0; 4],
                    emissive_rgb: [0.0; 3],
                    texture_index: 1,
                };
                transparent_instances
            ],
        };

        RuntimeDrawFrame {
            mode: RuntimeSceneMode::Spatial3d,
            view_2d: None,
            opaque_batches: if opaque_instances > 0 {
                vec![opaque_batch]
            } else {
                Vec::new()
            },
            transparent_batches: if transparent_instances > 0 {
                vec![transparent_batch]
            } else {
                Vec::new()
            },
            sprites: Vec::with_capacity(sprites),
            lights: Vec::with_capacity(lights),
            stats: DrawFrameStats {
                opaque_instances,
                transparent_instances,
                sprite_instances: sprites,
                light_instances: lights,
                opaque_batches: usize::from(opaque_instances > 0),
                transparent_batches: usize::from(transparent_instances > 0),
                total_draw_calls: usize::from(opaque_instances > 0)
                    + usize::from(transparent_instances > 0),
            },
        }
    }

    #[test]
    fn estimates_snapshot_bytes_from_frame_density() {
        let draw = sample_draw_frame(1024, 512, 4, 64);
        let estimated = MetalSceneRenderer::estimate_snapshot_bytes(&draw);
        let expected_transform_bytes =
            (1024_u64 + 512_u64) * size_of::<FrameInstanceTransform>() as u64;
        let expected_material_bytes = 2_u64 * size_of::<FrameMaterialRecord>() as u64;
        let expected_texture_bytes = 2_u64 * size_of::<FrameTextureRecord>() as u64;
        let expected_light_bytes = 4_u64 * size_of::<FrameLightRecord>() as u64;
        let expected_sprite_bytes = 64_u64 * 96;
        assert_eq!(
            estimated,
            expected_transform_bytes
                + expected_material_bytes
                + expected_texture_bytes
                + expected_light_bytes
                + expected_sprite_bytes
        );
    }
}
