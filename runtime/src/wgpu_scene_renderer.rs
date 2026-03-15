//! Minimal `wgpu` scene renderer for runtime draw frames.
//!
//! This backend consumes `RuntimeDrawFrame` and issues real render-pass draw calls for:
//! - opaque 3D batches
//! - transparent 3D batches
//! - sprite overlays (including telemetry HUD sprites)

use std::{collections::BTreeMap, fs, io::Cursor, path::Path};

use fbx::Property as FbxProperty;
use nalgebra::{Isometry3, Matrix4, Perspective3, Point3, Vector3};
use wgpu::util::DeviceExt;

use crate::draw_path::{DrawLane, RuntimeDrawFrame};
use crate::scene::{RayTracingMode, SceneLight, SceneLightKind, SpriteKind, MAX_SCENE_LIGHTS};
use crate::upscaler::{resolve_fsr_status, FsrConfig, FsrStatus};

const SPRITE_ATLAS_GRID_DIM: u32 = 4;
const SPRITE_ATLAS_TILE_SIZE: u32 = 64;
const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
const DEFAULT_SPHERE_FBX_BYTES: &[u8] = include_bytes!("../../docs/demos/sphere.fbx");
// Hysteresis avoids rapid mesh-mode flapping when instance counts hover around thresholds.
const SPHERE_LOD_ENABLE_THRESHOLD: usize = 2_500;
const SPHERE_LOD_DISABLE_THRESHOLD: usize = 1_800;
const RT_DYNAMIC_CAP: u32 = 1_024;

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
struct GpuVertex3d {
    position: [f32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
struct GpuInstance3d {
    model_col0: [f32; 4],
    model_col1: [f32; 4],
    model_col2: [f32; 4],
    model_col3: [f32; 4],
    base_color: [f32; 4],
    material_params: [f32; 4],
    emissive: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
struct GpuCameraUniform {
    view_proj: [f32; 16],
    camera_eye: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod, Default)]
struct GpuLight {
    position_kind: [f32; 4],
    direction_inner: [f32; 4],
    color_intensity: [f32; 4],
    params: [f32; 4], // range, outer_cos, softness, specular_strength
    shadow: [f32; 4], // casts_shadow, _, _, _
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
struct GpuLightingUniform {
    light_count: u32,
    rt_mode: u32,
    rt_active: u32,
    rt_dynamic_count: u32,
    rt_dynamic_cap: u32,
    _pad: [u32; 3],
}

impl Default for GpuLightingUniform {
    fn default() -> Self {
        Self {
            light_count: 0,
            rt_mode: 1,
            rt_active: 0,
            rt_dynamic_count: 0,
            rt_dynamic_cap: RT_DYNAMIC_CAP,
            _pad: [0; 3],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
struct GpuSpriteVertex {
    local_pos: [f32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
struct GpuSpriteInstance {
    translate_size: [f32; 4], // x, y, w, h
    rot_z: [f32; 4],          // rotation_rad, z, _, _
    color: [f32; 4],
    atlas_rect: [f32; 4],  // u0, v0, u1, v1
    kind_params: [f32; 4], // kind_code, texture_slot, atlas_flags, reserved
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct GpuBatchRange {
    lane: DrawLane,
    primitive_code: u8,
    start: u32,
    count: u32,
}

#[derive(Default)]
struct UploadPlan {
    instances_3d: Vec<GpuInstance3d>,
    ranges: Vec<GpuBatchRange>,
    sprites: Vec<GpuSpriteInstance>,
}

/// Upload summary for one frame.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct WgpuSceneRendererUploadStats {
    pub opaque_draw_calls: usize,
    pub transparent_draw_calls: usize,
    pub sprite_draw_calls: usize,
    pub total_draw_calls: usize,
    pub instance_3d_count: usize,
    pub sprite_count: usize,
    pub light_count: usize,
    pub rt_active: bool,
    pub rt_dynamic_count: u32,
    pub fsr_active: bool,
    pub fsr_scale: f32,
}

/// Runtime RT status snapshot from scene renderer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SceneRayTracingStatus {
    pub mode: RayTracingMode,
    pub active: bool,
    pub fallback_reason: String,
    pub rt_dynamic_count: u32,
    pub rt_dynamic_cap: u32,
    pub supports_ray_query: bool,
}

struct GpuMesh {
    vertex: wgpu::Buffer,
    index: wgpu::Buffer,
    index_count: u32,
    index_format: wgpu::IndexFormat,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum SphereLodMode {
    #[default]
    High,
    Low,
}

/// Runtime `wgpu` renderer for `RuntimeDrawFrame`.
pub struct WgpuSceneRenderer {
    adapter_backend: wgpu::Backend,
    camera_buffer: wgpu::Buffer,
    scene_bind_group: wgpu::BindGroup,
    light_buffer: wgpu::Buffer,
    lighting_buffer: wgpu::Buffer,
    pipeline_opaque: wgpu::RenderPipeline,
    pipeline_transparent: wgpu::RenderPipeline,
    pipeline_sprite: wgpu::RenderPipeline,
    _depth_texture: wgpu::Texture,
    depth_view: wgpu::TextureView,
    _sprite_atlas_texture: wgpu::Texture,
    sprite_atlas_bind_group: wgpu::BindGroup,
    box_mesh: GpuMesh,
    sphere_mesh_high: GpuMesh,
    sphere_mesh_low: GpuMesh,
    sphere_lod_mode: SphereLodMode,
    sprite_vertex_buffer: wgpu::Buffer,
    instance_3d_buffer: wgpu::Buffer,
    instance_3d_capacity_bytes: usize,
    sprite_instance_buffer: wgpu::Buffer,
    sprite_instance_capacity_bytes: usize,
    ranges: Vec<GpuBatchRange>,
    sprite_count: u32,
    light_count: u32,
    force_full_fbx_sphere: bool,
    camera_eye: [f32; 3],
    camera_target: [f32; 3],
    custom_mesh_slots: BTreeMap<u8, GpuMesh>,
    ray_tracing_status: SceneRayTracingStatus,
    fsr_config: FsrConfig,
    fsr_status: FsrStatus,
}

impl WgpuSceneRenderer {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        color_format: wgpu::TextureFormat,
        surface_width: u32,
        surface_height: u32,
        adapter_backend: wgpu::Backend,
    ) -> Self {
        let camera_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("runtime-scene-camera-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let camera_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("runtime-scene-camera-buffer"),
            size: std::mem::size_of::<GpuCameraUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let light_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("runtime-scene-light-buffer"),
            size: (std::mem::size_of::<GpuLight>() * MAX_SCENE_LIGHTS.max(1)) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let lighting_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("runtime-scene-lighting-buffer"),
            size: std::mem::size_of::<GpuLightingUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let scene_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("runtime-scene-bg"),
            layout: &camera_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: light_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: lighting_buffer.as_entire_binding(),
                },
            ],
        });

        let shader_3d = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("runtime-scene-3d-shader"),
            source: wgpu::ShaderSource::Wgsl(SCENE_3D_WGSL.into()),
        });
        let shader_sprite = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("runtime-scene-sprite-shader"),
            source: wgpu::ShaderSource::Wgsl(SCENE_SPRITE_WGSL.into()),
        });

        let sprite_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("runtime-scene-sprite-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let layout_3d = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("runtime-scene-3d-layout"),
            bind_group_layouts: &[&camera_bgl],
            immediate_size: 0,
        });
        let layout_sprite = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("runtime-scene-sprite-layout"),
            bind_group_layouts: &[&sprite_bgl, &camera_bgl],
            immediate_size: 0,
        });

        let pipeline_opaque = create_3d_pipeline(
            device,
            &layout_3d,
            &shader_3d,
            color_format,
            Some(wgpu::BlendState::REPLACE),
            true,
            Some(wgpu::Face::Back),
            "runtime-scene-opaque-pipeline",
        );
        let pipeline_transparent = create_3d_pipeline(
            device,
            &layout_3d,
            &shader_3d,
            color_format,
            Some(wgpu::BlendState::ALPHA_BLENDING),
            false,
            None,
            "runtime-scene-transparent-pipeline",
        );
        let pipeline_sprite =
            create_sprite_pipeline(device, &layout_sprite, &shader_sprite, color_format);
        let (depth_texture, depth_view) = create_depth_resources(
            device,
            surface_width.max(1),
            surface_height.max(1),
            "runtime-scene-depth",
        );
        let (sprite_atlas_texture, sprite_atlas_bind_group) =
            create_default_sprite_atlas_resources(device, queue, &sprite_bgl);

        let box_mesh = create_box_mesh(device);
        let sphere_mesh_high = create_sphere_mesh(device);
        let sphere_mesh_low = create_icosa_sphere_mesh(device);
        let sprite_vertex_buffer = create_sprite_quad_vertex_buffer(device);
        let instance_3d_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("runtime-scene-3d-instance-buffer"),
            size: std::mem::size_of::<GpuInstance3d>() as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let sprite_instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("runtime-scene-sprite-instance-buffer"),
            size: std::mem::size_of::<GpuSpriteInstance>() as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let renderer = Self {
            adapter_backend,
            camera_buffer,
            scene_bind_group,
            light_buffer,
            lighting_buffer,
            pipeline_opaque,
            pipeline_transparent,
            pipeline_sprite,
            _depth_texture: depth_texture,
            depth_view,
            _sprite_atlas_texture: sprite_atlas_texture,
            sprite_atlas_bind_group,
            box_mesh,
            sphere_mesh_high,
            sphere_mesh_low,
            sphere_lod_mode: SphereLodMode::High,
            sprite_vertex_buffer,
            instance_3d_buffer,
            instance_3d_capacity_bytes: std::mem::size_of::<GpuInstance3d>(),
            sprite_instance_buffer,
            sprite_instance_capacity_bytes: std::mem::size_of::<GpuSpriteInstance>(),
            ranges: Vec::new(),
            sprite_count: 0,
            light_count: 0,
            force_full_fbx_sphere: false,
            camera_eye: [0.0, 12.0, 36.0],
            camera_target: [0.0, 0.0, 0.0],
            custom_mesh_slots: BTreeMap::new(),
            ray_tracing_status: resolve_rt_status(
                RayTracingMode::Auto,
                supports_ray_query(device, adapter_backend),
            ),
            fsr_config: FsrConfig::default(),
            fsr_status: resolve_fsr_status(FsrConfig::default(), adapter_backend),
        };
        renderer.write_camera_uniform(queue, surface_width.max(1), surface_height.max(1));
        renderer.write_lighting_uniform(queue, 0);
        renderer
    }

    pub fn resize(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, width: u32, height: u32) {
        self.write_camera_uniform(queue, width.max(1), height.max(1));
        let (depth_texture, depth_view) =
            create_depth_resources(device, width.max(1), height.max(1), "runtime-scene-depth");
        self._depth_texture = depth_texture;
        self.depth_view = depth_view;
    }

    /// Overrides the active camera transform used by scene rendering.
    pub fn set_camera_view(
        &mut self,
        queue: &wgpu::Queue,
        width: u32,
        height: u32,
        eye: [f32; 3],
        target: [f32; 3],
    ) {
        self.camera_eye = eye;
        self.camera_target = target;
        self.write_camera_uniform(queue, width.max(1), height.max(1));
    }

    /// Bind an FBX mesh into a runtime mesh slot.
    ///
    /// Slots are consumed by `ScenePrimitive3d::Mesh { slot }` and mapped in draw batching.
    pub fn bind_fbx_mesh_slot_from_bytes(
        &mut self,
        device: &wgpu::Device,
        slot: u8,
        bytes: &[u8],
    ) -> Result<(), String> {
        let mesh_data = parse_first_mesh_from_fbx(bytes)?;
        let mesh = create_mesh_u32(
            device,
            &format!("runtime-scene-fbx-slot-{slot}"),
            &mesh_data.vertices,
            &mesh_data.indices,
        );
        self.custom_mesh_slots.insert(slot, mesh);
        Ok(())
    }

    /// Bind an FBX mesh from disk into a runtime mesh slot.
    pub fn bind_fbx_mesh_slot_from_path(
        &mut self,
        device: &wgpu::Device,
        slot: u8,
        path: &Path,
    ) -> Result<(), String> {
        let bytes = fs::read(path)
            .map_err(|err| format!("failed to read FBX '{}': {err}", path.display()))?;
        self.bind_fbx_mesh_slot_from_bytes(device, slot, &bytes)
    }

    /// Bind a built-in sphere mesh (high FBX or low icosa) into a runtime slot.
    pub fn bind_builtin_sphere_mesh_slot(
        &mut self,
        device: &wgpu::Device,
        slot: u8,
        high_quality: bool,
    ) {
        let mesh = if high_quality {
            create_sphere_mesh(device)
        } else {
            create_icosa_sphere_mesh(device)
        };
        self.custom_mesh_slots.insert(slot, mesh);
    }

    pub fn upload_draw_frame(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        draw: &RuntimeDrawFrame,
    ) -> WgpuSceneRendererUploadStats {
        let plan = build_upload_plan(draw);
        let required_3d_bytes = plan.instances_3d.len() * std::mem::size_of::<GpuInstance3d>();
        let required_sprite_bytes = plan.sprites.len() * std::mem::size_of::<GpuSpriteInstance>();

        ensure_buffer_capacity(
            device,
            &mut self.instance_3d_buffer,
            &mut self.instance_3d_capacity_bytes,
            required_3d_bytes.max(std::mem::size_of::<GpuInstance3d>()),
            wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            "runtime-scene-3d-instance-buffer",
        );
        ensure_buffer_capacity(
            device,
            &mut self.sprite_instance_buffer,
            &mut self.sprite_instance_capacity_bytes,
            required_sprite_bytes.max(std::mem::size_of::<GpuSpriteInstance>()),
            wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            "runtime-scene-sprite-instance-buffer",
        );

        if !plan.instances_3d.is_empty() {
            queue.write_buffer(
                &self.instance_3d_buffer,
                0,
                bytemuck::cast_slice(plan.instances_3d.as_slice()),
            );
        }
        if !plan.sprites.is_empty() {
            queue.write_buffer(
                &self.sprite_instance_buffer,
                0,
                bytemuck::cast_slice(plan.sprites.as_slice()),
            );
        }

        self.ranges = plan.ranges;
        self.sprite_count = plan.sprites.len() as u32;
        self.upload_lights(
            queue,
            draw.lights.as_slice(),
            draw.stats.opaque_instances + draw.stats.transparent_instances,
        );
        if self.force_full_fbx_sphere {
            self.sphere_lod_mode = SphereLodMode::High;
        } else {
            self.sphere_lod_mode = select_sphere_lod_mode(
                self.sphere_lod_mode,
                count_sphere_instances(self.ranges.as_slice()),
            );
        }

        let opaque_draw_calls = self
            .ranges
            .iter()
            .filter(|r| r.lane == DrawLane::Opaque)
            .count();
        let transparent_draw_calls = self
            .ranges
            .iter()
            .filter(|r| r.lane == DrawLane::Transparent)
            .count();

        WgpuSceneRendererUploadStats {
            opaque_draw_calls,
            transparent_draw_calls,
            sprite_draw_calls: usize::from(self.sprite_count > 0),
            total_draw_calls: opaque_draw_calls
                + transparent_draw_calls
                + usize::from(self.sprite_count > 0),
            instance_3d_count: draw.stats.opaque_instances + draw.stats.transparent_instances,
            sprite_count: draw.stats.sprite_instances,
            light_count: self.light_count as usize,
            rt_active: self.ray_tracing_status.active,
            rt_dynamic_count: self.ray_tracing_status.rt_dynamic_count,
            fsr_active: self.fsr_status.active,
            fsr_scale: self.fsr_status.render_scale,
        }
    }

    /// Force all sphere primitives to render with the full FBX mesh.
    ///
    /// When enabled, adaptive low-LOD sphere fallback is disabled.
    pub fn set_force_full_fbx_sphere(&mut self, force: bool) {
        self.force_full_fbx_sphere = force;
        if force {
            self.sphere_lod_mode = SphereLodMode::High;
        }
    }

    /// Configure RT mode (Off/Auto/On) with fail-soft fallback when unsupported.
    pub fn set_ray_tracing_mode(&mut self, queue: &wgpu::Queue, mode: RayTracingMode) {
        self.ray_tracing_status =
            resolve_rt_status(mode, self.ray_tracing_status.supports_ray_query);
        self.write_lighting_uniform(queue, self.light_count);
    }

    /// Current renderer RT status snapshot.
    pub fn ray_tracing_status(&self) -> SceneRayTracingStatus {
        self.ray_tracing_status.clone()
    }

    /// Configure FSR policy (mode/quality/sharpness) with fail-soft backend fallback.
    pub fn set_fsr_config(&mut self, config: FsrConfig) {
        self.fsr_config = config;
        self.fsr_status = resolve_fsr_status(config, self.adapter_backend);
    }

    /// Current effective FSR status snapshot.
    pub fn fsr_status(&self) -> FsrStatus {
        self.fsr_status.clone()
    }

    pub fn encode(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        clear: wgpu::Color,
    ) {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("runtime-scene-pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(clear),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &self.depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        if !self.ranges.is_empty() {
            pass.set_bind_group(0, &self.scene_bind_group, &[]);

            pass.set_pipeline(&self.pipeline_opaque);
            for range in self.ranges.iter().filter(|r| r.lane == DrawLane::Opaque) {
                draw_3d_range(
                    &mut pass,
                    &self.box_mesh,
                    &self.sphere_mesh_high,
                    &self.sphere_mesh_low,
                    &self.custom_mesh_slots,
                    self.sphere_lod_mode,
                    &self.instance_3d_buffer,
                    range,
                );
            }

            pass.set_pipeline(&self.pipeline_transparent);
            for range in self
                .ranges
                .iter()
                .filter(|r| r.lane == DrawLane::Transparent)
            {
                draw_3d_range(
                    &mut pass,
                    &self.box_mesh,
                    &self.sphere_mesh_high,
                    &self.sphere_mesh_low,
                    &self.custom_mesh_slots,
                    self.sphere_lod_mode,
                    &self.instance_3d_buffer,
                    range,
                );
            }
        }

        if self.sprite_count > 0 {
            pass.set_pipeline(&self.pipeline_sprite);
            pass.set_bind_group(0, &self.sprite_atlas_bind_group, &[]);
            pass.set_bind_group(1, &self.scene_bind_group, &[]);
            pass.set_vertex_buffer(0, self.sprite_vertex_buffer.slice(..));
            let sprite_bytes =
                (self.sprite_count as usize * std::mem::size_of::<GpuSpriteInstance>()) as u64;
            pass.set_vertex_buffer(1, self.sprite_instance_buffer.slice(0..sprite_bytes));
            pass.draw(0..6, 0..self.sprite_count);
        }
    }

    fn write_camera_uniform(&self, queue: &wgpu::Queue, width: u32, height: u32) {
        let aspect = (width.max(1) as f32) / (height.max(1) as f32);
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
        let mut uniform = GpuCameraUniform {
            view_proj: [0.0; 16],
            camera_eye: [
                self.camera_eye[0],
                self.camera_eye[1],
                self.camera_eye[2],
                1.0,
            ],
        };
        uniform.view_proj.copy_from_slice(view_proj.as_slice());
        queue.write_buffer(&self.camera_buffer, 0, bytemuck::bytes_of(&uniform));
    }

    fn upload_lights(
        &mut self,
        queue: &wgpu::Queue,
        lights: &[SceneLight],
        dynamic_instances: usize,
    ) {
        let selected = lights.len().min(MAX_SCENE_LIGHTS);
        let mut gpu_lights = vec![GpuLight::default(); selected.max(1)];
        for (index, light) in lights.iter().take(selected).enumerate() {
            gpu_lights[index] = gpu_light_from_scene(light);
        }
        if selected > 0 {
            queue.write_buffer(
                &self.light_buffer,
                0,
                bytemuck::cast_slice(gpu_lights.as_slice()),
            );
        }
        self.light_count = selected as u32;
        self.ray_tracing_status.rt_dynamic_count = if self.ray_tracing_status.active {
            (dynamic_instances as u32).min(self.ray_tracing_status.rt_dynamic_cap)
        } else {
            0
        };
        self.write_lighting_uniform(queue, self.light_count);
    }

    fn write_lighting_uniform(&self, queue: &wgpu::Queue, light_count: u32) {
        let uniform = GpuLightingUniform {
            light_count,
            rt_mode: ray_mode_to_u32(self.ray_tracing_status.mode),
            rt_active: u32::from(self.ray_tracing_status.active),
            rt_dynamic_count: self.ray_tracing_status.rt_dynamic_count,
            rt_dynamic_cap: self.ray_tracing_status.rt_dynamic_cap,
            _pad: [0; 3],
        };
        queue.write_buffer(&self.lighting_buffer, 0, bytemuck::bytes_of(&uniform));
    }
}

fn build_upload_plan(draw: &RuntimeDrawFrame) -> UploadPlan {
    let mut plan = UploadPlan::default();
    plan.instances_3d
        .reserve(draw.stats.opaque_instances + draw.stats.transparent_instances);
    plan.sprites.reserve(draw.stats.sprite_instances);
    plan.ranges
        .reserve(draw.opaque_batches.len() + draw.transparent_batches.len());

    for batch in &draw.opaque_batches {
        let start = plan.instances_3d.len() as u32;
        for instance in &batch.instances {
            plan.instances_3d.push(GpuInstance3d {
                model_col0: instance.model_cols[0],
                model_col1: instance.model_cols[1],
                model_col2: instance.model_cols[2],
                model_col3: instance.model_cols[3],
                base_color: instance.base_color_rgba,
                material_params: instance.material_params,
                emissive: [
                    instance.emissive_rgb[0],
                    instance.emissive_rgb[1],
                    instance.emissive_rgb[2],
                    0.0,
                ],
            });
        }
        let count = (plan.instances_3d.len() as u32).saturating_sub(start);
        if count > 0 {
            plan.ranges.push(GpuBatchRange {
                lane: DrawLane::Opaque,
                primitive_code: batch.key.primitive_code,
                start,
                count,
            });
        }
    }

    for batch in &draw.transparent_batches {
        let start = plan.instances_3d.len() as u32;
        for instance in &batch.instances {
            plan.instances_3d.push(GpuInstance3d {
                model_col0: instance.model_cols[0],
                model_col1: instance.model_cols[1],
                model_col2: instance.model_cols[2],
                model_col3: instance.model_cols[3],
                base_color: instance.base_color_rgba,
                material_params: instance.material_params,
                emissive: [
                    instance.emissive_rgb[0],
                    instance.emissive_rgb[1],
                    instance.emissive_rgb[2],
                    0.0,
                ],
            });
        }
        let count = (plan.instances_3d.len() as u32).saturating_sub(start);
        if count > 0 {
            plan.ranges.push(GpuBatchRange {
                lane: DrawLane::Transparent,
                primitive_code: batch.key.primitive_code,
                start,
                count,
            });
        }
    }

    for sprite in &draw.sprites {
        plan.sprites.push(GpuSpriteInstance {
            translate_size: [
                sprite.position[0],
                sprite.position[1],
                sprite.size[0],
                sprite.size[1],
            ],
            rot_z: [sprite.rotation_rad, sprite.position[2], 0.0, 0.0],
            color: sprite.color_rgba,
            atlas_rect: sprite_kind_atlas_rect(sprite.kind, sprite.texture_slot),
            kind_params: [
                sprite_kind_code(sprite.kind) as f32,
                sprite.texture_slot as f32,
                0.0,
                0.0,
            ],
        });
    }

    plan
}

fn draw_3d_range(
    pass: &mut wgpu::RenderPass<'_>,
    box_mesh: &GpuMesh,
    sphere_mesh_high: &GpuMesh,
    sphere_mesh_low: &GpuMesh,
    custom_mesh_slots: &BTreeMap<u8, GpuMesh>,
    sphere_lod_mode: SphereLodMode,
    instance_buffer: &wgpu::Buffer,
    range: &GpuBatchRange,
) {
    let mesh = match range.primitive_code {
        0 => match sphere_lod_mode {
            SphereLodMode::High => sphere_mesh_high,
            SphereLodMode::Low => sphere_mesh_low,
        },
        1 => box_mesh,
        code => {
            let slot = code.saturating_sub(2);
            custom_mesh_slots.get(&slot).unwrap_or(box_mesh)
        }
    };
    pass.set_vertex_buffer(0, mesh.vertex.slice(..));
    let bytes_per_instance = std::mem::size_of::<GpuInstance3d>() as u64;
    let start = range.start as u64 * bytes_per_instance;
    let end = (range.start as u64 + range.count as u64) * bytes_per_instance;
    pass.set_vertex_buffer(1, instance_buffer.slice(start..end));
    pass.set_index_buffer(mesh.index.slice(..), mesh.index_format);
    pass.draw_indexed(0..mesh.index_count, 0, 0..range.count);
}

fn count_sphere_instances(ranges: &[GpuBatchRange]) -> usize {
    ranges
        .iter()
        .filter(|range| range.primitive_code == 0)
        .map(|range| range.count as usize)
        .sum()
}

fn select_sphere_lod_mode(current: SphereLodMode, sphere_instances: usize) -> SphereLodMode {
    match current {
        SphereLodMode::High => {
            if sphere_instances >= SPHERE_LOD_ENABLE_THRESHOLD {
                SphereLodMode::Low
            } else {
                SphereLodMode::High
            }
        }
        SphereLodMode::Low => {
            if sphere_instances <= SPHERE_LOD_DISABLE_THRESHOLD {
                SphereLodMode::High
            } else {
                SphereLodMode::Low
            }
        }
    }
}

fn gpu_light_from_scene(light: &SceneLight) -> GpuLight {
    let kind = match light.kind {
        SceneLightKind::Point => 0.0,
        SceneLightKind::Spot => 1.0,
    };
    let direction = normalize_vec3(light.direction);
    let color = [
        light.color[0].clamp(0.0, 16.0),
        light.color[1].clamp(0.0, 16.0),
        light.color[2].clamp(0.0, 16.0),
    ];
    let inner = light.inner_cone_deg.to_radians().cos();
    let outer = light
        .outer_cone_deg
        .max(light.inner_cone_deg + 0.01)
        .to_radians()
        .cos();
    GpuLight {
        position_kind: [
            light.position[0],
            light.position[1],
            light.position[2],
            kind,
        ],
        direction_inner: [direction[0], direction[1], direction[2], inner],
        color_intensity: [color[0], color[1], color[2], light.intensity.max(0.0)],
        params: [
            light.range.max(0.05),
            outer,
            light.softness.clamp(0.0, 1.0),
            light.specular_strength.clamp(0.0, 8.0),
        ],
        shadow: [f32::from(light.casts_shadow), 0.0, 0.0, 0.0],
    }
}

fn normalize_vec3(input: [f32; 3]) -> [f32; 3] {
    let len = (input[0] * input[0] + input[1] * input[1] + input[2] * input[2]).sqrt();
    if len <= 1e-6 {
        [0.0, -1.0, 0.0]
    } else {
        [input[0] / len, input[1] / len, input[2] / len]
    }
}

fn supports_ray_query(device: &wgpu::Device, backend: wgpu::Backend) -> bool {
    backend == wgpu::Backend::Vulkan
        && device
            .features()
            .contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY)
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
                "ray query unsupported on current adapter; auto fallback to forward path"
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
                "mode=on requested but ray query unsupported; fail-soft fallback to auto forward"
                    .to_string()
            },
            rt_dynamic_count: 0,
            rt_dynamic_cap: RT_DYNAMIC_CAP,
            supports_ray_query,
        },
    }
}

#[inline]
fn ray_mode_to_u32(mode: RayTracingMode) -> u32 {
    match mode {
        RayTracingMode::Off => 0,
        RayTracingMode::Auto => 1,
        RayTracingMode::On => 2,
    }
}

fn create_3d_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
    color_format: wgpu::TextureFormat,
    blend: Option<wgpu::BlendState>,
    depth_write_enabled: bool,
    cull_mode: Option<wgpu::Face>,
    label: &str,
) -> wgpu::RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: Some("vs_main"),
            buffers: &[
                wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<GpuVertex3d>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x3],
                },
                wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<GpuInstance3d>() as u64,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &wgpu::vertex_attr_array![
                        1 => Float32x4,
                        2 => Float32x4,
                        3 => Float32x4,
                        4 => Float32x4,
                        5 => Float32x4,
                        6 => Float32x4,
                        7 => Float32x4
                    ],
                },
            ],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: shader,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format: color_format,
                blend,
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            cull_mode,
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: DEPTH_FORMAT,
            depth_write_enabled,
            depth_compare: wgpu::CompareFunction::LessEqual,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    })
}

fn create_sprite_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
    color_format: wgpu::TextureFormat,
) -> wgpu::RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("runtime-scene-sprite-pipeline"),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: Some("vs_main"),
            buffers: &[
                wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<GpuSpriteVertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x2],
                },
                wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<GpuSpriteInstance>() as u64,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &wgpu::vertex_attr_array![
                        1 => Float32x4,
                        2 => Float32x4,
                        3 => Float32x4,
                        4 => Float32x4,
                        5 => Float32x4
                    ],
                },
            ],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: shader,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format: color_format,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: DEPTH_FORMAT,
            depth_write_enabled: false,
            depth_compare: wgpu::CompareFunction::Always,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    })
}

#[inline]
fn sprite_kind_code(kind: SpriteKind) -> u32 {
    match kind {
        SpriteKind::Generic => 0,
        SpriteKind::Hud => 1,
        SpriteKind::Camera => 2,
        SpriteKind::Terrain => 3,
    }
}

#[inline]
fn sprite_kind_atlas_row(kind: SpriteKind) -> u32 {
    match kind {
        SpriteKind::Generic => 0,
        SpriteKind::Hud => 1,
        SpriteKind::Camera => 2,
        SpriteKind::Terrain => 3,
    }
}

#[inline]
fn sprite_kind_atlas_rect(kind: SpriteKind, texture_slot: u16) -> [f32; 4] {
    const ATLAS_COLS: u16 = 4;
    const INV_ATLAS: f32 = 0.25; // 1.0 / 4.0
    let col = texture_slot % ATLAS_COLS;
    let row = sprite_kind_atlas_row(kind);
    let u0 = col as f32 * INV_ATLAS;
    let v0 = row as f32 * INV_ATLAS;
    [u0, v0, u0 + INV_ATLAS, v0 + INV_ATLAS]
}

fn create_box_mesh(device: &wgpu::Device) -> GpuMesh {
    let vertices = [
        GpuVertex3d {
            position: [-0.5, -0.5, -0.5],
        },
        GpuVertex3d {
            position: [0.5, -0.5, -0.5],
        },
        GpuVertex3d {
            position: [0.5, 0.5, -0.5],
        },
        GpuVertex3d {
            position: [-0.5, 0.5, -0.5],
        },
        GpuVertex3d {
            position: [-0.5, -0.5, 0.5],
        },
        GpuVertex3d {
            position: [0.5, -0.5, 0.5],
        },
        GpuVertex3d {
            position: [0.5, 0.5, 0.5],
        },
        GpuVertex3d {
            position: [-0.5, 0.5, 0.5],
        },
    ];
    let indices: [u16; 36] = [
        0, 1, 2, 2, 3, 0, // back
        4, 6, 5, 6, 4, 7, // front
        0, 4, 5, 5, 1, 0, // bottom
        3, 2, 6, 6, 7, 3, // top
        1, 5, 6, 6, 2, 1, // right
        0, 3, 7, 7, 4, 0, // left
    ];
    create_mesh_u16(device, "runtime-scene-box", &vertices, &indices)
}

fn create_sphere_mesh(device: &wgpu::Device) -> GpuMesh {
    match parse_first_mesh_from_fbx(DEFAULT_SPHERE_FBX_BYTES) {
        Ok(mesh_data) => create_mesh_u32(
            device,
            "runtime-scene-sphere-fbx",
            &mesh_data.vertices,
            &mesh_data.indices,
        ),
        Err(_) => create_octa_sphere_mesh(device),
    }
}

fn create_octa_sphere_mesh(device: &wgpu::Device) -> GpuMesh {
    let vertices = [
        GpuVertex3d {
            position: [1.0, 0.0, 0.0],
        },
        GpuVertex3d {
            position: [-1.0, 0.0, 0.0],
        },
        GpuVertex3d {
            position: [0.0, 1.0, 0.0],
        },
        GpuVertex3d {
            position: [0.0, -1.0, 0.0],
        },
        GpuVertex3d {
            position: [0.0, 0.0, 1.0],
        },
        GpuVertex3d {
            position: [0.0, 0.0, -1.0],
        },
    ];
    let indices: [u16; 24] = [
        0, 2, 4, 4, 2, 1, 1, 2, 5, 5, 2, 0, 4, 3, 0, 1, 3, 4, 5, 3, 1, 0, 3, 5,
    ];
    create_mesh_u16(device, "runtime-scene-sphere", &vertices, &indices)
}

fn create_icosa_sphere_mesh(device: &wgpu::Device) -> GpuMesh {
    let t = (1.0 + 5.0_f32.sqrt()) * 0.5;
    let mut v = [
        [-1.0, t, 0.0],
        [1.0, t, 0.0],
        [-1.0, -t, 0.0],
        [1.0, -t, 0.0],
        [0.0, -1.0, t],
        [0.0, 1.0, t],
        [0.0, -1.0, -t],
        [0.0, 1.0, -t],
        [t, 0.0, -1.0],
        [t, 0.0, 1.0],
        [-t, 0.0, -1.0],
        [-t, 0.0, 1.0],
    ];

    for p in &mut v {
        let len = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt().max(1e-6);
        p[0] /= len;
        p[1] /= len;
        p[2] /= len;
    }

    let vertices = v.map(|position| GpuVertex3d { position });

    let indices: [u16; 60] = [
        0, 11, 5, 0, 5, 1, 0, 1, 7, 0, 7, 10, 0, 10, 11, 1, 5, 9, 5, 11, 4, 11, 10, 2, 10, 7, 6, 7,
        1, 8, 3, 9, 4, 3, 4, 2, 3, 2, 6, 3, 6, 8, 3, 8, 9, 4, 9, 5, 2, 4, 11, 6, 2, 10, 8, 6, 7, 9,
        8, 1,
    ];
    create_mesh_u16(device, "runtime-scene-sphere-icosa", &vertices, &indices)
}

#[derive(Debug, Clone)]
struct ParsedFbxMesh {
    vertices: Vec<GpuVertex3d>,
    indices: Vec<u32>,
}

fn parse_first_mesh_from_fbx(bytes: &[u8]) -> Result<ParsedFbxMesh, String> {
    let file = fbx::File::read_from(Cursor::new(bytes)).map_err(|err| format!("{err}"))?;
    let objects = file
        .children
        .iter()
        .find(|node| node.name == "Objects")
        .ok_or_else(|| "FBX objects node was not found".to_string())?;

    for geometry in objects
        .children
        .iter()
        .filter(|node| node.name == "Geometry")
    {
        let kind = geometry
            .properties
            .get(2)
            .and_then(fbx_property_as_string)
            .unwrap_or_default();
        if kind != "Mesh" {
            continue;
        }

        let vertices_f64 = geometry
            .children
            .iter()
            .find(|node| node.name == "Vertices")
            .and_then(|node| node.properties.first())
            .and_then(fbx_property_as_f64_slice)
            .ok_or_else(|| "FBX mesh does not contain vertices".to_string())?;
        let polygon_vertex_index = geometry
            .children
            .iter()
            .find(|node| node.name == "PolygonVertexIndex")
            .and_then(|node| node.properties.first())
            .and_then(fbx_property_as_i32_slice)
            .ok_or_else(|| "FBX mesh does not contain polygon vertex indices".to_string())?;

        let mut vertices = fbx_vertices_to_gpu(vertices_f64)?;
        let indices = fbx_polygon_indices_to_triangles(polygon_vertex_index, vertices.len())?;
        if indices.is_empty() {
            return Err("FBX mesh did not produce triangle indices".to_string());
        }
        normalize_vertices_to_unit_box(vertices.as_mut_slice());
        return Ok(ParsedFbxMesh { vertices, indices });
    }

    Err("No FBX mesh geometry node found".to_string())
}

fn fbx_vertices_to_gpu(vertices_f64: &[f64]) -> Result<Vec<GpuVertex3d>, String> {
    if vertices_f64.len() < 9 {
        return Err("FBX vertices array is too small".to_string());
    }
    if vertices_f64.len() % 3 != 0 {
        return Err("FBX vertices array is not 3-component aligned".to_string());
    }

    let mut vertices = Vec::with_capacity(vertices_f64.len() / 3);
    for chunk in vertices_f64.chunks_exact(3) {
        vertices.push(GpuVertex3d {
            position: [chunk[0] as f32, chunk[1] as f32, chunk[2] as f32],
        });
    }
    Ok(vertices)
}

/// Normalize imported FBX vertices into a centered unit box (`[-0.5, 0.5]` per axis).
///
/// This keeps mesh-slot scaling predictable so runtime X/Y/Z scale controls can shape panels
/// consistently even when source FBX files use arbitrary pivots or authoring units.
fn normalize_vertices_to_unit_box(vertices: &mut [GpuVertex3d]) {
    if vertices.is_empty() {
        return;
    }

    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    for v in vertices.iter() {
        for axis in 0..3 {
            min[axis] = min[axis].min(v.position[axis]);
            max[axis] = max[axis].max(v.position[axis]);
        }
    }

    let center = [
        (min[0] + max[0]) * 0.5,
        (min[1] + max[1]) * 0.5,
        (min[2] + max[2]) * 0.5,
    ];
    let extent = [max[0] - min[0], max[1] - min[1], max[2] - min[2]];
    let inv_extent = [
        if extent[0].abs() > 1e-6 {
            1.0 / extent[0]
        } else {
            0.0
        },
        if extent[1].abs() > 1e-6 {
            1.0 / extent[1]
        } else {
            0.0
        },
        if extent[2].abs() > 1e-6 {
            1.0 / extent[2]
        } else {
            0.0
        },
    ];

    for v in vertices.iter_mut() {
        for axis in 0..3 {
            if inv_extent[axis] > 0.0 {
                v.position[axis] = (v.position[axis] - center[axis]) * inv_extent[axis];
            } else {
                v.position[axis] = 0.0;
            }
        }
    }
}

fn fbx_polygon_indices_to_triangles(
    polygon_vertex_index: &[i32],
    vertex_count: usize,
) -> Result<Vec<u32>, String> {
    let mut triangles_u32 = Vec::<u32>::new();
    let mut polygon = Vec::<u32>::with_capacity(8);

    for &raw_index in polygon_vertex_index {
        let (resolved, is_polygon_end) = if raw_index < 0 {
            let corrected = raw_index
                .checked_neg()
                .and_then(|v| v.checked_sub(1))
                .ok_or_else(|| "FBX polygon index underflow".to_string())?;
            (corrected, true)
        } else {
            (raw_index, false)
        };

        let resolved_u32: u32 = resolved
            .try_into()
            .map_err(|_| "FBX polygon index is negative".to_string())?;
        if resolved_u32 as usize >= vertex_count {
            return Err("FBX polygon index exceeds vertex count".to_string());
        }
        polygon.push(resolved_u32);

        if is_polygon_end {
            triangulate_polygon_fan(&polygon, &mut triangles_u32);
            polygon.clear();
        }
    }

    if !polygon.is_empty() {
        triangulate_polygon_fan(&polygon, &mut triangles_u32);
    }
    if triangles_u32.is_empty() {
        return Err("FBX polygon list did not contain triangles".to_string());
    }

    Ok(triangles_u32)
}

fn triangulate_polygon_fan(polygon: &[u32], triangles_out: &mut Vec<u32>) {
    if polygon.len() < 3 {
        return;
    }
    let first = polygon[0];
    for i in 1..polygon.len() - 1 {
        triangles_out.push(first);
        triangles_out.push(polygon[i]);
        triangles_out.push(polygon[i + 1]);
    }
}

fn fbx_property_as_string(property: &FbxProperty) -> Option<&str> {
    match property {
        FbxProperty::String(value) => Some(value.as_str()),
        _ => None,
    }
}

fn fbx_property_as_f64_slice(property: &FbxProperty) -> Option<&[f64]> {
    match property {
        FbxProperty::F64Array(values) => Some(values.as_slice()),
        _ => None,
    }
}

fn fbx_property_as_i32_slice(property: &FbxProperty) -> Option<&[i32]> {
    match property {
        FbxProperty::I32Array(values) => Some(values.as_slice()),
        _ => None,
    }
}

fn create_mesh_u16(
    device: &wgpu::Device,
    label: &str,
    vertices: &[GpuVertex3d],
    indices: &[u16],
) -> GpuMesh {
    let vertex = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(&format!("{label}-vb")),
        contents: bytemuck::cast_slice(vertices),
        usage: wgpu::BufferUsages::VERTEX,
    });
    let index = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(&format!("{label}-ib")),
        contents: bytemuck::cast_slice(indices),
        usage: wgpu::BufferUsages::INDEX,
    });
    GpuMesh {
        vertex,
        index,
        index_count: indices.len() as u32,
        index_format: wgpu::IndexFormat::Uint16,
    }
}

fn create_mesh_u32(
    device: &wgpu::Device,
    label: &str,
    vertices: &[GpuVertex3d],
    indices: &[u32],
) -> GpuMesh {
    let vertex = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(&format!("{label}-vb")),
        contents: bytemuck::cast_slice(vertices),
        usage: wgpu::BufferUsages::VERTEX,
    });
    let index = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(&format!("{label}-ib")),
        contents: bytemuck::cast_slice(indices),
        usage: wgpu::BufferUsages::INDEX,
    });
    GpuMesh {
        vertex,
        index,
        index_count: indices.len() as u32,
        index_format: wgpu::IndexFormat::Uint32,
    }
}

fn create_sprite_quad_vertex_buffer(device: &wgpu::Device) -> wgpu::Buffer {
    let vertices = [
        GpuSpriteVertex {
            local_pos: [-0.5, -0.5],
        },
        GpuSpriteVertex {
            local_pos: [0.5, -0.5],
        },
        GpuSpriteVertex {
            local_pos: [0.5, 0.5],
        },
        GpuSpriteVertex {
            local_pos: [-0.5, -0.5],
        },
        GpuSpriteVertex {
            local_pos: [0.5, 0.5],
        },
        GpuSpriteVertex {
            local_pos: [-0.5, 0.5],
        },
    ];
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("runtime-scene-sprite-quad-vb"),
        contents: bytemuck::cast_slice(&vertices),
        usage: wgpu::BufferUsages::VERTEX,
    })
}

fn ensure_buffer_capacity(
    device: &wgpu::Device,
    buffer: &mut wgpu::Buffer,
    capacity_bytes: &mut usize,
    required_bytes: usize,
    usage: wgpu::BufferUsages,
    label: &str,
) {
    if required_bytes <= *capacity_bytes {
        return;
    }
    let mut new_capacity = (*capacity_bytes).max(256);
    while new_capacity < required_bytes {
        new_capacity = new_capacity.saturating_mul(2);
    }
    *buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: new_capacity as u64,
        usage,
        mapped_at_creation: false,
    });
    *capacity_bytes = new_capacity;
}

fn create_depth_resources(
    device: &wgpu::Device,
    width: u32,
    height: u32,
    label: &str,
) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: DEPTH_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

fn create_default_sprite_atlas_resources(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    bind_group_layout: &wgpu::BindGroupLayout,
) -> (wgpu::Texture, wgpu::BindGroup) {
    let width = SPRITE_ATLAS_GRID_DIM * SPRITE_ATLAS_TILE_SIZE;
    let height = SPRITE_ATLAS_GRID_DIM * SPRITE_ATLAS_TILE_SIZE;
    let pixels = build_default_sprite_atlas_pixels(width, height);
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("runtime-scene-sprite-atlas"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &pixels,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(width * 4),
            rows_per_image: Some(height),
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );

    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("runtime-scene-sprite-atlas-sampler"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::MipmapFilterMode::Nearest,
        ..Default::default()
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("runtime-scene-sprite-atlas-bg"),
        layout: bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&sampler),
            },
        ],
    });
    (texture, bind_group)
}

fn build_default_sprite_atlas_pixels(width: u32, height: u32) -> Vec<u8> {
    let mut pixels = vec![0u8; (width * height * 4) as usize];
    let tile = SPRITE_ATLAS_TILE_SIZE.max(1);

    for y in 0..height {
        for x in 0..width {
            let tile_x = (x / tile).min(SPRITE_ATLAS_GRID_DIM - 1);
            let tile_y = (y / tile).min(SPRITE_ATLAS_GRID_DIM - 1);
            let u = (x % tile) as f32 / (tile - 1) as f32;
            let v = (y % tile) as f32 / (tile - 1) as f32;
            let rgba = atlas_tile_rgba(tile_x, tile_y, u, v);
            let idx = ((y * width + x) * 4) as usize;
            pixels[idx] = rgba[0];
            pixels[idx + 1] = rgba[1];
            pixels[idx + 2] = rgba[2];
            pixels[idx + 3] = rgba[3];
        }
    }
    pixels
}

fn atlas_tile_rgba(tile_x: u32, tile_y: u32, u: f32, v: f32) -> [u8; 4] {
    let base_r = 40.0 + tile_x as f32 * 44.0;
    let base_g = 56.0 + tile_y as f32 * 40.0;
    let base_b = 72.0 + (tile_x ^ tile_y) as f32 * 30.0;

    let band = 0.82 + 0.18 * ((u * 14.0 + v * 12.0 + tile_x as f32).sin() * 0.5 + 0.5);
    let vignette = (1.0 - ((u - 0.5).abs() + (v - 0.5).abs()) * 0.6).clamp(0.65, 1.0);
    let shade = band * vignette;

    let r = (base_r * shade).clamp(0.0, 255.0) as u8;
    let g = (base_g * shade).clamp(0.0, 255.0) as u8;
    let b = (base_b * shade).clamp(0.0, 255.0) as u8;
    [r, g, b, 255]
}

const SCENE_3D_WGSL: &str = r#"
struct Camera {
    view_proj: mat4x4<f32>,
    camera_eye: vec4<f32>,
};

@group(0) @binding(0)
var<uniform> u_camera: Camera;

struct LightData {
    position_kind: vec4<f32>,
    direction_inner: vec4<f32>,
    color_intensity: vec4<f32>,
    params: vec4<f32>,
    shadow: vec4<f32>,
};

struct Lighting {
    light_count: u32,
    rt_mode: u32,
    rt_active: u32,
    rt_dynamic_count: u32,
    rt_dynamic_cap: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(1)
var<storage, read> u_lights: array<LightData, 32>;

@group(0) @binding(2)
var<uniform> u_lighting: Lighting;

struct VSIn {
    @location(0) position: vec3<f32>,
    @location(1) model_col0: vec4<f32>,
    @location(2) model_col1: vec4<f32>,
    @location(3) model_col2: vec4<f32>,
    @location(4) model_col3: vec4<f32>,
    @location(5) base_color: vec4<f32>,
    @location(6) material_params: vec4<f32>,
    @location(7) emissive: vec4<f32>,
};

struct VSOut {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) emissive: vec3<f32>,
    @location(2) world_pos: vec3<f32>,
    @location(3) local_pos: vec3<f32>,
    @location(4) primitive_code: f32,
};

@vertex
fn vs_main(input: VSIn) -> VSOut {
    let model = mat4x4<f32>(
        input.model_col0,
        input.model_col1,
        input.model_col2,
        input.model_col3
    );
    let world_pos = model * vec4<f32>(input.position, 1.0);

    var out: VSOut;
    out.position = u_camera.view_proj * world_pos;
    out.color = input.base_color;
    out.emissive = input.emissive.xyz;
    out.world_pos = world_pos.xyz;
    out.local_pos = input.position;
    out.primitive_code = input.material_params.w;
    return out;
}

fn evaluate_light(light: LightData, normal: vec3<f32>, world_pos: vec3<f32>, base_color: vec3<f32>) -> vec3<f32> {
    let to_light = light.position_kind.xyz - world_pos;
    let distance = max(length(to_light), 1e-4);
    let light_dir = to_light / distance;
    let range = max(light.params.x, 1e-3);
    if distance > range {
        return vec3<f32>(0.0);
    }

    var attenuation = 1.0 - smoothstep(range * 0.65, range, distance);
    attenuation *= attenuation;

    if light.position_kind.w > 0.5 {
        let spot_axis = normalize(-light.direction_inner.xyz);
        let cone = dot(spot_axis, light_dir);
        let inner = light.direction_inner.w;
        let outer = light.params.y;
        let cone_factor = smoothstep(outer, inner, cone);
        attenuation *= cone_factor;
    }

    let ndotl = max(dot(normal, light_dir), 0.0);
    if ndotl <= 0.0 || attenuation <= 1e-5 {
        return vec3<f32>(0.0);
    }

    let view_dir = normalize(u_camera.camera_eye.xyz - world_pos);
    let half_dir = normalize(light_dir + view_dir);
    let spec_power = 24.0 + light.params.w * 24.0;
    let specular = pow(max(dot(normal, half_dir), 0.0), spec_power) * light.params.w;

    var shadow_term = 1.0;
    if u_lighting.rt_active > 0u && light.shadow.x > 0.5 {
        let penumbra_floor = mix(0.35, 0.80, 1.0 - light.params.z);
        shadow_term = mix(penumbra_floor, 1.0, ndotl);
    }

    let diffuse = base_color * ndotl;
    let spec = vec3<f32>(specular);
    return (diffuse + spec) * light.color_intensity.rgb * light.color_intensity.w * attenuation * shadow_term;
}

@fragment
fn fs_main(input: VSOut, @builtin(front_facing) is_front: bool) -> @location(0) vec4<f32> {
    let dpx = dpdx(input.world_pos);
    let dpy = dpdy(input.world_pos);
    var normal = normalize(cross(dpx, dpy));
    if !is_front {
        normal = -normal;
    }

    var lit = input.color.rgb * 0.10;
    let light_count = min(u_lighting.light_count, 32u);
    if light_count == 0u {
        let fallback_dir = normalize(vec3<f32>(0.42, 0.74, 0.52));
        let fallback = max(dot(normal, fallback_dir), 0.0) * 0.76 + 0.24;
        lit += input.color.rgb * fallback;
    } else {
        for (var i: u32 = 0u; i < light_count; i = i + 1u) {
            lit += evaluate_light(u_lights[i], normal, input.world_pos, input.color.rgb);
        }
    }
    lit += input.emissive;
    var alpha = input.color.a;

    // Box primitives get extra edge contrast so the tank keeps a clear prism silhouette.
    if input.primitive_code > 0.5 {
        let edge = max(max(abs(input.local_pos.x), abs(input.local_pos.y)), abs(input.local_pos.z));
        let edge_boost = smoothstep(0.38, 0.50, edge);
        lit += vec3<f32>(0.08, 0.12, 0.18) * edge_boost;
        alpha = clamp(alpha + edge_boost * 0.32, 0.0, 1.0);
    }
    return vec4<f32>(lit, alpha);
}
"#;

const SCENE_SPRITE_WGSL: &str = r#"
struct VSIn {
    @location(0) local_pos: vec2<f32>,
    @location(1) translate_size: vec4<f32>,
    @location(2) rot_z: vec4<f32>,
    @location(3) color: vec4<f32>,
    @location(4) atlas_rect: vec4<f32>,
    @location(5) kind_params: vec4<f32>,
};

struct VSOut {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) atlas_uv: vec2<f32>,
    @location(3) kind_params: vec2<f32>,
};

@group(0) @binding(0) var sprite_tex: texture_2d<f32>;
@group(0) @binding(1) var sprite_smp: sampler;

struct Lighting {
    light_count: u32,
    rt_mode: u32,
    rt_active: u32,
    rt_dynamic_count: u32,
    rt_dynamic_cap: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(1) @binding(2) var<uniform> u_lighting: Lighting;

@vertex
fn vs_main(input: VSIn) -> VSOut {
    let c = cos(input.rot_z.x);
    let s = sin(input.rot_z.x);
    let scaled = vec2<f32>(
        input.local_pos.x * input.translate_size.z,
        input.local_pos.y * input.translate_size.w
    );
    let rotated = vec2<f32>(
        scaled.x * c - scaled.y * s,
        scaled.x * s + scaled.y * c
    );
    let pos = rotated + input.translate_size.xy;
    let uv = input.local_pos + vec2<f32>(0.5, 0.5);
    let atlas_uv = mix(input.atlas_rect.xy, input.atlas_rect.zw, uv);

    var out: VSOut;
    out.position = vec4<f32>(pos, input.rot_z.y, 1.0);
    out.color = input.color;
    out.uv = uv;
    out.atlas_uv = atlas_uv;
    out.kind_params = input.kind_params.xy;
    return out;
}

fn terrain_style(base_color: vec3<f32>, uv: vec2<f32>, atlas_uv: vec2<f32>, slot: f32) -> vec3<f32> {
    let stripe = 0.5 + 0.5 * sin((atlas_uv.x * 64.0) + slot * 0.37);
    let top = vec3<f32>(base_color.r * 1.06, base_color.g * 1.03, base_color.b * 0.90);
    let bottom = vec3<f32>(base_color.r * 0.85, base_color.g * 0.92, base_color.b * 0.78);
    let grad = mix(bottom, top, uv.y);
    return mix(grad, grad * vec3<f32>(0.72, 0.88, 0.72), stripe * 0.35);
}

fn camera_style(base_color: vec3<f32>, uv: vec2<f32>, atlas_uv: vec2<f32>, slot: f32) -> vec3<f32> {
    let centered = uv - vec2<f32>(0.5, 0.5);
    let dist = length(centered);
    let ring = 1.0 - smoothstep(0.26, 0.43, abs(dist - 0.31));
    let lens = 1.0 - smoothstep(0.07, 0.31, dist);
    let scan = 0.5 + 0.5 * sin((atlas_uv.y * 48.0) + slot * 0.21);
    let ring_color = vec3<f32>(0.95, 0.98, 1.0);
    let lens_color = mix(base_color * vec3<f32>(0.36, 0.52, 0.78), base_color, scan * 0.6);
    return lens_color * (0.55 + lens * 0.45) + ring_color * ring * 0.55;
}

@fragment
fn fs_main(input: VSOut) -> @location(0) vec4<f32> {
    let kind = i32(input.kind_params.x + 0.5);
    let slot = input.kind_params.y;
    let sampled = textureSample(sprite_tex, sprite_smp, input.atlas_uv);
    var color = input.color.rgb * sampled.rgb;
    let alpha = input.color.a * sampled.a;

    if kind == 2 {
        color = camera_style(color, input.uv, input.atlas_uv, slot);
    } else if kind == 3 {
        color = terrain_style(color, input.uv, input.atlas_uv, slot);
    } else if kind == 1 {
        let pulse = 0.93 + 0.07 * sin(input.atlas_uv.x * 28.0 + slot * 0.15);
        color = color * pulse;
    } else {
        let grain = 0.98 + 0.02 * sin((input.atlas_uv.x + input.atlas_uv.y) * 32.0 + slot * 0.11);
        color = color * grain;
    }

    // Optional light-aware glow billboard helper for sprite lanes.
    let centered = input.uv - vec2<f32>(0.5, 0.5);
    let radial = 1.0 - smoothstep(0.22, 0.66, length(centered));
    let light_gain = 1.0 + min(f32(u_lighting.light_count) / 16.0, 0.40);
    if kind == 0 || kind == 1 {
        color += vec3<f32>(0.06, 0.09, 0.14) * radial * light_gain;
    }

    return vec4<f32>(color, alpha);
}
"#;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::draw_path::{
        DrawBatch3d, DrawBatchKey, DrawFrameStats, DrawInstance3d, RuntimeDrawFrame,
    };
    use crate::scene::{SpriteInstance, SpriteKind};

    #[test]
    fn upload_plan_preserves_range_counts() {
        let draw = RuntimeDrawFrame {
            opaque_batches: vec![DrawBatch3d {
                lane: DrawLane::Opaque,
                key: DrawBatchKey {
                    primitive_code: 1,
                    shading_code: 0,
                    shadow_flags: 3,
                },
                instances: vec![make_instance(1), make_instance(2)],
            }],
            transparent_batches: vec![DrawBatch3d {
                lane: DrawLane::Transparent,
                key: DrawBatchKey {
                    primitive_code: 0,
                    shading_code: 0,
                    shadow_flags: 0,
                },
                instances: vec![make_instance(3)],
            }],
            sprites: Vec::new(),
            lights: Vec::new(),
            stats: DrawFrameStats {
                opaque_instances: 2,
                transparent_instances: 1,
                sprite_instances: 0,
                light_instances: 0,
                opaque_batches: 1,
                transparent_batches: 1,
                total_draw_calls: 2,
            },
        };

        let plan = build_upload_plan(&draw);
        assert_eq!(plan.instances_3d.len(), 3);
        assert_eq!(plan.ranges.len(), 2);
        assert_eq!(plan.ranges[0].count, 2);
        assert_eq!(plan.ranges[1].count, 1);
        assert_eq!(plan.ranges[1].start, 2);
    }

    #[test]
    fn upload_plan_encodes_sprite_kind_and_virtual_atlas_rect() {
        let draw = RuntimeDrawFrame {
            opaque_batches: Vec::new(),
            transparent_batches: Vec::new(),
            sprites: vec![SpriteInstance {
                sprite_id: 99,
                kind: SpriteKind::Camera,
                position: [0.0, 0.0, 0.0],
                size: [0.2, 0.2],
                rotation_rad: 0.0,
                color_rgba: [1.0, 1.0, 1.0, 1.0],
                texture_slot: 6,
                layer: 120,
            }],
            lights: Vec::new(),
            stats: DrawFrameStats {
                opaque_instances: 0,
                transparent_instances: 0,
                sprite_instances: 1,
                light_instances: 0,
                opaque_batches: 0,
                transparent_batches: 0,
                total_draw_calls: 1,
            },
        };

        let plan = build_upload_plan(&draw);
        assert_eq!(plan.sprites.len(), 1);
        let sprite = plan.sprites[0];
        assert!((sprite.kind_params[0] - 2.0).abs() < 1e-6); // camera
        assert!((sprite.kind_params[1] - 6.0).abs() < 1e-6); // texture_slot
        assert!((sprite.atlas_rect[0] - 0.5).abs() < 1e-6); // col 2 / 4
        assert!((sprite.atlas_rect[1] - 0.5).abs() < 1e-6); // row 2 / 4
        assert!((sprite.atlas_rect[2] - 0.75).abs() < 1e-6);
        assert!((sprite.atlas_rect[3] - 0.75).abs() < 1e-6);
    }

    #[test]
    fn default_sprite_atlas_pixels_are_non_uniform() {
        let width = SPRITE_ATLAS_GRID_DIM * SPRITE_ATLAS_TILE_SIZE;
        let height = SPRITE_ATLAS_GRID_DIM * SPRITE_ATLAS_TILE_SIZE;
        let pixels = build_default_sprite_atlas_pixels(width, height);
        assert_eq!(pixels.len(), (width * height * 4) as usize);
        let first = &pixels[0..4];
        let second = &pixels
            [(SPRITE_ATLAS_TILE_SIZE as usize * 4)..(SPRITE_ATLAS_TILE_SIZE as usize * 4 + 4)];
        assert_ne!(first, second);
    }

    #[test]
    fn embedded_sphere_fbx_is_parsed_into_indexed_triangles() {
        let mesh = parse_first_mesh_from_fbx(DEFAULT_SPHERE_FBX_BYTES)
            .expect("embedded sphere.fbx should parse");
        assert!(!mesh.vertices.is_empty());
        assert!(!mesh.indices.is_empty());
        assert_eq!(mesh.indices.len() % 3, 0);
        let max_index = mesh.indices.iter().copied().max().unwrap_or(0) as usize;
        assert!(max_index < mesh.vertices.len());
    }

    #[test]
    fn sphere_lod_hysteresis_switches_without_flapping() {
        let mut mode = SphereLodMode::High;
        mode = select_sphere_lod_mode(mode, SPHERE_LOD_ENABLE_THRESHOLD - 1);
        assert_eq!(mode, SphereLodMode::High);

        mode = select_sphere_lod_mode(mode, SPHERE_LOD_ENABLE_THRESHOLD);
        assert_eq!(mode, SphereLodMode::Low);

        // Between thresholds we should keep the current mode.
        mode = select_sphere_lod_mode(mode, SPHERE_LOD_DISABLE_THRESHOLD + 120);
        assert_eq!(mode, SphereLodMode::Low);

        mode = select_sphere_lod_mode(mode, SPHERE_LOD_DISABLE_THRESHOLD);
        assert_eq!(mode, SphereLodMode::High);
    }

    #[test]
    fn sphere_instance_counter_ignores_non_sphere_batches() {
        let ranges = vec![
            GpuBatchRange {
                lane: DrawLane::Opaque,
                primitive_code: 0,
                start: 0,
                count: 120,
            },
            GpuBatchRange {
                lane: DrawLane::Opaque,
                primitive_code: 1,
                start: 120,
                count: 44,
            },
            GpuBatchRange {
                lane: DrawLane::Transparent,
                primitive_code: 0,
                start: 164,
                count: 36,
            },
        ];
        assert_eq!(count_sphere_instances(ranges.as_slice()), 156);
    }

    #[test]
    fn rt_status_auto_and_on_fail_soft_without_support() {
        let auto = resolve_rt_status(RayTracingMode::Auto, false);
        assert_eq!(auto.mode, RayTracingMode::Auto);
        assert!(!auto.active);
        assert!(auto.fallback_reason.contains("ray query unsupported"));

        let on = resolve_rt_status(RayTracingMode::On, false);
        assert_eq!(on.mode, RayTracingMode::On);
        assert!(!on.active);
        assert!(on.fallback_reason.contains("mode=on requested"));
    }

    #[test]
    fn rt_status_on_activates_when_supported() {
        let status = resolve_rt_status(RayTracingMode::On, true);
        assert_eq!(status.mode, RayTracingMode::On);
        assert!(status.active);
        assert_eq!(status.rt_dynamic_cap, RT_DYNAMIC_CAP);
        assert!(status.fallback_reason.is_empty());
    }

    fn make_instance(id: u64) -> DrawInstance3d {
        DrawInstance3d {
            instance_id: id,
            model_cols: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            base_color_rgba: [1.0, 1.0, 1.0, 1.0],
            material_params: [0.5, 0.0, 0.0, 0.0],
            emissive_rgb: [0.0, 0.0, 0.0],
        }
    }
}
