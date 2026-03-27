//! Minimal `wgpu` scene renderer for runtime draw frames.
//!
//! This backend consumes `RuntimeDrawFrame` and issues real render-pass draw calls for:
//! - opaque 3D batches
//! - transparent 3D batches
//! - sprite overlays (including telemetry HUD sprites)

use std::{collections::BTreeMap, fs, io::Cursor, path::Path};

use fbx::Property as FbxProperty;
use font8x8::{UnicodeFonts, BASIC_FONTS};
use nalgebra::{Isometry3, Matrix4, Perspective3, Point3, Vector3};
use wgpu::util::DeviceExt;

use crate::draw_path::{DrawLane, RuntimeDrawFrame};
use crate::scene::{
    RayTracingMode, SceneLight, SceneLightKind, SpriteInstance, SpriteKind, MAX_SCENE_LIGHTS,
};
use crate::tlsprite::decode_sprite_texture_to_rgba;
use crate::upscaler::{resolve_fsr_status, FsrConfig, FsrStatus};

const SPRITE_ATLAS_GRID_DIM: u32 = 16;
const SPRITE_ATLAS_TILE_SIZE: u32 = 64;
const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
pub const DEFAULT_MSAA_SAMPLE_COUNT: u32 = 4;
/// Max shadow-casting lights that get their own shadow map each frame.
const MAX_SHADOW_LIGHTS: usize = 4;
/// Resolution of each shadow map (square). 1024 gives good quality for a flashlight at scene scale.
const SHADOW_MAP_SIZE: u32 = 1024;
const DEFAULT_SPHERE_FBX_BYTES: &[u8] = include_bytes!("../../../docs/demos/tlapp/sphere.fbx");
// Hysteresis avoids rapid mesh-mode flapping when instance counts hover around thresholds.
const SPHERE_LOD_ENABLE_THRESHOLD: usize = 2_500;
const SPHERE_LOD_DISABLE_THRESHOLD: usize = 1_800;
const RT_DYNAMIC_CAP: u32 = 16_384;
const EXTERNAL_IMAGE_SLOT_BASE: u16 = 64;
const TEXT_GLYPH_SLOT_BASE: u16 = 128;

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

/// Per-slot uniform written before each shadow depth pass. Contains the light's view-proj matrix.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod, Default)]
struct GpuShadowPassUniform {
    light_view_proj: [f32; 16],
}

/// Uniform holding all active shadow map light-space matrices + slot assignments.
/// Bound in group(0) binding(5) for the main 3D fragment shader.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
struct GpuShadowUniform {
    /// Up to MAX_SHADOW_LIGHTS light-space view-proj matrices (column-major, nalgebra layout).
    light_view_proj: [[f32; 16]; 4],
    /// For each shadow slot: which index in u_lights[] it corresponds to. -1 = unused.
    shadow_light_indices: [i32; 4],
    shadow_count: u32,
    _pad: [u32; 3],
}

impl Default for GpuShadowUniform {
    fn default() -> Self {
        Self {
            light_view_proj: [[0.0; 16]; 4],
            shadow_light_indices: [-1; 4],
            shadow_count: 0,
            _pad: [0; 3],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
struct GpuUpscaleUniform {
    inv_source_size: [f32; 2],
    source_uv_scale: [f32; 2],
    sharpness: f32,
    _pad: [f32; 3],
}

impl Default for GpuUpscaleUniform {
    fn default() -> Self {
        Self {
            inv_source_size: [1.0, 1.0],
            source_uv_scale: [1.0, 1.0],
            sharpness: 0.0,
            _pad: [0.0; 3],
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

/// Which BLAS a CPU-side RT instance references.
#[derive(Clone, Copy, PartialEq, Eq)]
enum RtBlasKind {
    Box,
    Sphere,
}

/// CPU-side record used each frame to populate the TLAS.
struct RtInstance {
    /// Row-major 3×4 affine transform (wgpu TlasInstance format).
    transform: [f32; 12],
    blas_kind: RtBlasKind,
}

#[derive(Default)]
struct UploadPlan {
    instances_3d: Vec<GpuInstance3d>,
    ranges: Vec<GpuBatchRange>,
    /// Regular (alpha-blended) sprites.
    sprites: Vec<GpuSpriteInstance>,
    /// Additive-blended light glow sprites, uploaded after regular sprites.
    glow_sprites: Vec<GpuSpriteInstance>,
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
    surface_color_format: wgpu::TextureFormat,
    camera_buffer: wgpu::Buffer,
    scene_bind_group: wgpu::BindGroup,
    light_buffer: wgpu::Buffer,
    lighting_buffer: wgpu::Buffer,
    pipeline_opaque: wgpu::RenderPipeline,
    pipeline_transparent: wgpu::RenderPipeline,
    pipeline_sprite: wgpu::RenderPipeline,
    pipeline_sprite_glow: wgpu::RenderPipeline,
    pipeline_sprite_overlay: wgpu::RenderPipeline,
    pipeline_upscale: wgpu::RenderPipeline,
    msaa_sample_count: u32,
    // Stored to allow pipeline rebuild when MSAA count changes at runtime.
    pipeline_layout_3d: wgpu::PipelineLayout,
    _shader_3d: wgpu::ShaderModule,
    pipeline_layout_sprite: wgpu::PipelineLayout,
    _shader_sprite: wgpu::ShaderModule,
    _depth_texture: wgpu::Texture,
    depth_view: wgpu::TextureView,
    _msaa_color_texture: wgpu::Texture,
    msaa_color_view: wgpu::TextureView,
    _msaa_depth_texture: wgpu::Texture,
    msaa_depth_view: wgpu::TextureView,
    _fsr_scene_texture: wgpu::Texture,
    fsr_scene_view: wgpu::TextureView,
    fsr_bind_group_layout: wgpu::BindGroupLayout,
    fsr_sampler: wgpu::Sampler,
    fsr_bind_group: wgpu::BindGroup,
    fsr_uniform_buffer: wgpu::Buffer,
    sprite_atlas_texture: wgpu::Texture,
    sprite_atlas_bind_group: wgpu::BindGroup,
    sprite_atlas_overlay_bind_group: wgpu::BindGroup,
    box_mesh: GpuMesh,
    sphere_mesh_high: GpuMesh,
    sphere_mesh_low: GpuMesh,
    sphere_lod_mode: SphereLodMode,
    sprite_vertex_buffer: wgpu::Buffer,
    instance_3d_buffer: wgpu::Buffer,
    instance_3d_capacity_bytes: usize,
    sprite_instance_buffer: wgpu::Buffer,
    sprite_instance_capacity_bytes: usize,
    overlay_sprite_instance_buffer: wgpu::Buffer,
    overlay_sprite_instance_capacity_bytes: usize,
    ranges: Vec<GpuBatchRange>,
    sprite_count: u32,
    glow_sprite_start: u32,
    glow_sprite_count: u32,
    overlay_sprite_count: u32,
    light_count: u32,
    force_full_fbx_sphere: bool,
    camera_eye: [f32; 3],
    camera_target: [f32; 3],
    custom_mesh_slots: BTreeMap<u8, GpuMesh>,
    ray_tracing_status: SceneRayTracingStatus,
    // ── Ray-tracing acceleration structures ────────────────────────────────
    /// BLAS for unit box geometry. None when RT is unsupported.
    rt_blas_box: Option<wgpu::Blas>,
    /// BLAS for unit sphere geometry. None when RT is unsupported.
    rt_blas_sphere: Option<wgpu::Blas>,
    /// Top-level acceleration structure rebuilt every frame. None when RT is unsupported.
    rt_tlas: Option<wgpu::Tlas>,
    /// Bind group layout for group 1 (RT-only pipelines): just the TLAS binding.
    _rt_tlas_bgl: Option<wgpu::BindGroupLayout>,
    /// Bind group for the TLAS (group 1 in RT 3D pipelines).
    rt_tlas_bg: Option<wgpu::BindGroup>,
    /// RT-enabled opaque 3D pipeline (uses scene_3d_rt.wgsl).
    pipeline_opaque_rt: Option<wgpu::RenderPipeline>,
    /// RT-enabled transparent 3D pipeline.
    pipeline_transparent_rt: Option<wgpu::RenderPipeline>,
    /// CPU-side instance list rebuilt every upload_draw_frame call; fed into the TLAS.
    rt_instances: Vec<RtInstance>,
    fsr_config: FsrConfig,
    fsr_status: FsrStatus,
    surface_width: u32,
    surface_height: u32,
    // ── Shadow map resources ────────────────────────────────────────────────
    _shadow_map_texture: wgpu::Texture,
    /// Per-layer views used as depth attachment during shadow depth passes.
    shadow_map_layer_views: Vec<wgpu::TextureView>,
    _shadow_sampler: wgpu::Sampler,
    shadow_uniform_buffer: wgpu::Buffer,
    /// Per-slot uniform buffers written with each light's view-proj before its depth pass.
    shadow_pass_buffers: Vec<wgpu::Buffer>,
    shadow_pass_bind_groups: Vec<wgpu::BindGroup>,
    pipeline_shadow: wgpu::RenderPipeline,
    /// Number of shadow slots filled this frame (≤ MAX_SHADOW_LIGHTS).
    shadow_active_count: u32,
}

impl WgpuSceneRenderer {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        color_format: wgpu::TextureFormat,
        surface_width: u32,
        surface_height: u32,
        adapter_backend: wgpu::Backend,
        msaa_sample_count: u32,
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
                // binding 3: shadow map depth texture array
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: None,
                },
                // binding 4: comparison sampler for PCF shadow reads
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
                    count: None,
                },
                // binding 5: shadow uniform (light-space matrices + slot mapping)
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
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
        // ── Shadow map resources ────────────────────────────────────────────────
        let shadow_map_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("runtime-shadow-map"),
            size: wgpu::Extent3d {
                width: SHADOW_MAP_SIZE,
                height: SHADOW_MAP_SIZE,
                depth_or_array_layers: MAX_SHADOW_LIGHTS as u32,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let shadow_map_array_view = shadow_map_texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("runtime-shadow-map-array"),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });
        let shadow_map_layer_views: Vec<wgpu::TextureView> = (0..MAX_SHADOW_LIGHTS)
            .map(|i| {
                shadow_map_texture.create_view(&wgpu::TextureViewDescriptor {
                    label: Some(&format!("runtime-shadow-map-layer-{i}")),
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    base_array_layer: i as u32,
                    array_layer_count: Some(1),
                    ..Default::default()
                })
            })
            .collect();
        let shadow_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("runtime-shadow-sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            // LessEqual: returns 1.0 when sampled_depth ≤ depth_ref.
            // In sample_shadow we invert the result so shadow_sum accumulates 1.0=lit, 0.0=shadow.
            compare: Some(wgpu::CompareFunction::LessEqual),
            ..Default::default()
        });
        let shadow_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("runtime-shadow-uniform-buffer"),
            size: std::mem::size_of::<GpuShadowUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // ── Shadow pass per-slot bind group layout + resources ──────────────────
        let shadow_pass_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("runtime-shadow-pass-bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let shadow_pass_buffers: Vec<wgpu::Buffer> = (0..MAX_SHADOW_LIGHTS)
            .map(|i| {
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("runtime-shadow-pass-buf-{i}")),
                    size: std::mem::size_of::<GpuShadowPassUniform>() as u64,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            })
            .collect();
        let shadow_pass_bind_groups: Vec<wgpu::BindGroup> = shadow_pass_buffers
            .iter()
            .enumerate()
            .map(|(i, buf)| {
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(&format!("runtime-shadow-pass-bg-{i}")),
                    layout: &shadow_pass_bgl,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buf.as_entire_binding(),
                    }],
                })
            })
            .collect();

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
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&shadow_map_array_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Sampler(&shadow_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: shadow_uniform_buffer.as_entire_binding(),
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
        let shader_shadow = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("runtime-scene-shadow-shader"),
            source: wgpu::ShaderSource::Wgsl(SCENE_SHADOW_WGSL.into()),
        });
        let shader_upscale = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("runtime-scene-upscale-shader"),
            source: wgpu::ShaderSource::Wgsl(SCENE_UPSCALE_WGSL.into()),
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
        let fsr_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("runtime-scene-fsr-bgl"),
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
        let layout_upscale = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("runtime-scene-upscale-layout"),
            bind_group_layouts: &[&fsr_bgl],
            immediate_size: 0,
        });
        let layout_shadow = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("runtime-shadow-depth-layout"),
            bind_group_layouts: &[&shadow_pass_bgl],
            immediate_size: 0,
        });
        let pipeline_shadow = create_shadow_pipeline(device, &layout_shadow, &shader_shadow);

        let pipeline_opaque = create_3d_pipeline(
            device,
            &layout_3d,
            &shader_3d,
            color_format,
            Some(wgpu::BlendState::REPLACE),
            true,
            Some(wgpu::Face::Back),
            "runtime-scene-opaque-pipeline",
            msaa_sample_count,
        );
        let pipeline_transparent = create_3d_pipeline(
            device,
            &layout_3d,
            &shader_3d,
            color_format,
            Some(wgpu::BlendState::ALPHA_BLENDING),
            false,
            Some(wgpu::Face::Back),
            "runtime-scene-transparent-pipeline",
            msaa_sample_count,
        );
        let pipeline_sprite = create_sprite_pipeline(
            device,
            &layout_sprite,
            &shader_sprite,
            color_format,
            msaa_sample_count,
        );
        let pipeline_sprite_glow = create_sprite_glow_pipeline(
            device,
            &layout_sprite,
            &shader_sprite,
            color_format,
            msaa_sample_count,
        );
        let pipeline_sprite_overlay =
            create_sprite_overlay_pipeline(device, &layout_sprite, &shader_sprite, color_format);
        let pipeline_upscale =
            create_upscale_pipeline(device, &layout_upscale, &shader_upscale, color_format);
        let (depth_texture, depth_view) = create_depth_resources(
            device,
            surface_width.max(1),
            surface_height.max(1),
            "runtime-scene-depth",
        );
        let (msaa_color_texture, msaa_color_view, msaa_depth_texture, msaa_depth_view) =
            create_msaa_resources(
                device,
                color_format,
                surface_width.max(1),
                surface_height.max(1),
                msaa_sample_count,
            );
        let (fsr_scene_texture, fsr_scene_view) = create_fsr_scene_resources(
            device,
            color_format,
            surface_width.max(1),
            surface_height.max(1),
            "runtime-scene-fsr-source",
        );
        let fsr_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("runtime-scene-fsr-uniform-buffer"),
            size: std::mem::size_of::<GpuUpscaleUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let fsr_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("runtime-scene-fsr-sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });
        let fsr_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("runtime-scene-fsr-bg"),
            layout: &fsr_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&fsr_scene_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&fsr_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: fsr_uniform_buffer.as_entire_binding(),
                },
            ],
        });
        let (sprite_atlas_texture, sprite_atlas_bind_group, sprite_atlas_overlay_bind_group) =
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
        let overlay_sprite_instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("runtime-scene-overlay-sprite-instance-buffer"),
            size: std::mem::size_of::<GpuSpriteInstance>() as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // ── RT infrastructure (only when both RT features are enabled) ─────────
        let rt_infra = if supports_ray_query(device, adapter_backend) {
            Some(create_rt_infrastructure(
                device,
                queue,
                &camera_bgl,
                color_format,
                msaa_sample_count,
            ))
        } else {
            None
        };
        let (
            rt_blas_box,
            rt_blas_sphere,
            rt_tlas,
            rt_tlas_bgl_opt,
            rt_tlas_bg,
            pipeline_opaque_rt,
            pipeline_transparent_rt,
        ) = match rt_infra {
            Some(r) => (
                Some(r.blas_box),
                Some(r.blas_sphere),
                Some(r.tlas),
                Some(r.tlas_bgl),
                Some(r.tlas_bg),
                Some(r.pipeline_opaque_rt),
                Some(r.pipeline_transparent_rt),
            ),
            None => (None, None, None, None, None, None, None),
        };

        let renderer = Self {
            adapter_backend,
            surface_color_format: color_format,
            camera_buffer,
            scene_bind_group,
            light_buffer,
            lighting_buffer,
            msaa_sample_count,
            pipeline_layout_3d: layout_3d,
            _shader_3d: shader_3d,
            pipeline_layout_sprite: layout_sprite,
            _shader_sprite: shader_sprite,
            pipeline_opaque,
            pipeline_transparent,
            pipeline_sprite,
            pipeline_sprite_glow,
            pipeline_sprite_overlay,
            pipeline_upscale,
            _depth_texture: depth_texture,
            depth_view,
            _msaa_color_texture: msaa_color_texture,
            msaa_color_view,
            _msaa_depth_texture: msaa_depth_texture,
            msaa_depth_view,
            _fsr_scene_texture: fsr_scene_texture,
            fsr_scene_view,
            fsr_bind_group_layout: fsr_bgl,
            fsr_sampler,
            fsr_bind_group,
            fsr_uniform_buffer,
            sprite_atlas_texture,
            sprite_atlas_bind_group,
            sprite_atlas_overlay_bind_group,
            box_mesh,
            sphere_mesh_high,
            sphere_mesh_low,
            sphere_lod_mode: SphereLodMode::High,
            sprite_vertex_buffer,
            instance_3d_buffer,
            instance_3d_capacity_bytes: std::mem::size_of::<GpuInstance3d>(),
            sprite_instance_buffer,
            sprite_instance_capacity_bytes: std::mem::size_of::<GpuSpriteInstance>(),
            overlay_sprite_instance_buffer,
            overlay_sprite_instance_capacity_bytes: std::mem::size_of::<GpuSpriteInstance>(),
            ranges: Vec::new(),
            sprite_count: 0,
            glow_sprite_start: 0,
            glow_sprite_count: 0,
            overlay_sprite_count: 0,
            light_count: 0,
            force_full_fbx_sphere: false,
            camera_eye: [0.0, 12.0, 36.0],
            camera_target: [0.0, 0.0, 0.0],
            custom_mesh_slots: BTreeMap::new(),
            ray_tracing_status: resolve_rt_status(
                RayTracingMode::Auto,
                supports_ray_query(device, adapter_backend),
            ),
            rt_blas_box,
            rt_blas_sphere,
            rt_tlas,
            _rt_tlas_bgl: rt_tlas_bgl_opt,
            rt_tlas_bg,
            pipeline_opaque_rt,
            pipeline_transparent_rt,
            rt_instances: Vec::new(),
            fsr_config: FsrConfig::default(),
            fsr_status: resolve_fsr_status(FsrConfig::default(), adapter_backend),
            surface_width: surface_width.max(1),
            surface_height: surface_height.max(1),
            _shadow_map_texture: shadow_map_texture,
            shadow_map_layer_views,
            _shadow_sampler: shadow_sampler,
            shadow_uniform_buffer,
            shadow_pass_buffers,
            shadow_pass_bind_groups,
            pipeline_shadow,
            shadow_active_count: 0,
        };
        renderer.write_camera_uniform(queue, surface_width.max(1), surface_height.max(1));
        renderer.write_lighting_uniform(queue, 0);
        renderer.write_fsr_uniform(queue);
        renderer
    }

    pub fn resize(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, width: u32, height: u32) {
        self.surface_width = width.max(1);
        self.surface_height = height.max(1);
        self.write_camera_uniform(queue, self.surface_width, self.surface_height);
        let (depth_texture, depth_view) = create_depth_resources(
            device,
            self.surface_width,
            self.surface_height,
            "runtime-scene-depth",
        );
        self._depth_texture = depth_texture;
        self.depth_view = depth_view;
        let (msaa_ct, msaa_cv, msaa_dt, msaa_dv) = create_msaa_resources(
            device,
            self.surface_color_format,
            self.surface_width,
            self.surface_height,
            self.msaa_sample_count,
        );
        self._msaa_color_texture = msaa_ct;
        self.msaa_color_view = msaa_cv;
        self._msaa_depth_texture = msaa_dt;
        self.msaa_depth_view = msaa_dv;
        let (fsr_scene_texture, fsr_scene_view) = create_fsr_scene_resources(
            device,
            self.surface_color_format,
            self.surface_width,
            self.surface_height,
            "runtime-scene-fsr-source",
        );
        self._fsr_scene_texture = fsr_scene_texture;
        self.fsr_scene_view = fsr_scene_view;
        self.fsr_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("runtime-scene-fsr-bg"),
            layout: &self.fsr_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&self.fsr_scene_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.fsr_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.fsr_uniform_buffer.as_entire_binding(),
                },
            ],
        });
        self.write_fsr_uniform(queue);
    }

    pub fn msaa_sample_count(&self) -> u32 {
        self.msaa_sample_count
    }

    /// Change MSAA sample count at runtime. Rebuilds MSAA textures and the four affected pipelines.
    /// Valid values: 1 (off), 2, or 4. Other values are clamped to the nearest supported count.
    pub fn set_msaa_sample_count(&mut self, device: &wgpu::Device, count: u32) {
        let count = match count {
            0 | 1 => 1,
            2 | 3 => 2,
            _ => 4,
        };
        if self.msaa_sample_count == count {
            return;
        }
        self.msaa_sample_count = count;
        let (ct, cv, dt, dv) = create_msaa_resources(
            device,
            self.surface_color_format,
            self.surface_width,
            self.surface_height,
            count,
        );
        self._msaa_color_texture = ct;
        self.msaa_color_view = cv;
        self._msaa_depth_texture = dt;
        self.msaa_depth_view = dv;
        self.pipeline_opaque = create_3d_pipeline(
            device,
            &self.pipeline_layout_3d,
            &self._shader_3d,
            self.surface_color_format,
            Some(wgpu::BlendState::REPLACE),
            true,
            Some(wgpu::Face::Back),
            "runtime-scene-opaque-pipeline",
            count,
        );
        self.pipeline_transparent = create_3d_pipeline(
            device,
            &self.pipeline_layout_3d,
            &self._shader_3d,
            self.surface_color_format,
            Some(wgpu::BlendState::ALPHA_BLENDING),
            false,
            Some(wgpu::Face::Back),
            "runtime-scene-transparent-pipeline",
            count,
        );
        self.pipeline_sprite = create_sprite_pipeline(
            device,
            &self.pipeline_layout_sprite,
            &self._shader_sprite,
            self.surface_color_format,
            count,
        );
        self.pipeline_sprite_glow = create_sprite_glow_pipeline(
            device,
            &self.pipeline_layout_sprite,
            &self._shader_sprite,
            self.surface_color_format,
            count,
        );
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

    /// Bind a 2D sprite source (`.png` / `.svg`) into one atlas slot.
    ///
    /// Slots in `[64..127]` are reserved for external image textures.
    pub fn bind_sprite_texture_slot_from_path(
        &mut self,
        queue: &wgpu::Queue,
        slot: u16,
        path: &Path,
    ) -> Result<(), String> {
        if slot >= TEXT_GLYPH_SLOT_BASE {
            return Err(format!(
                "slot {} is reserved for text glyph atlas lanes (>= {})",
                slot, TEXT_GLYPH_SLOT_BASE
            ));
        }
        let pixels =
            decode_sprite_texture_to_rgba(path, SPRITE_ATLAS_TILE_SIZE, SPRITE_ATLAS_TILE_SIZE)?;
        self.bind_sprite_texture_slot_from_rgba(queue, slot, &pixels)
    }

    /// Bind raw RGBA8 pixels into a sprite atlas slot.
    pub fn bind_sprite_texture_slot_from_rgba(
        &mut self,
        queue: &wgpu::Queue,
        slot: u16,
        rgba_pixels: &[u8],
    ) -> Result<(), String> {
        let expected = (SPRITE_ATLAS_TILE_SIZE * SPRITE_ATLAS_TILE_SIZE * 4) as usize;
        if rgba_pixels.len() != expected {
            return Err(format!(
                "invalid RGBA payload size for sprite atlas slot {}: got {}, expected {}",
                slot,
                rgba_pixels.len(),
                expected
            ));
        }

        let (origin_x, origin_y) = sprite_slot_origin_px(slot);
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.sprite_atlas_texture,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: origin_x,
                    y: origin_y,
                    z: 0,
                },
                aspect: wgpu::TextureAspect::All,
            },
            rgba_pixels,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(SPRITE_ATLAS_TILE_SIZE * 4),
                rows_per_image: Some(SPRITE_ATLAS_TILE_SIZE),
            },
            wgpu::Extent3d {
                width: SPRITE_ATLAS_TILE_SIZE,
                height: SPRITE_ATLAS_TILE_SIZE,
                depth_or_array_layers: 1,
            },
        );
        Ok(())
    }

    pub fn upload_draw_frame(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        draw: &RuntimeDrawFrame,
    ) -> WgpuSceneRendererUploadStats {
        let plan = build_upload_plan(draw);
        let required_3d_bytes = plan.instances_3d.len() * std::mem::size_of::<GpuInstance3d>();
        // Regular and glow sprites share one buffer: [regular..., glow...]
        let total_sprite_count = plan.sprites.len() + plan.glow_sprites.len();
        let required_sprite_bytes = total_sprite_count * std::mem::size_of::<GpuSpriteInstance>();

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
        let glow_byte_offset =
            (plan.sprites.len() * std::mem::size_of::<GpuSpriteInstance>()) as u64;
        if !plan.glow_sprites.is_empty() {
            queue.write_buffer(
                &self.sprite_instance_buffer,
                glow_byte_offset,
                bytemuck::cast_slice(plan.glow_sprites.as_slice()),
            );
        }

        self.ranges = plan.ranges;
        self.sprite_count = plan.sprites.len() as u32;
        self.glow_sprite_start = plan.sprites.len() as u32;
        self.glow_sprite_count = plan.glow_sprites.len() as u32;

        // Build CPU-side RT instance list for TLAS population this frame.
        if self.ray_tracing_status.active && self.rt_tlas.is_some() {
            let cap = self.ray_tracing_status.rt_dynamic_cap as usize;
            self.rt_instances.clear();
            'outer: for batch in &draw.opaque_batches {
                let blas_kind = match batch.key.primitive_code {
                    1 => RtBlasKind::Box,
                    _ => RtBlasKind::Sphere, // Sphere proxy for built-in + custom meshes.
                };
                for instance in &batch.instances {
                    if self.rt_instances.len() >= cap {
                        break 'outer;
                    }
                    self.rt_instances.push(RtInstance {
                        transform: col_major_4x4_to_row_major_3x4(
                            instance.model_cols[0],
                            instance.model_cols[1],
                            instance.model_cols[2],
                            instance.model_cols[3],
                        ),
                        blas_kind,
                    });
                }
            }
        } else {
            self.rt_instances.clear();
        }

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

    /// Populates the TLAS with the current frame's opaque instances and records the build
    /// command into `encoder`. Must be called before `encode()` when RT is active.
    pub fn build_rt_acceleration_structures(&mut self, encoder: &mut wgpu::CommandEncoder) {
        if !self.ray_tracing_status.active {
            return;
        }
        if self.rt_tlas.is_none() || self.rt_blas_box.is_none() || self.rt_blas_sphere.is_none() {
            return;
        }

        let cap = self.ray_tracing_status.rt_dynamic_cap as usize;
        let inst_count = self.rt_instances.len().min(cap);

        // Build TlasInstances from CPU-side data; TlasInstance::new clones the BLAS handle
        // so we can release the borrow on rt_blas_* before mutating rt_tlas.
        let tlas_instances: Vec<Option<wgpu::TlasInstance>> = self.rt_instances[..inst_count]
            .iter()
            .map(|inst| {
                let blas = match inst.blas_kind {
                    RtBlasKind::Sphere => self.rt_blas_sphere.as_ref().unwrap(),
                    RtBlasKind::Box => self.rt_blas_box.as_ref().unwrap(),
                };
                Some(wgpu::TlasInstance::new(blas, inst.transform, 0, 0xff))
            })
            .collect();

        let tlas = self.rt_tlas.as_mut().unwrap();
        let tlas_len = tlas.get().len();

        for (i, inst) in tlas_instances.into_iter().enumerate() {
            tlas[i] = inst;
        }
        // Clear any slots beyond the current instance count.
        for i in inst_count..tlas_len {
            tlas[i] = None;
        }

        encoder.build_acceleration_structures(
            std::iter::empty::<&wgpu::BlasBuildEntry>(),
            std::iter::once(&*tlas),
        );
    }

    /// Upload console/UI overlay sprites that must be rendered at native surface resolution.
    ///
    /// This bypasses the FSR render-scale path by being drawn in a separate pass after upscale.
    pub fn upload_overlay_sprites(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        sprites: &[SpriteInstance],
    ) {
        let mut overlay = Vec::with_capacity(sprites.len());
        for sprite in sprites {
            overlay.push(GpuSpriteInstance {
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

        let required_bytes = overlay.len() * std::mem::size_of::<GpuSpriteInstance>();
        ensure_buffer_capacity(
            device,
            &mut self.overlay_sprite_instance_buffer,
            &mut self.overlay_sprite_instance_capacity_bytes,
            required_bytes.max(std::mem::size_of::<GpuSpriteInstance>()),
            wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            "runtime-scene-overlay-sprite-instance-buffer",
        );

        if !overlay.is_empty() {
            queue.write_buffer(
                &self.overlay_sprite_instance_buffer,
                0,
                bytemuck::cast_slice(overlay.as_slice()),
            );
        }
        self.overlay_sprite_count = overlay.len() as u32;
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
    pub fn set_fsr_config(&mut self, queue: &wgpu::Queue, config: FsrConfig) {
        self.fsr_config = config;
        self.fsr_status = resolve_fsr_status(config, self.adapter_backend);
        self.write_fsr_uniform(queue);
    }

    /// Current effective FSR status snapshot.
    pub fn fsr_status(&self) -> FsrStatus {
        self.fsr_status.clone()
    }

    /// Current camera eye position (world space).
    pub fn camera_eye(&self) -> [f32; 3] {
        self.camera_eye
    }

    /// Project a world-space position to NDC using the renderer's current camera.
    /// Returns `Some([ndc_x, ndc_y, ndc_depth])` when the point is in front of the camera,
    /// or `None` when it is behind (clip.w ≤ 0).
    pub fn world_to_ndc(&self, world_pos: [f32; 3]) -> Option<[f32; 3]> {
        use nalgebra::{Isometry3, Matrix4, Perspective3, Point3, Vector3};
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
        let p = Point3::new(world_pos[0], world_pos[1], world_pos[2]);
        let clip = view_proj * p.to_homogeneous();
        if clip.w <= 0.0 {
            return None;
        }
        Some([clip.x / clip.w, clip.y / clip.w, clip.z / clip.w])
    }

    /// Compute the NDC half-size for a world-radius sphere at clip.w distance.
    /// Useful for sizing light glow sprites to be perspective-correct.
    pub fn world_radius_to_ndc_half_size(&self, world_radius: f32, clip_w: f32) -> f32 {
        let fov_y = 60f32.to_radians();
        let focal = 1.0 / (fov_y * 0.5).tan();
        (world_radius / clip_w.max(0.01)) * focal
    }

    pub fn encode(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        clear: wgpu::Color,
    ) {
        // When RT is active the fragment shader traces shadow rays, so PCF shadow map
        // passes are skipped.  The forward path still needs them.
        if !self.ray_tracing_status.active {
            self.encode_shadow_passes(encoder);
        }

        let (source_width, source_height, _, _, fsr_scene_active) = self.compute_fsr_source_dims();
        let scene_target = if fsr_scene_active {
            &self.fsr_scene_view
        } else {
            target
        };
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("runtime-scene-pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &self.msaa_color_view,
                depth_slice: None,
                resolve_target: Some(scene_target),
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(clear),
                    store: wgpu::StoreOp::Discard,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &self.msaa_depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Discard,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        if fsr_scene_active {
            pass.set_viewport(
                0.0,
                0.0,
                source_width as f32,
                source_height as f32,
                0.0,
                1.0,
            );
            pass.set_scissor_rect(0, 0, source_width, source_height);
        }

        if !self.ranges.is_empty() {
            pass.set_bind_group(0, &self.scene_bind_group, &[]);

            // Select RT vs standard pipelines; bind the TLAS as group 1 when RT is active.
            let rt_active = self.ray_tracing_status.active
                && self.pipeline_opaque_rt.is_some()
                && self.rt_tlas_bg.is_some();

            if rt_active {
                pass.set_bind_group(1, self.rt_tlas_bg.as_ref().unwrap(), &[]);
            }

            let pipeline_opaque = if rt_active {
                self.pipeline_opaque_rt.as_ref().unwrap()
            } else {
                &self.pipeline_opaque
            };
            let pipeline_transparent = if rt_active {
                self.pipeline_transparent_rt.as_ref().unwrap()
            } else {
                &self.pipeline_transparent
            };

            pass.set_pipeline(pipeline_opaque);
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

            pass.set_pipeline(pipeline_transparent);
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

        if self.sprite_count > 0 || self.glow_sprite_count > 0 {
            pass.set_bind_group(0, &self.sprite_atlas_bind_group, &[]);
            pass.set_bind_group(1, &self.scene_bind_group, &[]);
            pass.set_vertex_buffer(0, self.sprite_vertex_buffer.slice(..));
            let total_sprite_bytes = ((self.sprite_count + self.glow_sprite_count) as usize
                * std::mem::size_of::<GpuSpriteInstance>())
                as u64;
            pass.set_vertex_buffer(1, self.sprite_instance_buffer.slice(0..total_sprite_bytes));

            if self.sprite_count > 0 {
                pass.set_pipeline(&self.pipeline_sprite);
                pass.draw(0..6, 0..self.sprite_count);
            }

            if self.glow_sprite_count > 0 {
                pass.set_pipeline(&self.pipeline_sprite_glow);
                let glow_end = self.glow_sprite_start + self.glow_sprite_count;
                pass.draw(0..6, self.glow_sprite_start..glow_end);
            }
        }
        drop(pass);

        if fsr_scene_active {
            let mut upscale = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("runtime-scene-upscale-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(clear),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            upscale.set_pipeline(&self.pipeline_upscale);
            upscale.set_bind_group(0, &self.fsr_bind_group, &[]);
            upscale.draw(0..3, 0..1);
        }
    }

    /// Render native-resolution HUD/console overlay sprites after scene and optional FSR upscale.
    pub fn encode_overlay_sprites(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
    ) {
        if self.overlay_sprite_count == 0 {
            return;
        }
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("runtime-scene-overlay-pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&self.pipeline_sprite_overlay);
        pass.set_bind_group(0, &self.sprite_atlas_overlay_bind_group, &[]);
        pass.set_bind_group(1, &self.scene_bind_group, &[]);
        pass.set_vertex_buffer(0, self.sprite_vertex_buffer.slice(..));
        let sprite_bytes =
            (self.overlay_sprite_count as usize * std::mem::size_of::<GpuSpriteInstance>()) as u64;
        pass.set_vertex_buffer(
            1,
            self.overlay_sprite_instance_buffer.slice(0..sprite_bytes),
        );
        pass.draw(0..6, 0..self.overlay_sprite_count);
    }

    /// Render one depth pass per active shadow-casting light into the shadow map array.
    /// Must be called before `encode()` so shadow maps are ready for the scene pass.
    fn encode_shadow_passes(&self, encoder: &mut wgpu::CommandEncoder) {
        for slot in 0..self.shadow_active_count as usize {
            let label = format!("runtime-shadow-depth-pass-{slot}");
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some(&label),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.shadow_map_layer_views[slot],
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
            pass.set_pipeline(&self.pipeline_shadow);
            pass.set_bind_group(0, &self.shadow_pass_bind_groups[slot], &[]);
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
        // First pass: assign shadow map slots to shadow-casting spot lights.
        let mut shadow_slots = vec![-1i32; selected];
        let mut shadow_uniform = GpuShadowUniform::default();
        let mut slot = 0usize;
        for (i, light) in lights.iter().take(selected).enumerate() {
            if light.casts_shadow
                && matches!(light.kind, SceneLightKind::Spot)
                && slot < MAX_SHADOW_LIGHTS
            {
                shadow_slots[i] = slot as i32;
                shadow_uniform.shadow_light_indices[slot] = i as i32;
                shadow_uniform.light_view_proj[slot] = compute_spotlight_view_proj(light);
                // Upload per-slot pass uniform so the shadow depth pass VS has the matrix.
                queue.write_buffer(
                    &self.shadow_pass_buffers[slot],
                    0,
                    bytemuck::bytes_of(&GpuShadowPassUniform {
                        light_view_proj: shadow_uniform.light_view_proj[slot],
                    }),
                );
                slot += 1;
            }
        }
        shadow_uniform.shadow_count = slot as u32;
        self.shadow_active_count = slot as u32;

        queue.write_buffer(
            &self.shadow_uniform_buffer,
            0,
            bytemuck::bytes_of(&shadow_uniform),
        );

        // Second pass: build GPU light array with shadow slot encoded in shadow.y.
        let mut gpu_lights = vec![GpuLight::default(); selected.max(1)];
        for (index, light) in lights.iter().take(selected).enumerate() {
            gpu_lights[index] = gpu_light_from_scene(light, shadow_slots[index]);
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

    fn write_fsr_uniform(&self, queue: &wgpu::Queue) {
        let (source_width, source_height, source_uv_x, source_uv_y, active) =
            self.compute_fsr_source_dims();
        let uniform = GpuUpscaleUniform {
            inv_source_size: [1.0 / source_width as f32, 1.0 / source_height as f32],
            source_uv_scale: [source_uv_x, source_uv_y],
            sharpness: if active {
                self.fsr_status.sharpness
            } else {
                0.0
            },
            _pad: [0.0; 3],
        };
        queue.write_buffer(&self.fsr_uniform_buffer, 0, bytemuck::bytes_of(&uniform));
    }

    fn compute_fsr_source_dims(&self) -> (u32, u32, f32, f32, bool) {
        let active = self.fsr_status.active && self.fsr_status.render_scale < 0.999;
        if !active {
            return (
                self.surface_width.max(1),
                self.surface_height.max(1),
                1.0,
                1.0,
                false,
            );
        }
        let scaled_w = ((self.surface_width as f32 * self.fsr_status.render_scale)
            .round()
            .max(1.0)) as u32;
        let scaled_h = ((self.surface_height as f32 * self.fsr_status.render_scale)
            .round()
            .max(1.0)) as u32;
        let uv_x = scaled_w as f32 / self.surface_width.max(1) as f32;
        let uv_y = scaled_h as f32 / self.surface_height.max(1) as f32;
        (scaled_w, scaled_h, uv_x, uv_y, true)
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
        let gpu = GpuSpriteInstance {
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
        };
        // LightGlow sprites go into the separate additive-blend bucket.
        if sprite.kind == SpriteKind::LightGlow {
            plan.glow_sprites.push(gpu);
        } else {
            plan.sprites.push(gpu);
        }
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

fn gpu_light_from_scene(light: &SceneLight, shadow_slot: i32) -> GpuLight {
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
        color_intensity: [
            color[0],
            color[1],
            color[2],
            if light.enabled {
                light.intensity.max(0.0)
            } else {
                0.0
            },
        ],
        params: [
            light.range.max(0.05),
            outer,
            light.softness.clamp(0.0, 1.0),
            light.specular_strength.clamp(0.0, 8.0),
        ],
        // shadow.y encodes shadow slot: -1.0 = no shadow map, 0..3 = active slot.
        shadow: [f32::from(light.casts_shadow), shadow_slot as f32, 0.0, 0.0],
    }
}

/// Compute a view-proj matrix for a spotlight's shadow depth pass.
/// Returns column-major f32[16] in nalgebra layout (same as `write_camera_uniform`).
fn compute_spotlight_view_proj(light: &SceneLight) -> [f32; 16] {
    let pos = Point3::new(light.position[0], light.position[1], light.position[2]);
    let dir = Vector3::new(light.direction[0], light.direction[1], light.direction[2]).normalize();
    let up = if dir.y.abs() < 0.99 {
        Vector3::y()
    } else {
        Vector3::x()
    };
    let view = Isometry3::look_at_rh(&pos, &Point3::from(pos + dir), &up);
    // Extend FOV slightly beyond outer cone to avoid hard shadow map edges.
    let fov_y = ((light.outer_cone_deg * 2.0 + 6.0) as f32)
        .clamp(10.0, 170.0)
        .to_radians();
    let proj = Perspective3::new(1.0, fov_y, 0.5, light.range.max(1.0) * 1.1);
    let view_proj: Matrix4<f32> = proj.to_homogeneous() * view.to_homogeneous();
    let mut out = [0f32; 16];
    out.copy_from_slice(view_proj.as_slice());
    out
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
    sample_count: u32,
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
        multisample: wgpu::MultisampleState {
            count: sample_count,
            ..Default::default()
        },
        multiview_mask: None,
        cache: None,
    })
}

fn create_sprite_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
    color_format: wgpu::TextureFormat,
    sample_count: u32,
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
        multisample: wgpu::MultisampleState {
            count: sample_count,
            ..Default::default()
        },
        multiview_mask: None,
        cache: None,
    })
}

/// Additive-blend sprite pipeline for light glow billboards.
/// Result = src_color * src_alpha + dst_color (premultiplied-alpha additive).
fn create_sprite_glow_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
    color_format: wgpu::TextureFormat,
    sample_count: u32,
) -> wgpu::RenderPipeline {
    const ADDITIVE: wgpu::BlendState = wgpu::BlendState {
        color: wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::SrcAlpha,
            dst_factor: wgpu::BlendFactor::One,
            operation: wgpu::BlendOperation::Add,
        },
        alpha: wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::One,
            dst_factor: wgpu::BlendFactor::One,
            operation: wgpu::BlendOperation::Add,
        },
    };
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("runtime-scene-sprite-glow-pipeline"),
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
                blend: Some(ADDITIVE),
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
        multisample: wgpu::MultisampleState {
            count: sample_count,
            ..Default::default()
        },
        multiview_mask: None,
        cache: None,
    })
}

fn create_sprite_overlay_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
    color_format: wgpu::TextureFormat,
) -> wgpu::RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("runtime-scene-sprite-overlay-pipeline"),
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
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    })
}

fn create_upscale_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
    color_format: wgpu::TextureFormat,
) -> wgpu::RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("runtime-scene-upscale-pipeline"),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: Some("vs_main"),
            buffers: &[],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: shader,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format: color_format,
                blend: Some(wgpu::BlendState::REPLACE),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            ..Default::default()
        },
        depth_stencil: None,
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
        SpriteKind::LightGlow => 4,
    }
}

#[inline]
fn sprite_kind_atlas_row(kind: SpriteKind) -> u32 {
    match kind {
        SpriteKind::Generic => 0,
        SpriteKind::Hud => 1,
        SpriteKind::Camera => 2,
        SpriteKind::Terrain => 3,
        // LightGlow is fully procedural — atlas UV is unused; map to row 0.
        SpriteKind::LightGlow => 0,
    }
}

#[inline]
fn sprite_kind_atlas_rect(kind: SpriteKind, texture_slot: u16) -> [f32; 4] {
    const ATLAS_COLS: u16 = SPRITE_ATLAS_GRID_DIM as u16;
    const INV_ATLAS: f32 = 1.0 / SPRITE_ATLAS_GRID_DIM as f32;
    let (col, row) = if texture_slot >= EXTERNAL_IMAGE_SLOT_BASE {
        (
            texture_slot % ATLAS_COLS,
            (texture_slot / ATLAS_COLS).min(ATLAS_COLS - 1),
        )
    } else {
        (
            texture_slot % ATLAS_COLS,
            sprite_kind_atlas_row(kind) as u16,
        )
    };
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

fn create_fsr_scene_resources(
    device: &wgpu::Device,
    color_format: wgpu::TextureFormat,
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
        format: color_format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

fn create_msaa_resources(
    device: &wgpu::Device,
    color_format: wgpu::TextureFormat,
    width: u32,
    height: u32,
    sample_count: u32,
) -> (
    wgpu::Texture,
    wgpu::TextureView,
    wgpu::Texture,
    wgpu::TextureView,
) {
    let color_tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("runtime-scene-msaa-color"),
        size: wgpu::Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count,
        dimension: wgpu::TextureDimension::D2,
        format: color_format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let color_view = color_tex.create_view(&wgpu::TextureViewDescriptor::default());
    let depth_tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("runtime-scene-msaa-depth"),
        size: wgpu::Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count,
        dimension: wgpu::TextureDimension::D2,
        format: DEPTH_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let depth_view = depth_tex.create_view(&wgpu::TextureViewDescriptor::default());
    (color_tex, color_view, depth_tex, depth_view)
}

fn create_default_sprite_atlas_resources(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    bind_group_layout: &wgpu::BindGroupLayout,
) -> (wgpu::Texture, wgpu::BindGroup, wgpu::BindGroup) {
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
    let sampler_linear = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("runtime-scene-sprite-atlas-sampler-linear"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::MipmapFilterMode::Nearest,
        ..Default::default()
    });
    let sampler_nearest = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("runtime-scene-sprite-atlas-sampler-nearest"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Nearest,
        min_filter: wgpu::FilterMode::Nearest,
        mipmap_filter: wgpu::MipmapFilterMode::Nearest,
        ..Default::default()
    });

    let bind_group_linear = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("runtime-scene-sprite-atlas-bg-linear"),
        layout: bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&sampler_linear),
            },
        ],
    });

    let bind_group_nearest = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("runtime-scene-sprite-atlas-bg-nearest"),
        layout: bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&sampler_nearest),
            },
        ],
    });
    (texture, bind_group_linear, bind_group_nearest)
}

#[inline]
fn sprite_slot_origin_px(slot: u16) -> (u32, u32) {
    let total_slots = (SPRITE_ATLAS_GRID_DIM * SPRITE_ATLAS_GRID_DIM) as u16;
    let normalized = slot % total_slots.max(1);
    let col = (normalized % SPRITE_ATLAS_GRID_DIM as u16) as u32;
    let row = (normalized / SPRITE_ATLAS_GRID_DIM as u16) as u32;
    (col * SPRITE_ATLAS_TILE_SIZE, row * SPRITE_ATLAS_TILE_SIZE)
}

fn build_default_sprite_atlas_pixels(width: u32, height: u32) -> Vec<u8> {
    let mut pixels = vec![0u8; (width * height * 4) as usize];
    let tile = SPRITE_ATLAS_TILE_SIZE.max(1);

    for y in 0..height {
        for x in 0..width {
            let tile_x = (x / tile).min(SPRITE_ATLAS_GRID_DIM - 1);
            let tile_y = (y / tile).min(SPRITE_ATLAS_GRID_DIM - 1);
            let slot = (tile_y * SPRITE_ATLAS_GRID_DIM + tile_x) as u16;
            let u = (x % tile) as f32 / (tile - 1) as f32;
            let v = (y % tile) as f32 / (tile - 1) as f32;
            let mut rgba = atlas_tile_rgba(tile_x, tile_y, u, v);
            if let Some(glyph_rgba) =
                atlas_ascii_glyph_rgba(slot, (x % tile) as u32, (y % tile) as u32, tile)
            {
                rgba = glyph_rgba;
            }
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

fn atlas_ascii_glyph_rgba(slot: u16, px: u32, py: u32, tile: u32) -> Option<[u8; 4]> {
    if slot < TEXT_GLYPH_SLOT_BASE {
        return None;
    }
    let glyph_index = slot - TEXT_GLYPH_SLOT_BASE;
    if glyph_index >= 95 {
        return Some([0, 0, 0, 0]);
    }
    let ch = (glyph_index as u8 + 32) as char;
    let Some(bitmap) = BASIC_FONTS.get(ch) else {
        return Some([0, 0, 0, 0]);
    };
    let gx = ((px as usize * 8) / tile.max(1) as usize).min(7);
    let gy = ((py as usize * 8) / tile.max(1) as usize).min(7);
    // Sprite UV space samples glyph rows upside-down relative to font8x8 row order.
    // Flip only glyph Y here so CLI text is upright without changing other sprite lanes.
    let row = bitmap[7 - gy];
    let bit = (row >> gx) & 1;
    if bit == 0 {
        return Some([0, 0, 0, 0]);
    }
    // Soft edge to avoid aggressive aliasing at small sprite sizes.
    let edge = ((px % (tile / 8).max(1)) == 0) || ((py % (tile / 8).max(1)) == 0);
    if edge {
        Some([204, 255, 204, 220])
    } else {
        Some([232, 255, 232, 255])
    }
}

const SCENE_3D_WGSL: &str = include_str!("scene_3d.wgsl");

const SCENE_3D_RT_WGSL: &str = include_str!("scene_3d_rt.wgsl");

const SCENE_SPRITE_WGSL: &str = include_str!("scene_sprite.wgsl");

const SCENE_UPSCALE_WGSL: &str = include_str!("scene_upscale.wgsl");

/// Depth-only vertex shader for the shadow map pass.
/// Each draw uses the per-slot uniform (group 0, binding 0) containing the light view-proj.
const SCENE_SHADOW_WGSL: &str = include_str!("scene_shadow.wgsl");

// ── RT infrastructure ─────────────────────────────────────────────────────────

struct RtInfrastructure {
    blas_box: wgpu::Blas,
    blas_sphere: wgpu::Blas,
    tlas: wgpu::Tlas,
    tlas_bgl: wgpu::BindGroupLayout,
    tlas_bg: wgpu::BindGroup,
    pipeline_opaque_rt: wgpu::RenderPipeline,
    pipeline_transparent_rt: wgpu::RenderPipeline,
}

/// Converts a column-major 4×4 model matrix (stored as 4 column vecs) to a
/// row-major 3×4 affine matrix required by `wgpu::TlasInstance`.
fn col_major_4x4_to_row_major_3x4(
    c0: [f32; 4],
    c1: [f32; 4],
    c2: [f32; 4],
    c3: [f32; 4],
) -> [f32; 12] {
    [
        c0[0], c1[0], c2[0], c3[0], // row 0
        c0[1], c1[1], c2[1], c3[1], // row 1
        c0[2], c1[2], c2[2], c3[2], // row 2
    ]
}

/// Creates vertex + index buffers with `BLAS_INPUT` usage for the unit box geometry.
/// Returns (vertex_buffer, index_buffer).
fn create_rt_blas_buffers_box(device: &wgpu::Device) -> (wgpu::Buffer, wgpu::Buffer) {
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
        0, 1, 2, 2, 3, 0, 4, 6, 5, 6, 4, 7, 0, 4, 5, 5, 1, 0, 3, 2, 6, 6, 7, 3, 1, 5, 6, 6, 2, 1,
        0, 3, 7, 7, 4, 0,
    ];
    let vb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("rt-blas-box-vb"),
        contents: bytemuck::cast_slice(&vertices),
        usage: wgpu::BufferUsages::BLAS_INPUT,
    });
    let ib = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("rt-blas-box-ib"),
        contents: bytemuck::cast_slice(&indices),
        usage: wgpu::BufferUsages::BLAS_INPUT,
    });
    (vb, ib)
}

/// Creates vertex + index buffers with `BLAS_INPUT` usage for a subdivided icosphere proxy.
/// One subdivision of an icosahedron → 42 vertices, 80 triangles at radius 0.5.
/// Matches the FBX render mesh (normalized to [-0.5, 0.5] unit box).
/// Returns (vertex_buffer, index_buffer).
fn create_rt_blas_buffers_sphere(device: &wgpu::Device) -> (wgpu::Buffer, wgpu::Buffer) {
    use std::collections::HashMap;

    let t = (1.0 + 5.0_f32.sqrt()) * 0.5;
    let base_verts: [[f32; 3]; 12] = [
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
    let base_tris: [[u16; 3]; 20] = [
        [0, 11, 5],
        [0, 5, 1],
        [0, 1, 7],
        [0, 7, 10],
        [0, 10, 11],
        [1, 5, 9],
        [5, 11, 4],
        [11, 10, 2],
        [10, 7, 6],
        [7, 1, 8],
        [3, 9, 4],
        [3, 4, 2],
        [3, 2, 6],
        [3, 6, 8],
        [3, 8, 9],
        [4, 9, 5],
        [2, 4, 11],
        [6, 2, 10],
        [8, 6, 7],
        [9, 8, 1],
    ];

    // Normalize base vertices to radius 0.5.
    let mut verts: Vec<[f32; 3]> = base_verts
        .iter()
        .map(|p| {
            let len = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt().max(1e-6);
            [p[0] / len * 0.5, p[1] / len * 0.5, p[2] / len * 0.5]
        })
        .collect();
    let mut tris: Vec<[u16; 3]> = base_tris.to_vec();

    // One subdivision pass: split each triangle into 4 sub-triangles.
    let mut midpoint_cache: HashMap<(u16, u16), u16> = HashMap::new();
    let mut get_mid = |a: u16, b: u16, vs: &mut Vec<[f32; 3]>| -> u16 {
        let key = if a < b { (a, b) } else { (b, a) };
        if let Some(&idx) = midpoint_cache.get(&key) {
            return idx;
        }
        let pa = vs[a as usize];
        let pb = vs[b as usize];
        let mid = [
            (pa[0] + pb[0]) * 0.5,
            (pa[1] + pb[1]) * 0.5,
            (pa[2] + pb[2]) * 0.5,
        ];
        let len = (mid[0] * mid[0] + mid[1] * mid[1] + mid[2] * mid[2])
            .sqrt()
            .max(1e-6);
        let idx = vs.len() as u16;
        vs.push([mid[0] / len * 0.5, mid[1] / len * 0.5, mid[2] / len * 0.5]);
        midpoint_cache.insert(key, idx);
        idx
    };

    let old_tris = std::mem::take(&mut tris);
    for tri in &old_tris {
        let a = tri[0];
        let b = tri[1];
        let c = tri[2];
        let ab = get_mid(a, b, &mut verts);
        let bc = get_mid(b, c, &mut verts);
        let ca = get_mid(c, a, &mut verts);
        tris.push([a, ab, ca]);
        tris.push([b, bc, ab]);
        tris.push([c, ca, bc]);
        tris.push([ab, bc, ca]);
    }

    let vertices: Vec<GpuVertex3d> = verts.iter().map(|p| GpuVertex3d { position: *p }).collect();
    let indices: Vec<u16> = tris.iter().flat_map(|t| t.iter().copied()).collect();
    let vb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("rt-blas-sphere-vb"),
        contents: bytemuck::cast_slice(&vertices),
        usage: wgpu::BufferUsages::BLAS_INPUT,
    });
    let ib = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("rt-blas-sphere-ib"),
        contents: bytemuck::cast_slice(&indices),
        usage: wgpu::BufferUsages::BLAS_INPUT,
    });
    (vb, ib)
}

fn create_rt_infrastructure(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    camera_bgl: &wgpu::BindGroupLayout,
    color_format: wgpu::TextureFormat,
    msaa_sample_count: u32,
) -> RtInfrastructure {
    // Bind group layout for group 1: just the TLAS.
    let tlas_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("runtime-rt-tlas-bgl"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::AccelerationStructure {
                vertex_return: false,
            },
            count: None,
        }],
    });

    // RT pipeline layout: group 0 = scene data, group 1 = TLAS.
    let layout_3d_rt = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("runtime-scene-3d-rt-layout"),
        bind_group_layouts: &[camera_bgl, &tlas_bgl],
        immediate_size: 0,
    });

    let shader_3d_rt = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("runtime-scene-3d-rt-shader"),
        source: wgpu::ShaderSource::Wgsl(SCENE_3D_RT_WGSL.into()),
    });

    let pipeline_opaque_rt = create_3d_pipeline(
        device,
        &layout_3d_rt,
        &shader_3d_rt,
        color_format,
        Some(wgpu::BlendState::REPLACE),
        true,
        Some(wgpu::Face::Back),
        "runtime-scene-opaque-rt-pipeline",
        msaa_sample_count,
    );
    let pipeline_transparent_rt = create_3d_pipeline(
        device,
        &layout_3d_rt,
        &shader_3d_rt,
        color_format,
        Some(wgpu::BlendState::ALPHA_BLENDING),
        false,
        Some(wgpu::Face::Back),
        "runtime-scene-transparent-rt-pipeline",
        msaa_sample_count,
    );

    // BLAS: box (8 vertices, 36 u16 indices).
    let (box_vb, box_ib) = create_rt_blas_buffers_box(device);
    let blas_size_box = wgpu::BlasTriangleGeometrySizeDescriptor {
        vertex_format: wgpu::VertexFormat::Float32x3,
        vertex_count: 8,
        index_format: Some(wgpu::IndexFormat::Uint16),
        index_count: Some(36),
        flags: wgpu::AccelerationStructureGeometryFlags::OPAQUE,
    };
    let blas_box = device.create_blas(
        &wgpu::CreateBlasDescriptor {
            label: Some("rt-blas-box"),
            flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
            update_mode: wgpu::AccelerationStructureUpdateMode::Build,
        },
        wgpu::BlasGeometrySizeDescriptors::Triangles {
            descriptors: vec![blas_size_box.clone()],
        },
    );

    // BLAS: sphere (42 vertices, 240 u16 indices — subdivided icosphere at radius 0.5).
    let (sph_vb, sph_ib) = create_rt_blas_buffers_sphere(device);
    let sph_vertex_count = sph_vb.size() as u32 / std::mem::size_of::<GpuVertex3d>() as u32;
    let sph_index_count = sph_ib.size() as u32 / std::mem::size_of::<u16>() as u32;
    let blas_size_sphere = wgpu::BlasTriangleGeometrySizeDescriptor {
        vertex_format: wgpu::VertexFormat::Float32x3,
        vertex_count: sph_vertex_count,
        index_format: Some(wgpu::IndexFormat::Uint16),
        index_count: Some(sph_index_count),
        flags: wgpu::AccelerationStructureGeometryFlags::OPAQUE,
    };
    let blas_sphere = device.create_blas(
        &wgpu::CreateBlasDescriptor {
            label: Some("rt-blas-sphere"),
            flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
            update_mode: wgpu::AccelerationStructureUpdateMode::Build,
        },
        wgpu::BlasGeometrySizeDescriptors::Triangles {
            descriptors: vec![blas_size_sphere.clone()],
        },
    );

    // Build both BLASes in a one-shot command encoder.
    let mut blas_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("rt-blas-init-encoder"),
    });
    let build_box = wgpu::BlasBuildEntry {
        blas: &blas_box,
        geometry: wgpu::BlasGeometries::TriangleGeometries(vec![wgpu::BlasTriangleGeometry {
            size: &blas_size_box,
            vertex_buffer: &box_vb,
            first_vertex: 0,
            vertex_stride: std::mem::size_of::<GpuVertex3d>() as u64,
            index_buffer: Some(&box_ib),
            first_index: Some(0),
            transform_buffer: None,
            transform_buffer_offset: None,
        }]),
    };
    let build_sphere = wgpu::BlasBuildEntry {
        blas: &blas_sphere,
        geometry: wgpu::BlasGeometries::TriangleGeometries(vec![wgpu::BlasTriangleGeometry {
            size: &blas_size_sphere,
            vertex_buffer: &sph_vb,
            first_vertex: 0,
            vertex_stride: std::mem::size_of::<GpuVertex3d>() as u64,
            index_buffer: Some(&sph_ib),
            first_index: Some(0),
            transform_buffer: None,
            transform_buffer_offset: None,
        }]),
    };
    let blas_builds = [build_box, build_sphere];
    blas_encoder
        .build_acceleration_structures(blas_builds.iter(), std::iter::empty::<&wgpu::Tlas>());
    queue.submit(Some(blas_encoder.finish()));

    // TLAS rebuilt each frame with up to RT_DYNAMIC_CAP instances.
    let tlas = device.create_tlas(&wgpu::CreateTlasDescriptor {
        label: Some("rt-scene-tlas"),
        max_instances: RT_DYNAMIC_CAP,
        flags: wgpu::AccelerationStructureFlags::PREFER_FAST_BUILD,
        update_mode: wgpu::AccelerationStructureUpdateMode::Build,
    });

    // The TLAS bind group is stable across per-frame TLAS rebuilds.
    let tlas_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("runtime-rt-tlas-bg"),
        layout: &tlas_bgl,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: tlas.as_binding(),
        }],
    });

    RtInfrastructure {
        blas_box,
        blas_sphere,
        tlas,
        tlas_bgl,
        tlas_bg,
        pipeline_opaque_rt,
        pipeline_transparent_rt,
    }
}

fn create_shadow_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
) -> wgpu::RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("runtime-shadow-depth-pipeline"),
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
        fragment: None,
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            cull_mode: Some(wgpu::Face::Back),
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: DEPTH_FORMAT,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::LessEqual,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::draw_path::{
        DrawBatch3d, DrawBatchKey, DrawFrameStats, DrawInstance3d, RuntimeDrawFrame,
    };
    use crate::scene::{RuntimeSceneMode, SpriteInstance, SpriteKind};

    #[test]
    fn upload_plan_preserves_range_counts() {
        let draw = RuntimeDrawFrame {
            mode: RuntimeSceneMode::Spatial3d,
            view_2d: None,
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
            mode: RuntimeSceneMode::Spatial3d,
            view_2d: None,
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
        assert!((sprite.atlas_rect[0] - 0.375).abs() < 1e-6); // col 6 / 16
        assert!((sprite.atlas_rect[1] - 0.125).abs() < 1e-6); // row 2 / 16
        assert!((sprite.atlas_rect[2] - 0.4375).abs() < 1e-6);
        assert!((sprite.atlas_rect[3] - 0.1875).abs() < 1e-6);
    }

    #[test]
    fn external_image_slots_use_absolute_atlas_rows() {
        let rect = sprite_kind_atlas_rect(SpriteKind::Hud, EXTERNAL_IMAGE_SLOT_BASE);
        // slot 64 => col 0, row 4 in a 16x16 atlas grid.
        assert!((rect[0] - 0.0).abs() < 1e-6);
        assert!((rect[1] - 0.25).abs() < 1e-6);
        assert!((rect[2] - 0.0625).abs() < 1e-6);
        assert!((rect[3] - 0.3125).abs() < 1e-6);
    }

    #[test]
    fn sprite_slot_origin_px_maps_slot_to_expected_tile() {
        let (x, y) = sprite_slot_origin_px(EXTERNAL_IMAGE_SLOT_BASE);
        assert_eq!(x, 0);
        assert_eq!(y, SPRITE_ATLAS_TILE_SIZE * 4);

        let (x2, y2) = sprite_slot_origin_px(EXTERNAL_IMAGE_SLOT_BASE + 5);
        assert_eq!(x2, SPRITE_ATLAS_TILE_SIZE * 5);
        assert_eq!(y2, SPRITE_ATLAS_TILE_SIZE * 4);
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
            texture_index: 0,
        }
    }
}
