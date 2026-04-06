//! Backend-agnostic per-frame render snapshot records.
//!
//! These payload types are shared by Vulkan and Metal runtime backends.

/// Per-instance transform payload uploaded into backend-visible snapshot buffers.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct FrameInstanceTransform {
    pub model: [[f32; 4]; 4],
    pub color_rgba: [f32; 4],
    pub material_index: u32,
    pub texture_index: u32,
    pub flags: u32,
    pub _padding: u32,
}

/// One compact material record referenced by frame instances.
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct FrameMaterialRecord {
    pub material_params: [f32; 4],
    pub emissive_rgb: [f32; 3],
    pub shading_code: u32,
    pub texture_index: u32,
    pub flags: u32,
    pub _padding: [u32; 2],
}

/// One compact texture indirection record referenced by materials/instances.
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FrameTextureRecord {
    pub texture_slot: u32,
    pub sampler_code: u32,
    pub flags: u32,
    pub _padding: u32,
}

/// One compact scene light record prepared for renderer consumption.
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct FrameLightRecord {
    pub position_kind: [f32; 4],
    pub direction_inner: [f32; 4],
    pub color_intensity: [f32; 4],
    pub params: [f32; 4],
    pub shadow: [f32; 4],
}

/// Render-visible snapshot prepared by MPS / scene-build and consumed by render backends.
#[derive(Debug, Clone, Copy)]
pub struct RenderStateSnapshot<'a> {
    pub frame_id: u64,
    pub camera_view_proj: [[f32; 4]; 4],
    pub transforms: &'a [FrameInstanceTransform],
    pub materials: &'a [FrameMaterialRecord],
    pub textures: &'a [FrameTextureRecord],
    pub lights: &'a [FrameLightRecord],
}
