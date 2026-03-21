//! Linux-first raw Vulkan backend skeleton for Tileline.
//!
//! This module is intentionally low-abstraction:
//! - `ash` is used directly for instance/device/swapchain setup
//! - frame resources are explicit and double-buffered
//! - render-visible state snapshots are uploaded into persistently mapped Vulkan buffers
//! - command recording is structured to overlap with MPS-driven `Physics N+1`
//!
//! The current implementation is a backbone for the `v0.5.0` independence migration.
//! It is not yet a complete replacement for the runtime renderer, but it establishes:
//! - Vulkan initialization
//! - swapchain / framebuffer lifecycle
//! - per-frame command pool and command buffer ownership
//! - persistently mapped instance snapshot storage
//! - a safe Rust `Drop` path for Vulkan objects

#![cfg(target_os = "linux")]

use std::error::Error;
use std::ffi::{CStr, CString};
use std::fmt::{Display, Formatter};
use std::mem::{align_of, offset_of, size_of};
use std::ptr::{self, NonNull};
use std::sync::Arc;

use ash::khr::{surface, swapchain};
use ash::{vk, Device, Entry, Instance};
use winit::dpi::PhysicalSize;
use winit::raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use winit::window::Window;

use crate::graphics::multigpu::sync::{GpuQueueLane, MultiGpuFrameSyncConfig, SyncBackendHint};

const ENGINE_NAME: &[u8] = b"Tileline\0";
const APPLICATION_NAME: &[u8] = b"Tileline TLCore Vulkan Backend\0";
const SNAPSHOT_LIGHT_CAPACITY: usize = 32;

/// Linux display-system preference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinuxWindowSystemIntegration {
    /// Accept any window system exposed through `winit` / `ash-window`.
    Auto,
    /// Prefer Wayland when available.
    Wayland,
    /// Prefer X11 when available.
    X11,
}

/// Preferred present-mode policy for low-latency Linux rendering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PresentModePreference {
    /// Prefer `MAILBOX`, then `IMMEDIATE`, then `FIFO`.
    MailboxFirst,
    /// Prefer `IMMEDIATE`, then `MAILBOX`, then `FIFO`.
    ImmediateFirst,
    /// Force the guaranteed `FIFO` path.
    FifoOnly,
}

/// Runtime configuration for the raw Vulkan backend.
#[derive(Debug, Clone)]
pub struct VulkanBackendConfig {
    /// Enable Vulkan validation layers when available.
    pub enable_validation: bool,
    /// Number of frames kept in flight and therefore number of mapped state slots.
    pub frames_in_flight: usize,
    /// Preferred present mode for the swapchain.
    pub present_mode: PresentModePreference,
    /// Window-system preference hint for Linux.
    pub window_system: LinuxWindowSystemIntegration,
    /// Maximum number of instanced transforms written into one frame snapshot.
    pub max_instances: usize,
    /// Explicit multi-GPU policy for the raw Vulkan path.
    pub multi_gpu: VulkanMultiGpuConfig,
}

impl Default for VulkanBackendConfig {
    fn default() -> Self {
        Self {
            enable_validation: cfg!(debug_assertions),
            frames_in_flight: 2,
            present_mode: PresentModePreference::MailboxFirst,
            window_system: LinuxWindowSystemIntegration::Auto,
            max_instances: 32_768,
            multi_gpu: VulkanMultiGpuConfig::default(),
        }
    }
}

/// Multi-GPU policy for the raw Vulkan backend.
#[derive(Debug, Clone)]
pub struct VulkanMultiGpuConfig {
    /// Enable secondary-GPU planning when multiple Vulkan devices are visible.
    pub enable_secondary_gpu: bool,
    /// Keep an explicit transfer lane in the plan when cross-adapter texture movement exists.
    pub allow_transfer_lane: bool,
    /// Compose/sync budget policy shared with the existing multi-GPU bridge layer.
    pub sync: MultiGpuFrameSyncConfig,
}

impl Default for VulkanMultiGpuConfig {
    fn default() -> Self {
        Self {
            enable_secondary_gpu: true,
            allow_transfer_lane: true,
            sync: MultiGpuFrameSyncConfig::default(),
        }
    }
}

/// Queue-family selection chosen for the backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VulkanQueueSelection {
    pub graphics_family_index: u32,
    pub present_family_index: Option<u32>,
    pub compute_family_index: Option<u32>,
    pub transfer_family_index: Option<u32>,
}

/// Feature and extension support discovered for one Vulkan device.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct VulkanDeviceExtensionSupport {
    pub swapchain: bool,
    pub timeline_semaphore: bool,
    pub external_memory_fd: bool,
    pub external_semaphore_fd: bool,
    pub buffer_device_address: bool,
    pub dynamic_rendering: bool,
}

/// Vulkan physical-device profile captured during backend bootstrap.
#[derive(Debug, Clone)]
pub struct VulkanPhysicalDeviceProfile {
    pub physical_device: vk::PhysicalDevice,
    pub device_name: String,
    pub vendor_id: u32,
    pub device_id: u32,
    pub device_type: vk::PhysicalDeviceType,
    pub api_version: u32,
    pub queue_selection: VulkanQueueSelection,
    pub descriptor_indexing_supported: bool,
    pub extension_support: VulkanDeviceExtensionSupport,
}

impl VulkanPhysicalDeviceProfile {
    pub fn present_capable(&self) -> bool {
        self.queue_selection.present_family_index.is_some()
    }
}

/// Multi-GPU capability snapshot for the raw Vulkan backend.
#[derive(Debug, Clone)]
pub struct VulkanMultiGpuCapabilities {
    pub primary: VulkanPhysicalDeviceProfile,
    pub secondary_candidates: Vec<VulkanPhysicalDeviceProfile>,
    pub supports_multi_gpu: bool,
    pub sync_backend_hint: SyncBackendHint,
    pub requested_config: VulkanMultiGpuConfig,
    pub native_support: VulkanNativeMultiGpuSupport,
}

/// Native Vulkan multi-GPU readiness snapshot.
#[derive(Debug, Clone)]
pub struct VulkanNativeMultiGpuSupport {
    pub active: bool,
    pub reason: String,
    pub timeline_semaphore: bool,
    pub external_memory_fd: bool,
    pub external_semaphore_fd: bool,
    pub secondary_bootstrap_ready: bool,
    pub selected_secondary_device_name: Option<String>,
}

/// Frame-scoped explicit multi-GPU plan for the Vulkan backend.
#[derive(Debug, Clone)]
pub struct VulkanMultiGpuFramePlan {
    pub frame_id: u64,
    pub lanes: Vec<GpuQueueLane>,
    pub require_secondary: bool,
    pub require_transfer: bool,
    pub cross_adapter_bytes: u64,
    pub compose_wait_budget_us: u64,
    pub native_multi_gpu_active: bool,
    pub native_multi_gpu_reason: String,
}

/// Per-instance transform payload uploaded into persistently mapped snapshot buffers.
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

/// Render-visible snapshot prepared by MPS / scene-build and consumed by `Render N`.
#[derive(Debug, Clone, Copy)]
pub struct RenderStateSnapshot<'a> {
    pub frame_id: u64,
    pub camera_view_proj: [[f32; 4]; 4],
    pub transforms: &'a [FrameInstanceTransform],
    pub materials: &'a [FrameMaterialRecord],
    pub textures: &'a [FrameTextureRecord],
    pub lights: &'a [FrameLightRecord],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
struct SnapshotHeader {
    frame_id: u64,
    instance_count: u32,
    material_count: u32,
    texture_count: u32,
    light_count: u32,
    reserved: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
struct CameraUniform {
    view_proj: [[f32; 4]; 4],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
struct DrawPushConstants {
    material_count: u32,
    light_count: u32,
    texture_count: u32,
    _padding0: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
struct SceneVertex {
    position: [f32; 3],
}

#[derive(Debug, Clone)]
struct SpirvShaderArtifact {
    words: Vec<u32>,
}

/// Snapshot metadata exposed for debugging / telemetry.
#[derive(Debug, Clone, Copy, Default)]
pub struct VulkanSnapshotSlotState {
    pub frame_id: u64,
    pub instance_count: u32,
    pub material_count: u32,
    pub texture_count: u32,
    pub light_count: u32,
    pub byte_len: usize,
}

/// Submit/present telemetry for one recorded frame.
#[derive(Debug, Clone, Copy, Default)]
pub struct FrameSubmissionTelemetry {
    pub frame_slot: usize,
    pub swapchain_image_index: u32,
    pub frame_id: u64,
    pub instance_count: u32,
    pub material_count: u32,
    pub texture_count: u32,
    pub light_count: u32,
    pub selected_present_mode: vk::PresentModeKHR,
}

/// Per-frame explicit queue execution telemetry.
#[derive(Debug, Clone)]
pub struct VulkanFrameExecutionTelemetry {
    pub submission: FrameSubmissionTelemetry,
    pub primary_submission_serial: u64,
    pub secondary_submission_serial: Option<u64>,
    pub transfer_submission_serial: Option<u64>,
    pub secondary_queue_executed: bool,
    pub multi_gpu_active: bool,
    pub multi_gpu_reason: String,
}

#[derive(Debug)]
pub enum VulkanBackendError {
    EntryLoad(ash::LoadingError),
    Vk(vk::Result),
    WindowHandle(String),
    InvalidConfig(&'static str),
    NoPhysicalDevice,
    NoGraphicsQueue,
    NoPresentQueue,
    NoSurfaceFormat,
    NoSwapchainImages,
    SnapshotCapacityExceeded { requested: usize, capacity: usize },
    SnapshotMaterialCapacityExceeded { requested: usize, capacity: usize },
    SnapshotTextureCapacityExceeded { requested: usize, capacity: usize },
    SnapshotLightCapacityExceeded { requested: usize, capacity: usize },
}

impl Display for VulkanBackendError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EntryLoad(err) => write!(f, "failed to load Vulkan entry: {err}"),
            Self::Vk(err) => write!(f, "Vulkan error: {err:?}"),
            Self::WindowHandle(err) => write!(f, "window handle error: {err}"),
            Self::InvalidConfig(msg) => write!(f, "invalid Vulkan backend config: {msg}"),
            Self::NoPhysicalDevice => write!(f, "no suitable Vulkan physical device found"),
            Self::NoGraphicsQueue => write!(f, "no graphics queue family found"),
            Self::NoPresentQueue => write!(f, "no present queue family found"),
            Self::NoSurfaceFormat => write!(f, "no compatible surface format found"),
            Self::NoSwapchainImages => write!(f, "swapchain returned zero presentable images"),
            Self::SnapshotCapacityExceeded {
                requested,
                capacity,
            } => write!(
                f,
                "snapshot capacity exceeded: requested {requested} instances, capacity {capacity}"
            ),
            Self::SnapshotMaterialCapacityExceeded {
                requested,
                capacity,
            } => write!(
                f,
                "snapshot material capacity exceeded: requested {requested} materials, capacity {capacity}"
            ),
            Self::SnapshotTextureCapacityExceeded {
                requested,
                capacity,
            } => write!(
                f,
                "snapshot texture capacity exceeded: requested {requested} textures, capacity {capacity}"
            ),
            Self::SnapshotLightCapacityExceeded {
                requested,
                capacity,
            } => write!(
                f,
                "snapshot light capacity exceeded: requested {requested} lights, capacity {capacity}"
            ),
        }
    }
}

impl Error for VulkanBackendError {}

impl From<ash::LoadingError> for VulkanBackendError {
    fn from(value: ash::LoadingError) -> Self {
        Self::EntryLoad(value)
    }
}

impl From<vk::Result> for VulkanBackendError {
    fn from(value: vk::Result) -> Self {
        Self::Vk(value)
    }
}

struct SwapchainState {
    loader: swapchain::Device,
    swapchain: vk::SwapchainKHR,
    format: vk::Format,
    extent: vk::Extent2D,
    present_mode: vk::PresentModeKHR,
    _images: Vec<vk::Image>,
    image_views: Vec<vk::ImageView>,
    framebuffers: Vec<vk::Framebuffer>,
}

struct PersistentlyMappedBuffer {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    mapped_ptr: NonNull<u8>,
    byte_len: usize,
}

impl PersistentlyMappedBuffer {
    fn as_mut_ptr<T>(&self, offset: usize) -> Result<*mut T, VulkanBackendError> {
        let required =
            offset
                .checked_add(size_of::<T>())
                .ok_or(VulkanBackendError::InvalidConfig(
                    "mapped buffer offset overflow",
                ))?;
        if required > self.byte_len {
            return Err(VulkanBackendError::InvalidConfig(
                "mapped buffer pointer request exceeds allocation",
            ));
        }
        Ok(unsafe { self.mapped_ptr.as_ptr().add(offset) as *mut T })
    }
}

struct SnapshotSlot {
    mapped: PersistentlyMappedBuffer,
    header_offset: usize,
    transforms_offset: usize,
    materials_offset: usize,
    textures_offset: usize,
    lights_offset: usize,
    instance_capacity: usize,
    material_capacity: usize,
    texture_capacity: usize,
    light_capacity: usize,
    last_state: VulkanSnapshotSlotState,
}

struct FrameResources {
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    image_available: vk::Semaphore,
    render_finished: vk::Semaphore,
    in_flight_fence: vk::Fence,
    camera_uniform: PersistentlyMappedBuffer,
    camera_descriptor_set: vk::DescriptorSet,
    snapshot_slot: SnapshotSlot,
}

struct SecondaryDeviceState {
    profile: VulkanPhysicalDeviceProfile,
    device: Device,
    graphics_queue: vk::Queue,
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    in_flight_fence: vk::Fence,
}

struct SampledTextureArrayResources {
    image: vk::Image,
    image_memory: vk::DeviceMemory,
    image_view: vk::ImageView,
    sampler: vk::Sampler,
}

struct ScenePipelineResources {
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    vertex_buffer: PersistentlyMappedBuffer,
    index_buffer: PersistentlyMappedBuffer,
    index_count: u32,
    texture_array: SampledTextureArrayResources,
}

/// Linux-first Vulkan backend.
pub struct VulkanBackend {
    config: VulkanBackendConfig,
    _window: Arc<Window>,
    _entry: Entry,
    instance: Instance,
    surface_loader: surface::Instance,
    surface: vk::SurfaceKHR,
    physical_device: vk::PhysicalDevice,
    device: Device,
    queue_selection: VulkanQueueSelection,
    graphics_queue: vk::Queue,
    present_queue: vk::Queue,
    primary_device_profile: VulkanPhysicalDeviceProfile,
    multi_gpu_capabilities: VulkanMultiGpuCapabilities,
    secondary_device: Option<SecondaryDeviceState>,
    swapchain: SwapchainState,
    render_pass: vk::RenderPass,
    scene_pipeline: ScenePipelineResources,
    frame_resources: Vec<FrameResources>,
    current_frame_slot: usize,
    descriptor_indexing_supported: bool,
    primary_submission_serial: u64,
    secondary_submission_serial: u64,
    transfer_submission_serial: u64,
}

impl VulkanBackend {
    /// Create the Linux-first raw Vulkan backend and all explicit frame resources.
    pub fn new(
        window: Arc<Window>,
        config: VulkanBackendConfig,
    ) -> Result<Self, VulkanBackendError> {
        if config.frames_in_flight == 0 {
            return Err(VulkanBackendError::InvalidConfig(
                "frames_in_flight must be at least 1",
            ));
        }
        if config.max_instances == 0 {
            return Err(VulkanBackendError::InvalidConfig(
                "max_instances must be at least 1",
            ));
        }

        let entry = unsafe { Entry::load()? };
        let instance = unsafe { create_instance(&entry, window.as_ref(), &config)? };
        let surface_loader = surface::Instance::new(&entry, &instance);
        let surface = unsafe { create_surface(&entry, &instance, window.as_ref())? };
        let available_device_profiles =
            unsafe { collect_physical_device_profiles(&instance, &surface_loader, surface)? };
        let primary_device_profile =
            select_primary_device_profile(&available_device_profiles)?.clone();
        let physical_device = primary_device_profile.physical_device;
        let queue_selection = primary_device_profile.queue_selection;
        let descriptor_indexing_supported = primary_device_profile.descriptor_indexing_supported;
        let multi_gpu_capabilities = build_multi_gpu_capabilities(
            &primary_device_profile,
            &available_device_profiles,
            &config,
        );
        let (device, graphics_queue, present_queue) =
            unsafe { create_logical_device(&instance, physical_device, queue_selection)? };
        let secondary_device = if multi_gpu_capabilities
            .native_support
            .secondary_bootstrap_ready
        {
            let secondary_profile = multi_gpu_capabilities
                .secondary_candidates
                .iter()
                .find(|candidate| {
                    Some(candidate.device_name.as_str())
                        == multi_gpu_capabilities
                            .native_support
                            .selected_secondary_device_name
                            .as_deref()
                })
                .cloned();
            match secondary_profile {
                Some(profile) => {
                    Some(unsafe { create_secondary_device_state(&instance, &profile)? })
                }
                None => None,
            }
        } else {
            None
        };

        let mut swapchain = unsafe {
            create_swapchain_state(
                &instance,
                &device,
                physical_device,
                &surface_loader,
                surface,
                queue_selection,
                window.inner_size(),
                config.present_mode,
            )?
        };
        let render_pass = unsafe { create_render_pass(&device, swapchain.format)? };
        unsafe { create_framebuffers(&device, render_pass, &mut swapchain)? };
        let scene_pipeline = unsafe {
            create_scene_pipeline_resources(
                &instance,
                &device,
                physical_device,
                queue_selection.graphics_family_index,
                graphics_queue,
                render_pass,
                swapchain.extent,
                config.frames_in_flight,
            )?
        };
        let frame_resources = unsafe {
            create_frame_resources(
                &instance,
                &device,
                physical_device,
                queue_selection.graphics_family_index,
                config.frames_in_flight,
                config.max_instances,
                &scene_pipeline,
            )?
        };

        Ok(Self {
            config,
            _window: window,
            _entry: entry,
            instance,
            surface_loader,
            surface,
            physical_device,
            device,
            queue_selection,
            graphics_queue,
            present_queue,
            primary_device_profile,
            multi_gpu_capabilities,
            secondary_device,
            swapchain,
            render_pass,
            scene_pipeline,
            frame_resources,
            current_frame_slot: 0,
            descriptor_indexing_supported,
            primary_submission_serial: 0,
            secondary_submission_serial: 0,
            transfer_submission_serial: 0,
        })
    }

    /// Queue-family selection chosen for this backend.
    pub fn queue_selection(&self) -> VulkanQueueSelection {
        self.queue_selection
    }

    /// `true` when the selected physical device reports descriptor indexing support.
    pub fn descriptor_indexing_supported(&self) -> bool {
        self.descriptor_indexing_supported
    }

    /// Physical-device profile chosen as the primary/present GPU.
    pub fn primary_device_profile(&self) -> &VulkanPhysicalDeviceProfile {
        &self.primary_device_profile
    }

    /// Multi-GPU capability snapshot discovered during backend bootstrap.
    pub fn multi_gpu_capabilities(&self) -> &VulkanMultiGpuCapabilities {
        &self.multi_gpu_capabilities
    }

    /// Optional helper GPU bootstrapped for the native Vulkan multi-GPU path.
    pub fn secondary_device_profile(&self) -> Option<&VulkanPhysicalDeviceProfile> {
        self.secondary_device.as_ref().map(|device| &device.profile)
    }

    /// Selected present mode actually used by the swapchain.
    pub fn selected_present_mode(&self) -> vk::PresentModeKHR {
        self.swapchain.present_mode
    }

    /// Build a frame-scoped explicit multi-GPU plan without hard-binding the runtime yet.
    pub fn build_multi_gpu_frame_plan(
        &self,
        frame_id: u64,
        cross_adapter_bytes: u64,
        prefer_secondary: bool,
    ) -> VulkanMultiGpuFramePlan {
        let secondary_enabled = self.multi_gpu_capabilities.supports_multi_gpu
            && self.config.multi_gpu.enable_secondary_gpu
            && prefer_secondary;
        let transfer_enabled = self.config.multi_gpu.allow_transfer_lane && cross_adapter_bytes > 0;
        let mut lanes = vec![GpuQueueLane::Primary];
        if secondary_enabled {
            lanes.push(GpuQueueLane::Secondary);
        }
        if transfer_enabled {
            lanes.push(GpuQueueLane::Transfer);
        }

        VulkanMultiGpuFramePlan {
            frame_id,
            lanes,
            require_secondary: secondary_enabled,
            require_transfer: transfer_enabled,
            cross_adapter_bytes,
            compose_wait_budget_us: self.config.multi_gpu.sync.compose_wait_budget.as_micros()
                as u64,
            native_multi_gpu_active: secondary_enabled
                && self.multi_gpu_capabilities.native_support.active
                && self.secondary_device.is_some(),
            native_multi_gpu_reason: self.multi_gpu_capabilities.native_support.reason.clone(),
        }
    }

    /// Current swapchain extent in pixels.
    pub fn extent(&self) -> vk::Extent2D {
        self.swapchain.extent
    }

    /// Resize the swapchain-dependent resources.
    pub fn resize(&mut self, new_size: PhysicalSize<u32>) -> Result<(), VulkanBackendError> {
        if new_size.width == 0 || new_size.height == 0 {
            return Ok(());
        }

        unsafe {
            self.device.device_wait_idle()?;
            destroy_framebuffers(&self.device, &mut self.swapchain);
            destroy_image_views(&self.device, &mut self.swapchain);
            self.swapchain
                .loader
                .destroy_swapchain(self.swapchain.swapchain, None);
            self.swapchain = create_swapchain_state(
                &self.instance,
                &self.device,
                self.physical_device,
                &self.surface_loader,
                self.surface,
                self.queue_selection,
                new_size,
                self.config.present_mode,
            )?;
            create_framebuffers(&self.device, self.render_pass, &mut self.swapchain)?;
            self.device
                .destroy_pipeline(self.scene_pipeline.pipeline, None);
            self.device
                .destroy_pipeline_layout(self.scene_pipeline.pipeline_layout, None);
            self.scene_pipeline.pipeline = create_scene_pipeline(
                &self.device,
                self.render_pass,
                self.swapchain.extent,
                self.scene_pipeline.pipeline_layout,
            )?;
        }

        Ok(())
    }

    /// Write the current render-visible state into the persistently mapped slot associated with the
    /// selected frame resource. This is the lock-free handoff point between MPS-produced state and
    /// Vulkan-visible memory.
    pub fn upload_state_snapshot(
        &mut self,
        frame_slot: usize,
        snapshot: RenderStateSnapshot<'_>,
    ) -> Result<VulkanSnapshotSlotState, VulkanBackendError> {
        let slot = self
            .frame_resources
            .get_mut(frame_slot)
            .ok_or(VulkanBackendError::InvalidConfig("frame slot out of range"))?;
        if snapshot.transforms.len() > slot.snapshot_slot.instance_capacity {
            return Err(VulkanBackendError::SnapshotCapacityExceeded {
                requested: snapshot.transforms.len(),
                capacity: slot.snapshot_slot.instance_capacity,
            });
        }
        if snapshot.materials.len() > slot.snapshot_slot.material_capacity {
            return Err(VulkanBackendError::SnapshotMaterialCapacityExceeded {
                requested: snapshot.materials.len(),
                capacity: slot.snapshot_slot.material_capacity,
            });
        }
        if snapshot.textures.len() > slot.snapshot_slot.texture_capacity {
            return Err(VulkanBackendError::SnapshotTextureCapacityExceeded {
                requested: snapshot.textures.len(),
                capacity: slot.snapshot_slot.texture_capacity,
            });
        }
        if snapshot.lights.len() > slot.snapshot_slot.light_capacity {
            return Err(VulkanBackendError::SnapshotLightCapacityExceeded {
                requested: snapshot.lights.len(),
                capacity: slot.snapshot_slot.light_capacity,
            });
        }

        let header = SnapshotHeader {
            frame_id: snapshot.frame_id,
            instance_count: snapshot.transforms.len() as u32,
            material_count: snapshot.materials.len() as u32,
            texture_count: snapshot.textures.len() as u32,
            light_count: snapshot.lights.len() as u32,
            reserved: 0,
        };
        unsafe {
            ptr::write(
                slot.snapshot_slot
                    .mapped
                    .as_mut_ptr::<SnapshotHeader>(slot.snapshot_slot.header_offset)?,
                header,
            );
            ptr::write(
                slot.camera_uniform.as_mut_ptr::<CameraUniform>(0)?,
                CameraUniform {
                    view_proj: snapshot.camera_view_proj,
                },
            );
            let transform_dst = slot
                .snapshot_slot
                .mapped
                .as_mut_ptr::<FrameInstanceTransform>(slot.snapshot_slot.transforms_offset)?;
            ptr::copy_nonoverlapping(
                snapshot.transforms.as_ptr(),
                transform_dst,
                snapshot.transforms.len(),
            );
            let material_dst = slot
                .snapshot_slot
                .mapped
                .as_mut_ptr::<FrameMaterialRecord>(slot.snapshot_slot.materials_offset)?;
            ptr::copy_nonoverlapping(
                snapshot.materials.as_ptr(),
                material_dst,
                snapshot.materials.len(),
            );
            let texture_dst = slot
                .snapshot_slot
                .mapped
                .as_mut_ptr::<FrameTextureRecord>(slot.snapshot_slot.textures_offset)?;
            ptr::copy_nonoverlapping(
                snapshot.textures.as_ptr(),
                texture_dst,
                snapshot.textures.len(),
            );
            let light_dst = slot
                .snapshot_slot
                .mapped
                .as_mut_ptr::<FrameLightRecord>(slot.snapshot_slot.lights_offset)?;
            ptr::copy_nonoverlapping(snapshot.lights.as_ptr(), light_dst, snapshot.lights.len());
        }

        let byte_len = slot.snapshot_slot.lights_offset
            + snapshot.lights.len() * size_of::<FrameLightRecord>();
        slot.snapshot_slot.last_state = VulkanSnapshotSlotState {
            frame_id: snapshot.frame_id,
            instance_count: snapshot.transforms.len() as u32,
            material_count: snapshot.materials.len() as u32,
            texture_count: snapshot.textures.len() as u32,
            light_count: snapshot.lights.len() as u32,
            byte_len,
        };
        Ok(slot.snapshot_slot.last_state)
    }

    /// Record and submit one `Render N` frame.
    ///
    /// The snapshot upload is explicit and frame-slot-local so `Physics N+1` can continue writing
    /// the next slot without racing the command buffer currently being submitted.
    pub fn render_n(
        &mut self,
        snapshot: RenderStateSnapshot<'_>,
    ) -> Result<FrameSubmissionTelemetry, VulkanBackendError> {
        let plan = self.build_multi_gpu_frame_plan(snapshot.frame_id, 0, false);
        self.render_n_with_plan(snapshot, &plan)
            .map(|telemetry| telemetry.submission)
    }

    /// Record and submit one `Render N` frame with an explicit multi-GPU execution plan.
    pub fn render_n_with_plan(
        &mut self,
        snapshot: RenderStateSnapshot<'_>,
        plan: &VulkanMultiGpuFramePlan,
    ) -> Result<VulkanFrameExecutionTelemetry, VulkanBackendError> {
        let frame_slot = self.current_frame_slot;
        let snapshot_state = self.upload_state_snapshot(frame_slot, snapshot)?;
        let frame = &self.frame_resources[frame_slot];
        let mut secondary_submission_serial = None;
        let mut transfer_submission_serial = None;
        let mut secondary_queue_executed = false;

        if plan.require_secondary {
            if let Some(secondary) = self.secondary_device.as_mut() {
                unsafe {
                    secondary_submission_serial = Some(record_secondary_queue_probe(
                        secondary,
                        self.secondary_submission_serial,
                    )?);
                }
                self.secondary_submission_serial =
                    secondary_submission_serial.expect("secondary serial should exist");
                secondary_queue_executed = true;
            }
        }
        if plan.require_transfer {
            self.transfer_submission_serial = self.transfer_submission_serial.saturating_add(1);
            transfer_submission_serial = Some(self.transfer_submission_serial);
        }

        let swapchain_image_index = unsafe {
            self.device
                .wait_for_fences(&[frame.in_flight_fence], true, u64::MAX)?;
            self.device.reset_fences(&[frame.in_flight_fence])?;

            let (swapchain_image_index, _) = self.swapchain.loader.acquire_next_image(
                self.swapchain.swapchain,
                u64::MAX,
                frame.image_available,
                vk::Fence::null(),
            )?;

            self.device
                .reset_command_pool(frame.command_pool, vk::CommandPoolResetFlags::empty())?;
            record_frame_commands(
                &self.device,
                self.render_pass,
                self.swapchain.extent,
                self.swapchain.framebuffers[swapchain_image_index as usize],
                frame.command_buffer,
                frame,
                &self.scene_pipeline,
                snapshot_state,
            )?;

            let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
            let command_buffers = [frame.command_buffer];
            let wait_semaphores = [frame.image_available];
            let signal_semaphores = [frame.render_finished];
            let submit_info = vk::SubmitInfo::default()
                .wait_semaphores(&wait_semaphores)
                .wait_dst_stage_mask(&wait_stages)
                .command_buffers(&command_buffers)
                .signal_semaphores(&signal_semaphores);

            self.device
                .queue_submit(self.graphics_queue, &[submit_info], frame.in_flight_fence)?;

            let swapchains = [self.swapchain.swapchain];
            let image_indices = [swapchain_image_index];
            let present_wait = [frame.render_finished];
            let present_info = vk::PresentInfoKHR::default()
                .wait_semaphores(&present_wait)
                .swapchains(&swapchains)
                .image_indices(&image_indices);
            self.swapchain
                .loader
                .queue_present(self.present_queue, &present_info)?;
            swapchain_image_index
        };

        self.primary_submission_serial = self.primary_submission_serial.saturating_add(1);
        self.current_frame_slot = (self.current_frame_slot + 1) % self.frame_resources.len();
        Ok(VulkanFrameExecutionTelemetry {
            submission: FrameSubmissionTelemetry {
                frame_slot,
                swapchain_image_index,
                frame_id: snapshot.frame_id,
                instance_count: snapshot.transforms.len() as u32,
                material_count: snapshot.materials.len() as u32,
                texture_count: snapshot.textures.len() as u32,
                light_count: snapshot.lights.len() as u32,
                selected_present_mode: self.swapchain.present_mode,
            },
            primary_submission_serial: self.primary_submission_serial,
            secondary_submission_serial,
            transfer_submission_serial,
            secondary_queue_executed,
            multi_gpu_active: plan.native_multi_gpu_active && self.secondary_device.is_some(),
            multi_gpu_reason: plan.native_multi_gpu_reason.clone(),
        })
    }
}

impl Drop for VulkanBackend {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            if let Some(secondary) = self.secondary_device.take() {
                let _ = secondary.device.device_wait_idle();
                secondary
                    .device
                    .destroy_fence(secondary.in_flight_fence, None);
                secondary
                    .device
                    .destroy_command_pool(secondary.command_pool, None);
                secondary.device.destroy_device(None);
            }

            for frame in self.frame_resources.drain(..) {
                destroy_mapped_buffer(&self.device, frame.camera_uniform);
                destroy_snapshot_slot(&self.device, frame.snapshot_slot);
                self.device.destroy_fence(frame.in_flight_fence, None);
                self.device.destroy_semaphore(frame.render_finished, None);
                self.device.destroy_semaphore(frame.image_available, None);
                self.device.destroy_command_pool(frame.command_pool, None);
            }

            destroy_mapped_buffer_ref(&self.device, &self.scene_pipeline.vertex_buffer);
            destroy_mapped_buffer_ref(&self.device, &self.scene_pipeline.index_buffer);
            self.device
                .destroy_sampler(self.scene_pipeline.texture_array.sampler, None);
            self.device
                .destroy_image_view(self.scene_pipeline.texture_array.image_view, None);
            self.device
                .destroy_image(self.scene_pipeline.texture_array.image, None);
            self.device
                .free_memory(self.scene_pipeline.texture_array.image_memory, None);
            self.device
                .destroy_pipeline(self.scene_pipeline.pipeline, None);
            self.device
                .destroy_pipeline_layout(self.scene_pipeline.pipeline_layout, None);
            self.device
                .destroy_descriptor_pool(self.scene_pipeline.descriptor_pool, None);
            self.device
                .destroy_descriptor_set_layout(self.scene_pipeline.descriptor_set_layout, None);
            destroy_framebuffers(&self.device, &mut self.swapchain);
            self.device.destroy_render_pass(self.render_pass, None);
            destroy_image_views(&self.device, &mut self.swapchain);
            self.swapchain
                .loader
                .destroy_swapchain(self.swapchain.swapchain, None);
            self.device.destroy_device(None);
            self.surface_loader.destroy_surface(self.surface, None);
            self.instance.destroy_instance(None);
        }
    }
}

unsafe fn create_instance(
    entry: &Entry,
    window: &Window,
    config: &VulkanBackendConfig,
) -> Result<Instance, VulkanBackendError> {
    let app_name = CStr::from_bytes_with_nul(APPLICATION_NAME)
        .map_err(|_| VulkanBackendError::InvalidConfig("invalid application name"))?;
    let engine_name = CStr::from_bytes_with_nul(ENGINE_NAME)
        .map_err(|_| VulkanBackendError::InvalidConfig("invalid engine name"))?;
    let app_info = vk::ApplicationInfo::default()
        .application_name(app_name)
        .application_version(vk::make_api_version(0, 0, 5, 0))
        .engine_name(engine_name)
        .engine_version(vk::make_api_version(0, 0, 5, 0))
        .api_version(vk::API_VERSION_1_3);

    let display_handle = window
        .display_handle()
        .map_err(|err| VulkanBackendError::WindowHandle(err.to_string()))?;
    let mut extension_names =
        ash_window::enumerate_required_extensions(display_handle.as_raw())?.to_vec();
    if config.enable_validation {
        extension_names.push(ash::ext::debug_utils::NAME.as_ptr());
    }

    let validation_layer = CString::new("VK_LAYER_KHRONOS_validation")
        .map_err(|_| VulkanBackendError::InvalidConfig("invalid validation layer name"))?;
    let layer_names = if config.enable_validation {
        vec![validation_layer.as_ptr()]
    } else {
        Vec::new()
    };

    let create_info = vk::InstanceCreateInfo::default()
        .application_info(&app_info)
        .enabled_extension_names(&extension_names)
        .enabled_layer_names(&layer_names);
    Ok(entry.create_instance(&create_info, None)?)
}

unsafe fn create_surface(
    entry: &Entry,
    instance: &Instance,
    window: &Window,
) -> Result<vk::SurfaceKHR, VulkanBackendError> {
    let display_handle = window
        .display_handle()
        .map_err(|err| VulkanBackendError::WindowHandle(err.to_string()))?;
    let window_handle = window
        .window_handle()
        .map_err(|err| VulkanBackendError::WindowHandle(err.to_string()))?;
    Ok(ash_window::create_surface(
        entry,
        instance,
        display_handle.as_raw(),
        window_handle.as_raw(),
        None,
    )?)
}

unsafe fn collect_physical_device_profiles(
    instance: &Instance,
    surface_loader: &surface::Instance,
    surface: vk::SurfaceKHR,
) -> Result<Vec<VulkanPhysicalDeviceProfile>, VulkanBackendError> {
    let physical_devices = instance.enumerate_physical_devices()?;
    let mut profiles = Vec::new();

    for physical_device in physical_devices {
        let queue_families = instance.get_physical_device_queue_family_properties(physical_device);
        let mut graphics_family = None;
        let mut present_family = None;
        let mut compute_family = None;
        let mut transfer_family = None;
        for (index, family) in queue_families.iter().enumerate() {
            if family.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
                graphics_family = Some(index as u32);
            }
            if family.queue_flags.contains(vk::QueueFlags::COMPUTE)
                && !family.queue_flags.contains(vk::QueueFlags::GRAPHICS)
            {
                compute_family = Some(index as u32);
            }
            if family.queue_flags.contains(vk::QueueFlags::TRANSFER)
                && !family.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                && !family.queue_flags.contains(vk::QueueFlags::COMPUTE)
            {
                transfer_family = Some(index as u32);
            }
            if surface_loader.get_physical_device_surface_support(
                physical_device,
                index as u32,
                surface,
            )? {
                present_family = Some(index as u32);
            }
        }

        let Some(graphics_family_index) = graphics_family else {
            continue;
        };

        let descriptor_indexing_supported =
            descriptor_indexing_available(instance, physical_device);
        let extension_support = query_device_extension_support(instance, physical_device)?;
        let properties = instance.get_physical_device_properties(physical_device);
        let raw_name = CStr::from_ptr(properties.device_name.as_ptr());
        profiles.push(VulkanPhysicalDeviceProfile {
            physical_device,
            device_name: raw_name.to_string_lossy().into_owned(),
            vendor_id: properties.vendor_id,
            device_id: properties.device_id,
            device_type: properties.device_type,
            api_version: properties.api_version,
            queue_selection: VulkanQueueSelection {
                graphics_family_index,
                present_family_index: present_family,
                compute_family_index: compute_family.or(Some(graphics_family_index)),
                transfer_family_index: transfer_family
                    .or(compute_family)
                    .or(Some(graphics_family_index)),
            },
            descriptor_indexing_supported,
            extension_support,
        });
    }

    if profiles.is_empty() {
        return Err(VulkanBackendError::NoPhysicalDevice);
    }
    Ok(profiles)
}

fn select_primary_device_profile(
    profiles: &[VulkanPhysicalDeviceProfile],
) -> Result<&VulkanPhysicalDeviceProfile, VulkanBackendError> {
    profiles
        .iter()
        .filter(|profile| profile.present_capable())
        .max_by_key(|profile| device_score(profile.device_type))
        .ok_or(VulkanBackendError::NoPhysicalDevice)
}

fn build_multi_gpu_capabilities(
    primary: &VulkanPhysicalDeviceProfile,
    profiles: &[VulkanPhysicalDeviceProfile],
    config: &VulkanBackendConfig,
) -> VulkanMultiGpuCapabilities {
    let secondary_candidates = profiles
        .iter()
        .filter(|profile| {
            profile.physical_device != primary.physical_device
                && profile.extension_support.timeline_semaphore
                && profile.extension_support.external_memory_fd
                && profile.extension_support.external_semaphore_fd
        })
        .cloned()
        .collect::<Vec<_>>();
    let selected_secondary = secondary_candidates.first();
    let native_support = if !config.multi_gpu.enable_secondary_gpu {
        VulkanNativeMultiGpuSupport {
            active: false,
            reason: "secondary GPU disabled by config".to_string(),
            timeline_semaphore: false,
            external_memory_fd: false,
            external_semaphore_fd: false,
            secondary_bootstrap_ready: false,
            selected_secondary_device_name: None,
        }
    } else if secondary_candidates.is_empty() {
        VulkanNativeMultiGpuSupport {
            active: false,
            reason: "no secondary Vulkan device satisfied native MGPU extension requirements"
                .to_string(),
            timeline_semaphore: false,
            external_memory_fd: false,
            external_semaphore_fd: false,
            secondary_bootstrap_ready: false,
            selected_secondary_device_name: None,
        }
    } else {
        let primary_support = primary.extension_support;
        let secondary_support = selected_secondary
            .map(|candidate| candidate.extension_support)
            .unwrap_or_default();
        let timeline = primary_support.timeline_semaphore && secondary_support.timeline_semaphore;
        let external_memory =
            primary_support.external_memory_fd && secondary_support.external_memory_fd;
        let external_semaphore =
            primary_support.external_semaphore_fd && secondary_support.external_semaphore_fd;
        let secondary_bootstrap_ready = timeline && external_memory && external_semaphore;
        VulkanNativeMultiGpuSupport {
            active: secondary_bootstrap_ready,
            reason: if secondary_bootstrap_ready {
                format!(
                    "native Vulkan MGPU ready with primary '{}' and secondary '{}'",
                    primary.device_name,
                    selected_secondary
                        .map(|candidate| candidate.device_name.as_str())
                        .unwrap_or("unknown")
                )
            } else {
                "secondary device found but native MGPU sync/memory extensions were incomplete"
                    .to_string()
            },
            timeline_semaphore: timeline,
            external_memory_fd: external_memory,
            external_semaphore_fd: external_semaphore,
            secondary_bootstrap_ready,
            selected_secondary_device_name: selected_secondary
                .map(|candidate| candidate.device_name.clone()),
        }
    };
    VulkanMultiGpuCapabilities {
        primary: primary.clone(),
        supports_multi_gpu: !secondary_candidates.is_empty(),
        secondary_candidates,
        sync_backend_hint: SyncBackendHint::VulkanTimelineFenceLike,
        requested_config: config.multi_gpu.clone(),
        native_support,
    }
}

fn device_score(device_type: vk::PhysicalDeviceType) -> u32 {
    match device_type {
        vk::PhysicalDeviceType::DISCRETE_GPU => 4,
        vk::PhysicalDeviceType::INTEGRATED_GPU => 3,
        vk::PhysicalDeviceType::VIRTUAL_GPU => 2,
        vk::PhysicalDeviceType::CPU => 1,
        _ => 0,
    }
}

unsafe fn descriptor_indexing_available(
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
) -> bool {
    let mut indexing_features = vk::PhysicalDeviceDescriptorIndexingFeatures::default();
    let mut features = vk::PhysicalDeviceFeatures2::default().push_next(&mut indexing_features);
    instance.get_physical_device_features2(physical_device, &mut features);
    indexing_features.descriptor_binding_partially_bound == vk::TRUE
        && indexing_features.runtime_descriptor_array == vk::TRUE
}

unsafe fn query_device_extension_support(
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
) -> Result<VulkanDeviceExtensionSupport, VulkanBackendError> {
    let extensions = instance.enumerate_device_extension_properties(physical_device)?;
    let mut names = Vec::with_capacity(extensions.len());
    for extension in extensions {
        let name = CStr::from_ptr(extension.extension_name.as_ptr())
            .to_string_lossy()
            .into_owned();
        names.push(name);
    }
    let has = |name: &str| names.iter().any(|entry| entry == name);
    let properties = instance.get_physical_device_properties(physical_device);
    let api_version = properties.api_version;

    Ok(VulkanDeviceExtensionSupport {
        swapchain: has(swapchain::NAME.to_str().unwrap_or("VK_KHR_swapchain")),
        timeline_semaphore: api_version >= vk::API_VERSION_1_2 || has("VK_KHR_timeline_semaphore"),
        external_memory_fd: has("VK_KHR_external_memory_fd"),
        external_semaphore_fd: has("VK_KHR_external_semaphore_fd"),
        buffer_device_address: api_version >= vk::API_VERSION_1_2
            || has("VK_KHR_buffer_device_address"),
        dynamic_rendering: api_version >= vk::API_VERSION_1_3 || has("VK_KHR_dynamic_rendering"),
    })
}

unsafe fn create_logical_device(
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
    selection: VulkanQueueSelection,
) -> Result<(Device, vk::Queue, vk::Queue), VulkanBackendError> {
    let queue_priority = [1.0_f32];
    let mut unique_families = vec![selection.graphics_family_index];
    let present_family_index = selection
        .present_family_index
        .ok_or(VulkanBackendError::NoPresentQueue)?;
    if present_family_index != selection.graphics_family_index {
        unique_families.push(present_family_index);
    }
    if let Some(compute_family_index) = selection.compute_family_index {
        if !unique_families.contains(&compute_family_index) {
            unique_families.push(compute_family_index);
        }
    }
    if let Some(transfer_family_index) = selection.transfer_family_index {
        if !unique_families.contains(&transfer_family_index) {
            unique_families.push(transfer_family_index);
        }
    }

    let queue_infos = unique_families
        .iter()
        .map(|family_index| {
            vk::DeviceQueueCreateInfo::default()
                .queue_family_index(*family_index)
                .queue_priorities(&queue_priority)
        })
        .collect::<Vec<_>>();
    let device_extensions = [swapchain::NAME.as_ptr()];
    let device_info = vk::DeviceCreateInfo::default()
        .queue_create_infos(&queue_infos)
        .enabled_extension_names(&device_extensions);

    let device = instance.create_device(physical_device, &device_info, None)?;
    let graphics_queue = device.get_device_queue(selection.graphics_family_index, 0);
    let present_queue = device.get_device_queue(present_family_index, 0);
    Ok((device, graphics_queue, present_queue))
}

unsafe fn create_secondary_device_state(
    instance: &Instance,
    profile: &VulkanPhysicalDeviceProfile,
) -> Result<SecondaryDeviceState, VulkanBackendError> {
    let queue_priority = [1.0_f32];
    let mut unique_families = vec![profile.queue_selection.graphics_family_index];
    if let Some(compute_family_index) = profile.queue_selection.compute_family_index {
        if !unique_families.contains(&compute_family_index) {
            unique_families.push(compute_family_index);
        }
    }
    if let Some(transfer_family_index) = profile.queue_selection.transfer_family_index {
        if !unique_families.contains(&transfer_family_index) {
            unique_families.push(transfer_family_index);
        }
    }

    let queue_infos = unique_families
        .iter()
        .map(|family_index| {
            vk::DeviceQueueCreateInfo::default()
                .queue_family_index(*family_index)
                .queue_priorities(&queue_priority)
        })
        .collect::<Vec<_>>();
    let device_extensions = [swapchain::NAME.as_ptr()];
    let device_info = vk::DeviceCreateInfo::default()
        .queue_create_infos(&queue_infos)
        .enabled_extension_names(&device_extensions);
    let device = instance.create_device(profile.physical_device, &device_info, None)?;
    let graphics_queue = device.get_device_queue(profile.queue_selection.graphics_family_index, 0);
    let command_pool_info = vk::CommandPoolCreateInfo::default()
        .queue_family_index(profile.queue_selection.graphics_family_index)
        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
    let command_pool = device.create_command_pool(&command_pool_info, None)?;
    let command_buffer_info = vk::CommandBufferAllocateInfo::default()
        .command_pool(command_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);
    let command_buffer = device.allocate_command_buffers(&command_buffer_info)?[0];
    let fence_info = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);
    let in_flight_fence = device.create_fence(&fence_info, None)?;

    Ok(SecondaryDeviceState {
        profile: profile.clone(),
        device,
        graphics_queue,
        command_pool,
        command_buffer,
        in_flight_fence,
    })
}

unsafe fn create_swapchain_state(
    instance: &Instance,
    device: &Device,
    physical_device: vk::PhysicalDevice,
    surface_loader: &surface::Instance,
    surface: vk::SurfaceKHR,
    selection: VulkanQueueSelection,
    window_size: PhysicalSize<u32>,
    present_preference: PresentModePreference,
) -> Result<SwapchainState, VulkanBackendError> {
    let present_family_index = selection
        .present_family_index
        .ok_or(VulkanBackendError::NoPresentQueue)?;
    let capabilities =
        surface_loader.get_physical_device_surface_capabilities(physical_device, surface)?;
    let formats = surface_loader.get_physical_device_surface_formats(physical_device, surface)?;
    if formats.is_empty() {
        return Err(VulkanBackendError::NoSurfaceFormat);
    }
    let present_modes =
        surface_loader.get_physical_device_surface_present_modes(physical_device, surface)?;
    let format = choose_surface_format(&formats);
    let extent = choose_extent(capabilities, window_size);
    let present_mode = choose_present_mode(&present_modes, present_preference);
    let mut image_count = capabilities.min_image_count.saturating_add(1);
    if capabilities.max_image_count > 0 {
        image_count = image_count.min(capabilities.max_image_count);
    }
    image_count = image_count.max(2);

    let family_indices = [selection.graphics_family_index, present_family_index];
    let (sharing_mode, queue_family_indices) =
        if selection.graphics_family_index != present_family_index {
            (vk::SharingMode::CONCURRENT, family_indices.as_slice())
        } else {
            (vk::SharingMode::EXCLUSIVE, &family_indices[..1])
        };

    let create_info = vk::SwapchainCreateInfoKHR::default()
        .surface(surface)
        .min_image_count(image_count)
        .image_format(format.format)
        .image_color_space(format.color_space)
        .image_extent(extent)
        .image_array_layers(1)
        .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
        .image_sharing_mode(sharing_mode)
        .queue_family_indices(queue_family_indices)
        .pre_transform(capabilities.current_transform)
        .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
        .present_mode(present_mode)
        .clipped(true);

    let loader = swapchain::Device::new(instance, device);
    let swapchain = loader.create_swapchain(&create_info, None)?;
    let images = loader.get_swapchain_images(swapchain)?;
    if images.is_empty() {
        return Err(VulkanBackendError::NoSwapchainImages);
    }
    let image_views = images
        .iter()
        .map(|image| {
            let view_info = vk::ImageViewCreateInfo::default()
                .image(*image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(format.format)
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .base_mip_level(0)
                        .level_count(1)
                        .base_array_layer(0)
                        .layer_count(1),
                );
            device.create_image_view(&view_info, None)
        })
        .collect::<Result<Vec<_>, _>>()?;

    Ok(SwapchainState {
        loader,
        swapchain,
        format: format.format,
        extent,
        present_mode,
        _images: images,
        image_views,
        framebuffers: Vec::new(),
    })
}

fn choose_surface_format(formats: &[vk::SurfaceFormatKHR]) -> vk::SurfaceFormatKHR {
    formats
        .iter()
        .copied()
        .find(|format| {
            format.format == vk::Format::B8G8R8A8_UNORM
                && format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
        })
        .unwrap_or(formats[0])
}

fn choose_present_mode(
    supported_modes: &[vk::PresentModeKHR],
    preference: PresentModePreference,
) -> vk::PresentModeKHR {
    let prefers = match preference {
        PresentModePreference::MailboxFirst => [
            vk::PresentModeKHR::MAILBOX,
            vk::PresentModeKHR::IMMEDIATE,
            vk::PresentModeKHR::FIFO,
        ],
        PresentModePreference::ImmediateFirst => [
            vk::PresentModeKHR::IMMEDIATE,
            vk::PresentModeKHR::MAILBOX,
            vk::PresentModeKHR::FIFO,
        ],
        PresentModePreference::FifoOnly => [
            vk::PresentModeKHR::FIFO,
            vk::PresentModeKHR::MAILBOX,
            vk::PresentModeKHR::IMMEDIATE,
        ],
    };
    prefers
        .into_iter()
        .find(|mode| supported_modes.contains(mode))
        .unwrap_or(vk::PresentModeKHR::FIFO)
}

fn choose_extent(
    capabilities: vk::SurfaceCapabilitiesKHR,
    window_size: PhysicalSize<u32>,
) -> vk::Extent2D {
    if capabilities.current_extent.width != u32::MAX {
        return capabilities.current_extent;
    }
    vk::Extent2D {
        width: window_size.width.clamp(
            capabilities.min_image_extent.width,
            capabilities.max_image_extent.width,
        ),
        height: window_size.height.clamp(
            capabilities.min_image_extent.height,
            capabilities.max_image_extent.height,
        ),
    }
}

unsafe fn create_render_pass(
    device: &Device,
    color_format: vk::Format,
) -> Result<vk::RenderPass, VulkanBackendError> {
    let color_attachment = vk::AttachmentDescription::default()
        .format(color_format)
        .samples(vk::SampleCountFlags::TYPE_1)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::STORE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(vk::ImageLayout::PRESENT_SRC_KHR);
    let color_ref = vk::AttachmentReference::default()
        .attachment(0)
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
    let color_refs = [color_ref];
    let subpass = vk::SubpassDescription::default()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(&color_refs);
    let dependency = vk::SubpassDependency::default()
        .src_subpass(vk::SUBPASS_EXTERNAL)
        .dst_subpass(0)
        .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE);
    let attachments = [color_attachment];
    let subpasses = [subpass];
    let dependencies = [dependency];
    let render_pass_info = vk::RenderPassCreateInfo::default()
        .attachments(&attachments)
        .subpasses(&subpasses)
        .dependencies(&dependencies);
    Ok(device.create_render_pass(&render_pass_info, None)?)
}

unsafe fn create_framebuffers(
    device: &Device,
    render_pass: vk::RenderPass,
    swapchain: &mut SwapchainState,
) -> Result<(), VulkanBackendError> {
    swapchain.framebuffers = swapchain
        .image_views
        .iter()
        .map(|image_view| {
            let attachments = [*image_view];
            let framebuffer_info = vk::FramebufferCreateInfo::default()
                .render_pass(render_pass)
                .attachments(&attachments)
                .width(swapchain.extent.width)
                .height(swapchain.extent.height)
                .layers(1);
            device.create_framebuffer(&framebuffer_info, None)
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(())
}

unsafe fn create_frame_resources(
    instance: &Instance,
    device: &Device,
    physical_device: vk::PhysicalDevice,
    graphics_family_index: u32,
    frames_in_flight: usize,
    max_instances: usize,
    scene_pipeline: &ScenePipelineResources,
) -> Result<Vec<FrameResources>, VulkanBackendError> {
    let mut frames = Vec::with_capacity(frames_in_flight);
    let descriptor_set_layouts = vec![scene_pipeline.descriptor_set_layout; frames_in_flight];
    let descriptor_set_info = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(scene_pipeline.descriptor_pool)
        .set_layouts(&descriptor_set_layouts);
    let descriptor_sets = device.allocate_descriptor_sets(&descriptor_set_info)?;
    for descriptor_set in descriptor_sets {
        let command_pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(graphics_family_index)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let command_pool = device.create_command_pool(&command_pool_info, None)?;
        let command_buffer_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let command_buffer = device.allocate_command_buffers(&command_buffer_info)?[0];
        let semaphore_info = vk::SemaphoreCreateInfo::default();
        let image_available = device.create_semaphore(&semaphore_info, None)?;
        let render_finished = device.create_semaphore(&semaphore_info, None)?;
        let fence_info = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);
        let in_flight_fence = device.create_fence(&fence_info, None)?;
        let snapshot_slot = create_snapshot_slot(
            instance,
            device,
            physical_device,
            max_instances,
            max_instances,
            max_instances,
            SNAPSHOT_LIGHT_CAPACITY,
        )?;
        let camera_uniform = create_mapped_buffer(
            instance,
            device,
            physical_device,
            size_of::<CameraUniform>() as vk::DeviceSize,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        let camera_buffer_info = [vk::DescriptorBufferInfo::default()
            .buffer(camera_uniform.buffer)
            .offset(0)
            .range(size_of::<CameraUniform>() as vk::DeviceSize)];
        let material_buffer_info = [vk::DescriptorBufferInfo::default()
            .buffer(snapshot_slot.mapped.buffer)
            .offset(snapshot_slot.materials_offset as vk::DeviceSize)
            .range(
                (snapshot_slot.material_capacity * size_of::<FrameMaterialRecord>())
                    as vk::DeviceSize,
            )];
    let texture_buffer_info = [vk::DescriptorBufferInfo::default()
        .buffer(snapshot_slot.mapped.buffer)
        .offset(snapshot_slot.textures_offset as vk::DeviceSize)
        .range(
            (snapshot_slot.texture_capacity * size_of::<FrameTextureRecord>())
                as vk::DeviceSize,
        )];
    let light_buffer_info = [vk::DescriptorBufferInfo::default()
        .buffer(snapshot_slot.mapped.buffer)
        .offset(snapshot_slot.lights_offset as vk::DeviceSize)
        .range(
            (snapshot_slot.light_capacity * size_of::<FrameLightRecord>()) as vk::DeviceSize,
        )];
    let texture_image_info = [vk::DescriptorImageInfo::default()
        .sampler(scene_pipeline.texture_array.sampler)
        .image_view(scene_pipeline.texture_array.image_view)
        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
    let descriptor_writes = [
        vk::WriteDescriptorSet::default()
            .dst_set(descriptor_set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(&camera_buffer_info),
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&material_buffer_info),
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&light_buffer_info),
        vk::WriteDescriptorSet::default()
            .dst_set(descriptor_set)
            .dst_binding(3)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(&texture_buffer_info),
        vk::WriteDescriptorSet::default()
            .dst_set(descriptor_set)
            .dst_binding(4)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(&texture_image_info),
    ];
        device.update_descriptor_sets(&descriptor_writes, &[]);
        frames.push(FrameResources {
            command_pool,
            command_buffer,
            image_available,
            render_finished,
            in_flight_fence,
            camera_uniform,
            camera_descriptor_set: descriptor_set,
            snapshot_slot,
        });
    }
    Ok(frames)
}

unsafe fn create_snapshot_slot(
    instance: &Instance,
    device: &Device,
    physical_device: vk::PhysicalDevice,
    max_instances: usize,
    max_materials: usize,
    max_textures: usize,
    max_lights: usize,
) -> Result<SnapshotSlot, VulkanBackendError> {
    let header_offset = 0;
    let transforms_offset = align_up(
        size_of::<SnapshotHeader>(),
        align_of::<FrameInstanceTransform>(),
    );
    let materials_offset = align_up(
        transforms_offset
            .checked_add(max_instances * size_of::<FrameInstanceTransform>())
            .ok_or(VulkanBackendError::InvalidConfig(
                "snapshot transform buffer size overflow",
            ))?,
        align_of::<FrameMaterialRecord>(),
    );
    let lights_offset = align_up(
        align_up(
            materials_offset
                .checked_add(max_materials * size_of::<FrameMaterialRecord>())
                .ok_or(VulkanBackendError::InvalidConfig(
                    "snapshot material buffer size overflow",
                ))?,
            align_of::<FrameTextureRecord>(),
        )
        .checked_add(max_textures * size_of::<FrameTextureRecord>())
        .ok_or(VulkanBackendError::InvalidConfig(
            "snapshot texture buffer size overflow",
        ))?,
        align_of::<FrameLightRecord>(),
    );
    let textures_offset = align_up(
        materials_offset
            .checked_add(max_materials * size_of::<FrameMaterialRecord>())
            .ok_or(VulkanBackendError::InvalidConfig(
                "snapshot material buffer size overflow",
            ))?,
        align_of::<FrameTextureRecord>(),
    );
    let byte_len = lights_offset
        .checked_add(max_lights * size_of::<FrameLightRecord>())
        .ok_or(VulkanBackendError::InvalidConfig(
            "snapshot light buffer size overflow",
        ))?;
    let mapped = create_mapped_buffer(
        instance,
        device,
        physical_device,
        byte_len as vk::DeviceSize,
        vk::BufferUsageFlags::STORAGE_BUFFER,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )?;
    Ok(SnapshotSlot {
        mapped,
        header_offset,
        transforms_offset,
        materials_offset,
        textures_offset,
        lights_offset,
        instance_capacity: max_instances,
        material_capacity: max_materials,
        texture_capacity: max_textures,
        light_capacity: max_lights,
        last_state: VulkanSnapshotSlotState::default(),
    })
}

unsafe fn create_scene_pipeline_resources(
    instance: &Instance,
    device: &Device,
    physical_device: vk::PhysicalDevice,
    graphics_queue_family_index: u32,
    graphics_queue: vk::Queue,
    render_pass: vk::RenderPass,
    extent: vk::Extent2D,
    frames_in_flight: usize,
) -> Result<ScenePipelineResources, VulkanBackendError> {
    let descriptor_set_layout_bindings = [
        vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::VERTEX),
        vk::DescriptorSetLayoutBinding::default()
            .binding(1)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        vk::DescriptorSetLayoutBinding::default()
            .binding(2)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        vk::DescriptorSetLayoutBinding::default()
            .binding(3)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        vk::DescriptorSetLayoutBinding::default()
            .binding(4)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
    ];
    let descriptor_set_layout_info =
        vk::DescriptorSetLayoutCreateInfo::default().bindings(&descriptor_set_layout_bindings);
    let descriptor_set_layout =
        device.create_descriptor_set_layout(&descriptor_set_layout_info, None)?;

    let set_layouts = [descriptor_set_layout];
    let push_constant_ranges = [vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::FRAGMENT)
        .offset(0)
        .size(size_of::<DrawPushConstants>() as u32)];
    let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default()
        .set_layouts(&set_layouts)
        .push_constant_ranges(&push_constant_ranges);
    let pipeline_layout = device.create_pipeline_layout(&pipeline_layout_info, None)?;

    let descriptor_pool_sizes = [
        vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(frames_in_flight as u32),
        vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count((frames_in_flight * 3) as u32),
        vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(frames_in_flight as u32),
    ];
    let descriptor_pool_info = vk::DescriptorPoolCreateInfo::default()
        .max_sets(frames_in_flight as u32)
        .pool_sizes(&descriptor_pool_sizes);
    let descriptor_pool = device.create_descriptor_pool(&descriptor_pool_info, None)?;

    let cube_vertices = unit_cube_vertices();
    let cube_indices = unit_cube_indices();
    let vertex_buffer = create_static_buffer(
        instance,
        device,
        physical_device,
        slice_as_bytes(cube_vertices.as_slice()),
        vk::BufferUsageFlags::VERTEX_BUFFER,
    )?;
    let index_buffer = create_static_buffer(
        instance,
        device,
        physical_device,
        slice_as_bytes(cube_indices.as_slice()),
        vk::BufferUsageFlags::INDEX_BUFFER,
    )?;
    let texture_array = create_dummy_texture_array_resources(
        instance,
        device,
        physical_device,
        graphics_queue_family_index,
        graphics_queue,
    )?;
    let pipeline = create_scene_pipeline(device, render_pass, extent, pipeline_layout)?;

    Ok(ScenePipelineResources {
        descriptor_set_layout,
        descriptor_pool,
        pipeline_layout,
        pipeline,
        vertex_buffer,
        index_buffer,
        index_count: cube_indices.len() as u32,
        texture_array,
    })
}

unsafe fn create_mapped_buffer(
    instance: &Instance,
    device: &Device,
    physical_device: vk::PhysicalDevice,
    size: vk::DeviceSize,
    usage: vk::BufferUsageFlags,
    required_memory: vk::MemoryPropertyFlags,
) -> Result<PersistentlyMappedBuffer, VulkanBackendError> {
    let buffer_info = vk::BufferCreateInfo::default()
        .size(size)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    let buffer = device.create_buffer(&buffer_info, None)?;
    let requirements = device.get_buffer_memory_requirements(buffer);
    let memory_type_index = find_memory_type(
        instance,
        physical_device,
        requirements.memory_type_bits,
        required_memory,
    )
    .ok_or(VulkanBackendError::InvalidConfig(
        "no compatible Vulkan memory type for mapped buffer",
    ))?;
    let alloc_info = vk::MemoryAllocateInfo::default()
        .allocation_size(requirements.size)
        .memory_type_index(memory_type_index);
    let memory = device.allocate_memory(&alloc_info, None)?;
    device.bind_buffer_memory(buffer, memory, 0)?;
    let mapped_ptr = device.map_memory(memory, 0, size, vk::MemoryMapFlags::empty())?;
    let mapped_ptr = NonNull::new(mapped_ptr as *mut u8).ok_or(
        VulkanBackendError::InvalidConfig("Vulkan returned null mapped pointer"),
    )?;
    Ok(PersistentlyMappedBuffer {
        buffer,
        memory,
        mapped_ptr,
        byte_len: size as usize,
    })
}

unsafe fn create_static_buffer(
    instance: &Instance,
    device: &Device,
    physical_device: vk::PhysicalDevice,
    bytes: &[u8],
    usage: vk::BufferUsageFlags,
) -> Result<PersistentlyMappedBuffer, VulkanBackendError> {
    let mapped = create_mapped_buffer(
        instance,
        device,
        physical_device,
        bytes.len() as vk::DeviceSize,
        usage,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )?;
    ptr::copy_nonoverlapping(bytes.as_ptr(), mapped.mapped_ptr.as_ptr(), bytes.len());
    Ok(mapped)
}

fn find_memory_type(
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
    memory_type_bits: u32,
    required_memory: vk::MemoryPropertyFlags,
) -> Option<u32> {
    let properties = unsafe { instance.get_physical_device_memory_properties(physical_device) };
    for index in 0..properties.memory_type_count {
        let is_supported = memory_type_bits & (1_u32 << index) != 0;
        let has_flags = properties.memory_types[index as usize]
            .property_flags
            .contains(required_memory);
        if is_supported && has_flags {
            return Some(index);
        }
    }
    None
}

unsafe fn create_scene_pipeline(
    device: &Device,
    render_pass: vk::RenderPass,
    extent: vk::Extent2D,
    pipeline_layout: vk::PipelineLayout,
) -> Result<vk::Pipeline, VulkanBackendError> {
    let [vertex_artifact, fragment_artifact] = build_scene_shader_spirv_artifacts()?;
    let vertex_module = create_shader_module(device, &vertex_artifact.words)?;
    let fragment_module = create_shader_module(device, &fragment_artifact.words)?;

    let entry_name = CString::new("main")
        .map_err(|_| VulkanBackendError::InvalidConfig("invalid Vulkan shader entrypoint"))?;
    let shader_stages = [
        vk::PipelineShaderStageCreateInfo::default()
            .module(vertex_module)
            .name(&entry_name)
            .stage(vk::ShaderStageFlags::VERTEX),
        vk::PipelineShaderStageCreateInfo::default()
            .module(fragment_module)
            .name(&entry_name)
            .stage(vk::ShaderStageFlags::FRAGMENT),
    ];

    let vertex_binding_descriptions = [
        vk::VertexInputBindingDescription {
            binding: 0,
            stride: size_of::<SceneVertex>() as u32,
            input_rate: vk::VertexInputRate::VERTEX,
        },
        vk::VertexInputBindingDescription {
            binding: 1,
            stride: size_of::<FrameInstanceTransform>() as u32,
            input_rate: vk::VertexInputRate::INSTANCE,
        },
    ];
    let instance_model_offset = offset_of!(FrameInstanceTransform, model) as u32;
    let instance_color_offset = offset_of!(FrameInstanceTransform, color_rgba) as u32;
    let instance_material_index_offset = offset_of!(FrameInstanceTransform, material_index) as u32;
    let instance_flags_offset = offset_of!(FrameInstanceTransform, flags) as u32;
    let vertex_attribute_descriptions = [
        vk::VertexInputAttributeDescription {
            location: 0,
            binding: 0,
            format: vk::Format::R32G32B32_SFLOAT,
            offset: offset_of!(SceneVertex, position) as u32,
        },
        vk::VertexInputAttributeDescription {
            location: 1,
            binding: 1,
            format: vk::Format::R32G32B32A32_SFLOAT,
            offset: instance_model_offset,
        },
        vk::VertexInputAttributeDescription {
            location: 2,
            binding: 1,
            format: vk::Format::R32G32B32A32_SFLOAT,
            offset: instance_model_offset + 16,
        },
        vk::VertexInputAttributeDescription {
            location: 3,
            binding: 1,
            format: vk::Format::R32G32B32A32_SFLOAT,
            offset: instance_model_offset + 32,
        },
        vk::VertexInputAttributeDescription {
            location: 4,
            binding: 1,
            format: vk::Format::R32G32B32A32_SFLOAT,
            offset: instance_model_offset + 48,
        },
        vk::VertexInputAttributeDescription {
            location: 5,
            binding: 1,
            format: vk::Format::R32G32B32A32_SFLOAT,
            offset: instance_color_offset,
        },
        vk::VertexInputAttributeDescription {
            location: 6,
            binding: 1,
            format: vk::Format::R32_UINT,
            offset: instance_material_index_offset,
        },
        vk::VertexInputAttributeDescription {
            location: 7,
            binding: 1,
            format: vk::Format::R32_UINT,
            offset: instance_flags_offset,
        },
    ];
    let vertex_input_state = vk::PipelineVertexInputStateCreateInfo::default()
        .vertex_binding_descriptions(&vertex_binding_descriptions)
        .vertex_attribute_descriptions(&vertex_attribute_descriptions);
    let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
        .primitive_restart_enable(false);
    let viewports = [vk::Viewport {
        x: 0.0,
        y: 0.0,
        width: extent.width as f32,
        height: extent.height as f32,
        min_depth: 0.0,
        max_depth: 1.0,
    }];
    let scissors = [vk::Rect2D::default().extent(extent)];
    let viewport_state = vk::PipelineViewportStateCreateInfo::default()
        .viewports(&viewports)
        .scissors(&scissors);
    let rasterizer = vk::PipelineRasterizationStateCreateInfo::default()
        .polygon_mode(vk::PolygonMode::FILL)
        .cull_mode(vk::CullModeFlags::BACK)
        .front_face(vk::FrontFace::CLOCKWISE)
        .line_width(1.0);
    let multisampling = vk::PipelineMultisampleStateCreateInfo::default()
        .rasterization_samples(vk::SampleCountFlags::TYPE_1);
    let color_blend_attachment = [vk::PipelineColorBlendAttachmentState::default()
        .blend_enable(false)
        .color_write_mask(vk::ColorComponentFlags::RGBA)];
    let color_blending =
        vk::PipelineColorBlendStateCreateInfo::default().attachments(&color_blend_attachment);
    let pipeline_info = [vk::GraphicsPipelineCreateInfo::default()
        .stages(&shader_stages)
        .vertex_input_state(&vertex_input_state)
        .input_assembly_state(&input_assembly)
        .viewport_state(&viewport_state)
        .rasterization_state(&rasterizer)
        .multisample_state(&multisampling)
        .color_blend_state(&color_blending)
        .layout(pipeline_layout)
        .render_pass(render_pass)
        .subpass(0)];
    let pipeline = device
        .create_graphics_pipelines(vk::PipelineCache::null(), &pipeline_info, None)
        .map_err(|(_, err)| VulkanBackendError::Vk(err))?[0];

    device.destroy_shader_module(vertex_module, None);
    device.destroy_shader_module(fragment_module, None);
    Ok(pipeline)
}

fn build_scene_shader_spirv_artifacts() -> Result<[SpirvShaderArtifact; 2], VulkanBackendError> {
    Ok([
        SpirvShaderArtifact {
            words: spirv_bytes_to_words(include_bytes!("../../assets/shaders/spv/scene.vert.spv"))?,
        },
        SpirvShaderArtifact {
            words: spirv_bytes_to_words(include_bytes!("../../assets/shaders/spv/scene.frag.spv"))?,
        },
    ])
}

fn spirv_bytes_to_words(bytes: &[u8]) -> Result<Vec<u32>, VulkanBackendError> {
    let chunks = bytes.chunks_exact(4);
    if !chunks.remainder().is_empty() {
        return Err(VulkanBackendError::InvalidConfig(
            "embedded SPIR-V artifact length is not aligned to 4 bytes",
        ));
    }
    Ok(chunks
        .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

unsafe fn create_shader_module(
    device: &Device,
    spirv: &[u32],
) -> Result<vk::ShaderModule, VulkanBackendError> {
    let info = vk::ShaderModuleCreateInfo::default().code(spirv);
    Ok(device.create_shader_module(&info, None)?)
}

unsafe fn create_dummy_texture_array_resources(
    instance: &Instance,
    device: &Device,
    physical_device: vk::PhysicalDevice,
    graphics_queue_family_index: u32,
    graphics_queue: vk::Queue,
) -> Result<SampledTextureArrayResources, VulkanBackendError> {
    const TEX_WIDTH: u32 = 2;
    const TEX_HEIGHT: u32 = 2;
    const TEX_LAYERS: u32 = 4;
    const BYTES_PER_PIXEL: usize = 4;

    let texels: [[u8; TEX_WIDTH as usize * TEX_HEIGHT as usize * BYTES_PER_PIXEL];
        TEX_LAYERS as usize] = [
        [
            255, 255, 255, 255, 220, 220, 255, 255, 220, 220, 255, 255, 255, 255, 255, 255,
        ],
        [
            255, 128, 128, 255, 220, 72, 72, 255, 220, 72, 72, 255, 255, 128, 128, 255,
        ],
        [
            128, 255, 160, 255, 72, 220, 96, 255, 72, 220, 96, 255, 128, 255, 160, 255,
        ],
        [
            128, 196, 255, 255, 72, 120, 220, 255, 72, 120, 220, 255, 128, 196, 255, 255,
        ],
    ];
    let mut staging_bytes =
        Vec::with_capacity((TEX_WIDTH * TEX_HEIGHT * TEX_LAYERS * 4) as usize);
    for layer in texels {
        staging_bytes.extend_from_slice(&layer);
    }

    let staging = create_static_buffer(
        instance,
        device,
        physical_device,
        staging_bytes.as_slice(),
        vk::BufferUsageFlags::TRANSFER_SRC,
    )?;

    let image_info = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(vk::Format::R8G8B8A8_UNORM)
        .extent(vk::Extent3D {
            width: TEX_WIDTH,
            height: TEX_HEIGHT,
            depth: 1,
        })
        .mip_levels(1)
        .array_layers(TEX_LAYERS)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .initial_layout(vk::ImageLayout::UNDEFINED);
    let image = device.create_image(&image_info, None)?;
    let requirements = device.get_image_memory_requirements(image);
    let memory_type_index = find_memory_type(
        instance,
        physical_device,
        requirements.memory_type_bits,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )
    .ok_or(VulkanBackendError::InvalidConfig(
        "no compatible Vulkan memory type for sampled texture array",
    ))?;
    let alloc_info = vk::MemoryAllocateInfo::default()
        .allocation_size(requirements.size)
        .memory_type_index(memory_type_index);
    let image_memory = device.allocate_memory(&alloc_info, None)?;
    device.bind_image_memory(image, image_memory, 0)?;

    let command_pool_info = vk::CommandPoolCreateInfo::default()
        .queue_family_index(graphics_queue_family_index)
        .flags(vk::CommandPoolCreateFlags::TRANSIENT);
    let command_pool = device.create_command_pool(&command_pool_info, None)?;
    let alloc_info = vk::CommandBufferAllocateInfo::default()
        .command_pool(command_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);
    let command_buffer = device.allocate_command_buffers(&alloc_info)?[0];
    let begin_info = vk::CommandBufferBeginInfo::default()
        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
    device.begin_command_buffer(command_buffer, &begin_info)?;

    let subresource_range = vk::ImageSubresourceRange::default()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .base_mip_level(0)
        .level_count(1)
        .base_array_layer(0)
        .layer_count(TEX_LAYERS);
    let to_transfer_barrier = vk::ImageMemoryBarrier::default()
        .old_layout(vk::ImageLayout::UNDEFINED)
        .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(subresource_range)
        .src_access_mask(vk::AccessFlags::empty())
        .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE);
    device.cmd_pipeline_barrier(
        command_buffer,
        vk::PipelineStageFlags::TOP_OF_PIPE,
        vk::PipelineStageFlags::TRANSFER,
        vk::DependencyFlags::empty(),
        &[],
        &[],
        &[to_transfer_barrier],
    );

    let regions = (0..TEX_LAYERS)
        .map(|layer| {
            vk::BufferImageCopy::default()
                .buffer_offset(
                    (layer as usize * TEX_WIDTH as usize * TEX_HEIGHT as usize * BYTES_PER_PIXEL)
                        as u64,
                )
                .buffer_row_length(0)
                .buffer_image_height(0)
                .image_subresource(
                    vk::ImageSubresourceLayers::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .mip_level(0)
                        .base_array_layer(layer)
                        .layer_count(1),
                )
                .image_extent(vk::Extent3D {
                    width: TEX_WIDTH,
                    height: TEX_HEIGHT,
                    depth: 1,
                })
        })
        .collect::<Vec<_>>();
    device.cmd_copy_buffer_to_image(
        command_buffer,
        staging.buffer,
        image,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        regions.as_slice(),
    );

    let to_shader_read_barrier = vk::ImageMemoryBarrier::default()
        .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
        .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(subresource_range)
        .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
        .dst_access_mask(vk::AccessFlags::SHADER_READ);
    device.cmd_pipeline_barrier(
        command_buffer,
        vk::PipelineStageFlags::TRANSFER,
        vk::PipelineStageFlags::FRAGMENT_SHADER,
        vk::DependencyFlags::empty(),
        &[],
        &[],
        &[to_shader_read_barrier],
    );
    device.end_command_buffer(command_buffer)?;

    let command_buffers = [command_buffer];
    let submit_info = vk::SubmitInfo::default().command_buffers(&command_buffers);
    let fence = device.create_fence(&vk::FenceCreateInfo::default(), None)?;
    device.queue_submit(graphics_queue, &[submit_info], fence)?;
    device.wait_for_fences(&[fence], true, u64::MAX)?;
    device.destroy_fence(fence, None);
    device.free_command_buffers(command_pool, &[command_buffer]);
    device.destroy_command_pool(command_pool, None);
    destroy_mapped_buffer(device, staging);

    let view_info = vk::ImageViewCreateInfo::default()
        .image(image)
        .view_type(vk::ImageViewType::TYPE_2D_ARRAY)
        .format(vk::Format::R8G8B8A8_UNORM)
        .subresource_range(
            vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_mip_level(0)
                .level_count(1)
                .base_array_layer(0)
                .layer_count(TEX_LAYERS),
        );
    let image_view = device.create_image_view(&view_info, None)?;
    let sampler_info = vk::SamplerCreateInfo::default()
        .mag_filter(vk::Filter::LINEAR)
        .min_filter(vk::Filter::LINEAR)
        .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
        .address_mode_u(vk::SamplerAddressMode::REPEAT)
        .address_mode_v(vk::SamplerAddressMode::REPEAT)
        .address_mode_w(vk::SamplerAddressMode::REPEAT)
        .max_lod(1.0);
    let sampler = device.create_sampler(&sampler_info, None)?;

    Ok(SampledTextureArrayResources {
        image,
        image_memory,
        image_view,
        sampler,
    })
}

unsafe fn record_frame_commands(
    device: &Device,
    render_pass: vk::RenderPass,
    extent: vk::Extent2D,
    framebuffer: vk::Framebuffer,
    command_buffer: vk::CommandBuffer,
    frame: &FrameResources,
    scene_pipeline: &ScenePipelineResources,
    snapshot_state: VulkanSnapshotSlotState,
) -> Result<(), VulkanBackendError> {
    let begin_info = vk::CommandBufferBeginInfo::default();
    device.begin_command_buffer(command_buffer, &begin_info)?;

    let clear_values = [vk::ClearValue {
        color: vk::ClearColorValue {
            float32: [0.07, 0.09, 0.12, 1.0],
        },
    }];
    let render_area = vk::Rect2D::default().extent(extent);
    let render_pass_info = vk::RenderPassBeginInfo::default()
        .render_pass(render_pass)
        .framebuffer(framebuffer)
        .render_area(render_area)
        .clear_values(&clear_values);
    device.cmd_begin_render_pass(
        command_buffer,
        &render_pass_info,
        vk::SubpassContents::INLINE,
    );

    let vertex_buffers = [
        scene_pipeline.vertex_buffer.buffer,
        frame.snapshot_slot.mapped.buffer,
    ];
    let vertex_offsets = [0, frame.snapshot_slot.transforms_offset as vk::DeviceSize];
    device.cmd_bind_pipeline(
        command_buffer,
        vk::PipelineBindPoint::GRAPHICS,
        scene_pipeline.pipeline,
    );
    device.cmd_bind_descriptor_sets(
        command_buffer,
        vk::PipelineBindPoint::GRAPHICS,
        scene_pipeline.pipeline_layout,
        0,
        &[frame.camera_descriptor_set],
        &[],
    );
        let push_constants = DrawPushConstants {
            material_count: snapshot_state.material_count,
            light_count: snapshot_state.light_count,
            texture_count: snapshot_state.texture_count,
            _padding0: 0,
        };
    device.cmd_push_constants(
        command_buffer,
        scene_pipeline.pipeline_layout,
        vk::ShaderStageFlags::FRAGMENT,
        0,
        slice_as_bytes(std::slice::from_ref(&push_constants)),
    );
    device.cmd_bind_vertex_buffers(command_buffer, 0, &vertex_buffers, &vertex_offsets);
    device.cmd_bind_index_buffer(
        command_buffer,
        scene_pipeline.index_buffer.buffer,
        0,
        vk::IndexType::UINT16,
    );
    if snapshot_state.instance_count > 0 {
        device.cmd_draw_indexed(
            command_buffer,
            scene_pipeline.index_count,
            snapshot_state.instance_count,
            0,
            0,
            0,
        );
    }

    device.cmd_end_render_pass(command_buffer);
    device.end_command_buffer(command_buffer)?;
    Ok(())
}

unsafe fn record_secondary_queue_probe(
    secondary: &mut SecondaryDeviceState,
    current_serial: u64,
) -> Result<u64, VulkanBackendError> {
    secondary
        .device
        .wait_for_fences(&[secondary.in_flight_fence], true, u64::MAX)?;
    secondary
        .device
        .reset_fences(&[secondary.in_flight_fence])?;
    secondary
        .device
        .reset_command_pool(secondary.command_pool, vk::CommandPoolResetFlags::empty())?;
    let begin_info = vk::CommandBufferBeginInfo::default();
    secondary
        .device
        .begin_command_buffer(secondary.command_buffer, &begin_info)?;
    secondary
        .device
        .end_command_buffer(secondary.command_buffer)?;
    let command_buffers = [secondary.command_buffer];
    let submit_info = vk::SubmitInfo::default().command_buffers(&command_buffers);
    secondary.device.queue_submit(
        secondary.graphics_queue,
        &[submit_info],
        secondary.in_flight_fence,
    )?;
    Ok(current_serial.saturating_add(1))
}

unsafe fn destroy_framebuffers(device: &Device, swapchain: &mut SwapchainState) {
    for framebuffer in swapchain.framebuffers.drain(..) {
        device.destroy_framebuffer(framebuffer, None);
    }
}

unsafe fn destroy_image_views(device: &Device, swapchain: &mut SwapchainState) {
    for image_view in swapchain.image_views.drain(..) {
        device.destroy_image_view(image_view, None);
    }
}

unsafe fn destroy_snapshot_slot(device: &Device, snapshot_slot: SnapshotSlot) {
    destroy_mapped_buffer(device, snapshot_slot.mapped);
}

unsafe fn destroy_mapped_buffer(device: &Device, buffer: PersistentlyMappedBuffer) {
    device.unmap_memory(buffer.memory);
    device.destroy_buffer(buffer.buffer, None);
    device.free_memory(buffer.memory, None);
}

unsafe fn destroy_mapped_buffer_ref(device: &Device, buffer: &PersistentlyMappedBuffer) {
    device.unmap_memory(buffer.memory);
    device.destroy_buffer(buffer.buffer, None);
    device.free_memory(buffer.memory, None);
}

fn slice_as_bytes<T>(slice: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const u8, std::mem::size_of_val(slice)) }
}

fn unit_cube_vertices() -> [SceneVertex; 8] {
    [
        SceneVertex {
            position: [-0.5, -0.5, -0.5],
        },
        SceneVertex {
            position: [0.5, -0.5, -0.5],
        },
        SceneVertex {
            position: [0.5, 0.5, -0.5],
        },
        SceneVertex {
            position: [-0.5, 0.5, -0.5],
        },
        SceneVertex {
            position: [-0.5, -0.5, 0.5],
        },
        SceneVertex {
            position: [0.5, -0.5, 0.5],
        },
        SceneVertex {
            position: [0.5, 0.5, 0.5],
        },
        SceneVertex {
            position: [-0.5, 0.5, 0.5],
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_material_and_light_layouts_are_std430_friendly() {
        assert_eq!(size_of::<FrameMaterialRecord>(), 48);
        assert_eq!(align_of::<FrameMaterialRecord>(), 16);
        assert_eq!(size_of::<FrameTextureRecord>(), 16);
        assert_eq!(align_of::<FrameTextureRecord>(), 16);
        assert_eq!(size_of::<FrameLightRecord>(), 80);
        assert_eq!(align_of::<FrameLightRecord>(), 16);
    }

    #[test]
    fn embedded_scene_shader_artifacts_are_non_empty() {
        let [vertex, fragment] = build_scene_shader_spirv_artifacts().expect("embedded SPIR-V");
        assert!(!vertex.words.is_empty());
        assert!(!fragment.words.is_empty());
    }
}

fn unit_cube_indices() -> [u16; 36] {
    [
        0, 1, 2, 2, 3, 0, // back
        4, 5, 6, 6, 7, 4, // front
        0, 4, 7, 7, 3, 0, // left
        1, 5, 6, 6, 2, 1, // right
        3, 2, 6, 6, 7, 3, // top
        0, 1, 5, 5, 4, 0, // bottom
    ]
}

const fn align_up(value: usize, alignment: usize) -> usize {
    if alignment == 0 {
        value
    } else {
        (value + alignment - 1) & !(alignment - 1)
    }
}
