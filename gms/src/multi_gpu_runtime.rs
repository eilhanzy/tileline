//! Portable explicit multi-GPU runtime helpers for GMS.
//!
//! This module executes a synthetic secondary GPU workload (for benchmarking and bring-up)
//! while relying on `bridge::MultiGpuDispatcher` for lane planning. Portable `wgpu` does not
//! expose direct cross-device texture sharing, so cross-adapter transfer is modeled with a
//! host-visible bridge (readback + upload ring buffers).

use std::collections::VecDeque;
use std::error::Error;
use std::fmt;
use std::fmt::Write as _;
use std::process::Command;
use std::sync::OnceLock;
use std::time::Duration;

use crate::bridge::{MultiGpuDispatchPlan, MultiGpuDispatcher, MultiGpuWorkloadRequest};
use crate::hardware::{
    safe_default_required_limits_for_adapter, GpuAdapterProfile, GpuInventory, MemoryTopology,
};
use crate::SharedTransferKind;
use wgpu::{Color, TextureFormat};

/// Initialization policy for the multi-GPU executor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MultiGpuInitPolicy {
    /// Best-effort enablement. Returns `Ok(None)` if no meaningful helper lane can be created.
    Auto,
    /// Require a secondary physical GPU helper lane. Returns an error when unavailable.
    Force,
}

/// Input parameters required to build a portable multi-GPU executor.
pub struct MultiGpuExecutorConfig {
    /// Auto vs force behavior when a valid secondary GPU is not available.
    pub policy: MultiGpuInitPolicy,
    /// Present GPU adapter info (used to avoid selecting the same physical GPU as secondary).
    pub primary_adapter_info: wgpu::AdapterInfo,
    /// Full discovered inventory (may include duplicate backends for the same physical GPU).
    pub inventory: GpuInventory,
    /// Primary/present GPU device handle used to allocate upload bridge resources.
    pub primary_device: wgpu::Device,
    /// Current frame width in pixels.
    pub frame_width: u32,
    /// Current frame height in pixels.
    pub frame_height: u32,
    /// Format used for the synthetic secondary offscreen target.
    pub secondary_offscreen_format: TextureFormat,
    /// Primary benchmark work units per present (used to scale the helper lane).
    pub primary_work_units_per_present: u32,
    /// Planner workload request for the multi-GPU split.
    pub workload_request: MultiGpuWorkloadRequest,
    /// Auto mode minimum projected gain threshold. Below this, the executor disables itself.
    pub auto_min_projected_gain_pct: f64,
}

/// Runtime/telemetry summary produced by the portable multi-GPU executor.
#[derive(Debug, Clone)]
pub struct MultiGpuExecutorSummary {
    pub primary_adapter_name: String,
    pub secondary_adapter_name: String,
    pub secondary_memory_topology: MemoryTopology,
    pub secondary_work_units_per_present: u32,
    pub total_secondary_work_units: u64,
    pub total_secondary_submissions: u64,
    pub projected_score_gain_pct: f64,
    pub meets_target_gain_20pct: bool,
    pub estimated_single_gpu_frame_ms: f64,
    pub estimated_multi_gpu_frame_ms: f64,
    pub bridge_kind: Option<SharedTransferKind>,
    pub bridge_bytes_per_frame: u64,
    pub bridge_chunk_bytes: u64,
    pub sync_frames_in_flight: u32,
    pub sync_queue_waits: u64,
    pub sync_queue_polls: u64,
    pub aggressive_integrated_preallocation: bool,
    pub integrated_encoder_pool: u32,
    pub integrated_ring_segments: u32,
    /// True when Vulkan version compatibility gating was applied.
    pub vulkan_version_gate_enabled: bool,
    /// Primary adapter Vulkan API version (major.minor), if detected.
    pub primary_vulkan_api_version: Option<String>,
    /// Secondary adapter Vulkan API version (major.minor), if detected.
    pub secondary_vulkan_api_version: Option<String>,
}

/// Result of one secondary helper-lane frame submission.
#[derive(Debug, Clone)]
pub struct MultiGpuFrameSubmitResult {
    /// Number of synthetic secondary work units submitted this present interval.
    pub work_units: u32,
    /// Submission index returned by `wgpu::Queue::submit` for sync registration.
    pub submission_index: Option<wgpu::SubmissionIndex>,
}

/// Portable explicit multi-GPU helper runtime.
///
/// This executor opens a secondary adapter/device and submits synthetic GPU work to it using the
/// planner output from `MultiGpuDispatcher`. It is intended to live in `gms/src` so benchmark code
/// can remain thin while real runtime integration evolves.
pub struct MultiGpuExecutor {
    plan: MultiGpuDispatchPlan,
    secondary_profile: GpuAdapterProfile,
    secondary_device: wgpu::Device,
    secondary_queue: wgpu::Queue,
    secondary_texture: wgpu::Texture,
    frame_width: u32,
    frame_height: u32,
    secondary_format: TextureFormat,
    secondary_clear_phase: f64,
    secondary_work_units_per_present: u32,
    bridge_runtime: Option<MultiGpuBridgeRuntime>,
    sync_state: MultiGpuSyncState,
    total_secondary_work_units: u64,
    total_secondary_submissions: u64,
    vulkan_version_gate_enabled: bool,
    primary_vulkan_api_version: Option<VulkanApiVersion>,
    secondary_vulkan_api_version: Option<VulkanApiVersion>,
}

struct MultiGpuBridgeRuntime {
    host_ring: Vec<Vec<u8>>,
    _producer_readback_buffers: Vec<wgpu::Buffer>,
    _consumer_upload_buffers: Vec<wgpu::Buffer>,
    ring_cursor: usize,
}

#[derive(Default)]
struct MultiGpuSyncState {
    pending_secondary_submissions: VecDeque<wgpu::SubmissionIndex>,
    poll_count: u64,
    wait_count: u64,
}

impl MultiGpuExecutor {
    /// Attempt to build a portable multi-GPU helper runtime.
    ///
    /// Returns `Ok(None)` in auto mode when no meaningful secondary physical GPU helper lane can
    /// be established.
    pub fn try_new(config: MultiGpuExecutorConfig) -> Result<Option<Self>, Box<dyn Error>> {
        let MultiGpuExecutorConfig {
            policy,
            primary_adapter_info,
            inventory,
            primary_device,
            frame_width,
            frame_height,
            secondary_offscreen_format,
            primary_work_units_per_present,
            workload_request,
            auto_min_projected_gain_pct,
        } = config;

        let present_profile = match_inventory_profile(&inventory, &primary_adapter_info);
        let multi_gpu_inventory = dedupe_inventory_by_physical_gpu(&inventory);
        let present_profile_mgpu = present_profile.as_ref().and_then(|present| {
            multi_gpu_inventory
                .adapters
                .iter()
                .find(|profile| same_physical_gpu_profile(profile, present))
                .cloned()
        });

        let dispatcher = MultiGpuDispatcher::new(multi_gpu_inventory.clone());
        let plan = dispatcher.plan_dispatch(workload_request);

        let Some(secondary_assignment) = plan.secondary() else {
            return match policy {
                MultiGpuInitPolicy::Auto => Ok(None),
                MultiGpuInitPolicy::Force => Err(Box::new(SimpleError(
                    build_no_secondary_assignment_diagnostic(
                        &primary_adapter_info,
                        &inventory,
                        &multi_gpu_inventory,
                        &plan,
                    ),
                ))),
            };
        };

        let Some(secondary_profile) = multi_gpu_inventory
            .adapters
            .iter()
            .find(|profile| profile.index == secondary_assignment.adapter_index)
            .cloned()
        else {
            return match policy {
                MultiGpuInitPolicy::Auto => Ok(None),
                MultiGpuInitPolicy::Force => Err(Box::new(SimpleError(
                    "multi-GPU requested but secondary adapter profile was not found".into(),
                ))),
            };
        };

        if let Some(primary_profile) = present_profile_mgpu.as_ref() {
            if same_physical_gpu_profile(primary_profile, &secondary_profile) {
                return match policy {
                    MultiGpuInitPolicy::Auto => Ok(None),
                    MultiGpuInitPolicy::Force => Err(Box::new(SimpleError(
                        "multi-GPU requested but planner secondary maps to the same physical adapter"
                            .into(),
                    ))),
                };
            }
        }

        if secondary_assignment.total_jobs == 0 {
            return match policy {
                MultiGpuInitPolicy::Auto => Ok(None),
                MultiGpuInitPolicy::Force => Err(Box::new(SimpleError(
                    "multi-GPU requested but secondary adapter received zero jobs".into(),
                ))),
            };
        }

        let instance = wgpu::Instance::default();
        let adapters = pollster::block_on(instance.enumerate_adapters(wgpu::Backends::all()));
        let secondary_adapter = adapters
            .into_iter()
            .find(|adapter| {
                let info = adapter.get_info();
                adapter_info_matches_profile(&secondary_profile, &info)
            })
            .ok_or_else(|| {
                Box::new(SimpleError(format!(
                    "failed to open secondary adapter '{}'",
                    secondary_profile.name
                ))) as Box<dyn Error>
            })?;
        let secondary_adapter_info = secondary_adapter.get_info();
        let version_gate = match validate_vulkan_version_compatibility(
            &primary_adapter_info,
            &secondary_adapter_info,
        ) {
            Ok(gate) => gate,
            Err(err) => {
                return match policy {
                    MultiGpuInitPolicy::Auto => Ok(None),
                    MultiGpuInitPolicy::Force => Err(Box::new(SimpleError(err))),
                };
            }
        };

        // Some mobile/embedded stacks (including Panthor-class ARM drivers) can reject the
        // default requested texture limits (notably `max_texture_dimension_3d = 2048`) even when
        // the adapter is otherwise usable. Use a conservative default profile clamped to support.
        let (secondary_required_limits, _secondary_limit_clamp_report) =
            safe_default_required_limits_for_adapter(&secondary_adapter);
        let (secondary_device, secondary_queue) =
            pollster::block_on(secondary_adapter.request_device(&wgpu::DeviceDescriptor {
                label: Some("gms-multi-gpu-secondary-device"),
                required_limits: secondary_required_limits,
                ..Default::default()
            }))?;

        let secondary_texture = create_secondary_offscreen_texture(
            &secondary_device,
            frame_width,
            frame_height,
            secondary_offscreen_format,
            Some("gms-multi-gpu-secondary-offscreen"),
        );

        let primary_assignment_jobs = present_profile_mgpu
            .as_ref()
            .and_then(|profile| {
                plan.assignments
                    .iter()
                    .find(|assignment| assignment.adapter_index == profile.index)
            })
            .map(|assignment| assignment.total_jobs)
            .unwrap_or_else(|| {
                plan.primary()
                    .map(|assignment| assignment.total_jobs)
                    .unwrap_or(1)
            });

        let primary_units = primary_work_units_per_present.max(1);
        let secondary_work_units_per_present = derive_secondary_work_units_per_present(
            primary_units,
            primary_assignment_jobs,
            secondary_assignment.total_jobs,
            &plan,
        );

        if matches!(policy, MultiGpuInitPolicy::Auto) && secondary_work_units_per_present == 0 {
            return Ok(None);
        }

        if matches!(policy, MultiGpuInitPolicy::Auto)
            && plan.projected_score_gain_pct < auto_min_projected_gain_pct
        {
            return Ok(None);
        }

        let bridge_runtime = plan
            .shared_texture_bridge
            .as_ref()
            .map(|bridge| {
                build_multi_gpu_bridge_runtime(
                    bridge,
                    &secondary_device,
                    &primary_device,
                    &secondary_profile,
                )
            })
            .transpose()?;

        let mut sync_state = MultiGpuSyncState::default();
        let _ = secondary_device.poll(wgpu::PollType::Poll);
        sync_state.poll_count = sync_state.poll_count.saturating_add(1);

        Ok(Some(Self {
            plan,
            secondary_profile,
            secondary_device,
            secondary_queue,
            secondary_texture,
            frame_width,
            frame_height,
            secondary_format: secondary_offscreen_format,
            secondary_clear_phase: 0.0,
            secondary_work_units_per_present,
            bridge_runtime,
            sync_state,
            total_secondary_work_units: 0,
            total_secondary_submissions: 0,
            vulkan_version_gate_enabled: version_gate.enabled,
            primary_vulkan_api_version: version_gate.primary_version,
            secondary_vulkan_api_version: version_gate.secondary_version,
        }))
    }

    /// Resize the synthetic secondary render target.
    pub fn resize(&mut self, frame_width: u32, frame_height: u32) {
        self.frame_width = frame_width;
        self.frame_height = frame_height;
        self.secondary_texture = create_secondary_offscreen_texture(
            &self.secondary_device,
            frame_width,
            frame_height,
            self.secondary_format,
            Some("gms-multi-gpu-secondary-offscreen"),
        );
    }

    /// Submit a synthetic secondary workload frame and return completed work units.
    pub fn submit_frame(&mut self) -> u32 {
        self.submit_frame_recording_submission().work_units
    }

    /// Submit a synthetic secondary workload frame and return work units + submission index.
    pub fn submit_frame_recording_submission(&mut self) -> MultiGpuFrameSubmitResult {
        if self.frame_width == 0 || self.frame_height == 0 {
            return MultiGpuFrameSubmitResult {
                work_units: 0,
                submission_index: None,
            };
        }

        let frames_in_flight_limit = self.plan.sync.frames_in_flight.max(1) as usize;
        if self.sync_state.pending_secondary_submissions.len() >= frames_in_flight_limit {
            if let Some(oldest_submission) =
                self.sync_state.pending_secondary_submissions.pop_front()
            {
                let _ = self.secondary_device.poll(wgpu::PollType::Wait {
                    submission_index: Some(oldest_submission),
                    timeout: Some(Duration::from_micros(250)),
                });
                self.sync_state.wait_count = self.sync_state.wait_count.saturating_add(1);
            }
        }

        let view = self
            .secondary_texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder =
            self.secondary_device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("gms-multi-gpu-secondary-encoder"),
                });

        for _ in 0..self.secondary_work_units_per_present {
            self.secondary_clear_phase =
                (self.secondary_clear_phase + 0.0175) % std::f64::consts::TAU;
            encode_clear_pass(
                &mut encoder,
                &view,
                synthetic_clear_color(self.secondary_clear_phase),
            );
        }

        // Portable `wgpu` cannot directly share textures across devices. Keep a warm host ring so
        // a real runtime can replace this with readback/upload copies without per-frame allocation.
        if let Some(bridge_runtime) = self.bridge_runtime.as_mut() {
            bridge_runtime.touch_host_ring();
        }

        let submission = self.secondary_queue.submit(Some(encoder.finish()));
        self.sync_state
            .pending_secondary_submissions
            .push_back(submission.clone());
        let _ = self.secondary_device.poll(wgpu::PollType::Poll);
        self.sync_state.poll_count = self.sync_state.poll_count.saturating_add(1);

        self.total_secondary_work_units = self
            .total_secondary_work_units
            .saturating_add(self.secondary_work_units_per_present as u64);
        self.total_secondary_submissions = self.total_secondary_submissions.saturating_add(1);

        MultiGpuFrameSubmitResult {
            work_units: self.secondary_work_units_per_present,
            submission_index: Some(submission),
        }
    }

    /// Secondary work units submitted per present interval.
    pub fn secondary_work_units_per_present(&self) -> u32 {
        self.secondary_work_units_per_present
    }

    /// Access the planner result used by the runtime.
    pub fn plan(&self) -> &MultiGpuDispatchPlan {
        &self.plan
    }

    /// Build a runtime summary suitable for logging/benchmark output.
    pub fn summary(&self) -> MultiGpuExecutorSummary {
        let primary_adapter_name = self
            .plan
            .primary()
            .map(|assignment| assignment.adapter_name.clone())
            .unwrap_or_else(|| "Unknown".to_owned());
        let bridge_kind = self
            .plan
            .shared_texture_bridge
            .as_ref()
            .map(|bridge| bridge.transfer_kind);
        let bridge_bytes_per_frame = self
            .plan
            .shared_texture_bridge
            .as_ref()
            .map(|bridge| bridge.bytes_per_frame)
            .unwrap_or(0);
        let bridge_chunk_bytes = self
            .plan
            .shared_texture_bridge
            .as_ref()
            .map(|bridge| bridge.chunk_bytes)
            .unwrap_or(0);

        MultiGpuExecutorSummary {
            primary_adapter_name,
            secondary_adapter_name: self.secondary_profile.name.clone(),
            secondary_memory_topology: self.secondary_profile.memory_topology,
            secondary_work_units_per_present: self.secondary_work_units_per_present,
            total_secondary_work_units: self.total_secondary_work_units,
            total_secondary_submissions: self.total_secondary_submissions,
            projected_score_gain_pct: self.plan.projected_score_gain_pct,
            meets_target_gain_20pct: self.plan.meets_target_gain_20pct,
            estimated_single_gpu_frame_ms: self.plan.estimated_single_gpu_frame_ms,
            estimated_multi_gpu_frame_ms: self.plan.estimated_multi_gpu_frame_ms,
            bridge_kind,
            bridge_bytes_per_frame,
            bridge_chunk_bytes,
            sync_frames_in_flight: self.plan.sync.frames_in_flight,
            sync_queue_waits: self.sync_state.wait_count,
            sync_queue_polls: self.sync_state.poll_count,
            aggressive_integrated_preallocation: self.plan.sync.aggressive_integrated_preallocation,
            integrated_encoder_pool: self.plan.sync.integrated_encoder_pool,
            integrated_ring_segments: self.plan.sync.integrated_ring_segments,
            vulkan_version_gate_enabled: self.vulkan_version_gate_enabled,
            primary_vulkan_api_version: self.primary_vulkan_api_version.map(|v| v.to_string()),
            secondary_vulkan_api_version: self.secondary_vulkan_api_version.map(|v| v.to_string()),
        }
    }

    /// Access the secondary device handle for explicit sync reconciliation.
    pub fn secondary_device(&self) -> &wgpu::Device {
        &self.secondary_device
    }

    /// Access the secondary adapter profile used by the helper lane.
    pub fn secondary_profile(&self) -> &GpuAdapterProfile {
        &self.secondary_profile
    }
}

impl MultiGpuBridgeRuntime {
    fn touch_host_ring(&mut self) {
        if self.host_ring.is_empty() {
            return;
        }
        let cursor = self.ring_cursor % self.host_ring.len();
        if let Some(segment) = self.host_ring.get_mut(cursor) {
            if !segment.is_empty() {
                let marker = (self.ring_cursor as u8).wrapping_mul(17).wrapping_add(3);
                segment[0] = marker;
                if segment.len() > 64 {
                    segment[64] = marker.rotate_left(1);
                }
            }
        }
        self.ring_cursor = self.ring_cursor.wrapping_add(1);
    }
}

fn derive_secondary_work_units_per_present(
    primary_units: u32,
    primary_jobs: u32,
    secondary_jobs: u32,
    plan: &MultiGpuDispatchPlan,
) -> u32 {
    if secondary_jobs == 0 {
        return 0;
    }

    let ratio = if primary_jobs == 0 {
        1.0
    } else {
        secondary_jobs as f64 / primary_jobs as f64
    };

    let projected_gain_bias =
        (1.0 + plan.projected_score_gain_pct.max(0.0) / 100.0).clamp(1.0, 2.5);
    let raw = (primary_units.max(1) as f64) * ratio * projected_gain_bias.sqrt();
    raw.round().clamp(1.0, 256.0) as u32
}

fn build_multi_gpu_bridge_runtime(
    bridge: &crate::bridge::SharedTextureBridgePlan,
    secondary_device: &wgpu::Device,
    primary_device: &wgpu::Device,
    secondary_profile: &GpuAdapterProfile,
) -> Result<MultiGpuBridgeRuntime, Box<dyn Error>> {
    let ring_segments = bridge.ring_segments.max(1) as usize;
    let segment_bytes = bridge.chunk_bytes.max(256);

    let host_ring = (0..ring_segments)
        .map(|_| vec![0u8; segment_bytes as usize])
        .collect::<Vec<_>>();

    let producer_readback_buffers = (0..ring_segments)
        .map(|_slot| {
            secondary_device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(match bridge.transfer_kind {
                    SharedTransferKind::HostMappedBridge => "gms-mgpu-host-bridge-readback",
                    SharedTransferKind::UnifiedMemoryMirror => "gms-mgpu-unified-bridge-readback",
                }),
                size: segment_bytes,
                usage: bridge.producer_readback_buffer_usages,
                mapped_at_creation: false,
            })
        })
        .collect::<Vec<_>>();

    let mut consumer_upload_usage = bridge.consumer_upload_buffer_usages;
    if matches!(secondary_profile.memory_topology, MemoryTopology::Unified) {
        consumer_upload_usage |= wgpu::BufferUsages::COPY_DST;
    }

    let consumer_upload_buffers = (0..ring_segments)
        .map(|_| {
            primary_device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("gms-mgpu-host-bridge-upload"),
                size: segment_bytes,
                usage: consumer_upload_usage,
                mapped_at_creation: false,
            })
        })
        .collect::<Vec<_>>();

    Ok(MultiGpuBridgeRuntime {
        host_ring,
        _producer_readback_buffers: producer_readback_buffers,
        _consumer_upload_buffers: consumer_upload_buffers,
        ring_cursor: 0,
    })
}

fn create_secondary_offscreen_texture(
    device: &wgpu::Device,
    width: u32,
    height: u32,
    format: TextureFormat,
    label: Option<&str>,
) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label,
        size: wgpu::Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    })
}

fn encode_clear_pass(encoder: &mut wgpu::CommandEncoder, view: &wgpu::TextureView, color: Color) {
    let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("gms-multi-gpu-secondary-pass"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(color),
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });
}

fn synthetic_clear_color(phase: f64) -> Color {
    let r = 0.10 + 0.30 * (phase * 1.11).sin().abs();
    let g = 0.08 + 0.34 * (phase * 0.91).cos().abs();
    let b = 0.12 + 0.26 * (phase * 1.49).sin().abs();
    Color { r, g, b, a: 1.0 }
}

fn adapter_info_matches_profile(profile: &GpuAdapterProfile, info: &wgpu::AdapterInfo) -> bool {
    profile.vendor_id == info.vendor
        && profile.device_id == info.device
        && profile.backend == info.backend
        && profile.name == info.name
}

fn match_inventory_profile(
    inventory: &GpuInventory,
    adapter_info: &wgpu::AdapterInfo,
) -> Option<GpuAdapterProfile> {
    inventory
        .adapters
        .iter()
        .find(|profile| {
            profile.vendor_id == adapter_info.vendor
                && profile.device_id == adapter_info.device
                && profile.backend == adapter_info.backend
                && profile.name == adapter_info.name
        })
        .cloned()
        .or_else(|| {
            inventory
                .adapters
                .iter()
                .find(|profile| {
                    profile.backend == adapter_info.backend
                        && profile.vendor_id == adapter_info.vendor
                        && profile.name == adapter_info.name
                })
                .cloned()
        })
}

fn same_physical_gpu_profile(left: &GpuAdapterProfile, right: &GpuAdapterProfile) -> bool {
    let same_vendor = left.vendor_id == right.vendor_id;
    let same_device_id = left.device_id != 0 && left.device_id == right.device_id;
    let same_name = normalized_physical_name(&left.name) == normalized_physical_name(&right.name);

    same_vendor && (same_device_id || same_name)
}

fn dedupe_inventory_by_physical_gpu(inventory: &GpuInventory) -> GpuInventory {
    let mut deduped = Vec::with_capacity(inventory.adapters.len());
    for profile in &inventory.adapters {
        if deduped
            .iter()
            .any(|existing| same_physical_gpu_profile(existing, profile))
        {
            continue;
        }
        deduped.push(profile.clone());
    }
    GpuInventory { adapters: deduped }
}

fn normalized_physical_name(name: &str) -> String {
    let lower = name.to_ascii_lowercase();
    let base = lower.split('/').next().unwrap_or(&lower).trim();
    base.to_owned()
}

#[derive(Debug, Clone)]
struct SimpleError(String);

impl fmt::Display for SimpleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl Error for SimpleError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct VulkanApiVersion {
    major: u32,
    minor: u32,
}

impl fmt::Display for VulkanApiVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}", self.major, self.minor)
    }
}

#[derive(Debug, Clone, Copy)]
struct VulkanVersionGate {
    enabled: bool,
    primary_version: Option<VulkanApiVersion>,
    secondary_version: Option<VulkanApiVersion>,
}

fn validate_vulkan_version_compatibility(
    primary: &wgpu::AdapterInfo,
    secondary: &wgpu::AdapterInfo,
) -> Result<VulkanVersionGate, String> {
    if !matches!(primary.backend, wgpu::Backend::Vulkan)
        || !matches!(secondary.backend, wgpu::Backend::Vulkan)
    {
        return Ok(VulkanVersionGate {
            enabled: false,
            primary_version: None,
            secondary_version: None,
        });
    }

    let primary_version = parse_adapter_vulkan_api_version(primary);
    let secondary_version = parse_adapter_vulkan_api_version(secondary);

    match (primary_version, secondary_version) {
        (Some(left), Some(right)) if left == right => Ok(VulkanVersionGate {
            enabled: true,
            primary_version: Some(left),
            secondary_version: Some(right),
        }),
        (Some(left), Some(right)) => Err(format!(
            "multi-GPU requested but Vulkan API versions are incompatible: primary '{}' uses {}, secondary '{}' uses {}",
            primary.name, left, secondary.name, right
        )),
        (None, Some(right)) => Err(format!(
            "multi-GPU requested but primary adapter '{}' Vulkan API version could not be parsed (secondary parsed as {})",
            primary.name, right
        )),
        (Some(left), None) => Err(format!(
            "multi-GPU requested but secondary adapter '{}' Vulkan API version could not be parsed (primary parsed as {})",
            secondary.name, left
        )),
        (None, None) => Err(format!(
            "multi-GPU requested but Vulkan API versions could not be parsed for either adapter ('{}' and '{}')",
            primary.name, secondary.name
        )),
    }
}

fn parse_adapter_vulkan_api_version(info: &wgpu::AdapterInfo) -> Option<VulkanApiVersion> {
    if !matches!(info.backend, wgpu::Backend::Vulkan) {
        return None;
    }
    parse_vulkan_api_version_from_text(&info.driver_info)
        .or_else(|| parse_vulkan_api_version_from_text(&info.driver))
        .or_else(|| parse_vulkan_api_version_from_text(&info.name))
        .or_else(|| probe_vulkaninfo_api_version(info))
}

fn parse_vulkan_api_version_from_text(text: &str) -> Option<VulkanApiVersion> {
    if text.is_empty() {
        return None;
    }
    let lower = text.to_ascii_lowercase();
    for key in ["vulkan", "api version", "apiversion"] {
        let mut offset = 0usize;
        while let Some(rel_idx) = lower[offset..].find(key) {
            let abs_idx = offset + rel_idx;
            let window_end = (abs_idx + key.len() + 32).min(text.len());
            if let Some(parsed) = parse_vulkan_api_version_in_range(
                text.as_bytes(),
                abs_idx,
                window_end,
            ) {
                return Some(parsed);
            }
            offset = abs_idx + key.len();
        }
    }
    None
}

fn parse_vulkan_api_version_anywhere(text: &str) -> Option<VulkanApiVersion> {
    if text.is_empty() {
        return None;
    }
    parse_vulkan_api_version_in_range(text.as_bytes(), 0, text.len())
}

fn parse_vulkan_api_version_in_range(
    bytes: &[u8],
    start: usize,
    end: usize,
) -> Option<VulkanApiVersion> {
    if start >= end || start >= bytes.len() {
        return None;
    }
    let end = end.min(bytes.len());
    let mut i = start;
    while i < end {
        if !bytes[i].is_ascii_digit() {
            i += 1;
            continue;
        }

        let (major, next_i) = match parse_u32_at(bytes, i) {
            Some(parsed) => parsed,
            None => {
                i += 1;
                continue;
            }
        };
        if next_i >= end || bytes[next_i] != b'.' {
            i += 1;
            continue;
        }
        let minor_start = next_i + 1;
        if minor_start >= end || !bytes[minor_start].is_ascii_digit() {
            i += 1;
            continue;
        }
        let (minor, _) = match parse_u32_at(bytes, minor_start) {
            Some(parsed) => parsed,
            None => {
                i += 1;
                continue;
            }
        };

        // Vulkan API major/minor should be >= 1.x in modern stacks.
        if (1..=5).contains(&major) && minor <= 10 {
            return Some(VulkanApiVersion { major, minor });
        }
        i += 1;
    }
    None
}

fn parse_u32_at(bytes: &[u8], start: usize) -> Option<(u32, usize)> {
    let mut i = start;
    let mut value: u32 = 0;
    let mut consumed = 0usize;
    while i < bytes.len() && bytes[i].is_ascii_digit() {
        value = value
            .checked_mul(10)?
            .checked_add(u32::from(bytes[i] - b'0'))?;
        i += 1;
        consumed += 1;
    }
    if consumed == 0 {
        None
    } else {
        Some((value, i))
    }
}

#[derive(Debug, Clone)]
struct VulkanInfoApiVersionEntry {
    normalized_name: String,
    vendor: Option<u32>,
    device: Option<u32>,
    api_version: VulkanApiVersion,
}

fn probe_vulkaninfo_api_version(info: &wgpu::AdapterInfo) -> Option<VulkanApiVersion> {
    static CACHE: OnceLock<Vec<VulkanInfoApiVersionEntry>> = OnceLock::new();
    let entries = CACHE.get_or_init(load_vulkaninfo_api_version_entries);
    if entries.is_empty() {
        return None;
    }

    let target_name = normalize_gpu_name(&info.name);
    entries
        .iter()
        .find(|entry| {
            entry.vendor == Some(info.vendor)
                && entry.device == Some(info.device)
                && !entry.normalized_name.is_empty()
                && (target_name.contains(&entry.normalized_name)
                    || entry.normalized_name.contains(&target_name))
        })
        .or_else(|| {
            entries
                .iter()
                .find(|entry| entry.vendor == Some(info.vendor) && entry.device == Some(info.device))
        })
        .or_else(|| {
            entries
                .iter()
                .filter(|entry| {
                    !entry.normalized_name.is_empty()
                        && (target_name.contains(&entry.normalized_name)
                            || entry.normalized_name.contains(&target_name))
                })
                .max_by_key(|entry| entry.normalized_name.len())
        })
        .map(|entry| entry.api_version)
}

fn load_vulkaninfo_api_version_entries() -> Vec<VulkanInfoApiVersionEntry> {
    let output = match Command::new("vulkaninfo").arg("--summary").output() {
        Ok(output) => output,
        Err(_) => return Vec::new(),
    };
    if !output.status.success() {
        return Vec::new();
    }
    let text = String::from_utf8_lossy(&output.stdout);
    parse_vulkaninfo_api_version_entries(&text)
}

fn parse_vulkaninfo_api_version_entries(text: &str) -> Vec<VulkanInfoApiVersionEntry> {
    let mut entries = Vec::new();
    let mut current_name: Option<String> = None;
    let mut current_vendor: Option<u32> = None;
    let mut current_device: Option<u32> = None;
    let mut current_api: Option<VulkanApiVersion> = None;

    let flush = |entries: &mut Vec<VulkanInfoApiVersionEntry>,
                 current_name: &mut Option<String>,
                 current_vendor: &mut Option<u32>,
                 current_device: &mut Option<u32>,
                 current_api: &mut Option<VulkanApiVersion>| {
        let Some(api_version) = current_api.take() else {
            *current_name = None;
            *current_vendor = None;
            *current_device = None;
            return;
        };
        let normalized_name = current_name
            .take()
            .map(|name| normalize_gpu_name(&name))
            .unwrap_or_default();
        entries.push(VulkanInfoApiVersionEntry {
            normalized_name,
            vendor: current_vendor.take(),
            device: current_device.take(),
            api_version,
        });
    };

    for raw_line in text.lines() {
        let line = raw_line.trim();
        if line.is_empty() {
            continue;
        }
        if line.starts_with("GPU") && line.ends_with(':') {
            flush(
                &mut entries,
                &mut current_name,
                &mut current_vendor,
                &mut current_device,
                &mut current_api,
            );
            continue;
        }
        if let Some(name) = parse_vulkaninfo_gpu_name_line(line) {
            current_name = Some(name);
            continue;
        }
        let Some((key, value)) = split_vulkaninfo_kv_line(line) else {
            continue;
        };
        let key = normalize_gpu_name(key).replace(' ', "");
        if key == "devicename" {
            current_name = Some(value.trim().to_owned());
            continue;
        }
        if key == "vendorid" {
            current_vendor = parse_first_u32_hex_or_decimal(value);
            continue;
        }
        if key == "deviceid" {
            current_device = parse_first_u32_hex_or_decimal(value);
            continue;
        }
        if key == "apiversion" {
            current_api = parse_vulkan_api_version_anywhere(value);
        }
    }

    flush(
        &mut entries,
        &mut current_name,
        &mut current_vendor,
        &mut current_device,
        &mut current_api,
    );
    entries
}

fn normalize_gpu_name(name: &str) -> String {
    let mut out = String::with_capacity(name.len());
    let mut last_space = false;
    for ch in name.chars().flat_map(|c| c.to_lowercase()) {
        if ch.is_ascii_alphanumeric() {
            out.push(ch);
            last_space = false;
        } else if !last_space {
            out.push(' ');
            last_space = true;
        }
    }
    out.trim().to_owned()
}

fn parse_vulkaninfo_gpu_name_line(line: &str) -> Option<String> {
    if line.to_ascii_lowercase().starts_with("gpu id") {
        let open = line.find('(')?;
        let close = line.rfind(')')?;
        if close > open + 1 {
            return Some(line[open + 1..close].trim().to_owned());
        }
    }
    None
}

fn split_vulkaninfo_kv_line(line: &str) -> Option<(&str, &str)> {
    line.split_once('=').or_else(|| line.split_once(':'))
}

fn parse_first_u32_hex_or_decimal(text: &str) -> Option<u32> {
    let lower = text.to_ascii_lowercase();
    if let Some(idx) = lower.find("0x") {
        let mut end = idx + 2;
        let bytes = lower.as_bytes();
        while end < bytes.len() && bytes[end].is_ascii_hexdigit() {
            end += 1;
        }
        if end > idx + 2 {
            return u32::from_str_radix(&lower[idx + 2..end], 16).ok();
        }
    }
    let mut start = None;
    for (i, ch) in text.char_indices() {
        if ch.is_ascii_digit() {
            start = Some(i);
            break;
        }
    }
    let start = start?;
    let rest = &text[start..];
    let end = rest
        .char_indices()
        .find_map(|(i, ch)| (!ch.is_ascii_digit()).then_some(i))
        .unwrap_or(rest.len());
    rest[..end].parse::<u32>().ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use wgpu::{AdapterInfo, Backend, DeviceType};

    fn vk_info(name: &str, driver: &str, driver_info: &str) -> AdapterInfo {
        AdapterInfo {
            name: name.to_owned(),
            vendor: 0x1002,
            device: 0x73BF,
            device_type: DeviceType::DiscreteGpu,
            device_pci_bus_id: "0000:01:00.0".to_owned(),
            driver: driver.to_owned(),
            driver_info: driver_info.to_owned(),
            backend: Backend::Vulkan,
            subgroup_min_size: 32,
            subgroup_max_size: 32,
            transient_saves_memory: false,
        }
    }

    #[test]
    fn parses_vulkan_major_minor_from_driver_info() {
        let parsed = parse_vulkan_api_version_from_text("Mesa RADV Vulkan 1.3.302");
        assert_eq!(parsed, Some(VulkanApiVersion { major: 1, minor: 3 }));
    }

    #[test]
    fn rejects_non_vulkan_driver_like_versions() {
        let parsed = parse_vulkan_api_version_from_text("NVIDIA 590.48.01");
        assert_eq!(parsed, None);
    }

    #[test]
    fn parses_vulkaninfo_summary_api_versions() {
        let text = r#"
GPU0:
    apiVersion         = 1.4.325
    vendorID           = 0x10de
    deviceID           = 0x2d04
    deviceName         = NVIDIA GeForce RTX 5060 Ti
GPU1:
    apiVersion         = 1.4.335
    vendorID           = 0x1002
    deviceID           = 0x164e
    deviceName         = AMD Ryzen 9 7900 12-Core Processor (RADV RAPHAEL_MENDOCINO)
"#;
        let entries = parse_vulkaninfo_api_version_entries(text);
        assert_eq!(entries.len(), 2);
        assert_eq!(
            entries[0].api_version,
            VulkanApiVersion { major: 1, minor: 4 }
        );
        assert_eq!(entries[0].vendor, Some(0x10de));
        assert_eq!(entries[0].device, Some(0x2d04));
        assert_eq!(
            entries[1].api_version,
            VulkanApiVersion { major: 1, minor: 4 }
        );
    }

    #[test]
    fn vulkan_gate_accepts_matching_major_minor() {
        let primary = vk_info("Primary", "Mesa", "Vulkan 1.3.290");
        let secondary = vk_info("Secondary", "NVIDIA", "Vulkan 1.3.280");
        let gate = validate_vulkan_version_compatibility(&primary, &secondary)
            .expect("matching Vulkan versions should pass");
        assert!(gate.enabled);
        assert_eq!(
            gate.primary_version,
            Some(VulkanApiVersion { major: 1, minor: 3 })
        );
        assert_eq!(
            gate.secondary_version,
            Some(VulkanApiVersion { major: 1, minor: 3 })
        );
    }

    #[test]
    fn vulkan_gate_rejects_major_minor_mismatch() {
        let primary = vk_info("Primary", "Mesa", "Vulkan 1.3.290");
        let secondary = vk_info("Secondary", "Mesa", "Vulkan 1.2.250");
        let err = validate_vulkan_version_compatibility(&primary, &secondary)
            .expect_err("mismatched Vulkan versions should fail");
        assert!(err.contains("incompatible"));
        assert!(err.contains("1.3"));
        assert!(err.contains("1.2"));
    }

    #[test]
    fn vulkan_gate_rejects_when_version_unparseable() {
        let primary = vk_info("Primary", "Mesa", "Vulkan 1.3.290");
        let secondary = vk_info("Secondary", "Mesa", "unknown");
        let err = validate_vulkan_version_compatibility(&primary, &secondary)
            .expect_err("unparseable Vulkan version should fail");
        assert!(err.contains("could not be parsed"));
    }
}

fn build_no_secondary_assignment_diagnostic(
    primary_adapter_info: &wgpu::AdapterInfo,
    raw_inventory: &GpuInventory,
    deduped_inventory: &GpuInventory,
    plan: &MultiGpuDispatchPlan,
) -> String {
    let mut msg = String::new();
    let _ = writeln!(
        msg,
        "multi-GPU requested but no secondary adapter assignment was produced"
    );
    let _ = writeln!(
        msg,
        "present adapter: {} | backend: {:?} | type: {:?} | vendor: {:#06x} | device: {:#06x}",
        primary_adapter_info.name,
        primary_adapter_info.backend,
        primary_adapter_info.device_type,
        primary_adapter_info.vendor,
        primary_adapter_info.device
    );

    let raw_count = raw_inventory.adapters.len();
    let deduped_count = deduped_inventory.adapters.len();
    let usable_count = deduped_inventory
        .adapters
        .iter()
        .filter(|adapter| adapter.is_usable_gpu())
        .count();

    let _ = writeln!(
        msg,
        "inventory: raw={} | deduped_physical={} | deduped_usable={} | planner_assignments={}",
        raw_count,
        deduped_count,
        usable_count,
        plan.assignments.len()
    );

    if raw_count != deduped_count {
        let _ = writeln!(
            msg,
            "note: duplicate backends for the same physical GPU were collapsed before multi-GPU planning"
        );
    }

    if deduped_inventory.adapters.is_empty() {
        let _ = writeln!(msg, "deduped adapters: <none>");
    } else {
        let _ = writeln!(msg, "deduped adapters:");
        for (idx, adapter) in deduped_inventory.adapters.iter().enumerate() {
            let _ = writeln!(
                msg,
                "  [{}] {} | backend: {:?} | type: {:?} | topo: {:?} | score: {} | usable: {} | vendor: {:#06x} | device: {:#06x}",
                idx,
                adapter.name,
                adapter.backend,
                adapter.device_type,
                adapter.memory_topology,
                adapter.score,
                adapter.is_usable_gpu(),
                adapter.vendor_id,
                adapter.device_id
            );
        }
    }

    if !plan.assignments.is_empty() {
        let _ = writeln!(msg, "planner assignments:");
        for assignment in &plan.assignments {
            let _ = writeln!(
                msg,
                "  - {} | role: {:?} | total_jobs: {} | score: {}",
                assignment.adapter_name, assignment.role, assignment.total_jobs, assignment.score
            );
        }
    }

    msg
}
