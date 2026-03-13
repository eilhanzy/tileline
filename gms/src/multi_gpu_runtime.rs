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
use std::time::Duration;

use crate::bridge::{MultiGpuDispatchPlan, MultiGpuDispatcher, MultiGpuWorkloadRequest};
use crate::hardware::{GpuAdapterProfile, GpuInventory, MemoryTopology};
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
}

/// Secondary helper-lane submission result compatible with bridge tracking paths.
#[derive(Debug, Clone)]
pub struct MultiGpuFrameSubmitResult {
    /// Number of synthetic helper work units submitted for this frame.
    pub work_units: u32,
    /// Submission index emitted by the secondary queue (if any work was submitted).
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

        // `wgpu` 28 can reject `DeviceDescriptor::default()` on some adapters due to limit
        // negotiation quirks. Request the adapter's advertised limits explicitly.
        let secondary_required_limits = secondary_adapter.limits();
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

    /// Submit a synthetic helper frame and return both work count and submission index.
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

    /// Borrow the secondary helper device used by the executor.
    pub fn secondary_device(&self) -> &wgpu::Device {
        &self.secondary_device
    }

    /// Borrow the selected secondary GPU profile.
    pub fn secondary_profile(&self) -> &GpuAdapterProfile {
        &self.secondary_profile
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
        }
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
