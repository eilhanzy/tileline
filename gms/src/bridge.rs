//! GMS bridge/dispatcher planning.
//!
//! This module does not execute GPU kernels directly yet. It builds a scheduling plan that
//! can be consumed by the later render/compute runtime. The plan is weighted by the GPU
//! scores discovered in `hardware.rs`.

use crate::hardware::{
    ComputeUnitEstimateSource, ComputeUnitKind, GpuAdapterProfile, GpuInventory, MemoryTopology,
};
use wgpu::{BufferUsages, TextureUsages};

/// Input workload description for a frame or simulation step.
#[derive(Debug, Clone, Copy)]
pub struct WorkloadRequest {
    /// Number of 3D object update jobs (transforms, culling metadata, etc.).
    pub object_updates: u32,
    /// Number of physics compute jobs/chunks.
    pub physics_jobs: u32,
    /// Approximate bytes uploaded per object update.
    pub bytes_per_object: u64,
    /// Approximate bytes uploaded per physics job.
    pub bytes_per_physics_job: u64,
    /// Preferred base workgroup size before per-GPU scaling.
    pub base_workgroup_size: u32,
}

impl Default for WorkloadRequest {
    fn default() -> Self {
        Self {
            object_updates: 0,
            physics_jobs: 0,
            bytes_per_object: 256,
            bytes_per_physics_job: 1024,
            base_workgroup_size: 64,
        }
    }
}

/// Buffer strategy for minimizing transfers and favoring `MAP_WRITE` uploads.
#[derive(Debug, Clone, Copy)]
pub struct ZeroCopyBufferPlan {
    /// Host-visible upload buffer usages (staging ring / mapped upload path).
    pub upload_buffer_usages: BufferUsages,
    /// GPU execution buffer usages (storage + copy endpoints).
    pub device_buffer_usages: BufferUsages,
    /// True when the adapter can prefer directly mapped primary buffers.
    pub prefer_mapped_primary: bool,
    /// Suggested host staging bytes for this adapter's assigned workload.
    pub staging_bytes: u64,
    /// Suggested device storage bytes for this adapter's assigned workload.
    pub device_bytes: u64,
    /// Suggested number of upload ring segments.
    pub ring_segments: u32,
}

/// Work assigned to one adapter.
#[derive(Debug, Clone)]
pub struct GpuWorkAssignment {
    pub adapter_index: usize,
    pub adapter_name: String,
    pub score: u64,
    pub compute_unit_kind: ComputeUnitKind,
    pub compute_units: u32,
    pub compute_unit_source: ComputeUnitEstimateSource,
    pub score_share_pct: f64,
    pub object_updates: u32,
    pub physics_jobs: u32,
    pub total_jobs: u32,
    pub workgroup_size: u32,
    pub dispatch_workgroups: u32,
    pub zero_copy: ZeroCopyBufferPlan,
}

/// Full dispatch plan for the current workload.
#[derive(Debug, Clone, Default)]
pub struct DispatchPlan {
    pub total_object_updates: u32,
    pub total_physics_jobs: u32,
    pub total_jobs: u32,
    pub assignments: Vec<GpuWorkAssignment>,
}

impl DispatchPlan {
    /// Return the strongest adapter assignment (first item, inventory is score-sorted).
    pub fn primary(&self) -> Option<&GpuWorkAssignment> {
        self.assignments.first()
    }
}

/// Multi-GPU execution role for a participating adapter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MultiGpuRole {
    /// Owns the final composition and swapchain presentation.
    PrimaryPresent,
    /// Runs latency-sensitive secondary lanes (UI/Post-FX) when beneficial.
    SecondaryLatency,
    /// Additional compute helper for heavy lanes.
    AuxiliaryCompute,
}

/// High-level workload classes used by the multi-GPU balancer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskClass {
    /// Sampled processing / texture-heavy compute that benefits from the strongest GPU.
    SampledProcessing,
    /// Object updates (transforms, culling metadata, animation packing).
    ObjectUpdate,
    /// Physics/particle or simulation compute.
    Physics,
    /// UI composition and widgets, usually latency-sensitive and lighter.
    Ui,
    /// Post-processing or final compositing passes.
    PostFx,
}

/// Portable cross-adapter transfer strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SharedTransferKind {
    /// Portable `wgpu` path: secondary GPU writes to a readback buffer, host mirrors bytes into
    /// a primary-GPU upload ring, then the primary copies into its composition texture.
    HostMappedBridge,
    /// Unified-memory-friendly mirror path (still host-orchestrated in portable `wgpu`).
    UnifiedMemoryMirror,
}

/// `wgpu` synchronization equivalents used by the planner.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyncEquivalent {
    /// Host-tracked queue submission sequence IDs used like a timeline semaphore.
    QueueSubmissionTimeline,
    /// Queue completion wait/poll used like a fence.
    QueueCompletionFence,
}

/// Detailed per-lane allocation for a single adapter in the multi-GPU plan.
#[derive(Debug, Clone)]
pub struct MultiGpuLaneAssignment {
    pub adapter_index: usize,
    pub adapter_name: String,
    pub role: MultiGpuRole,
    pub score: u64,
    pub compute_unit_kind: ComputeUnitKind,
    pub compute_units: u32,
    pub compute_unit_source: ComputeUnitEstimateSource,
    pub score_share_pct: f64,
    pub sampled_processing_jobs: u32,
    pub object_updates: u32,
    pub physics_jobs: u32,
    pub ui_jobs: u32,
    pub post_fx_jobs: u32,
    pub total_jobs: u32,
    pub workgroup_size: u32,
    pub dispatch_workgroups: u32,
    /// Heuristic frame-time estimate for the adapter's assigned work.
    pub estimated_work_ms: f64,
    /// Optional spillback count when latency-sensitive tasks overflow to the primary.
    pub latency_spillback_jobs: u32,
    pub zero_copy: ZeroCopyBufferPlan,
}

/// Portable shared-memory interface plan for cross-adapter texture transport.
#[derive(Debug, Clone, Copy)]
pub struct SharedTextureBridgePlan {
    /// Secondary producer adapter index (usually the iGPU running UI/Post-FX).
    pub producer_adapter_index: usize,
    /// Primary consumer adapter index (final present GPU).
    pub consumer_adapter_index: usize,
    pub transfer_kind: SharedTransferKind,
    /// Source texture usages on the producer adapter.
    pub producer_texture_usages: TextureUsages,
    /// Destination/composite texture usages on the consumer adapter.
    pub consumer_texture_usages: TextureUsages,
    /// Producer-side readback staging buffer usages.
    pub producer_readback_buffer_usages: BufferUsages,
    /// Consumer-side upload staging buffer usages.
    pub consumer_upload_buffer_usages: BufferUsages,
    /// Estimated bytes moved per frame for secondary outputs.
    pub bytes_per_frame: u64,
    /// Chunk size for chunked copies to keep latency predictable.
    pub chunk_bytes: u64,
    /// Number of persistent ring segments for overlap.
    pub ring_segments: u32,
    /// Frames that may be in flight concurrently through the bridge.
    pub frames_in_flight: u32,
    /// Heuristic host-mediated transfer latency estimate.
    pub estimated_transfer_ms: f64,
}

/// Synchronization and preallocation plan for multi-GPU frame overlap.
#[derive(Debug, Clone, Copy)]
pub struct MultiGpuSyncPlan {
    /// Fence-equivalent primitive in portable `wgpu`.
    pub fence_equivalent: SyncEquivalent,
    /// Semaphore-equivalent primitive in portable `wgpu`.
    pub semaphore_equivalent: SyncEquivalent,
    /// Frames that can overlap before backpressure should trigger.
    pub frames_in_flight: u32,
    /// Number of queue timeline stages (secondary compute -> transfer -> primary composite/present).
    pub queue_timeline_stages: u32,
    /// Preallocated command buffers for the primary GPU.
    pub primary_command_buffers_preallocated: u32,
    /// Preallocated command buffers for the secondary GPU.
    pub secondary_command_buffers_preallocated: u32,
    /// Preallocated command buffers for copy/upload stages.
    pub transfer_command_buffers_preallocated: u32,
    /// Extra encoder/command-buffer reservation for integrated GPUs to reduce jitter.
    pub aggressive_integrated_preallocation: bool,
    /// Suggested secondary command encoder pool size.
    pub integrated_encoder_pool: u32,
    /// Secondary upload/readback ring segments reserved for stability.
    pub integrated_ring_segments: u32,
    /// Target budget for the secondary latency lane.
    pub secondary_budget_ms: f64,
    /// Estimated secondary lane cost after spillback.
    pub estimated_secondary_ms: f64,
    /// Recommended slack before forcing spillback to the primary.
    pub secondary_slack_ms: f64,
}

/// Multi-GPU workload request that includes heavy and latency-sensitive lanes.
#[derive(Debug, Clone, Copy)]
pub struct MultiGpuWorkloadRequest {
    /// Texture-heavy sampled processing / shading jobs.
    pub sampled_processing_jobs: u32,
    /// Object update jobs.
    pub object_updates: u32,
    /// Physics jobs.
    pub physics_jobs: u32,
    /// UI jobs.
    pub ui_jobs: u32,
    /// Post-processing jobs.
    pub post_fx_jobs: u32,
    /// Bytes per sampled processing job.
    pub bytes_per_sampled_job: u64,
    /// Bytes per object update job.
    pub bytes_per_object: u64,
    /// Bytes per physics job.
    pub bytes_per_physics_job: u64,
    /// Bytes per UI job.
    pub bytes_per_ui_job: u64,
    /// Bytes per post-FX job.
    pub bytes_per_post_fx_job: u64,
    /// Estimated processed texture bytes transferred from secondary to primary per frame.
    pub processed_texture_bytes_per_frame: u64,
    /// Preferred base workgroup size before per-GPU scaling.
    pub base_workgroup_size: u32,
    /// Target frame budget used to cap secondary latency work (e.g. 0.26ms or 16.67ms).
    pub target_frame_budget_ms: f64,
}

impl Default for MultiGpuWorkloadRequest {
    fn default() -> Self {
        Self {
            sampled_processing_jobs: 0,
            object_updates: 0,
            physics_jobs: 0,
            ui_jobs: 0,
            post_fx_jobs: 0,
            bytes_per_sampled_job: 4096,
            bytes_per_object: 256,
            bytes_per_physics_job: 1024,
            bytes_per_ui_job: 512,
            bytes_per_post_fx_job: 1024,
            processed_texture_bytes_per_frame: 8 * 1024 * 1024,
            base_workgroup_size: 64,
            target_frame_budget_ms: 16.67,
        }
    }
}

/// Full explicit multi-GPU dispatch and synchronization plan.
#[derive(Debug, Clone)]
pub struct MultiGpuDispatchPlan {
    pub primary_adapter_index: Option<usize>,
    pub secondary_adapter_index: Option<usize>,
    pub assignments: Vec<MultiGpuLaneAssignment>,
    pub shared_texture_bridge: Option<SharedTextureBridgePlan>,
    pub sync: MultiGpuSyncPlan,
    /// Single-GPU baseline estimate on the primary adapter.
    pub estimated_single_gpu_frame_ms: f64,
    /// Multi-GPU overlapped frame estimate after transfer and sync overheads.
    pub estimated_multi_gpu_frame_ms: f64,
    /// Projected render-benchmark score improvement from overlap.
    pub projected_score_gain_pct: f64,
    /// True when the heuristic plan predicts >= 20% score gain.
    pub meets_target_gain_20pct: bool,
}

impl MultiGpuDispatchPlan {
    /// Return the adapter assignment that owns final present.
    pub fn primary(&self) -> Option<&MultiGpuLaneAssignment> {
        self.assignments
            .iter()
            .find(|assignment| assignment.role == MultiGpuRole::PrimaryPresent)
    }

    /// Return the adapter assignment used as latency helper.
    pub fn secondary(&self) -> Option<&MultiGpuLaneAssignment> {
        self.assignments
            .iter()
            .find(|assignment| assignment.role == MultiGpuRole::SecondaryLatency)
    }
}

/// Explicit multi-GPU dispatcher planner for heterogeneous (dGPU + iGPU) rendering.
///
/// This planner remains portable across `wgpu` backends by modeling cross-adapter resource
/// movement through a host-visible bridge. Portable `wgpu` does not expose direct cross-device
/// texture sharing or native semaphore/fence objects, so the runtime is expected to implement the
/// sync plan using queue submission sequencing + queue completion waits.
#[derive(Debug, Clone)]
pub struct MultiGpuDispatcher {
    inventory: GpuInventory,
}

impl MultiGpuDispatcher {
    /// Discover adapters and build a multi-GPU dispatcher.
    pub fn discover() -> Self {
        Self {
            inventory: GpuInventory::discover(),
        }
    }

    /// Build from a precomputed inventory.
    pub fn new(inventory: GpuInventory) -> Self {
        Self { inventory }
    }

    /// Access discovered inventory.
    pub fn inventory(&self) -> &GpuInventory {
        &self.inventory
    }

    /// Plan an explicit multi-GPU workload split.
    ///
    /// Heuristic policy:
    /// - Heavy sampled/physics lanes are biased toward the strongest adapter score.
    /// - UI/Post-FX are biased to the secondary adapter for overlap.
    /// - Secondary latency work is capped by a target frame budget; overflow spills back.
    pub fn plan_dispatch(&self, request: MultiGpuWorkloadRequest) -> MultiGpuDispatchPlan {
        let usable = self.inventory.usable_adapters().collect::<Vec<_>>();
        let total_usable_score = self.inventory.total_usable_score().max(1);

        if usable.is_empty() {
            return MultiGpuDispatchPlan {
                primary_adapter_index: None,
                secondary_adapter_index: None,
                assignments: Vec::new(),
                shared_texture_bridge: None,
                sync: default_sync_plan(request.target_frame_budget_ms),
                estimated_single_gpu_frame_ms: 0.0,
                estimated_multi_gpu_frame_ms: 0.0,
                projected_score_gain_pct: 0.0,
                meets_target_gain_20pct: false,
            };
        }

        let primary_slot = select_primary_adapter_slot(&usable);
        let primary_gpu = usable[primary_slot];
        let secondary_slot = select_secondary_adapter_slot(&usable, primary_slot, primary_gpu);
        let secondary_gpu = secondary_slot.map(|slot| usable[slot]);
        let best_score = usable.iter().map(|gpu| gpu.score).max().unwrap_or(1);

        // Heavy lanes: strongest score gets the largest share. Physics is biased slightly more
        // toward discrete VRAM due to bandwidth and latency behavior.
        let sampled_weights = usable
            .iter()
            .map(|gpu| heavy_weight_for(gpu, TaskClass::SampledProcessing))
            .collect::<Vec<_>>();
        let object_weights = usable
            .iter()
            .map(|gpu| heavy_weight_for(gpu, TaskClass::ObjectUpdate))
            .collect::<Vec<_>>();
        let physics_weights = usable
            .iter()
            .map(|gpu| heavy_weight_for(gpu, TaskClass::Physics))
            .collect::<Vec<_>>();

        let mut sampled_distribution =
            allocate_weighted_counts(request.sampled_processing_jobs, &sampled_weights);
        let mut object_distribution =
            allocate_weighted_counts(request.object_updates, &object_weights);
        let mut physics_distribution =
            allocate_weighted_counts(request.physics_jobs, &physics_weights);

        let mut ui_distribution = vec![0u32; usable.len()];
        let mut post_fx_distribution = vec![0u32; usable.len()];
        let mut latency_spillback_by_slot = vec![0u32; usable.len()];

        if let Some(sec_slot) = secondary_slot {
            let sec_gpu = usable[sec_slot];

            // Latency-sensitive work is intentionally routed to the secondary adapter so the
            // primary can focus on heavy throughput. We cap the secondary lane using a conservative
            // frame-budget heuristic and spill overflow back to the primary to avoid missed frames.
            let ui_target = request.ui_jobs;
            let post_fx_target = request.post_fx_jobs;

            let mut sec_ui = ui_target;
            let mut sec_post_fx = post_fx_target;

            let secondary_budget_ms = secondary_latency_budget_ms(request.target_frame_budget_ms);
            let estimated_secondary_ms =
                estimate_lane_ms(sec_gpu, sec_ui, sec_post_fx, 0, 0, 0, request);

            if estimated_secondary_ms > secondary_budget_ms {
                let ratio = (secondary_budget_ms / estimated_secondary_ms).clamp(0.0, 1.0);
                sec_ui = ((sec_ui as f64) * ratio).floor() as u32;
                sec_post_fx = ((sec_post_fx as f64) * ratio).floor() as u32;
            }

            ui_distribution[sec_slot] = sec_ui;
            post_fx_distribution[sec_slot] = sec_post_fx;

            let ui_spill = ui_target.saturating_sub(sec_ui);
            let post_fx_spill = post_fx_target.saturating_sub(sec_post_fx);
            ui_distribution[primary_slot] = ui_distribution[primary_slot].saturating_add(ui_spill);
            post_fx_distribution[primary_slot] =
                post_fx_distribution[primary_slot].saturating_add(post_fx_spill);
            latency_spillback_by_slot[primary_slot] =
                latency_spillback_by_slot[primary_slot].saturating_add(ui_spill + post_fx_spill);

            // Multi-GPU aggressive balancing:
            // Ensure the secondary lane receives a minimum total job share so helper adapters
            // (especially iGPU or 2nd dGPU) are not underutilized while the primary stays saturated.
            enforce_secondary_minimum_job_share(
                primary_slot,
                sec_slot,
                primary_gpu,
                sec_gpu,
                request,
                &mut sampled_distribution,
                &mut object_distribution,
                &mut physics_distribution,
                &ui_distribution,
                &post_fx_distribution,
            );
        } else {
            // Single-GPU fallback: run all latency work on the primary adapter.
            ui_distribution[primary_slot] = request.ui_jobs;
            post_fx_distribution[primary_slot] = request.post_fx_jobs;
        }

        let mut assignments = Vec::with_capacity(usable.len());
        for (slot, gpu) in usable.iter().enumerate() {
            let role = if slot == primary_slot {
                MultiGpuRole::PrimaryPresent
            } else if Some(slot) == secondary_slot {
                MultiGpuRole::SecondaryLatency
            } else {
                MultiGpuRole::AuxiliaryCompute
            };

            let sampled_processing_jobs = sampled_distribution[slot];
            let object_updates = object_distribution[slot];
            let physics_jobs = physics_distribution[slot];
            let ui_jobs = ui_distribution[slot];
            let post_fx_jobs = post_fx_distribution[slot];
            let total_jobs = sampled_processing_jobs
                .saturating_add(object_updates)
                .saturating_add(physics_jobs)
                .saturating_add(ui_jobs)
                .saturating_add(post_fx_jobs);

            let workgroup_size =
                select_workgroup_size(gpu, best_score, request.base_workgroup_size);
            let dispatch_workgroups = ceil_div_u32(total_jobs.max(1), workgroup_size.max(1));
            let score_share_pct = (gpu.score as f64 / total_usable_score as f64) * 100.0;
            let assigned_bytes = estimate_multi_gpu_assigned_bytes(
                gpu,
                request,
                sampled_processing_jobs,
                object_updates,
                physics_jobs,
                ui_jobs,
                post_fx_jobs,
            );
            let zero_copy = build_zero_copy_plan(gpu, assigned_bytes);
            let estimated_work_ms = estimate_lane_ms(
                gpu,
                sampled_processing_jobs,
                object_updates,
                physics_jobs,
                ui_jobs,
                post_fx_jobs,
                request,
            );

            assignments.push(MultiGpuLaneAssignment {
                adapter_index: gpu.index,
                adapter_name: gpu.name.clone(),
                role,
                score: gpu.score,
                compute_unit_kind: gpu.compute_unit_kind,
                compute_units: gpu.estimated_compute_units,
                compute_unit_source: gpu.compute_unit_source,
                score_share_pct,
                sampled_processing_jobs,
                object_updates,
                physics_jobs,
                ui_jobs,
                post_fx_jobs,
                total_jobs,
                workgroup_size,
                dispatch_workgroups,
                estimated_work_ms,
                latency_spillback_jobs: latency_spillback_by_slot[slot],
                zero_copy,
            });
        }

        let shared_texture_bridge =
            build_shared_texture_bridge_plan(primary_gpu, secondary_gpu, request, &assignments);
        let transfer_ms = shared_texture_bridge
            .as_ref()
            .map(|plan| plan.estimated_transfer_ms)
            .unwrap_or(0.0);

        let primary_ms = assignments
            .iter()
            .find(|assignment| assignment.role == MultiGpuRole::PrimaryPresent)
            .map(|assignment| assignment.estimated_work_ms)
            .unwrap_or(0.0);
        let secondary_ms = assignments
            .iter()
            .find(|assignment| assignment.role == MultiGpuRole::SecondaryLatency)
            .map(|assignment| assignment.estimated_work_ms)
            .unwrap_or(0.0);

        let sync = build_multi_gpu_sync_plan(
            request,
            primary_gpu,
            secondary_gpu,
            secondary_ms,
            shared_texture_bridge.as_ref(),
        );

        let baseline_single_ms = estimate_lane_ms(
            primary_gpu,
            request.sampled_processing_jobs,
            request.object_updates,
            request.physics_jobs,
            request.ui_jobs,
            request.post_fx_jobs,
            request,
        );

        // Overlap model: total frame time is the slower of (primary lane) and
        // (secondary lane + bridge + sync overhead). We add a small orchestration cost.
        let multi_ms = if secondary_gpu.is_some() {
            let secondary_pipeline_ms =
                secondary_ms + transfer_ms + sync.secondary_slack_ms.max(0.0) * 0.25;
            primary_ms.max(secondary_pipeline_ms) + 0.03
        } else {
            baseline_single_ms
        };

        let projected_score_gain_pct = if baseline_single_ms > 0.0 && multi_ms > 0.0 {
            ((baseline_single_ms / multi_ms) - 1.0) * 100.0
        } else {
            0.0
        };

        MultiGpuDispatchPlan {
            primary_adapter_index: Some(primary_gpu.index),
            secondary_adapter_index: secondary_gpu.map(|gpu| gpu.index),
            assignments,
            shared_texture_bridge,
            sync,
            estimated_single_gpu_frame_ms: baseline_single_ms,
            estimated_multi_gpu_frame_ms: multi_ms,
            projected_score_gain_pct,
            meets_target_gain_20pct: projected_score_gain_pct >= 20.0,
        }
    }
}

/// High-level GPU dispatcher planner.
#[derive(Debug, Clone)]
pub struct GmsDispatcher {
    inventory: GpuInventory,
}

fn default_sync_plan(target_frame_budget_ms: f64) -> MultiGpuSyncPlan {
    MultiGpuSyncPlan {
        fence_equivalent: SyncEquivalent::QueueCompletionFence,
        semaphore_equivalent: SyncEquivalent::QueueSubmissionTimeline,
        frames_in_flight: 2,
        queue_timeline_stages: 3,
        primary_command_buffers_preallocated: 4,
        secondary_command_buffers_preallocated: 0,
        transfer_command_buffers_preallocated: 0,
        aggressive_integrated_preallocation: false,
        integrated_encoder_pool: 0,
        integrated_ring_segments: 0,
        secondary_budget_ms: secondary_latency_budget_ms(target_frame_budget_ms),
        estimated_secondary_ms: 0.0,
        secondary_slack_ms: 0.0,
    }
}

fn select_primary_adapter_slot(usable: &[&GpuAdapterProfile]) -> usize {
    usable
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| {
            primary_priority(left)
                .partial_cmp(&primary_priority(right))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(idx, _)| idx)
        .unwrap_or(0)
}

fn primary_priority(gpu: &GpuAdapterProfile) -> f64 {
    let topology_bonus = match gpu.memory_topology {
        MemoryTopology::DiscreteVram => 1.25,
        MemoryTopology::Unified => 0.95,
        MemoryTopology::Virtualized => 0.70,
        MemoryTopology::System => 0.30,
        MemoryTopology::Unknown => 0.90,
    };
    (gpu.score.max(1) as f64) * topology_bonus
}

fn select_secondary_adapter_slot(
    usable: &[&GpuAdapterProfile],
    primary_slot: usize,
    primary_gpu: &GpuAdapterProfile,
) -> Option<usize> {
    let has_discrete_helper = usable.iter().enumerate().any(|(idx, gpu)| {
        idx != primary_slot && matches!(gpu.memory_topology, MemoryTopology::DiscreteVram)
    });

    // Future-proof for dual-dGPU systems:
    // If the present adapter is discrete and another discrete adapter exists, select that helper
    // first so heavy secondary lanes can scale with higher VRAM bandwidth before falling back to
    // integrated-latency routing.
    if matches!(primary_gpu.memory_topology, MemoryTopology::DiscreteVram) && has_discrete_helper {
        return usable
            .iter()
            .enumerate()
            .filter(|(idx, gpu)| {
                *idx != primary_slot && matches!(gpu.memory_topology, MemoryTopology::DiscreteVram)
            })
            .max_by(|(_, left), (_, right)| {
                secondary_discrete_priority(left)
                    .partial_cmp(&secondary_discrete_priority(right))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(idx, _)| idx);
    }

    usable
        .iter()
        .enumerate()
        .filter(|(idx, _)| *idx != primary_slot)
        .max_by(|(_, left), (_, right)| {
            secondary_priority(left)
                .partial_cmp(&secondary_priority(right))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(idx, _)| idx)
}

fn secondary_discrete_priority(gpu: &GpuAdapterProfile) -> f64 {
    let score = gpu.score.max(1) as f64;
    let vram_factor = ((gpu.estimated_vram_mb.max(1024) as f64) / 8_192.0)
        .sqrt()
        .clamp(0.80, 1.60);
    let map_factor = if gpu.supports_mappable_primary_buffers {
        1.05
    } else {
        1.0
    };
    score * vram_factor * map_factor
}

fn secondary_priority(gpu: &GpuAdapterProfile) -> f64 {
    let latency_bonus = match gpu.memory_topology {
        MemoryTopology::Unified => 1.30,
        MemoryTopology::DiscreteVram => 0.95,
        MemoryTopology::Virtualized => 0.60,
        MemoryTopology::System => 0.25,
        MemoryTopology::Unknown => 0.80,
    };
    let mappable_bonus = if gpu.supports_mappable_primary_buffers {
        1.10
    } else {
        1.0
    };
    (gpu.score.max(1) as f64) * latency_bonus * mappable_bonus
}

fn heavy_weight_for(gpu: &GpuAdapterProfile, class: TaskClass) -> f64 {
    let score = gpu.score.max(1) as f64;
    let topology_factor = match gpu.memory_topology {
        MemoryTopology::DiscreteVram => match class {
            TaskClass::SampledProcessing => 1.20,
            TaskClass::Physics => 1.15,
            TaskClass::ObjectUpdate => 1.05,
            TaskClass::Ui | TaskClass::PostFx => 0.95,
        },
        MemoryTopology::Unified => match class {
            TaskClass::SampledProcessing => 0.95,
            TaskClass::Physics => 0.92,
            TaskClass::ObjectUpdate => 1.05,
            TaskClass::Ui | TaskClass::PostFx => 1.15,
        },
        MemoryTopology::Virtualized => 0.60,
        MemoryTopology::System => 0.20,
        MemoryTopology::Unknown => 0.85,
    };
    let exponent = match class {
        TaskClass::SampledProcessing => 1.18,
        TaskClass::Physics => 1.15,
        TaskClass::ObjectUpdate => 1.05,
        TaskClass::Ui | TaskClass::PostFx => 1.0,
    };
    score.powf(exponent) * topology_factor * unit_parallelism_factor(gpu, class)
}

fn secondary_latency_budget_ms(target_frame_budget_ms: f64) -> f64 {
    let target = if target_frame_budget_ms.is_finite() && target_frame_budget_ms > 0.0 {
        target_frame_budget_ms
    } else {
        16.67
    };
    // Reserve margin for interop/copy/present. The latency helper should finish early enough that
    // the primary can absorb jitter without stalling the frame.
    (target * 0.70).clamp(0.10, 12.0)
}

fn enforce_secondary_minimum_job_share(
    primary_slot: usize,
    secondary_slot: usize,
    primary_gpu: &GpuAdapterProfile,
    secondary_gpu: &GpuAdapterProfile,
    request: MultiGpuWorkloadRequest,
    sampled_distribution: &mut [u32],
    object_distribution: &mut [u32],
    physics_distribution: &mut [u32],
    ui_distribution: &[u32],
    post_fx_distribution: &[u32],
) {
    let total_jobs_requested = request
        .sampled_processing_jobs
        .saturating_add(request.object_updates)
        .saturating_add(request.physics_jobs)
        .saturating_add(request.ui_jobs)
        .saturating_add(request.post_fx_jobs);
    if total_jobs_requested == 0 {
        return;
    }

    let secondary_current_jobs = sampled_distribution
        .get(secondary_slot)
        .copied()
        .unwrap_or(0)
        .saturating_add(
            object_distribution
                .get(secondary_slot)
                .copied()
                .unwrap_or(0),
        )
        .saturating_add(
            physics_distribution
                .get(secondary_slot)
                .copied()
                .unwrap_or(0),
        )
        .saturating_add(ui_distribution.get(secondary_slot).copied().unwrap_or(0))
        .saturating_add(
            post_fx_distribution
                .get(secondary_slot)
                .copied()
                .unwrap_or(0),
        );

    let target_share = aggressive_secondary_target_share(primary_gpu, secondary_gpu);
    let target_secondary_jobs = ((total_jobs_requested as f64) * target_share)
        .ceil()
        .clamp(1.0, total_jobs_requested as f64) as u32;
    if secondary_current_jobs >= target_secondary_jobs {
        return;
    }

    let mut needed = target_secondary_jobs.saturating_sub(secondary_current_jobs);

    // Move heavy lanes from primary to secondary according to topology:
    // - secondary dGPU: sampled/physics first
    // - secondary UMA/iGPU: object/physics first
    match secondary_gpu.memory_topology {
        MemoryTopology::DiscreteVram => {
            needed = move_lane_jobs(sampled_distribution, primary_slot, secondary_slot, needed);
            needed = move_lane_jobs(physics_distribution, primary_slot, secondary_slot, needed);
            let _ = move_lane_jobs(object_distribution, primary_slot, secondary_slot, needed);
        }
        MemoryTopology::Unified => {
            needed = move_lane_jobs(object_distribution, primary_slot, secondary_slot, needed);
            needed = move_lane_jobs(physics_distribution, primary_slot, secondary_slot, needed);
            let _ = move_lane_jobs(sampled_distribution, primary_slot, secondary_slot, needed);
        }
        MemoryTopology::Virtualized | MemoryTopology::System | MemoryTopology::Unknown => {
            needed = move_lane_jobs(object_distribution, primary_slot, secondary_slot, needed);
            let _ = move_lane_jobs(physics_distribution, primary_slot, secondary_slot, needed);
        }
    }
}

fn aggressive_secondary_target_share(
    primary_gpu: &GpuAdapterProfile,
    secondary_gpu: &GpuAdapterProfile,
) -> f64 {
    let primary_score = primary_gpu.score.max(1) as f64;
    let secondary_score = secondary_gpu.score.max(1) as f64;
    let relative = (secondary_score / primary_score).clamp(0.05, 1.60);

    let (base, max_share) = match secondary_gpu.memory_topology {
        MemoryTopology::DiscreteVram => (0.24, 0.58),
        MemoryTopology::Unified => (0.18, 0.36),
        MemoryTopology::Virtualized => (0.10, 0.22),
        MemoryTopology::System | MemoryTopology::Unknown => (0.08, 0.18),
    };
    let dual_discrete_bonus = if matches!(primary_gpu.memory_topology, MemoryTopology::DiscreteVram)
        && matches!(secondary_gpu.memory_topology, MemoryTopology::DiscreteVram)
    {
        0.08
    } else {
        0.0
    };

    (base + relative.sqrt() * 0.18 + dual_discrete_bonus).clamp(base, max_share)
}

fn move_lane_jobs(
    lane_distribution: &mut [u32],
    from_slot: usize,
    to_slot: usize,
    needed: u32,
) -> u32 {
    if needed == 0 {
        return 0;
    }
    let available = lane_distribution.get(from_slot).copied().unwrap_or(0);
    if available == 0 {
        return needed;
    }
    let moved = available.min(needed);
    if let Some(from) = lane_distribution.get_mut(from_slot) {
        *from = from.saturating_sub(moved);
    }
    if let Some(to) = lane_distribution.get_mut(to_slot) {
        *to = to.saturating_add(moved);
    }
    needed.saturating_sub(moved)
}

fn estimate_lane_ms(
    gpu: &GpuAdapterProfile,
    sampled_processing_jobs: u32,
    object_updates: u32,
    physics_jobs: u32,
    ui_jobs: u32,
    post_fx_jobs: u32,
    request: MultiGpuWorkloadRequest,
) -> f64 {
    let score = gpu.score.max(1) as f64;

    // Weighted job units approximate relative GPU cost. Byte payload contributes a smaller term
    // so large texture-heavy jobs reflect memory pressure without fully dominating the estimate.
    let weighted_job_units = (sampled_processing_jobs as f64) * 24.0
        + (object_updates as f64) * 4.0
        + (physics_jobs as f64) * 10.0
        + (ui_jobs as f64) * 3.0
        + (post_fx_jobs as f64) * 5.0;

    let bytes = estimate_multi_gpu_assigned_bytes(
        gpu,
        request,
        sampled_processing_jobs,
        object_updates,
        physics_jobs,
        ui_jobs,
        post_fx_jobs,
    ) as f64;
    let byte_units = (bytes / (1024.0 * 1024.0)).sqrt() * 1.5;

    let topology_penalty = match gpu.memory_topology {
        MemoryTopology::DiscreteVram => 1.0,
        MemoryTopology::Unified => 0.95,
        MemoryTopology::Virtualized => 1.35,
        MemoryTopology::System => 2.0,
        MemoryTopology::Unknown => 1.15,
    };

    // Score already includes CU+bandwidth terms; this is a bounded correction so per-lane
    // estimates reflect SM/CU/CoreCluster parallelism without overfitting the heuristic.
    let unit_parallelism_boost = average_unit_parallelism_factor(gpu).clamp(0.75, 1.65);
    let per_score_scale = 6.0; // Tuned to keep planner estimates in a practical frame-time range.
    ((weighted_job_units + byte_units) * per_score_scale / (score * unit_parallelism_boost))
        * 1000.0
        * topology_penalty
}

fn estimate_multi_gpu_assigned_bytes(
    gpu: &GpuAdapterProfile,
    request: MultiGpuWorkloadRequest,
    sampled_processing_jobs: u32,
    object_updates: u32,
    physics_jobs: u32,
    ui_jobs: u32,
    post_fx_jobs: u32,
) -> u64 {
    let _ = gpu;
    let total = (sampled_processing_jobs as u64)
        .saturating_mul(request.bytes_per_sampled_job)
        .saturating_add((object_updates as u64).saturating_mul(request.bytes_per_object))
        .saturating_add((physics_jobs as u64).saturating_mul(request.bytes_per_physics_job))
        .saturating_add((ui_jobs as u64).saturating_mul(request.bytes_per_ui_job))
        .saturating_add((post_fx_jobs as u64).saturating_mul(request.bytes_per_post_fx_job));
    round_up_u64(total, 256)
}

fn build_shared_texture_bridge_plan(
    primary: &GpuAdapterProfile,
    secondary: Option<&GpuAdapterProfile>,
    request: MultiGpuWorkloadRequest,
    assignments: &[MultiGpuLaneAssignment],
) -> Option<SharedTextureBridgePlan> {
    let secondary = secondary?;
    let secondary_assignment = assignments
        .iter()
        .find(|assignment| assignment.adapter_index == secondary.index)?;

    if secondary_assignment.ui_jobs == 0 && secondary_assignment.post_fx_jobs == 0 {
        return None;
    }

    let bytes_per_frame = round_up_u64(
        request.processed_texture_bytes_per_frame.max(
            secondary_assignment
                .zero_copy
                .device_bytes
                .min(64 * 1024 * 1024),
        ),
        256,
    );

    let transfer_kind = if matches!(primary.memory_topology, MemoryTopology::Unified)
        && matches!(secondary.memory_topology, MemoryTopology::Unified)
    {
        SharedTransferKind::UnifiedMemoryMirror
    } else {
        SharedTransferKind::HostMappedBridge
    };

    let ring_segments = if matches!(secondary.memory_topology, MemoryTopology::Unified) {
        4
    } else {
        3
    };
    let frames_in_flight = if matches!(secondary.memory_topology, MemoryTopology::Unified) {
        4
    } else {
        3
    };
    let chunk_bytes = round_up_u64(
        (bytes_per_frame / ring_segments.max(1) as u64).clamp(256 * 1024, 16 * 1024 * 1024),
        256,
    );

    // Heuristic transfer estimate. Unified mirrors are cheaper; host-bridge adds memcpy/upload cost.
    let effective_gbps = match transfer_kind {
        SharedTransferKind::UnifiedMemoryMirror => 60.0,
        SharedTransferKind::HostMappedBridge => 18.0,
    };
    let estimated_transfer_ms =
        ((bytes_per_frame as f64) / (effective_gbps * 1_000_000_000.0)).max(0.0) * 1000.0 + 0.05;

    Some(SharedTextureBridgePlan {
        producer_adapter_index: secondary.index,
        consumer_adapter_index: primary.index,
        transfer_kind,
        producer_texture_usages: TextureUsages::STORAGE_BINDING
            | TextureUsages::RENDER_ATTACHMENT
            | TextureUsages::COPY_SRC
            | TextureUsages::TEXTURE_BINDING,
        consumer_texture_usages: TextureUsages::COPY_DST
            | TextureUsages::RENDER_ATTACHMENT
            | TextureUsages::TEXTURE_BINDING,
        producer_readback_buffer_usages: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        consumer_upload_buffer_usages: BufferUsages::MAP_WRITE | BufferUsages::COPY_SRC,
        bytes_per_frame,
        chunk_bytes,
        ring_segments,
        frames_in_flight,
        estimated_transfer_ms,
    })
}

fn build_multi_gpu_sync_plan(
    request: MultiGpuWorkloadRequest,
    _primary: &GpuAdapterProfile,
    secondary: Option<&GpuAdapterProfile>,
    estimated_secondary_ms: f64,
    bridge: Option<&SharedTextureBridgePlan>,
) -> MultiGpuSyncPlan {
    let Some(secondary) = secondary else {
        return default_sync_plan(request.target_frame_budget_ms);
    };

    let unified_secondary = matches!(secondary.memory_topology, MemoryTopology::Unified);
    let frames_in_flight = if unified_secondary { 4 } else { 3 };
    let integrated_ring_segments = if unified_secondary { 4 } else { 3 };
    let aggressive_integrated_preallocation = unified_secondary;
    let integrated_encoder_pool = if unified_secondary { 8 } else { 4 };

    // For integrated GPUs with observed stability issues (e.g. ~0.350 stability factor), we keep
    // a larger reserve of preallocated command buffers/encoders to avoid per-frame allocation jitter.
    let primary_command_buffers_preallocated = frames_in_flight * 3;
    let secondary_command_buffers_preallocated = if unified_secondary {
        frames_in_flight * 6
    } else {
        frames_in_flight * 4
    };
    let transfer_command_buffers_preallocated = frames_in_flight * 2;

    let secondary_budget_ms = secondary_latency_budget_ms(request.target_frame_budget_ms);
    let bridge_ms = bridge.map(|plan| plan.estimated_transfer_ms).unwrap_or(0.0);
    let secondary_slack_ms = (secondary_budget_ms - (estimated_secondary_ms + bridge_ms)).max(0.0);

    MultiGpuSyncPlan {
        fence_equivalent: SyncEquivalent::QueueCompletionFence,
        semaphore_equivalent: SyncEquivalent::QueueSubmissionTimeline,
        frames_in_flight,
        queue_timeline_stages: 3,
        primary_command_buffers_preallocated,
        secondary_command_buffers_preallocated,
        transfer_command_buffers_preallocated,
        aggressive_integrated_preallocation,
        integrated_encoder_pool,
        integrated_ring_segments,
        secondary_budget_ms,
        estimated_secondary_ms,
        secondary_slack_ms,
    }
}

impl GmsDispatcher {
    /// Discover adapters and build a dispatcher.
    pub fn discover() -> Self {
        Self {
            inventory: GpuInventory::discover(),
        }
    }

    /// Build from a precomputed inventory.
    pub fn new(inventory: GpuInventory) -> Self {
        Self { inventory }
    }

    /// Access discovered inventory.
    pub fn inventory(&self) -> &GpuInventory {
        &self.inventory
    }

    /// Build a CU/SM-score-weighted dispatch plan.
    pub fn plan_dispatch(&self, request: WorkloadRequest) -> DispatchPlan {
        let usable = self.inventory.usable_adapters().collect::<Vec<_>>();
        if usable.is_empty() {
            return DispatchPlan {
                total_object_updates: request.object_updates,
                total_physics_jobs: request.physics_jobs,
                total_jobs: request.object_updates.saturating_add(request.physics_jobs),
                assignments: Vec::new(),
            };
        }

        let object_weights = usable
            .iter()
            .map(|gpu| heavy_weight_for(gpu, TaskClass::ObjectUpdate))
            .collect::<Vec<_>>();

        // Physics is typically the heavier lane. We bias distribution to stronger adapters so
        // the fastest discrete GPU absorbs a larger share of expensive work.
        let physics_weights = usable
            .iter()
            .map(|gpu| heavy_weight_for(gpu, TaskClass::Physics))
            .collect::<Vec<_>>();

        let object_distribution = allocate_weighted_counts(request.object_updates, &object_weights);
        let physics_distribution = allocate_weighted_counts(request.physics_jobs, &physics_weights);
        let best_score = usable.iter().map(|gpu| gpu.score).max().unwrap_or(1);

        let assignments = usable
            .iter()
            .enumerate()
            .map(|(slot, gpu)| {
                let object_updates = object_distribution[slot];
                let physics_jobs = physics_distribution[slot];
                let total_jobs = object_updates.saturating_add(physics_jobs);
                let workgroup_size =
                    select_workgroup_size(*gpu, best_score, request.base_workgroup_size);
                let dispatch_workgroups = ceil_div_u32(total_jobs.max(1), workgroup_size.max(1));
                let score_share_pct =
                    (gpu.score as f64 / self.inventory.total_usable_score().max(1) as f64) * 100.0;
                let data_bytes =
                    estimate_assigned_bytes(gpu, request, object_updates, physics_jobs);
                let zero_copy = build_zero_copy_plan(gpu, data_bytes);

                GpuWorkAssignment {
                    adapter_index: gpu.index,
                    adapter_name: gpu.name.clone(),
                    score: gpu.score,
                    compute_unit_kind: gpu.compute_unit_kind,
                    compute_units: gpu.estimated_compute_units,
                    compute_unit_source: gpu.compute_unit_source,
                    score_share_pct,
                    object_updates,
                    physics_jobs,
                    total_jobs,
                    workgroup_size,
                    dispatch_workgroups,
                    zero_copy,
                }
            })
            .collect::<Vec<_>>();

        DispatchPlan {
            total_object_updates: request.object_updates,
            total_physics_jobs: request.physics_jobs,
            total_jobs: request.object_updates.saturating_add(request.physics_jobs),
            assignments,
        }
    }
}

fn allocate_weighted_counts(total: u32, weights: &[f64]) -> Vec<u32> {
    if total == 0 || weights.is_empty() {
        return vec![0; weights.len()];
    }

    let normalized_sum = weights.iter().copied().sum::<f64>().max(f64::EPSILON);
    let mut base = Vec::with_capacity(weights.len());
    let mut remainders = Vec::with_capacity(weights.len());
    let mut assigned = 0u32;

    for (index, weight) in weights.iter().copied().enumerate() {
        let exact = (total as f64) * (weight / normalized_sum);
        let floor = exact.floor() as u32;
        base.push(floor);
        remainders.push((index, exact - floor as f64));
        assigned = assigned.saturating_add(floor);
    }

    remainders.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut remaining = total.saturating_sub(assigned);
    for (index, _) in remainders {
        if remaining == 0 {
            break;
        }
        base[index] = base[index].saturating_add(1);
        remaining -= 1;
    }

    base
}

fn select_workgroup_size(
    gpu: &GpuAdapterProfile,
    best_score: u64,
    base_workgroup_size: u32,
) -> u32 {
    let base = clamp_pow2(base_workgroup_size.max(32), 32, 1024);
    let score_ratio = gpu.score_ratio_against(best_score.max(1)).clamp(0.1, 1.0);
    let unit_factor = average_unit_parallelism_factor(gpu)
        .mul_add(0.75, 0.25)
        .sqrt()
        .clamp(0.5, 2.5);
    let scaled = (base as f64) * score_ratio.sqrt() * unit_factor;

    let device_cap = gpu
        .limits
        .max_compute_invocations_per_workgroup
        .clamp(32, 1024);

    clamp_pow2(scaled.round() as u32, 32, device_cap)
}

fn estimate_assigned_bytes(
    gpu: &GpuAdapterProfile,
    request: WorkloadRequest,
    object_updates: u32,
    physics_jobs: u32,
) -> u64 {
    let _ = gpu;
    let payload_bytes = (object_updates as u64)
        .saturating_mul(request.bytes_per_object)
        .saturating_add((physics_jobs as u64).saturating_mul(request.bytes_per_physics_job));
    round_up_u64(payload_bytes, 256)
}

fn build_zero_copy_plan(gpu: &GpuAdapterProfile, total_bytes: u64) -> ZeroCopyBufferPlan {
    let device_buffer_usages =
        BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC;

    match gpu.memory_topology {
        MemoryTopology::Unified => {
            // Unified memory path: prefer MAP_WRITE heavily and reduce duplicate copies.
            let upload = BufferUsages::MAP_WRITE
                | BufferUsages::COPY_SRC
                | BufferUsages::COPY_DST
                | if gpu.supports_mappable_primary_buffers {
                    BufferUsages::STORAGE
                } else {
                    BufferUsages::empty()
                };

            ZeroCopyBufferPlan {
                upload_buffer_usages: upload,
                device_buffer_usages,
                prefer_mapped_primary: gpu.supports_mappable_primary_buffers,
                staging_bytes: round_up_u64(total_bytes, 256),
                device_bytes: round_up_u64(total_bytes, 256),
                ring_segments: 2,
            }
        }
        MemoryTopology::DiscreteVram | MemoryTopology::Virtualized => {
            // Discrete path: keep a persistent MAP_WRITE upload ring and batch copies to minimize PCIe traffic.
            let upload = BufferUsages::MAP_WRITE | BufferUsages::COPY_SRC;
            let staging = round_up_u64(total_bytes, 256);
            let device = round_up_u64(total_bytes, 256).max(256);

            ZeroCopyBufferPlan {
                upload_buffer_usages: upload,
                device_buffer_usages,
                prefer_mapped_primary: false,
                staging_bytes: staging,
                device_bytes: device,
                ring_segments: 3,
            }
        }
        MemoryTopology::System | MemoryTopology::Unknown => ZeroCopyBufferPlan {
            upload_buffer_usages: BufferUsages::MAP_WRITE | BufferUsages::COPY_SRC,
            device_buffer_usages,
            prefer_mapped_primary: false,
            staging_bytes: round_up_u64(total_bytes, 256),
            device_bytes: round_up_u64(total_bytes, 256).max(256),
            ring_segments: 2,
        },
    }
}

fn clamp_pow2(value: u32, min_value: u32, max_value: u32) -> u32 {
    let max_value = max_value.max(min_value);
    let clamped = value.clamp(min_value, max_value);
    let pow2 = clamped.next_power_of_two();
    if pow2 > max_value {
        prev_power_of_two(max_value)
    } else {
        pow2.max(min_value)
    }
}

fn average_unit_parallelism_factor(gpu: &GpuAdapterProfile) -> f64 {
    (unit_parallelism_factor(gpu, TaskClass::SampledProcessing)
        + unit_parallelism_factor(gpu, TaskClass::ObjectUpdate)
        + unit_parallelism_factor(gpu, TaskClass::Physics))
        / 3.0
}

fn unit_parallelism_factor(gpu: &GpuAdapterProfile, class: TaskClass) -> f64 {
    let units = gpu.estimated_compute_units.max(1) as f64;
    let baseline_units = match gpu.compute_unit_kind {
        ComputeUnitKind::Sm => 32.0,
        ComputeUnitKind::Cu => 24.0,
        ComputeUnitKind::CoreCluster => 10.0,
        ComputeUnitKind::Unknown => 16.0,
    };

    let class_exponent = match class {
        TaskClass::SampledProcessing => 0.60,
        TaskClass::Physics => 0.55,
        TaskClass::ObjectUpdate => 0.35,
        TaskClass::Ui | TaskClass::PostFx => 0.20,
    };

    let kind_bias = match gpu.compute_unit_kind {
        ComputeUnitKind::Sm => match class {
            TaskClass::SampledProcessing => 1.08,
            TaskClass::Physics => 1.03,
            TaskClass::ObjectUpdate => 1.00,
            TaskClass::Ui | TaskClass::PostFx => 0.98,
        },
        ComputeUnitKind::Cu => match class {
            TaskClass::SampledProcessing => 1.02,
            TaskClass::Physics => 1.08,
            TaskClass::ObjectUpdate => 1.00,
            TaskClass::Ui | TaskClass::PostFx => 0.98,
        },
        ComputeUnitKind::CoreCluster => match class {
            TaskClass::SampledProcessing | TaskClass::Physics => 0.97,
            TaskClass::ObjectUpdate | TaskClass::Ui | TaskClass::PostFx => 1.03,
        },
        ComputeUnitKind::Unknown => 1.0,
    };

    let source_confidence = match gpu.compute_unit_source {
        ComputeUnitEstimateSource::NativeProbe => 1.00,
        ComputeUnitEstimateSource::DeviceNameTable => 0.92,
        ComputeUnitEstimateSource::DriverLimitsHeuristic => 0.55,
    };

    let raw = (units / baseline_units).powf(class_exponent) * kind_bias;
    let blended = 1.0 + (raw - 1.0) * source_confidence;
    blended.clamp(0.70, 2.40)
}

fn prev_power_of_two(value: u32) -> u32 {
    if value <= 1 {
        1
    } else {
        1 << (31 - value.leading_zeros())
    }
}

fn ceil_div_u32(numerator: u32, denominator: u32) -> u32 {
    if denominator == 0 {
        0
    } else {
        (numerator.saturating_add(denominator - 1)) / denominator
    }
}

fn round_up_u64(value: u64, alignment: u64) -> u64 {
    if alignment == 0 {
        return value;
    }
    let rem = value % alignment;
    if rem == 0 {
        value
    } else {
        value.saturating_add(alignment - rem)
    }
}
