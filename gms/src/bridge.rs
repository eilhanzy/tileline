//! GMS bridge/dispatcher planning.
//!
//! This module does not execute GPU kernels directly yet. It builds a scheduling plan that
//! can be consumed by the later render/compute runtime. The plan is weighted by the GPU
//! scores discovered in `hardware.rs`.

use crate::hardware::{GpuAdapterProfile, GpuInventory, MemoryTopology};
use wgpu::BufferUsages;

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

/// High-level GPU dispatcher planner.
#[derive(Debug, Clone)]
pub struct GmsDispatcher {
    inventory: GpuInventory,
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
            .map(|gpu| gpu.score as f64)
            .collect::<Vec<_>>();

        // Physics is typically the heavier lane. We bias distribution to stronger adapters so
        // the fastest discrete GPU absorbs a larger share of expensive work.
        let physics_weights = usable
            .iter()
            .map(|gpu| {
                let topology_bias = match gpu.memory_topology {
                    MemoryTopology::DiscreteVram => 1.10,
                    MemoryTopology::Unified => 0.95,
                    MemoryTopology::Virtualized => 0.70,
                    MemoryTopology::System => 0.30,
                    MemoryTopology::Unknown => 0.85,
                };
                (gpu.score.max(1) as f64).powf(1.15) * topology_bias
            })
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
    let cu_factor = (gpu.estimated_compute_units.max(1) as f64 / 16.0)
        .sqrt()
        .clamp(0.5, 2.5);
    let scaled = (base as f64) * score_ratio.sqrt() * cu_factor;

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
