//! Runtime helpers that convert scene workload estimates into bridge submissions.
//!
//! This module keeps scene-driven CPU->GPU planning in `runtime/src`:
//! - chunk scene workload estimates into bridge task descriptors
//! - submit tasks through MPS via `FrameLoopRuntime`
//! - seal the frame so `MpsGmsBridge` can publish a frame plan on the next pump

use gms::SceneWorkloadEstimate;
use mps::{CorePreference, NativeTask, TaskPriority};
use tl_core::{
    BridgeFrameId, BridgeGpuTaskKind, BridgeMpsSubmission, BridgeSubmitReceipt,
    BridgeTaskDescriptor, BridgeTaskRouting,
};

use crate::frame_loop::FrameLoopRuntime;

/// Runtime tuning for scene-workload dispatch into the bridge.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SceneDispatchBridgeConfig {
    /// Max sampled-processing jobs per submitted task descriptor.
    pub sampled_chunk_jobs: u32,
    /// Max object-update jobs per submitted task descriptor.
    pub object_chunk_jobs: u32,
    /// Max physics jobs per submitted task descriptor.
    pub physics_chunk_jobs: u32,
    /// Max AI/ML jobs per submitted task descriptor.
    pub ai_ml_chunk_jobs: u32,
    /// Max UI jobs per submitted task descriptor.
    pub ui_chunk_jobs: u32,
    /// Max post-FX jobs per submitted task descriptor.
    pub post_fx_chunk_jobs: u32,
    /// Collect `BridgeSubmitReceipt` entries in the result for debugging/telemetry.
    pub collect_receipts: bool,
}

impl Default for SceneDispatchBridgeConfig {
    fn default() -> Self {
        Self {
            sampled_chunk_jobs: 256,
            object_chunk_jobs: 512,
            physics_chunk_jobs: 256,
            ai_ml_chunk_jobs: 192,
            ui_chunk_jobs: 128,
            post_fx_chunk_jobs: 128,
            collect_receipts: false,
        }
    }
}

/// Per-lane submission counters for one scene dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SceneDispatchLaneSummary {
    pub lane: BridgeGpuTaskKind,
    pub submitted_tasks: u32,
    pub total_jobs: u32,
    pub total_payload_bytes: u64,
}

/// Aggregate submission result for one frame.
#[derive(Debug, Clone)]
pub struct SceneDispatchSubmission {
    pub frame_id: BridgeFrameId,
    pub total_submitted_tasks: u32,
    pub total_jobs: u32,
    pub lanes: Vec<SceneDispatchLaneSummary>,
    pub receipts: Vec<BridgeSubmitReceipt>,
    pub frame_sealed: bool,
}

/// Submit a scene workload estimate into the MPS<->GMS bridge and seal the frame.
///
/// This function emits lightweight native tasks so planning stays aligned with the MPS completion
/// path. The closures are intentionally small placeholders; callers can replace this with real
/// CPU preprocess work later while preserving descriptor mapping.
pub fn submit_scene_estimate_to_bridge(
    frame_loop: &mut FrameLoopRuntime,
    frame_id: BridgeFrameId,
    estimate: &SceneWorkloadEstimate,
    config: SceneDispatchBridgeConfig,
) -> SceneDispatchSubmission {
    let mut total_submitted_tasks = 0u32;
    let mut total_jobs = 0u32;
    let mut lanes = Vec::with_capacity(6);
    let mut receipts = Vec::new();

    let base_workgroup = Some(estimate.multi_gpu.base_workgroup_size.max(32));

    submit_lane(
        frame_loop,
        frame_id,
        LaneSubmitRequest {
            kind: BridgeGpuTaskKind::SampledMesh,
            routing: BridgeTaskRouting::ForcePrimaryHeavy,
            total_jobs: estimate.multi_gpu.sampled_processing_jobs,
            chunk_jobs: config.sampled_chunk_jobs,
            bytes_per_job: estimate.multi_gpu.bytes_per_sampled_job,
            cpu_priority: TaskPriority::High,
            cpu_preference: CorePreference::Performance,
            base_workgroup,
            processed_texture_total: 0,
            collect_receipts: config.collect_receipts,
        },
        &mut total_submitted_tasks,
        &mut total_jobs,
        &mut lanes,
        &mut receipts,
    );

    submit_lane(
        frame_loop,
        frame_id,
        LaneSubmitRequest {
            kind: BridgeGpuTaskKind::ObjectUpdate,
            routing: BridgeTaskRouting::Auto,
            total_jobs: estimate.multi_gpu.object_updates,
            chunk_jobs: config.object_chunk_jobs,
            bytes_per_job: estimate.multi_gpu.bytes_per_object,
            cpu_priority: TaskPriority::Normal,
            cpu_preference: CorePreference::Performance,
            base_workgroup,
            processed_texture_total: 0,
            collect_receipts: config.collect_receipts,
        },
        &mut total_submitted_tasks,
        &mut total_jobs,
        &mut lanes,
        &mut receipts,
    );

    submit_lane(
        frame_loop,
        frame_id,
        LaneSubmitRequest {
            kind: BridgeGpuTaskKind::TractionVfPhysics,
            routing: BridgeTaskRouting::ForcePrimaryHeavy,
            total_jobs: estimate.multi_gpu.physics_jobs,
            chunk_jobs: config.physics_chunk_jobs,
            bytes_per_job: estimate.multi_gpu.bytes_per_physics_job,
            cpu_priority: TaskPriority::High,
            cpu_preference: CorePreference::Performance,
            base_workgroup,
            processed_texture_total: 0,
            collect_receipts: config.collect_receipts,
        },
        &mut total_submitted_tasks,
        &mut total_jobs,
        &mut lanes,
        &mut receipts,
    );

    submit_lane(
        frame_loop,
        frame_id,
        LaneSubmitRequest {
            kind: BridgeGpuTaskKind::AiMl,
            routing: BridgeTaskRouting::ForcePrimaryHeavy,
            total_jobs: estimate.multi_gpu.ai_ml_jobs,
            chunk_jobs: config.ai_ml_chunk_jobs,
            bytes_per_job: estimate.multi_gpu.bytes_per_ai_ml_job,
            cpu_priority: TaskPriority::High,
            cpu_preference: CorePreference::Performance,
            base_workgroup,
            processed_texture_total: 0,
            collect_receipts: config.collect_receipts,
        },
        &mut total_submitted_tasks,
        &mut total_jobs,
        &mut lanes,
        &mut receipts,
    );

    submit_lane(
        frame_loop,
        frame_id,
        LaneSubmitRequest {
            kind: BridgeGpuTaskKind::Ui,
            routing: BridgeTaskRouting::ForceSecondaryLatency,
            total_jobs: estimate.multi_gpu.ui_jobs,
            chunk_jobs: config.ui_chunk_jobs,
            bytes_per_job: estimate.multi_gpu.bytes_per_ui_job,
            cpu_priority: TaskPriority::High,
            cpu_preference: CorePreference::Efficient,
            base_workgroup,
            processed_texture_total: 0,
            collect_receipts: config.collect_receipts,
        },
        &mut total_submitted_tasks,
        &mut total_jobs,
        &mut lanes,
        &mut receipts,
    );

    submit_lane(
        frame_loop,
        frame_id,
        LaneSubmitRequest {
            kind: BridgeGpuTaskKind::PostFx,
            routing: BridgeTaskRouting::ForceSecondaryLatency,
            total_jobs: estimate.multi_gpu.post_fx_jobs,
            chunk_jobs: config.post_fx_chunk_jobs,
            bytes_per_job: estimate.multi_gpu.bytes_per_post_fx_job,
            cpu_priority: TaskPriority::High,
            cpu_preference: CorePreference::Efficient,
            base_workgroup,
            processed_texture_total: estimate.multi_gpu.processed_texture_bytes_per_frame,
            collect_receipts: config.collect_receipts,
        },
        &mut total_submitted_tasks,
        &mut total_jobs,
        &mut lanes,
        &mut receipts,
    );

    let frame_sealed = total_submitted_tasks > 0;
    if frame_sealed {
        frame_loop.seal_frame(frame_id);
    }

    SceneDispatchSubmission {
        frame_id,
        total_submitted_tasks,
        total_jobs,
        lanes,
        receipts,
        frame_sealed,
    }
}

#[derive(Debug, Clone, Copy)]
struct LaneSubmitRequest {
    kind: BridgeGpuTaskKind,
    routing: BridgeTaskRouting,
    total_jobs: u32,
    chunk_jobs: u32,
    bytes_per_job: u64,
    cpu_priority: TaskPriority,
    cpu_preference: CorePreference,
    base_workgroup: Option<u32>,
    processed_texture_total: u64,
    collect_receipts: bool,
}

fn submit_lane(
    frame_loop: &FrameLoopRuntime,
    frame_id: BridgeFrameId,
    request: LaneSubmitRequest,
    total_submitted_tasks: &mut u32,
    total_jobs: &mut u32,
    lanes: &mut Vec<SceneDispatchLaneSummary>,
    receipts: &mut Vec<BridgeSubmitReceipt>,
) {
    if request.total_jobs == 0 {
        return;
    }

    let chunk_size = request.chunk_jobs.max(1);
    let mut remaining = request.total_jobs;
    let mut lane_tasks = 0u32;
    let mut lane_payload = 0u64;
    let mut lane_processed_texture_remaining = request.processed_texture_total;

    while remaining > 0 {
        let jobs = remaining.min(chunk_size);
        let payload_bytes = round_up_u64(
            (request.bytes_per_job.saturating_mul(jobs as u64)).max(256),
            256,
        );
        let processed_texture_bytes = if request.processed_texture_total > 0 {
            let estimated = (request.processed_texture_total / request.total_jobs.max(1) as u64)
                .saturating_mul(jobs as u64);
            let chunk = estimated.min(lane_processed_texture_remaining);
            lane_processed_texture_remaining =
                lane_processed_texture_remaining.saturating_sub(chunk);
            chunk
        } else {
            0
        };

        let descriptor = BridgeTaskDescriptor {
            frame_id,
            gpu_kind: request.kind,
            routing: request.routing,
            cpu_priority: request.cpu_priority,
            cpu_preference: request.cpu_preference,
            job_count: jobs,
            payload_bytes,
            processed_texture_bytes,
            base_workgroup_size_hint: request.base_workgroup,
        };

        let submission = BridgeMpsSubmission {
            descriptor,
            task: noop_native_task(),
        };
        let receipt = frame_loop.submit_cpu_task(submission);
        if request.collect_receipts {
            receipts.push(receipt);
        }

        lane_tasks = lane_tasks.saturating_add(1);
        lane_payload = lane_payload.saturating_add(payload_bytes);
        *total_submitted_tasks = total_submitted_tasks.saturating_add(1);
        *total_jobs = total_jobs.saturating_add(jobs);
        remaining -= jobs;
    }

    lanes.push(SceneDispatchLaneSummary {
        lane: request.kind,
        submitted_tasks: lane_tasks,
        total_jobs: request.total_jobs,
        total_payload_bytes: lane_payload,
    });
}

fn noop_native_task() -> NativeTask {
    Box::new(|| {})
}

fn round_up_u64(value: u64, alignment: u64) -> u64 {
    if alignment <= 1 {
        return value;
    }
    let rem = value % alignment;
    if rem == 0 {
        value
    } else {
        value.saturating_add(alignment - rem)
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use gms::{MultiGpuWorkloadRequest, SceneWorkloadEstimate, WorkloadRequest};
    use tl_core::MpsGmsBridgeConfig;

    use crate::frame_loop::FrameLoopRuntimeConfig;

    use super::*;

    #[test]
    fn submits_scene_estimate_and_publishes_bridge_plan() {
        let mut runtime = FrameLoopRuntime::new(
            FrameLoopRuntimeConfig::default(),
            MpsGmsBridgeConfig::default(),
        );
        let estimate = SceneWorkloadEstimate {
            single_gpu: WorkloadRequest {
                object_updates: 512,
                physics_jobs: 256,
                bytes_per_object: 256,
                bytes_per_physics_job: 1024,
                base_workgroup_size: 64,
            },
            multi_gpu: MultiGpuWorkloadRequest {
                sampled_processing_jobs: 512,
                object_updates: 768,
                physics_jobs: 384,
                ai_ml_jobs: 128,
                ui_jobs: 64,
                post_fx_jobs: 96,
                bytes_per_sampled_job: 2048,
                bytes_per_object: 256,
                bytes_per_physics_job: 1024,
                bytes_per_ai_ml_job: 2048,
                bytes_per_ui_job: 512,
                bytes_per_post_fx_job: 1024,
                processed_texture_bytes_per_frame: 4 * 1024 * 1024,
                base_workgroup_size: 64,
                target_frame_budget_ms: 16.67,
            },
            complexity_score: 0.0,
            estimated_frame_bytes: 0,
        };

        let submission = submit_scene_estimate_to_bridge(
            &mut runtime,
            1,
            &estimate,
            SceneDispatchBridgeConfig::default(),
        );
        assert!(submission.frame_sealed);
        assert!(submission.total_submitted_tasks > 0);

        // Wait for no-op MPS tasks to complete so bridge pump can publish the plan.
        let _ = runtime.wait_for_cpu_idle(Duration::from_millis(500));

        let mut got_plan = false;
        for _ in 0..8 {
            let _ = runtime.tick();
            if runtime.pop_next_frame_plan().is_some() {
                got_plan = true;
                break;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        assert!(got_plan);
    }
}
