//! Runtime helpers that convert mobile scene hints into MPS<->MGS bridge submissions.
//!
//! Keeps mobile path orchestration in `runtime/src` and mirrors the GMS dispatch helper semantics.

use mgs::MpsWorkloadHint;
use mps::{CorePreference, NativeTask, TaskPriority};
use tl_core::{
    MgsBridgeFrameId, MgsBridgeMpsSubmission, MgsBridgeSubmitReceipt, MgsBridgeTaskDescriptor,
};

use crate::mgs_frame_loop::MgsFrameLoopRuntime;

/// Runtime tuning for MGS workload-hint dispatch into the bridge.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MobileSceneDispatchConfig {
    /// Max object count represented by one submitted descriptor.
    pub object_chunk_count: u32,
    /// Collect submit receipts in the result for diagnostics.
    pub collect_receipts: bool,
    /// Priority for MPS preprocessing placeholders.
    pub cpu_priority: TaskPriority,
    /// Preferred CPU class for MPS preprocessing placeholders.
    pub cpu_preference: CorePreference,
}

impl Default for MobileSceneDispatchConfig {
    fn default() -> Self {
        Self {
            object_chunk_count: 2_048,
            collect_receipts: false,
            cpu_priority: TaskPriority::High,
            cpu_preference: CorePreference::Auto,
        }
    }
}

/// Aggregate submission result for one MGS frame.
#[derive(Debug, Clone)]
pub struct MobileSceneDispatchSubmission {
    pub frame_id: MgsBridgeFrameId,
    pub total_submitted_tasks: u32,
    pub total_objects: u32,
    pub total_transfer_kb: u32,
    pub receipts: Vec<MgsBridgeSubmitReceipt>,
    pub frame_sealed: bool,
}

/// Submit an MGS workload hint into the MPS<->MGS bridge and seal the frame.
///
/// Uses lightweight no-op closures so runtime can exercise canonical bridge flow
/// while keeping CPU placeholder cost negligible.
pub fn submit_mobile_hint_to_bridge(
    frame_loop: &mut MgsFrameLoopRuntime,
    frame_id: MgsBridgeFrameId,
    hint: MpsWorkloadHint,
    config: MobileSceneDispatchConfig,
) -> MobileSceneDispatchSubmission {
    let total_objects = hint.object_count.max(1);
    let chunk_size = config.object_chunk_count.max(1);

    let mut remaining = total_objects;
    let mut submitted = 0u32;
    let mut receipts = Vec::new();
    let mut transfer_remaining = hint.transfer_size_kb.max(1);

    while remaining > 0 {
        let objects = remaining.min(chunk_size);
        // Distribute transfer estimate proportionally per chunk, preserving total.
        let proportional_transfer = ((hint.transfer_size_kb.max(1) as u64)
            .saturating_mul(objects as u64)
            / total_objects as u64) as u32;
        let transfer_kb = proportional_transfer.max(1).min(transfer_remaining.max(1));
        transfer_remaining = transfer_remaining.saturating_sub(transfer_kb);

        let descriptor = MgsBridgeTaskDescriptor {
            frame_id,
            cpu_priority: config.cpu_priority,
            cpu_preference: config.cpu_preference,
            object_count: objects,
            transfer_size_kb: transfer_kb,
            target_width: hint.target_width,
            target_height: hint.target_height,
            latency_budget_ms: hint.latency_budget_ms,
        };

        let receipt = frame_loop.submit_cpu_task(MgsBridgeMpsSubmission {
            descriptor,
            task: make_noop_preprocess_task(),
        });

        if config.collect_receipts {
            receipts.push(receipt);
        }

        submitted = submitted.saturating_add(1);
        remaining = remaining.saturating_sub(objects);
    }

    let frame_sealed = submitted > 0;
    if frame_sealed {
        frame_loop.seal_frame(frame_id);
    }

    MobileSceneDispatchSubmission {
        frame_id,
        total_submitted_tasks: submitted,
        total_objects: total_objects,
        total_transfer_kb: hint.transfer_size_kb.max(1),
        receipts,
        frame_sealed,
    }
}

fn make_noop_preprocess_task() -> NativeTask {
    Box::new(|| {
        std::sync::atomic::compiler_fence(std::sync::atomic::Ordering::SeqCst);
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use mgs::MobileGpuProfile;

    #[test]
    fn submits_and_seals_mobile_hint() {
        let profile = MobileGpuProfile::detect("Mali-G610");
        let mut frame_loop = MgsFrameLoopRuntime::new(
            crate::MgsFrameLoopRuntimeConfig::default(),
            tl_core::MpsMgsBridgeConfig::default(),
            profile,
        );

        let hint = MpsWorkloadHint {
            transfer_size_kb: 512,
            object_count: 4_096,
            target_width: 1280,
            target_height: 720,
            latency_budget_ms: 16,
        };

        let submission = submit_mobile_hint_to_bridge(
            &mut frame_loop,
            3,
            hint,
            MobileSceneDispatchConfig::default(),
        );
        assert!(submission.frame_sealed);
        assert!(submission.total_submitted_tasks > 0);

        let mut published_any = false;
        let mut plan = None;
        for _ in 0..128 {
            let tick = frame_loop.tick();
            published_any |= tick.bridge_published_frames > 0;
            if let Some(next_plan) = frame_loop.pop_next_frame_plan() {
                plan = Some(next_plan);
                break;
            }
            std::thread::yield_now();
        }
        assert!(
            published_any || plan.is_some(),
            "expected at least one published/queued plan after sealing"
        );
        let plan = plan.expect("expected plan");
        assert_eq!(plan.frame_id, 3);
    }
}
