//! Runtime frame-loop integration for the Tileline MPS<->GMS bridge.
//!
//! This module keeps benchmark/demo code thin by hosting the canonical frame orchestration flow in
//! `runtime/src`:
//! 1. Pump the lock-free MPS->GMS bridge without blocking CPU workers.
//! 2. Drain published GPU frame plans into a render-thread local queue.
//! 3. Record primary/secondary/transfer queue submissions after `wgpu::Queue::submit`.
//! 4. Reconcile present readiness using the bridge's bounded multi-GPU sync policy.
//!
//! The orchestration state is single-thread-owned (render/runtime thread), so no mutexes are used.

use std::collections::{BTreeSet, VecDeque};
use std::time::Duration;

use tl_core::{
    AdaptiveBufferDecision, BridgeFrameId, BridgeFramePlan, BridgeMpsSubmission, GpuQueueLane,
    GpuSubmissionHandle, GpuSubmissionWaiter, MpsGmsBridge, MpsGmsBridgeConfig,
    MpsGmsBridgeMetrics,
};

/// Runtime orchestrator configuration.
#[derive(Debug, Clone, Copy)]
pub struct FrameLoopRuntimeConfig {
    /// Max number of sealed frames the bridge may publish during one `tick` call.
    pub max_bridge_frames_per_tick: usize,
    /// Max number of published plans drained from the bridge queue into the local runtime queue.
    pub max_plan_drains_per_tick: usize,
}

impl Default for FrameLoopRuntimeConfig {
    fn default() -> Self {
        Self {
            max_bridge_frames_per_tick: 4,
            max_plan_drains_per_tick: 16,
        }
    }
}

/// Aggregate result of a `tick` call.
#[derive(Debug, Clone, Copy, Default)]
pub struct RuntimeTickResult {
    /// Number of frame plans published by `MpsGmsBridge::pump` during this tick.
    pub bridge_published_frames: usize,
    /// Number of frame plans drained into the runtime-local queue.
    pub drained_frame_plans: usize,
    /// Current number of frame plans waiting to be consumed by the render loop.
    pub queued_frame_plans: usize,
}

/// Result of recording queue submissions for a frame.
#[derive(Debug, Clone, Copy, Default)]
pub struct FrameSubmissionRecordResult {
    /// `true` if the primary queue submission index was accepted by the bridge sync tracker.
    pub primary_recorded: bool,
    /// `true` if the secondary/helper queue submission index was accepted by the tracker.
    pub secondary_recorded: bool,
    /// `true` if the transfer queue submission index was accepted by the tracker.
    pub transfer_recorded: bool,
}

/// Snapshot of runtime-level counters plus bridge metrics.
#[derive(Debug, Clone)]
pub struct FrameLoopRuntimeMetrics {
    /// Number of `tick()` calls executed by the runtime coordinator.
    pub tick_count: u64,
    /// Total number of frame plans published by bridge pumps across ticks.
    pub bridge_tick_published_frames: u64,
    /// Total number of frame plans drained into the runtime-local queue.
    pub bridge_tick_drained_plans: u64,
    /// Number of frame plans popped by the render loop for execution.
    pub frame_plans_popped: u64,
    /// Number of primary queue submission records forwarded to the bridge.
    pub primary_submission_records: u64,
    /// Number of secondary queue submission records forwarded to the bridge.
    pub secondary_submission_records: u64,
    /// Number of transfer queue submission records forwarded to the bridge.
    pub transfer_submission_records: u64,
    /// Number of present reconcile calls attempted by the runtime.
    pub present_reconcile_calls: u64,
    /// Number of reconcile calls that reported the frame ready for present.
    pub present_reconcile_ready: u64,
    /// Number of reconcile calls that timed out within the bounded wait budget.
    pub present_reconcile_timeouts: u64,
    /// Current count of queued frame plans waiting in the runtime-local queue.
    pub queued_frame_plans: usize,
    /// Snapshot of underlying bridge counters and state.
    pub bridge: MpsGmsBridgeMetrics,
}

/// Render-thread-owned runtime integration layer for the MPS<->GMS bridge.
pub struct FrameLoopRuntime {
    config: FrameLoopRuntimeConfig,
    bridge: MpsGmsBridge,
    queued_plans: VecDeque<BridgeFramePlan>,
    queued_plan_ids: BTreeSet<BridgeFrameId>,
    tick_count: u64,
    bridge_tick_published_frames: u64,
    bridge_tick_drained_plans: u64,
    frame_plans_popped: u64,
    primary_submission_records: u64,
    secondary_submission_records: u64,
    transfer_submission_records: u64,
    present_reconcile_calls: u64,
    present_reconcile_ready: u64,
    present_reconcile_timeouts: u64,
}

impl FrameLoopRuntime {
    /// Build a runtime orchestrator with a newly constructed bridge.
    pub fn new(runtime_config: FrameLoopRuntimeConfig, bridge_config: MpsGmsBridgeConfig) -> Self {
        Self::with_bridge(runtime_config, MpsGmsBridge::new(bridge_config))
    }

    /// Build a runtime orchestrator from an existing bridge instance.
    pub fn with_bridge(runtime_config: FrameLoopRuntimeConfig, bridge: MpsGmsBridge) -> Self {
        Self {
            config: runtime_config,
            bridge,
            queued_plans: VecDeque::new(),
            queued_plan_ids: BTreeSet::new(),
            tick_count: 0,
            bridge_tick_published_frames: 0,
            bridge_tick_drained_plans: 0,
            frame_plans_popped: 0,
            primary_submission_records: 0,
            secondary_submission_records: 0,
            transfer_submission_records: 0,
            present_reconcile_calls: 0,
            present_reconcile_ready: 0,
            present_reconcile_timeouts: 0,
        }
    }

    /// Access the underlying bridge for advanced operations.
    pub fn bridge(&self) -> &MpsGmsBridge {
        &self.bridge
    }

    /// Mutable access to the underlying bridge for advanced integrations.
    pub fn bridge_mut(&mut self) -> &mut MpsGmsBridge {
        &mut self.bridge
    }

    /// Submit a CPU-side preprocessing task into MPS via the bridge.
    pub fn submit_cpu_task(&self, submission: BridgeMpsSubmission) -> tl_core::BridgeSubmitReceipt {
        self.bridge.submit_mps_task(submission)
    }

    /// Seal a frame so the bridge may publish a GMS plan once all CPU completions are drained.
    pub fn seal_frame(&mut self, frame_id: BridgeFrameId) {
        self.bridge.seal_frame(frame_id);
    }

    /// Pump the lock-free bridge and drain published frame plans into the local queue.
    ///
    /// This method does not block CPU workers. The only potentially bounded wait in the pipeline
    /// remains the explicit present reconcile step.
    pub fn tick(&mut self) -> RuntimeTickResult {
        self.tick_count = self.tick_count.saturating_add(1);

        let published = self
            .bridge
            .pump(self.config.max_bridge_frames_per_tick.max(1));
        self.bridge_tick_published_frames = self
            .bridge_tick_published_frames
            .saturating_add(published as u64);

        let mut drained = 0usize;
        let drain_limit = self.config.max_plan_drains_per_tick.max(1);
        while drained < drain_limit {
            let Some(plan) = self.bridge.try_pop_frame_plan() else {
                break;
            };

            // De-duplicate by frame id in case callers keep old plans around and re-pump quickly.
            if self.queued_plan_ids.insert(plan.frame_id) {
                self.queued_plans.push_back(plan);
                drained += 1;
            }
        }

        self.bridge_tick_drained_plans = self
            .bridge_tick_drained_plans
            .saturating_add(drained as u64);

        RuntimeTickResult {
            bridge_published_frames: published,
            drained_frame_plans: drained,
            queued_frame_plans: self.queued_plans.len(),
        }
    }

    /// Non-destructively inspect the next planned frame.
    pub fn peek_next_frame_plan(&self) -> Option<&BridgeFramePlan> {
        self.queued_plans.front()
    }

    /// Pop the next planned frame for render/compute submission.
    pub fn pop_next_frame_plan(&mut self) -> Option<BridgeFramePlan> {
        let plan = self.queued_plans.pop_front()?;
        self.queued_plan_ids.remove(&plan.frame_id);
        self.frame_plans_popped = self.frame_plans_popped.saturating_add(1);
        Some(plan)
    }

    /// Number of frame plans waiting for the render loop.
    pub fn queued_frame_plan_count(&self) -> usize {
        self.queued_plans.len()
    }

    /// Record queue submissions generated for a frame after actual `wgpu::Queue::submit` calls.
    pub fn record_frame_submissions(
        &mut self,
        frame_id: BridgeFrameId,
        primary_submission: Option<GpuSubmissionHandle>,
        secondary_submission: Option<GpuSubmissionHandle>,
        transfer_submission: Option<GpuSubmissionHandle>,
    ) -> FrameSubmissionRecordResult {
        let mut result = FrameSubmissionRecordResult::default();

        if let Some(submission) = primary_submission {
            result.primary_recorded =
                self.bridge
                    .record_gpu_submission(frame_id, GpuQueueLane::Primary, submission);
            if result.primary_recorded {
                self.primary_submission_records = self.primary_submission_records.saturating_add(1);
            }
        }

        if let Some(submission) = secondary_submission {
            result.secondary_recorded =
                self.bridge
                    .record_gpu_submission(frame_id, GpuQueueLane::Secondary, submission);
            if result.secondary_recorded {
                self.secondary_submission_records =
                    self.secondary_submission_records.saturating_add(1);
            }
        }

        if let Some(submission) = transfer_submission {
            result.transfer_recorded =
                self.bridge
                    .record_gpu_submission(frame_id, GpuQueueLane::Transfer, submission);
            if result.transfer_recorded {
                self.transfer_submission_records =
                    self.transfer_submission_records.saturating_add(1);
            }
        }

        result
    }

    /// Non-blocking present readiness probe (portable fence poll equivalent).
    pub fn try_reconcile_present_nonblocking(
        &mut self,
        primary_waiter: &dyn GpuSubmissionWaiter,
        secondary_waiter: Option<&dyn GpuSubmissionWaiter>,
        transfer_waiter: Option<&dyn GpuSubmissionWaiter>,
    ) -> Option<tl_core::ComposeBarrierState> {
        self.present_reconcile_calls = self.present_reconcile_calls.saturating_add(1);
        let state = self.bridge.try_reconcile_present_nonblocking(
            primary_waiter,
            secondary_waiter,
            transfer_waiter,
        );
        if let Some(state) = state.as_ref() {
            if state.ready {
                self.present_reconcile_ready = self.present_reconcile_ready.saturating_add(1);
            }
            if state.timed_out {
                self.present_reconcile_timeouts = self.present_reconcile_timeouts.saturating_add(1);
            }
        }
        state
    }

    /// Budgeted present reconcile (bounded wait, default bridge sync policy is 0.8ms).
    pub fn reconcile_present(
        &mut self,
        primary_waiter: &dyn GpuSubmissionWaiter,
        secondary_waiter: Option<&dyn GpuSubmissionWaiter>,
        transfer_waiter: Option<&dyn GpuSubmissionWaiter>,
    ) -> Option<tl_core::ComposeBarrierState> {
        self.present_reconcile_calls = self.present_reconcile_calls.saturating_add(1);
        let state =
            self.bridge
                .reconcile_present(primary_waiter, secondary_waiter, transfer_waiter);
        if let Some(state) = state.as_ref() {
            if state.ready {
                self.present_reconcile_ready = self.present_reconcile_ready.saturating_add(1);
            }
            if state.timed_out {
                self.present_reconcile_timeouts = self.present_reconcile_timeouts.saturating_add(1);
            }
        }
        state
    }

    /// Block until MPS is idle (utility for shutdown/tests). Not used in the hot frame path.
    pub fn wait_for_cpu_idle(&self, timeout: Duration) -> bool {
        self.bridge.mps().wait_for_idle(timeout)
    }

    /// Feed Apple UMA telemetry into the bridge's adaptive controller when available.
    pub fn reconcile_apple_uma(
        &mut self,
        telemetry: tl_core::AdaptiveFrameTelemetry,
    ) -> Option<AdaptiveBufferDecision> {
        self.bridge.reconcile_apple_uma(telemetry)
    }

    /// Snapshot runtime and bridge counters.
    pub fn metrics(&self) -> FrameLoopRuntimeMetrics {
        FrameLoopRuntimeMetrics {
            tick_count: self.tick_count,
            bridge_tick_published_frames: self.bridge_tick_published_frames,
            bridge_tick_drained_plans: self.bridge_tick_drained_plans,
            frame_plans_popped: self.frame_plans_popped,
            primary_submission_records: self.primary_submission_records,
            secondary_submission_records: self.secondary_submission_records,
            transfer_submission_records: self.transfer_submission_records,
            present_reconcile_calls: self.present_reconcile_calls,
            present_reconcile_ready: self.present_reconcile_ready,
            present_reconcile_timeouts: self.present_reconcile_timeouts,
            queued_frame_plans: self.queued_plans.len(),
            bridge: self.bridge.metrics(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_runtime_config_has_small_tick_batch() {
        let cfg = FrameLoopRuntimeConfig::default();
        assert!(cfg.max_bridge_frames_per_tick >= 1);
        assert!(cfg.max_plan_drains_per_tick >= 1);
    }
}
