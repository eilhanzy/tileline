//! `wgpu` render-loop integration for Tileline's MPS<->GMS bridge.
//!
//! This module is the engine-side (non-benchmark) glue that wires real queue submissions and
//! present synchronization to `tl-core::MpsGmsBridge` and the portable multi-GPU sync layer.
//! It keeps the hot path data-oriented and avoids blocking outside the explicit bounded present
//! reconcile step.

use std::collections::BTreeMap;
use std::io;
use std::time::Instant;

use crate::frame_loop::{
    FrameLoopRuntime, FrameLoopRuntimeConfig, FrameLoopRuntimeMetrics, FrameSubmissionRecordResult,
    RuntimeTickResult,
};
use crate::network_transport::{NetworkPumpResult, NetworkTransportRuntime};
use crate::pre_alpha_loop::{
    RuntimeFramePhase, RuntimePhaseOrderMetrics, RuntimePhaseOrderTracker, PRE_ALPHA_PHASE_ORDER,
};
use crate::scene_dispatch::{
    submit_scene_estimate_to_bridge, SceneDispatchBridgeConfig, SceneDispatchSubmission,
};
use crate::tlscript_parallel::TlscriptParallelRuntimeCoordinator;
use gms::{MultiGpuExecutor, MultiGpuFrameSubmitResult, SceneWorkloadEstimate};
use nps::DecodedPacketEvent;
use paradoxpe::PhysicsWorld;
use tl_core::{
    AdaptiveBufferDecision, AdaptiveFrameTelemetry, BridgeFrameId, BridgeFramePlan,
    ComposeBarrierState, MpsGmsBridgeConfig,
};
use tokio::net::UdpSocket;

/// Per-frame runtime execution hints and counters used for telemetry and sync decisions.
#[derive(Debug, Clone, Copy)]
pub struct FrameExecutionTelemetry {
    /// Wall-clock frame time in milliseconds (present interval or work interval, caller policy).
    pub frame_time_ms: f64,
    /// Primary queue encoder submissions issued for the frame.
    pub primary_submitted_encoders: u32,
    /// Transfer queue encoder submissions issued for the frame.
    pub transfer_submitted_encoders: u32,
}

impl Default for FrameExecutionTelemetry {
    fn default() -> Self {
        Self {
            frame_time_ms: 0.0,
            primary_submitted_encoders: 1,
            transfer_submitted_encoders: 0,
        }
    }
}

/// Result of a secondary helper submission routed through [`MultiGpuExecutor`].
#[derive(Debug, Clone)]
pub struct SecondaryHelperSubmitOutcome {
    /// Frame id this helper submission is associated with.
    pub frame_id: BridgeFrameId,
    /// Raw helper submission result returned by the GMS multi-GPU executor.
    pub submit: MultiGpuFrameSubmitResult,
    /// `true` when the returned submission index was recorded into the bridge synchronizer.
    pub recorded_to_bridge: bool,
}

/// Result snapshot from one canonical pre-alpha frame execution.
#[derive(Debug, Clone)]
pub struct PreAlphaFrameExecution {
    /// Monotonic sequence assigned by the runtime phase-order tracker.
    pub frame_sequence: u64,
    /// Canonical order applied for this execution.
    pub phase_order: [RuntimeFramePhase; 5],
    /// `tick_bridge()` result collected during RenderPlan phase.
    pub tick_result: RuntimeTickResult,
    /// Frame id from `begin_next_frame_plan()`, when a bridge plan was available.
    pub planned_frame_id: Option<BridgeFrameId>,
    /// Whether render submission callback was run for this frame.
    pub render_plan_executed: bool,
    /// Present reconcile result at the end of the frame.
    pub present_state: Option<ComposeBarrierState>,
    /// `true` when the phase tracker observed no ordering violation.
    pub phase_order_valid: bool,
}

/// Result snapshot for pre-alpha frame execution with runtime systems attached.
#[derive(Debug, Clone)]
pub struct PreAlphaSystemsExecution {
    /// Canonical pre-alpha frame execution result.
    pub frame: PreAlphaFrameExecution,
    /// Network transport pump counters for the network phase.
    pub network_pump: NetworkPumpResult,
    /// Number of decoded packet events drained during network phase.
    pub decoded_events: usize,
    /// Number of decode failures drained during network phase.
    pub decode_failures: usize,
    /// Number of encode failures drained during network phase.
    pub encode_failures: usize,
    /// Number of bootstrap hello packets queued during network phase.
    pub bootstrap_packets_queued: usize,
    /// Number of ParadoxPE fixed substeps executed in physics phase.
    pub physics_substeps: u32,
    /// Number of authoritative snapshot packets queued in physics phase.
    pub snapshot_packets_queued: usize,
}

/// Runtime-level counters for the `wgpu` integration glue.
#[derive(Debug, Clone)]
pub struct WgpuRenderLoopMetrics {
    /// Number of bridge tick iterations run by the coordinator.
    pub ticks: u64,
    /// Number of frame plans begun and tracked as active.
    pub plans_begun: u64,
    /// Current number of active frames tracked for submission/present state.
    pub active_frames: usize,
    /// Primary queue submissions recorded through the coordinator.
    pub primary_submit_records: u64,
    /// Secondary/helper queue submissions recorded through the coordinator.
    pub secondary_submit_records: u64,
    /// Transfer queue submissions recorded through the coordinator.
    pub transfer_submit_records: u64,
    /// Calls to the helper submission path.
    pub helper_submit_calls: u64,
    /// Total synthetic helper work units submitted via the helper lane.
    pub helper_work_units_submitted: u64,
    /// Calls to present reconciliation.
    pub present_reconcile_calls: u64,
    /// Reconcile calls that returned ready.
    pub present_reconcile_ready: u64,
    /// Reconcile calls that timed out before readiness.
    pub present_reconcile_timeouts: u64,
    /// Number of Apple UMA adaptive reconcile passes executed.
    pub apple_uma_reconciles: u64,
    /// Last Apple UMA adaptive decision emitted by the bridge sync layer.
    pub apple_uma_last_decision: Option<AdaptiveBufferDecision>,
    /// Number of frames executed through `run_pre_alpha_frame(...)`.
    pub pre_alpha_frame_calls: u64,
    /// Number of pre-alpha frame executions that had a bridge frame plan to render.
    pub pre_alpha_frames_with_plan: u64,
    /// Snapshot of canonical runtime phase-order tracking.
    pub phase_order: RuntimePhaseOrderMetrics,
    /// Snapshot of `tl-core` bridge metrics.
    pub frame_loop: tl_core::MpsGmsBridgeMetrics,
    /// Snapshot of `runtime::frame_loop` orchestration metrics.
    pub frame_loop_runtime: FrameLoopRuntimeMetrics,
}

#[derive(Debug, Clone)]
struct ActiveFrameState {
    _frame_id: BridgeFrameId,
    _plan_created_at: Instant,
    primary_submission_recorded: bool,
    secondary_submission_recorded: bool,
    transfer_submission_recorded: bool,
    last_present_reconcile: Option<ComposeBarrierState>,
}

/// Engine-facing `wgpu` render-loop coordinator around [`FrameLoopRuntime`].
///
/// Typical per-frame flow:
/// 1. `tick_bridge()`
/// 2. `begin_next_frame_plan()`
/// 3. Submit primary GPU work, call `record_primary_submission(...)`
/// 4. Optionally call `submit_secondary_helper_for_frame(...)` if a helper lane exists
/// 5. Optionally submit bridge copies, call `record_transfer_submission(...)`
/// 6. `reconcile_present(...)` before `frame.present()`
/// 7. `report_frame_telemetry(...)` to feed Apple UMA adaptive logic (when active)
pub struct WgpuRenderLoopCoordinator {
    frame_loop: FrameLoopRuntime,
    multi_gpu_executor: Option<MultiGpuExecutor>,
    active_frames: BTreeMap<BridgeFrameId, ActiveFrameState>,
    ticks: u64,
    plans_begun: u64,
    primary_submit_records: u64,
    secondary_submit_records: u64,
    transfer_submit_records: u64,
    helper_submit_calls: u64,
    helper_work_units_submitted: u64,
    present_reconcile_calls: u64,
    present_reconcile_ready: u64,
    present_reconcile_timeouts: u64,
    pre_alpha_frame_calls: u64,
    pre_alpha_frames_with_plan: u64,
    apple_uma_reconciles: u64,
    apple_uma_last_decision: Option<AdaptiveBufferDecision>,
    phase_order: RuntimePhaseOrderTracker,
}

impl WgpuRenderLoopCoordinator {
    /// Create a coordinator with a fresh bridge/runtime stack.
    pub fn new(
        frame_loop_config: FrameLoopRuntimeConfig,
        bridge_config: MpsGmsBridgeConfig,
    ) -> Self {
        Self::with_frame_loop(FrameLoopRuntime::new(frame_loop_config, bridge_config))
    }

    /// Create a coordinator from an existing frame-loop runtime.
    pub fn with_frame_loop(frame_loop: FrameLoopRuntime) -> Self {
        Self {
            frame_loop,
            multi_gpu_executor: None,
            active_frames: BTreeMap::new(),
            ticks: 0,
            plans_begun: 0,
            primary_submit_records: 0,
            secondary_submit_records: 0,
            transfer_submit_records: 0,
            helper_submit_calls: 0,
            helper_work_units_submitted: 0,
            present_reconcile_calls: 0,
            present_reconcile_ready: 0,
            present_reconcile_timeouts: 0,
            pre_alpha_frame_calls: 0,
            pre_alpha_frames_with_plan: 0,
            apple_uma_reconciles: 0,
            apple_uma_last_decision: None,
            phase_order: RuntimePhaseOrderTracker::default(),
        }
    }

    /// Access the underlying frame-loop runtime.
    pub fn frame_loop(&self) -> &FrameLoopRuntime {
        &self.frame_loop
    }

    /// Mutable access to the underlying frame-loop runtime.
    pub fn frame_loop_mut(&mut self) -> &mut FrameLoopRuntime {
        &mut self.frame_loop
    }

    /// Install or replace the optional GMS helper-lane executor.
    pub fn set_multi_gpu_executor(&mut self, executor: Option<MultiGpuExecutor>) {
        self.multi_gpu_executor = executor;
    }

    /// Access the optional helper-lane executor.
    pub fn multi_gpu_executor(&self) -> Option<&MultiGpuExecutor> {
        self.multi_gpu_executor.as_ref()
    }

    /// Mutable access to the optional helper-lane executor.
    pub fn multi_gpu_executor_mut(&mut self) -> Option<&mut MultiGpuExecutor> {
        self.multi_gpu_executor.as_mut()
    }

    /// Pump the bridge and drain newly published frame plans.
    pub fn tick_bridge(&mut self) -> RuntimeTickResult {
        self.ticks = self.ticks.saturating_add(1);
        self.frame_loop.tick()
    }

    /// Pop the next frame plan and register it as active for submission tracking.
    pub fn begin_next_frame_plan(&mut self) -> Option<BridgeFramePlan> {
        let plan = self.frame_loop.pop_next_frame_plan()?;
        let state = ActiveFrameState {
            _frame_id: plan.frame_id,
            _plan_created_at: plan.created_at,
            primary_submission_recorded: false,
            secondary_submission_recorded: false,
            transfer_submission_recorded: false,
            last_present_reconcile: None,
        };
        self.active_frames.insert(plan.frame_id, state);
        self.plans_begun = self.plans_begun.saturating_add(1);
        Some(plan)
    }

    /// Submit scene-derived workload estimates into bridge queues for one frame.
    ///
    /// This is the runtime-owned path for scene->bridge integration without benchmark glue.
    pub fn submit_scene_workload_for_frame(
        &mut self,
        frame_id: BridgeFrameId,
        estimate: &SceneWorkloadEstimate,
        config: SceneDispatchBridgeConfig,
    ) -> SceneDispatchSubmission {
        submit_scene_estimate_to_bridge(self.frame_loop_mut(), frame_id, estimate, config)
    }

    /// Execute one canonical pre-alpha frame in fixed order:
    /// `network -> script -> physics -> render_plan -> present`.
    ///
    /// This API is the runtime-owned integration path intended to reduce ordering regressions
    /// during the pre-alpha phase. Existing lower-level methods remain available for advanced use.
    pub fn run_pre_alpha_frame<NetworkPhase, ScriptPhase, PhysicsPhase, RenderPhase>(
        &mut self,
        primary_device: &wgpu::Device,
        transfer_device: Option<&wgpu::Device>,
        mut network_phase: NetworkPhase,
        mut script_phase: ScriptPhase,
        mut physics_phase: PhysicsPhase,
        mut render_phase: RenderPhase,
    ) -> PreAlphaFrameExecution
    where
        NetworkPhase: FnMut(&mut Self),
        ScriptPhase: FnMut(&mut Self),
        PhysicsPhase: FnMut(&mut Self),
        RenderPhase: FnMut(&BridgeFramePlan, &mut Self),
    {
        self.pre_alpha_frame_calls = self.pre_alpha_frame_calls.saturating_add(1);
        let frame_sequence = self.phase_order.begin_frame();
        let mut phase_order_valid = true;

        phase_order_valid &= self
            .phase_order
            .enter_phase(RuntimeFramePhase::Network)
            .is_ok();
        network_phase(self);

        phase_order_valid &= self
            .phase_order
            .enter_phase(RuntimeFramePhase::Script)
            .is_ok();
        script_phase(self);

        phase_order_valid &= self
            .phase_order
            .enter_phase(RuntimeFramePhase::Physics)
            .is_ok();
        physics_phase(self);

        phase_order_valid &= self
            .phase_order
            .enter_phase(RuntimeFramePhase::RenderPlan)
            .is_ok();
        let tick_result = self.tick_bridge();
        let mut planned_frame_id = None;
        let mut render_plan_executed = false;
        if let Some(plan) = self.begin_next_frame_plan() {
            planned_frame_id = Some(plan.frame_id);
            render_phase(&plan, self);
            render_plan_executed = true;
            self.pre_alpha_frames_with_plan = self.pre_alpha_frames_with_plan.saturating_add(1);
        }

        phase_order_valid &= self
            .phase_order
            .enter_phase(RuntimeFramePhase::Present)
            .is_ok();
        let present_state = self.reconcile_present(primary_device, transfer_device);
        phase_order_valid &= self.phase_order.finish_frame();

        PreAlphaFrameExecution {
            frame_sequence,
            phase_order: PRE_ALPHA_PHASE_ORDER,
            tick_result,
            planned_frame_id,
            render_plan_executed,
            present_state,
            phase_order_valid,
        }
    }

    /// Execute one canonical pre-alpha frame with runtime-owned systems:
    /// network transport -> `.tlscript` routing -> ParadoxPE fixed step -> render plan -> present.
    pub fn run_pre_alpha_frame_with_systems<ScriptPhase, RenderPhase>(
        &mut self,
        primary_device: &wgpu::Device,
        transfer_device: Option<&wgpu::Device>,
        network_transport: &mut NetworkTransportRuntime,
        network_socket: &UdpSocket,
        tlscript_runtime: &mut TlscriptParallelRuntimeCoordinator,
        physics_world: &mut PhysicsWorld,
        frame_dt: f32,
        mut script_phase: ScriptPhase,
        mut render_phase: RenderPhase,
    ) -> io::Result<PreAlphaSystemsExecution>
    where
        ScriptPhase: FnMut(
            &mut TlscriptParallelRuntimeCoordinator,
            &mut PhysicsWorld,
            &[DecodedPacketEvent],
            &mut Self,
        ),
        RenderPhase: FnMut(&BridgeFramePlan, &mut Self, &PhysicsWorld),
    {
        self.pre_alpha_frame_calls = self.pre_alpha_frame_calls.saturating_add(1);
        let frame_sequence = self.phase_order.begin_frame();
        let mut phase_order_valid = true;

        phase_order_valid &= self
            .phase_order
            .enter_phase(RuntimeFramePhase::Network)
            .is_ok();

        let bootstrap_tick = physics_world
            .fixed_step_clock()
            .tick()
            .min(u64::from(u32::MAX));
        let bootstrap_packets_queued =
            network_transport.begin_bootstrap_for_all_peers(bootstrap_tick as u32);

        let network_pump = match network_transport.pump_nonblocking(network_socket) {
            Ok(pump) => pump,
            Err(err) => {
                let _ = self.phase_order.finish_frame();
                return Err(err);
            }
        };
        let decoded_events = network_transport.drain_decoded_packets(256);
        let decode_failures = network_transport.drain_decode_failures(128).len();
        let encode_failures = network_transport.drain_encode_failures(128).len();
        let decoded_event_count = decoded_events.len();

        phase_order_valid &= self
            .phase_order
            .enter_phase(RuntimeFramePhase::Script)
            .is_ok();
        script_phase(tlscript_runtime, physics_world, &decoded_events, self);

        phase_order_valid &= self
            .phase_order
            .enter_phase(RuntimeFramePhase::Physics)
            .is_ok();
        let physics_substeps = physics_world.step(frame_dt.max(0.0));
        let snapshot_packets_queued =
            network_transport.queue_paradox_snapshot_if_due(physics_world);

        phase_order_valid &= self
            .phase_order
            .enter_phase(RuntimeFramePhase::RenderPlan)
            .is_ok();
        let tick_result = self.tick_bridge();
        let mut planned_frame_id = None;
        let mut render_plan_executed = false;
        if let Some(plan) = self.begin_next_frame_plan() {
            planned_frame_id = Some(plan.frame_id);
            render_phase(&plan, self, physics_world);
            render_plan_executed = true;
            self.pre_alpha_frames_with_plan = self.pre_alpha_frames_with_plan.saturating_add(1);
        }

        phase_order_valid &= self
            .phase_order
            .enter_phase(RuntimeFramePhase::Present)
            .is_ok();
        let present_state = self.reconcile_present(primary_device, transfer_device);
        phase_order_valid &= self.phase_order.finish_frame();

        Ok(PreAlphaSystemsExecution {
            frame: PreAlphaFrameExecution {
                frame_sequence,
                phase_order: PRE_ALPHA_PHASE_ORDER,
                tick_result,
                planned_frame_id,
                render_plan_executed,
                present_state,
                phase_order_valid,
            },
            network_pump,
            decoded_events: decoded_event_count,
            decode_failures,
            encode_failures,
            bootstrap_packets_queued,
            physics_substeps,
            snapshot_packets_queued,
        })
    }

    /// Record a primary queue submission for a frame.
    pub fn record_primary_submission(
        &mut self,
        frame_id: BridgeFrameId,
        submission: wgpu::SubmissionIndex,
    ) -> bool {
        let result =
            self.frame_loop
                .record_frame_submissions(frame_id, Some(submission), None, None);
        if result.primary_recorded {
            self.primary_submit_records = self.primary_submit_records.saturating_add(1);
            if let Some(state) = self.active_frames.get_mut(&frame_id) {
                state.primary_submission_recorded = true;
            }
        }
        result.primary_recorded
    }

    /// Record a transfer queue submission for a frame.
    pub fn record_transfer_submission(
        &mut self,
        frame_id: BridgeFrameId,
        submission: wgpu::SubmissionIndex,
    ) -> bool {
        let result =
            self.frame_loop
                .record_frame_submissions(frame_id, None, None, Some(submission));
        if result.transfer_recorded {
            self.transfer_submit_records = self.transfer_submit_records.saturating_add(1);
            if let Some(state) = self.active_frames.get_mut(&frame_id) {
                state.transfer_submission_recorded = true;
            }
        }
        result.transfer_recorded
    }

    /// Submit synthetic helper work on the secondary GPU and record the resulting submission index.
    ///
    /// Returns `None` when no helper executor is installed.
    pub fn submit_secondary_helper_for_frame(
        &mut self,
        frame_id: BridgeFrameId,
    ) -> Option<SecondaryHelperSubmitOutcome> {
        let executor = self.multi_gpu_executor.as_mut()?;
        self.helper_submit_calls = self.helper_submit_calls.saturating_add(1);

        let submit = executor.submit_frame_recording_submission();
        let mut record = FrameSubmissionRecordResult::default();
        if let Some(submission_index) = submit.submission_index.clone() {
            record = self.frame_loop.record_frame_submissions(
                frame_id,
                None,
                Some(submission_index),
                None,
            );
        }

        if record.secondary_recorded {
            self.secondary_submit_records = self.secondary_submit_records.saturating_add(1);
            if let Some(state) = self.active_frames.get_mut(&frame_id) {
                state.secondary_submission_recorded = true;
            }
        }
        self.helper_work_units_submitted = self
            .helper_work_units_submitted
            .saturating_add(submit.work_units as u64);

        Some(SecondaryHelperSubmitOutcome {
            frame_id,
            submit,
            recorded_to_bridge: record.secondary_recorded,
        })
    }

    /// Non-blocking readiness probe before present (portable fence poll path).
    pub fn try_reconcile_present_nonblocking(
        &mut self,
        primary_device: &wgpu::Device,
        transfer_device: Option<&wgpu::Device>,
    ) -> Option<ComposeBarrierState> {
        self.present_reconcile_calls = self.present_reconcile_calls.saturating_add(1);
        let secondary_device = self
            .multi_gpu_executor
            .as_ref()
            .map(|e| e.secondary_device());
        let state = self.frame_loop.try_reconcile_present_nonblocking(
            primary_device,
            secondary_device,
            transfer_device,
        );
        self.accumulate_present_reconcile_counters(state.as_ref());
        state
    }

    /// Bounded present reconcile before `SurfaceTexture::present` (default 0.8ms budget).
    pub fn reconcile_present(
        &mut self,
        primary_device: &wgpu::Device,
        transfer_device: Option<&wgpu::Device>,
    ) -> Option<ComposeBarrierState> {
        self.present_reconcile_calls = self.present_reconcile_calls.saturating_add(1);
        let secondary_device = self
            .multi_gpu_executor
            .as_ref()
            .map(|e| e.secondary_device());
        let state =
            self.frame_loop
                .reconcile_present(primary_device, secondary_device, transfer_device);
        self.accumulate_present_reconcile_counters(state.as_ref());
        if let Some(state_value) = state.as_ref() {
            if let Some(frame_id) = state_value.frame_id {
                if let Some(active) = self.active_frames.get_mut(&frame_id) {
                    active.last_present_reconcile = Some(state_value.clone());
                    if state_value.ready {
                        self.active_frames.remove(&frame_id);
                    }
                }
            }
        }
        state
    }

    /// Feed frame timing telemetry into the Apple UMA adaptive controller when active.
    ///
    /// Call this once per rendered frame (typically after present or after compose completion).
    pub fn report_frame_telemetry(
        &mut self,
        frame_id: BridgeFrameId,
        telemetry: FrameExecutionTelemetry,
    ) -> Option<AdaptiveBufferDecision> {
        let secondary_encoder_count = self
            .multi_gpu_executor
            .as_ref()
            .map(|executor| u32::from(executor.secondary_work_units_per_present() > 0))
            .unwrap_or(0);
        let in_flight_encoders = self
            .frame_loop
            .metrics()
            .bridge
            .sync
            .as_ref()
            .map(|sync| sync.tracked_frames as u32)
            .unwrap_or(0)
            .saturating_add(secondary_encoder_count);
        let igpu_score = self
            .multi_gpu_executor
            .as_ref()
            .map(|executor| executor.secondary_profile().score)
            .unwrap_or(0);

        let adaptive_input = AdaptiveFrameTelemetry {
            frame_time_ms: telemetry.frame_time_ms,
            submitted_encoders: telemetry
                .primary_submitted_encoders
                .saturating_add(telemetry.transfer_submitted_encoders)
                .saturating_add(secondary_encoder_count)
                .max(1),
            in_flight_encoders: in_flight_encoders.max(1),
            igpu_gms_hardware_score: igpu_score,
            // FrameExecutionTelemetry does not track swap-chain queue depth directly.
            // GMS congestion is already inferred from in_flight_encoders vs the encoder
            // window; pending_frame_count is left to be wired up if a dedicated
            // present-queue depth counter is added to FrameExecutionTelemetry later.
            pending_frame_count: 0,
        };

        let decision = self.frame_loop.reconcile_apple_uma(adaptive_input);
        if let Some(decision) = decision {
            self.apple_uma_reconciles = self.apple_uma_reconciles.saturating_add(1);
            self.apple_uma_last_decision = Some(decision);
        }

        // If the frame is no longer tracked and we only needed telemetry delivery, drop stale state.
        if let Some(active) = self.active_frames.get(&frame_id) {
            if active
                .last_present_reconcile
                .as_ref()
                .map(|s| s.ready)
                .unwrap_or(false)
            {
                self.active_frames.remove(&frame_id);
            }
        }

        decision
    }

    /// Snapshot metrics for profiling and integration validation.
    pub fn metrics(&self) -> WgpuRenderLoopMetrics {
        WgpuRenderLoopMetrics {
            ticks: self.ticks,
            plans_begun: self.plans_begun,
            active_frames: self.active_frames.len(),
            primary_submit_records: self.primary_submit_records,
            secondary_submit_records: self.secondary_submit_records,
            transfer_submit_records: self.transfer_submit_records,
            helper_submit_calls: self.helper_submit_calls,
            helper_work_units_submitted: self.helper_work_units_submitted,
            present_reconcile_calls: self.present_reconcile_calls,
            present_reconcile_ready: self.present_reconcile_ready,
            present_reconcile_timeouts: self.present_reconcile_timeouts,
            pre_alpha_frame_calls: self.pre_alpha_frame_calls,
            pre_alpha_frames_with_plan: self.pre_alpha_frames_with_plan,
            phase_order: self.phase_order.metrics(),
            apple_uma_reconciles: self.apple_uma_reconciles,
            apple_uma_last_decision: self.apple_uma_last_decision,
            frame_loop: self.frame_loop.bridge().metrics(),
            frame_loop_runtime: self.frame_loop.metrics(),
        }
    }

    fn accumulate_present_reconcile_counters(&mut self, state: Option<&ComposeBarrierState>) {
        if let Some(state) = state {
            if state.ready {
                self.present_reconcile_ready = self.present_reconcile_ready.saturating_add(1);
            }
            if state.timed_out {
                self.present_reconcile_timeouts = self.present_reconcile_timeouts.saturating_add(1);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_execution_telemetry_defaults_are_safe() {
        let t = FrameExecutionTelemetry::default();
        assert!(t.primary_submitted_encoders >= 1);
        assert!(t.frame_time_ms >= 0.0);
    }
}
