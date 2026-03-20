//! Runtime frame-loop integration for the Tileline MPS<->MGS bridge.
//!
//! Mirrors `frame_loop.rs` semantics so runtime code can switch between GMS and
//! MGS bridge paths without changing orchestration flow.

use std::collections::{BTreeSet, VecDeque};

use mgs::MobileGpuProfile;
use tl_core::{
    MgsBridgeFrameId, MgsBridgeFramePlan, MgsBridgeMpsSubmission, MpsMgsBridge, MpsMgsBridgeConfig,
    MpsMgsBridgeMetrics,
};

/// Runtime orchestrator configuration for MGS bridge path.
#[derive(Debug, Clone, Copy)]
pub struct MgsFrameLoopRuntimeConfig {
    /// Max number of sealed frames the bridge may publish during one `tick`.
    pub max_bridge_frames_per_tick: usize,
    /// Max number of published plans drained into runtime-local queue per `tick`.
    pub max_plan_drains_per_tick: usize,
}

impl Default for MgsFrameLoopRuntimeConfig {
    fn default() -> Self {
        Self {
            max_bridge_frames_per_tick: 4,
            max_plan_drains_per_tick: 16,
        }
    }
}

/// Aggregate result of one `tick` call on MGS frame-loop runtime.
#[derive(Debug, Clone, Copy, Default)]
pub struct MgsRuntimeTickResult {
    /// Number of frame plans published by `MpsMgsBridge::pump` during this tick.
    pub bridge_published_frames: usize,
    /// Number of frame plans drained into runtime-local queue.
    pub drained_frame_plans: usize,
    /// Current queued frame plans waiting for runtime consumption.
    pub queued_frame_plans: usize,
}

/// Snapshot of runtime-level counters plus MGS bridge metrics.
#[derive(Debug, Clone)]
pub struct MgsFrameLoopRuntimeMetrics {
    /// Number of `tick()` calls executed.
    pub tick_count: u64,
    /// Total frame plans published by bridge pumps across ticks.
    pub bridge_tick_published_frames: u64,
    /// Total frame plans drained into runtime-local queue.
    pub bridge_tick_drained_plans: u64,
    /// Number of frame plans popped by runtime.
    pub frame_plans_popped: u64,
    /// Current queued frame plans.
    pub queued_frame_plans: usize,
    /// Snapshot of underlying MPS<->MGS bridge counters/state.
    pub bridge: MpsMgsBridgeMetrics,
}

/// Runtime-owned integration layer for the MPS<->MGS bridge.
pub struct MgsFrameLoopRuntime {
    config: MgsFrameLoopRuntimeConfig,
    bridge: MpsMgsBridge,
    queued_plans: VecDeque<MgsBridgeFramePlan>,
    queued_plan_ids: BTreeSet<MgsBridgeFrameId>,
    tick_count: u64,
    bridge_tick_published_frames: u64,
    bridge_tick_drained_plans: u64,
    frame_plans_popped: u64,
}

impl MgsFrameLoopRuntime {
    /// Build from runtime + bridge config and auto-detected mobile profile.
    pub fn new(
        runtime_config: MgsFrameLoopRuntimeConfig,
        bridge_config: MpsMgsBridgeConfig,
        profile: MobileGpuProfile,
    ) -> Self {
        Self::with_bridge(runtime_config, MpsMgsBridge::new(profile, bridge_config))
    }

    /// Build from caller-provided bridge instance.
    pub fn with_bridge(runtime_config: MgsFrameLoopRuntimeConfig, bridge: MpsMgsBridge) -> Self {
        Self {
            config: runtime_config,
            bridge,
            queued_plans: VecDeque::new(),
            queued_plan_ids: BTreeSet::new(),
            tick_count: 0,
            bridge_tick_published_frames: 0,
            bridge_tick_drained_plans: 0,
            frame_plans_popped: 0,
        }
    }

    /// Access underlying bridge.
    pub fn bridge(&self) -> &MpsMgsBridge {
        &self.bridge
    }

    /// Mutable access to underlying bridge.
    pub fn bridge_mut(&mut self) -> &mut MpsMgsBridge {
        &mut self.bridge
    }

    /// Submit a CPU preprocessing task through MPS into MGS bridge.
    pub fn submit_cpu_task(
        &self,
        submission: MgsBridgeMpsSubmission,
    ) -> tl_core::MgsBridgeSubmitReceipt {
        self.bridge.submit_mps_task(submission)
    }

    /// Seal a frame so bridge may publish an MGS plan when completions are drained.
    pub fn seal_frame(&mut self, frame_id: MgsBridgeFrameId) {
        self.bridge.seal_frame(frame_id);
    }

    /// Pump bridge and drain published plans into runtime-local queue.
    pub fn tick(&mut self) -> MgsRuntimeTickResult {
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

            if self.queued_plan_ids.insert(plan.frame_id) {
                self.queued_plans.push_back(plan);
                drained += 1;
            }
        }

        self.bridge_tick_drained_plans = self
            .bridge_tick_drained_plans
            .saturating_add(drained as u64);

        MgsRuntimeTickResult {
            bridge_published_frames: published,
            drained_frame_plans: drained,
            queued_frame_plans: self.queued_plans.len(),
        }
    }

    /// Inspect next queued frame plan.
    pub fn peek_next_frame_plan(&self) -> Option<&MgsBridgeFramePlan> {
        self.queued_plans.front()
    }

    /// Pop next queued frame plan.
    pub fn pop_next_frame_plan(&mut self) -> Option<MgsBridgeFramePlan> {
        let plan = self.queued_plans.pop_front()?;
        self.queued_plan_ids.remove(&plan.frame_id);
        self.frame_plans_popped = self.frame_plans_popped.saturating_add(1);
        Some(plan)
    }

    /// Number of queued plans currently waiting for runtime consumption.
    pub fn queued_frame_plan_count(&self) -> usize {
        self.queued_plans.len()
    }

    /// Snapshot runtime + bridge metrics.
    pub fn metrics(&self) -> MgsFrameLoopRuntimeMetrics {
        MgsFrameLoopRuntimeMetrics {
            tick_count: self.tick_count,
            bridge_tick_published_frames: self.bridge_tick_published_frames,
            bridge_tick_drained_plans: self.bridge_tick_drained_plans,
            frame_plans_popped: self.frame_plans_popped,
            queued_frame_plans: self.queued_plans.len(),
            bridge: self.bridge.metrics(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_runtime() -> MgsFrameLoopRuntime {
        let profile = MobileGpuProfile::detect("Mali-G610");
        MgsFrameLoopRuntime::new(
            MgsFrameLoopRuntimeConfig::default(),
            MpsMgsBridgeConfig::default(),
            profile,
        )
    }

    #[test]
    fn publishes_and_pops_after_seal() {
        let mut rt = make_runtime();
        let descriptor = tl_core::MgsBridgeTaskDescriptor {
            frame_id: 1,
            object_count: 64,
            transfer_size_kb: 256,
            ..Default::default()
        };
        rt.bridge().enqueue_completed_descriptor(descriptor);
        rt.seal_frame(1);

        let tick = rt.tick();
        assert_eq!(tick.bridge_published_frames, 1);
        assert_eq!(tick.drained_frame_plans, 1);
        let plan = rt.pop_next_frame_plan().expect("expected plan");
        assert_eq!(plan.frame_id, 1);
    }

    #[test]
    fn dedupes_plans_by_frame_id() {
        let mut rt = make_runtime();
        let descriptor = tl_core::MgsBridgeTaskDescriptor {
            frame_id: 7,
            object_count: 10,
            transfer_size_kb: 10,
            ..Default::default()
        };
        rt.bridge().enqueue_completed_descriptor(descriptor);
        rt.seal_frame(7);
        rt.tick();

        // No duplicate re-insert should occur on next tick when queue is empty.
        rt.tick();
        assert_eq!(rt.queued_frame_plan_count(), 1);
    }
}
