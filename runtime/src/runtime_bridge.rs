//! Runtime-owned unified bridge orchestration for GMS/MGS paths.
//!
//! This module provides one frame-pipeline surface over two bridge backends:
//! - `FrameLoopRuntime` (MPS<->GMS)
//! - `MgsFrameLoopRuntime` (MPS<->MGS)
//!
//! It enables TLApp to keep one overlapped frame loop while selecting GPU planner
//! path by scheduler policy.

use gms::SceneWorkloadEstimate;
use mgs::MobileGpuProfile;
use tl_core::{BridgeFramePlan, MgsBridgeFramePlan, MpsGmsBridgeConfig, MpsMgsBridgeConfig};

use crate::frame_loop::{FrameLoopRuntime, FrameLoopRuntimeConfig};
use crate::mgs_frame_loop::{MgsFrameLoopRuntime, MgsFrameLoopRuntimeConfig};
use crate::mobile_scene_dispatch::{
    submit_mobile_hint_to_bridge, MobileSceneDispatchConfig, MobileSceneDispatchSubmission,
};
use crate::mobile_scene_workload::{
    estimate_mobile_workload_hint, MobileSceneWorkloadBridgeConfig,
};
use crate::scene::SceneFrameInstances;
use crate::scene_dispatch::{
    submit_scene_estimate_to_bridge, SceneDispatchBridgeConfig, SceneDispatchSubmission,
};
use crate::scene_workload::{estimate_scene_workload_requests, SceneWorkloadBridgeConfig};
use crate::GraphicsSchedulerPath;

/// Active runtime bridge path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeBridgePath {
    /// Desktop/full multi-GPU planning path.
    GmsPath,
    /// Mobile/TBDR planning path.
    MgsPath,
}

impl RuntimeBridgePath {
    /// Human-readable short label for telemetry overlays.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::GmsPath => "gms",
            Self::MgsPath => "mgs",
        }
    }
}

/// Unified frame plan wrapper returned by runtime bridge orchestration.
#[derive(Debug, Clone)]
pub enum RuntimeFramePlan {
    Gms(BridgeFramePlan),
    Mgs(MgsBridgeFramePlan),
}

impl RuntimeFramePlan {
    /// Frame id attached to the wrapped bridge plan.
    pub fn frame_id(&self) -> u64 {
        match self {
            Self::Gms(plan) => plan.frame_id,
            Self::Mgs(plan) => plan.frame_id,
        }
    }

    /// Source path of this plan.
    pub fn path(&self) -> RuntimeBridgePath {
        match self {
            Self::Gms(_) => RuntimeBridgePath::GmsPath,
            Self::Mgs(_) => RuntimeBridgePath::MgsPath,
        }
    }
}

/// Unified scene submission result.
#[derive(Debug, Clone)]
pub enum RuntimeBridgeSubmission {
    Gms(SceneDispatchSubmission),
    Mgs(MobileSceneDispatchSubmission),
}

impl RuntimeBridgeSubmission {
    /// Frame id associated with this submission.
    pub fn frame_id(&self) -> u64 {
        match self {
            Self::Gms(s) => s.frame_id,
            Self::Mgs(s) => s.frame_id,
        }
    }

    /// Number of submitted CPU placeholder tasks.
    pub fn submitted_tasks(&self) -> u32 {
        match self {
            Self::Gms(s) => s.total_submitted_tasks,
            Self::Mgs(s) => s.total_submitted_tasks,
        }
    }

    /// Whether frame was sealed in this dispatch.
    pub fn frame_sealed(&self) -> bool {
        match self {
            Self::Gms(s) => s.frame_sealed,
            Self::Mgs(s) => s.frame_sealed,
        }
    }
}

/// Runtime bridge tick result with fallback diagnostics.
#[derive(Debug, Clone)]
pub struct RuntimeBridgeTick {
    pub bridge_path: RuntimeBridgePath,
    pub bridge_pump_published: usize,
    pub bridge_pump_drained: usize,
    pub queued_plan_depth: usize,
    pub used_fallback_plan: bool,
    pub plan: Option<RuntimeFramePlan>,
}

impl RuntimeBridgeTick {
    fn empty(path: RuntimeBridgePath) -> Self {
        Self {
            bridge_path: path,
            bridge_pump_published: 0,
            bridge_pump_drained: 0,
            queued_plan_depth: 0,
            used_fallback_plan: false,
            plan: None,
        }
    }
}

/// Path-independent bridge metrics used by runtime telemetry.
#[derive(Debug, Clone)]
pub struct RuntimeBridgeMetrics {
    pub bridge_path: RuntimeBridgePath,
    pub queued_plan_depth: usize,
    pub bridge_tick_published_frames: u64,
    pub bridge_tick_drained_plans: u64,
    pub frame_plans_popped: u64,
}

/// Runtime bridge configuration.
#[derive(Debug, Clone)]
pub struct RuntimeBridgeConfig {
    pub gms_frame_loop: FrameLoopRuntimeConfig,
    pub gms_bridge: MpsGmsBridgeConfig,
    pub gms_scene_workload: SceneWorkloadBridgeConfig,
    pub gms_scene_dispatch: SceneDispatchBridgeConfig,
    pub mgs_frame_loop: MgsFrameLoopRuntimeConfig,
    pub mgs_bridge: MpsMgsBridgeConfig,
    pub mgs_scene_workload: MobileSceneWorkloadBridgeConfig,
    pub mgs_scene_dispatch: MobileSceneDispatchConfig,
    /// Reuse last valid plan when no new bridge plan is ready for current frame.
    pub allow_fallback_reuse: bool,
}

impl Default for RuntimeBridgeConfig {
    fn default() -> Self {
        Self {
            gms_frame_loop: FrameLoopRuntimeConfig::default(),
            gms_bridge: MpsGmsBridgeConfig::default(),
            gms_scene_workload: SceneWorkloadBridgeConfig::default(),
            gms_scene_dispatch: SceneDispatchBridgeConfig::default(),
            mgs_frame_loop: MgsFrameLoopRuntimeConfig::default(),
            mgs_bridge: MpsMgsBridgeConfig::default(),
            mgs_scene_workload: MobileSceneWorkloadBridgeConfig::default(),
            mgs_scene_dispatch: MobileSceneDispatchConfig::default(),
            allow_fallback_reuse: true,
        }
    }
}

enum RuntimeBridgeLoop {
    Gms(FrameLoopRuntime),
    Mgs(MgsFrameLoopRuntime),
}

/// Runtime-owned unified bridge orchestrator for parallel frame pipeline.
pub struct RuntimeBridgeOrchestrator {
    config: RuntimeBridgeConfig,
    path: RuntimeBridgePath,
    inner: RuntimeBridgeLoop,
    last_plan: Option<RuntimeFramePlan>,
}

impl RuntimeBridgeOrchestrator {
    /// Build orchestrator from selected scheduler path.
    pub fn new_for_scheduler(
        scheduler: GraphicsSchedulerPath,
        adapter_name: &str,
        mut config: RuntimeBridgeConfig,
        viewport_width: u32,
        viewport_height: u32,
    ) -> Self {
        config.mgs_bridge.default_target_width = viewport_width.max(1);
        config.mgs_bridge.default_target_height = viewport_height.max(1);

        match runtime_bridge_path_from_scheduler(scheduler) {
            RuntimeBridgePath::GmsPath => Self {
                path: RuntimeBridgePath::GmsPath,
                inner: RuntimeBridgeLoop::Gms(FrameLoopRuntime::new(
                    config.gms_frame_loop,
                    config.gms_bridge,
                )),
                config,
                last_plan: None,
            },
            RuntimeBridgePath::MgsPath => {
                let profile = MobileGpuProfile::detect(adapter_name);
                Self {
                    path: RuntimeBridgePath::MgsPath,
                    inner: RuntimeBridgeLoop::Mgs(MgsFrameLoopRuntime::new(
                        config.mgs_frame_loop,
                        config.mgs_bridge.clone(),
                        profile,
                    )),
                    config,
                    last_plan: None,
                }
            }
        }
    }

    /// Selected runtime bridge path.
    pub fn path(&self) -> RuntimeBridgePath {
        self.path
    }

    /// Submit one frame workload into active bridge path and seal frame.
    pub fn submit_scene_workload(
        &mut self,
        frame_id: u64,
        frame: &SceneFrameInstances,
        dynamic_body_count: usize,
        viewport_width: u32,
        viewport_height: u32,
    ) -> RuntimeBridgeSubmission {
        match &mut self.inner {
            RuntimeBridgeLoop::Gms(frame_loop) => {
                let estimate: SceneWorkloadEstimate = estimate_scene_workload_requests(
                    frame,
                    dynamic_body_count,
                    viewport_width,
                    viewport_height,
                    self.config.gms_scene_workload,
                );
                RuntimeBridgeSubmission::Gms(submit_scene_estimate_to_bridge(
                    frame_loop,
                    frame_id,
                    &estimate,
                    self.config.gms_scene_dispatch,
                ))
            }
            RuntimeBridgeLoop::Mgs(frame_loop) => {
                let hint = estimate_mobile_workload_hint(
                    frame,
                    dynamic_body_count,
                    viewport_width,
                    viewport_height,
                    self.config.mgs_scene_workload,
                );
                RuntimeBridgeSubmission::Mgs(submit_mobile_hint_to_bridge(
                    frame_loop,
                    frame_id,
                    hint,
                    self.config.mgs_scene_dispatch,
                ))
            }
        }
    }

    /// Tick active bridge loop, pop next plan, and optionally reuse last plan as fail-soft fallback.
    pub fn tick_and_plan(&mut self) -> RuntimeBridgeTick {
        let mut tick = RuntimeBridgeTick::empty(self.path);

        let next_plan = match &mut self.inner {
            RuntimeBridgeLoop::Gms(frame_loop) => {
                let result = frame_loop.tick();
                tick.bridge_pump_published = result.bridge_published_frames;
                tick.bridge_pump_drained = result.drained_frame_plans;
                tick.queued_plan_depth = result.queued_frame_plans;
                frame_loop.pop_next_frame_plan().map(RuntimeFramePlan::Gms)
            }
            RuntimeBridgeLoop::Mgs(frame_loop) => {
                let result = frame_loop.tick();
                tick.bridge_pump_published = result.bridge_published_frames;
                tick.bridge_pump_drained = result.drained_frame_plans;
                tick.queued_plan_depth = result.queued_frame_plans;
                frame_loop.pop_next_frame_plan().map(RuntimeFramePlan::Mgs)
            }
        };

        if let Some(plan) = next_plan {
            self.last_plan = Some(plan.clone());
            tick.plan = Some(plan);
            return tick;
        }

        if self.config.allow_fallback_reuse {
            if let Some(last) = self.last_plan.clone() {
                tick.used_fallback_plan = true;
                tick.plan = Some(last);
            }
        }

        tick
    }

    /// Snapshot unified path-independent metrics.
    pub fn metrics(&self) -> RuntimeBridgeMetrics {
        match &self.inner {
            RuntimeBridgeLoop::Gms(frame_loop) => {
                let m = frame_loop.metrics();
                RuntimeBridgeMetrics {
                    bridge_path: self.path,
                    queued_plan_depth: m.queued_frame_plans,
                    bridge_tick_published_frames: m.bridge_tick_published_frames,
                    bridge_tick_drained_plans: m.bridge_tick_drained_plans,
                    frame_plans_popped: m.frame_plans_popped,
                }
            }
            RuntimeBridgeLoop::Mgs(frame_loop) => {
                let m = frame_loop.metrics();
                RuntimeBridgeMetrics {
                    bridge_path: self.path,
                    queued_plan_depth: m.queued_frame_plans,
                    bridge_tick_published_frames: m.bridge_tick_published_frames,
                    bridge_tick_drained_plans: m.bridge_tick_drained_plans,
                    frame_plans_popped: m.frame_plans_popped,
                }
            }
        }
    }
}

/// Map scheduler policy to runtime bridge path.
pub fn runtime_bridge_path_from_scheduler(scheduler: GraphicsSchedulerPath) -> RuntimeBridgePath {
    match scheduler {
        GraphicsSchedulerPath::Gms => RuntimeBridgePath::GmsPath,
        GraphicsSchedulerPath::Mgs => RuntimeBridgePath::MgsPath,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn scheduler_mapping_prefers_matching_path() {
        assert_eq!(
            runtime_bridge_path_from_scheduler(GraphicsSchedulerPath::Gms),
            RuntimeBridgePath::GmsPath
        );
        assert_eq!(
            runtime_bridge_path_from_scheduler(GraphicsSchedulerPath::Mgs),
            RuntimeBridgePath::MgsPath
        );
    }

    #[test]
    fn tick_reuses_last_plan_when_fallback_enabled() {
        let mut orchestrator = RuntimeBridgeOrchestrator::new_for_scheduler(
            GraphicsSchedulerPath::Mgs,
            "Mali-G610",
            RuntimeBridgeConfig::default(),
            1280,
            720,
        );

        let frame = SceneFrameInstances::default();
        let _submission = orchestrator.submit_scene_workload(1, &frame, 128, 1280, 720);
        let mut first_tick = orchestrator.tick_and_plan();
        for _ in 0..128 {
            if first_tick.plan.is_some() {
                break;
            }
            std::thread::sleep(Duration::from_millis(1));
            first_tick = orchestrator.tick_and_plan();
        }
        assert!(
            first_tick.plan.is_some(),
            "expected at least one plan after submission/seal"
        );

        // No submission for frame 2, so fallback should reuse latest valid plan.
        let second_tick = orchestrator.tick_and_plan();
        assert!(second_tick.plan.is_some());
        assert!(second_tick.used_fallback_plan);
    }
}
