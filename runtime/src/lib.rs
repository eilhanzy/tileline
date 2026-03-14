//! Tileline runtime integration crate.
//!
//! This crate hosts engine-level orchestration code that wires `tl-core` bridge primitives into a
//! render loop without pushing integration logic into benchmarks/examples.
//!
//! Modules:
//! - `frame_loop`: bridge pumping and frame-plan queue management
//! - `mobile_scene_workload`: scene->MGS hint synthesis helpers
//! - `pre_alpha_loop`: canonical pre-alpha runtime phase ordering
//! - `scheduler_path`: automatic runtime selection policy for GMS vs MGS
//! - `scene`: runtime scene/sprite payloads and bounce showcase orchestration helpers
//! - `scene_dispatch`: scene workload -> bridge task submission helpers
//! - `scene_workload`: scene->GMS workload synthesis helpers
//! - `tlscript_showcase`: `.tlscript` showcase compile/evaluate bootstrap
//! - `wgpu_render_loop`: canonical `wgpu` submit/present integration hooks

mod frame_loop;
mod mobile_scene_workload;
mod network_transport;
mod pre_alpha_loop;
mod scene;
mod scene_dispatch;
mod scene_workload;
mod scheduler_path;
mod tlscript_parallel;
mod tlscript_showcase;
mod wgpu_render_loop;

pub use frame_loop::{
    FrameLoopRuntime, FrameLoopRuntimeConfig, FrameLoopRuntimeMetrics, FrameSubmissionRecordResult,
    RuntimeTickResult,
};
pub use mobile_scene_workload::{
    build_mobile_scene_snapshot, estimate_mobile_workload_hint, MobileSceneWorkloadBridgeConfig,
};
pub use network_transport::{
    LaneTrafficCounter, NetworkLaneMetrics, NetworkPeerMetrics, NetworkPumpResult,
    NetworkTransportConfig, NetworkTransportMetrics, NetworkTransportRuntime,
    SnapshotCadenceConfig,
};
pub use pre_alpha_loop::{
    RuntimeFramePhase, RuntimePhaseOrderMetrics, RuntimePhaseOrderTracker, RuntimePhaseViolation,
    PRE_ALPHA_PHASE_ORDER,
};
pub use scene::{
    BounceTankPatchMetrics, BounceTankRuntimePatch, BounceTankSceneConfig,
    BounceTankSceneController, BounceTankTickMetrics, RenderSyncMode, SceneFrameInstances,
    SceneInstance3d, SceneMaterial, ScenePrimitive3d, SceneTransform3d, ShadingModel,
    SpriteInstance, TickRatePolicy,
};
pub use scene_dispatch::{
    submit_scene_estimate_to_bridge, SceneDispatchBridgeConfig, SceneDispatchLaneSummary,
    SceneDispatchSubmission,
};
pub use scene_workload::{
    build_scene_workload_snapshot, estimate_scene_workload_requests, SceneWorkloadBridgeConfig,
};
pub use scheduler_path::{choose_scheduler_path, GraphicsSchedulerDecision, GraphicsSchedulerPath};
pub use tlscript_parallel::{
    TlscriptDispatchSubmission, TlscriptMpsDispatchConfig, TlscriptParallelRuntimeCoordinator,
    TlscriptParallelRuntimeMetrics, TlscriptWorkChunk,
};
pub use tlscript_showcase::{
    compile_tlscript_showcase, TlscriptShowcaseCompileOutcome, TlscriptShowcaseConfig,
    TlscriptShowcaseFrameInput, TlscriptShowcaseFrameOutput, TlscriptShowcaseProgram,
};
pub use wgpu_render_loop::{
    FrameExecutionTelemetry, PreAlphaFrameExecution, PreAlphaSystemsExecution,
    SecondaryHelperSubmitOutcome, WgpuRenderLoopCoordinator, WgpuRenderLoopMetrics,
};
