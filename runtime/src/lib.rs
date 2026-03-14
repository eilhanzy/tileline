//! Tileline runtime integration crate.
//!
//! This crate hosts engine-level orchestration code that wires `tl-core` bridge primitives into a
//! render loop without pushing integration logic into benchmarks/examples.
//!
//! Modules:
//! - `frame_loop`: bridge pumping and frame-plan queue management
//! - `pre_alpha_loop`: canonical pre-alpha runtime phase ordering
//! - `scene`: runtime scene/sprite payloads and bounce showcase orchestration helpers
//! - `scene_workload`: scene->GMS workload synthesis helpers
//! - `wgpu_render_loop`: canonical `wgpu` submit/present integration hooks

mod frame_loop;
mod network_transport;
mod pre_alpha_loop;
mod scene;
mod scene_workload;
mod tlscript_parallel;
mod wgpu_render_loop;

pub use frame_loop::{
    FrameLoopRuntime, FrameLoopRuntimeConfig, FrameLoopRuntimeMetrics, FrameSubmissionRecordResult,
    RuntimeTickResult,
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
    BounceTankSceneConfig, BounceTankSceneController, BounceTankTickMetrics, RenderSyncMode,
    SceneFrameInstances, SceneInstance3d, SceneMaterial, ScenePrimitive3d, SceneTransform3d,
    ShadingModel, SpriteInstance, TickRatePolicy,
};
pub use scene_workload::{
    build_scene_workload_snapshot, estimate_scene_workload_requests, SceneWorkloadBridgeConfig,
};
pub use tlscript_parallel::{
    TlscriptDispatchSubmission, TlscriptMpsDispatchConfig, TlscriptParallelRuntimeCoordinator,
    TlscriptParallelRuntimeMetrics, TlscriptWorkChunk,
};
pub use wgpu_render_loop::{
    FrameExecutionTelemetry, PreAlphaFrameExecution, PreAlphaSystemsExecution,
    SecondaryHelperSubmitOutcome, WgpuRenderLoopCoordinator, WgpuRenderLoopMetrics,
};
