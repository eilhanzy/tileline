//! Tileline runtime integration crate.
//!
//! This crate hosts engine-level orchestration code that wires `tl-core` bridge primitives into a
//! render loop without pushing integration logic into benchmarks/examples.
//!
//! Modules:
//! - `frame_loop`: bridge pumping and frame-plan queue management
//! - `mobile_scene_workload`: scene->MGS hint synthesis helpers
//! - `mas`: Multi Audio Scheduler (MAS) primitives integrated with MPS
//! - `pre_alpha_loop`: canonical pre-alpha runtime phase ordering
//! - `scheduler_path`: automatic runtime selection policy for GMS vs MGS
//! - `scene`: runtime scene/sprite payloads and bounce showcase orchestration helpers
//! - `scene_dispatch`: scene workload -> bridge task submission helpers
//! - `scene_workload`: scene->GMS workload synthesis helpers
//! - `draw_path`: scene payload -> deterministic backend draw batches
//! - `telemetry_hud`: telemetry -> HUD sprite overlay composition
//! - `tlapp_app`: canonical TLApp runtime entry moved from examples into core runtime
//! - `tlsprite`: `.tlsprite` sprite program parser and frame emitter
//! - `tlscript_showcase`: `.tlscript` showcase compile/evaluate bootstrap
//! - `tlsprite_editor`: list-mode `.tlsprite` editor model + lavender Alpha theme
//! - `wgpu_scene_renderer`: backend implementation for draw-path + HUD sprite rendering
//! - `wgpu_render_loop`: canonical `wgpu` submit/present integration hooks

mod draw_path;
mod frame_loop;
mod mas;
mod mobile_scene_workload;
mod network_transport;
mod pre_alpha_loop;
mod scene;
mod scene_dispatch;
mod scene_workload;
mod scheduler_path;
mod telemetry_hud;
mod tlapp_app;
mod tlscript_parallel;
mod tlscript_showcase;
mod tlsprite;
mod tlsprite_editor;
mod wgpu_render_loop;
mod wgpu_scene_renderer;

pub use draw_path::{
    DrawBatch3d, DrawBatchKey, DrawFrameStats, DrawInstance3d, DrawLane, DrawPathCompiler,
    RuntimeDrawFrame,
};
pub use frame_loop::{
    FrameLoopRuntime, FrameLoopRuntimeConfig, FrameLoopRuntimeMetrics, FrameSubmissionRecordResult,
    RuntimeTickResult,
};
pub use mas::{
    AudioBufferBlock, MasConfig, MasCoreAffinity, MasMetrics, MasPriority, MasSubmission,
    MultiAudioScheduler,
};
pub use mobile_scene_workload::{
    build_mobile_scene_snapshot, estimate_mobile_workload_hint, MobileSceneWorkloadBridgeConfig,
};
pub use network_transport::{
    LaneTrafficCounter, NetworkBootstrapConfig, NetworkBootstrapRole, NetworkLaneMetrics,
    NetworkPeerMetrics, NetworkPeerSessionState, NetworkPumpResult, NetworkSessionPhase,
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
    SpriteInstance, SpriteKind, TickRatePolicy,
};
pub use scene_dispatch::{
    submit_scene_estimate_to_bridge, SceneDispatchBridgeConfig, SceneDispatchLaneSummary,
    SceneDispatchSubmission,
};
pub use scene_workload::{
    build_scene_workload_snapshot, estimate_scene_workload_requests, SceneWorkloadBridgeConfig,
};
pub use scheduler_path::{choose_scheduler_path, GraphicsSchedulerDecision, GraphicsSchedulerPath};
pub use telemetry_hud::{
    TelemetryHudComposer, TelemetryHudConfig, TelemetryHudMetrics, TelemetryHudSample,
};
pub use tlapp_app::run_from_env as run_tlapp_from_env;
pub use tlscript_parallel::{
    TlscriptDispatchSubmission, TlscriptMpsDispatchConfig, TlscriptParallelRuntimeCoordinator,
    TlscriptParallelRuntimeMetrics, TlscriptWorkChunk,
};
pub use tlscript_showcase::{
    compile_tlscript_showcase, TlscriptShowcaseCompileOutcome, TlscriptShowcaseConfig,
    TlscriptShowcaseControlInput, TlscriptShowcaseFrameInput, TlscriptShowcaseFrameOutput,
    TlscriptShowcaseProgram,
};
pub use tlsprite::{
    compile_tlsprite, compile_tlsprite_pack, load_tlsprite_pack, TlspriteCacheLoadOutcome,
    TlspriteCacheLoadSource, TlspriteCompileOutcome, TlspriteDiagnostic, TlspriteDiagnosticLevel,
    TlspriteFrameContext, TlspriteHotReloadConfig, TlspriteHotReloadEvent, TlspriteHotReloader,
    TlspritePack, TlspriteProgram, TlspriteProgramCache, TlspriteProgramCacheStats,
    TlspriteScaleAxis, TlspriteScaleSource, TlspriteSpriteDef, TlspriteWatchBackend,
    TlspriteWatchConfig, TlspriteWatchReloader,
};
pub use tlsprite_editor::{
    TlspriteEditorPalette, TlspriteEditorTheme, TlspriteListDocument, TlspriteListRow,
};
pub use wgpu_render_loop::{
    FrameExecutionTelemetry, PreAlphaFrameExecution, PreAlphaSystemsExecution,
    SecondaryHelperSubmitOutcome, WgpuRenderLoopCoordinator, WgpuRenderLoopMetrics,
};
pub use wgpu_scene_renderer::{WgpuSceneRenderer, WgpuSceneRendererUploadStats};
