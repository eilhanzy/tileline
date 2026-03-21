//! Tileline runtime integration crate.
//!
//! This crate hosts engine-level orchestration code that wires `tl-core` bridge primitives into a
//! render loop without pushing integration logic into benchmarks/examples.
//!
//! Modules:
//! - `frame_loop`: bridge pumping and frame-plan queue management
//! - `mobile_scene_workload`: scene->MGS hint synthesis helpers
//! - `mas`: Multi Audio Scheduler (MAS) primitives integrated with MPS
//! - `pak`: deterministic `.pak` asset packaging/list/unpack support
//! - `pre_alpha_loop`: canonical pre-alpha runtime phase ordering
//! - `scheduler_path`: automatic runtime selection policy for GMS vs MGS
//! - `scene`: runtime scene/sprite payloads and bounce showcase orchestration helpers
//! - `scene_dispatch`: scene workload -> bridge task submission helpers
//! - `scene_workload`: scene->GMS workload synthesis helpers
//! - `draw_path`: scene payload -> deterministic backend draw batches
//! - `telemetry_hud`: telemetry -> HUD sprite overlay composition
//! - `tile_world_2d`: chunked side-view tile world storage + visibility telemetry
//! - `tlapp_app`: canonical TLApp runtime entry moved from examples into core runtime
//! - `tlpfile`: project manifest that unifies `.tlscript` / `.tlsprite` / `.tljoint`
//! - `tlpfile_gui`: general-purpose GUI shell driven by `.tlpfile`
//! - `upscaler`: runtime FSR policy and fail-soft render-scale resolution
//! - `tljoint`: scene-based multi `.tlscript` + multi `.tlsprite` binding manifest
//! - `tlsprite`: `.tlsprite` sprite program parser and frame emitter
//! - `tlsprite_editor_cli`: runtime-owned CLI entry for list-mode `.tlsprite` editing
//! - `tlscript_showcase`: `.tlscript` showcase compile/evaluate bootstrap
//! - `tlsprite_editor`: list-mode `.tlsprite` editor model + lavender Alpha theme
//! - `vulkan_snapshot`: draw-frame -> raw Vulkan snapshot translation helpers
//! - `vulkan_scene_renderer`: runtime adapter over `tl-core` raw Vulkan backend
//! - `wgpu_scene_renderer`: backend implementation for draw-path + HUD sprite rendering
//! - `wgpu_render_loop`: canonical `wgpu` submit/present integration hooks

mod app_runner;
mod draw_path;
mod frame_loop;
mod mas;
mod mgs_frame_loop;
mod mobile_scene_dispatch;
mod mobile_scene_workload;
mod network_transport;
mod pak;
mod physics_mps_runner;
mod pre_alpha_loop;
mod runtime_bridge;
mod scene;
mod scene_dispatch;
mod scene_workload;
mod scheduler_path;
mod telemetry_hud;
mod tile_world_2d;
mod tlapp_app;
mod tljoint;
mod tlpfile;
mod tlpfile_gui;
mod tlscript_parallel;
mod tlscript_showcase;
mod tlsprite;
mod tlsprite_editor;
mod tlsprite_editor_cli;
mod upscaler;
mod versioning;
#[cfg(target_os = "linux")]
mod vulkan_scene_renderer;
mod vulkan_snapshot;
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
    MasWavError, MultiAudioScheduler, WavClip, WavPlaybackCursor, WavPlaybackParams,
};
pub use mgs_frame_loop::{
    MgsFrameLoopRuntime, MgsFrameLoopRuntimeConfig, MgsFrameLoopRuntimeMetrics,
    MgsRuntimeTickResult,
};
pub use mobile_scene_dispatch::{
    submit_mobile_hint_to_bridge, MobileSceneDispatchConfig, MobileSceneDispatchSubmission,
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
pub use nps::{MeshFanoutConfig, NetworkTopology};
pub use pak::{
    create_pak_from_dir, list_pak, read_file_from_pak, unpack_pak, PakBuildReport, PakEntry,
    PakIndex, PakUnpackReport,
};
pub use pre_alpha_loop::{
    RuntimeFramePhase, RuntimePhaseOrderMetrics, RuntimePhaseOrderTracker, RuntimePhaseViolation,
    PRE_ALPHA_PHASE_ORDER,
};
pub use runtime_bridge::{
    runtime_bridge_path_from_scheduler, RuntimeBridgeConfig, RuntimeBridgeMetrics,
    RuntimeBridgeOrchestrator, RuntimeBridgePath, RuntimeBridgeSubmission, RuntimeBridgeTick,
    RuntimeFramePlan,
};
pub use scene::{
    apply_scene_light_overrides, clamp_scene_lights_for_camera, BounceTankPatchMetrics,
    BounceTankRuntimePatch, BounceTankSceneConfig, BounceTankSceneController,
    BounceTankTickMetrics, RayTracingMode, RenderSyncMode, RuntimeSceneMode, SceneAudioTrack,
    SceneFrameInstances, SceneInstance3d, SceneLight, SceneLightKind, SceneLightOverride,
    SceneMaterial, ScenePrimitive3d, SceneTransform3d, SceneView2d, ShadingModel, SpriteInstance,
    SpriteKind, TickRatePolicy, MAX_SCENE_LIGHTS,
};
pub use scene_dispatch::{
    submit_scene_estimate_to_bridge, SceneDispatchBridgeConfig, SceneDispatchLaneSummary,
    SceneDispatchSubmission,
};
pub use scene_workload::{
    build_scene_workload_snapshot, estimate_scene_workload_requests, SceneWorkloadBridgeConfig,
};
pub use scheduler_path::{
    choose_scheduler_path, choose_scheduler_path_for_platform, GraphicsSchedulerDecision,
    GraphicsSchedulerPath, RuntimePlatform,
};
pub use telemetry_hud::{
    TelemetryHudComposer, TelemetryHudConfig, TelemetryHudMetrics, TelemetryHudSample,
};
pub use tile_world_2d::{
    ChunkedTileWorld2d, TileChunkCoord2d, TileCoord2d, TileMutation2d, TileView2d,
    TileVisibleInstance2d, TileVisibleSet2d, TileWorld2dConfig, TileWorldFrameTelemetry,
    TILE_ID_EMPTY,
};
pub use tlapp_app::run_from_env as run_tlapp_from_env;
#[cfg(target_os = "android")]
pub use tlapp_app::run_with_android_app as run_tlapp_with_android_app;
pub use tljoint::{
    compile_tljoint_scene_from_path, load_tljoint, parse_tljoint, TljointDiagnostic,
    TljointDiagnosticLevel, TljointManifest, TljointParseOutcome, TljointSceneBinding,
    TljointSceneBundle, TljointSceneCompileOutcome,
};
pub use tlpfile::{
    compile_tlpfile_scene_from_path, load_tlpfile, parse_tlpfile, TlpfileDiagnostic,
    TlpfileDiagnosticLevel, TlpfileGraphicsScheduler, TlpfileParseOutcome, TlpfileProject,
    TlpfileSceneBinding, TlpfileSceneBundle, TlpfileSceneCompileOutcome,
};
pub use tlpfile_gui::run_from_env as run_tlproject_gui_from_env;
#[cfg(target_os = "android")]
pub use tlpfile_gui::run_with_android_app as run_tlproject_gui_with_android_app;
pub use tlscript_parallel::{
    TlscriptDispatchSubmission, TlscriptMpsDispatchConfig, TlscriptParallelRuntimeCoordinator,
    TlscriptParallelRuntimeMetrics, TlscriptWorkChunk,
};
pub use tlscript_showcase::{
    compile_tlscript_showcase, TlscriptCoordinateSpace, TlscriptGfxProfile,
    TlscriptOverlayTileLookup, TlscriptPerformancePreset, TlscriptShowcaseCompileOutcome,
    TlscriptShowcaseConfig, TlscriptShowcaseControlInput, TlscriptShowcaseFrameInput,
    TlscriptShowcaseFrameOutput, TlscriptShowcaseProgram, TlscriptTileFill, TlscriptTileLookup,
};
pub use tlsprite::{
    compile_tlsprite, compile_tlsprite_pack, compile_tlsprite_with_extra_roots, load_tlsprite_pack,
    TlspriteCacheLoadOutcome, TlspriteCacheLoadSource, TlspriteCompileOutcome, TlspriteDiagnostic,
    TlspriteDiagnosticLevel, TlspriteFrameContext, TlspriteHotReloadConfig, TlspriteHotReloadEvent,
    TlspriteHotReloader, TlspriteLightDef, TlspritePack, TlspriteProgram, TlspriteProgramCache,
    TlspriteProgramCacheStats, TlspriteScaleAxis, TlspriteScaleSource, TlspriteSpriteDef,
    TlspriteWatchBackend, TlspriteWatchConfig, TlspriteWatchReloader,
};
pub use tlsprite_editor::{
    TlspriteEditorPalette, TlspriteEditorTheme, TlspriteListDocument, TlspriteListRow,
};
pub use tlsprite_editor_cli::run_from_env as run_tlsprite_editor_from_env;
pub use upscaler::{
    resolve_fsr_status, FsrConfig, FsrDynamoConfig, FsrMode, FsrQualityPreset, FsrStatus,
};
pub use versioning::{
    resolve_tileline_version_query, tileline_version_entries, TilelineVersionEntry, ENGINE_ID,
    ENGINE_VERSION,
};
#[cfg(target_os = "linux")]
pub use vulkan_scene_renderer::{
    VulkanSceneRenderer, VulkanSceneRendererConfig, VulkanSceneRendererError,
    VulkanSceneRendererFrameResult,
};
pub use vulkan_snapshot::{build_vulkan_render_snapshot, VulkanSnapshotBuildStats};
pub use wgpu_render_loop::{
    FrameExecutionTelemetry, PreAlphaFrameExecution, PreAlphaSystemsExecution,
    SecondaryHelperSubmitOutcome, WgpuRenderLoopCoordinator, WgpuRenderLoopMetrics,
};
pub use wgpu_scene_renderer::{
    SceneRayTracingStatus, WgpuSceneRenderer, WgpuSceneRendererUploadStats,
    DEFAULT_MSAA_SAMPLE_COUNT,
};
