//! Tileline core orchestration crate.
//!
//! `tl-core` is the integration boundary between:
//! - `mps` (CPU scheduling / WASM execution)
//! - `gms` (GPU discovery, planning, and multi-GPU heuristics)
//! - runtime/render-loop code that records explicit GPU submissions and present synchronization
//!
//! The crate exposes:
//! - [`core::bridge`] for MPS<->GMS frame planning
//! - [`graphics::multigpu::sync`] for portable explicit multi-GPU synchronization and UMA hooks
//! - [`graphics::vulkan_backend`] for the Linux-first raw Vulkan backend skeleton
//! - [`tlscript`] for the in-memory, zero-copy `.tlscript` frontend lexer/token layer

/// Canonical module id used by runtime version commands.
pub const MODULE_ID: &str = "tl-core";
/// Crate version resolved at compile time.
pub const MODULE_VERSION: &str = env!("CARGO_PKG_VERSION");

pub mod core;
pub mod graphics;
pub mod tlscript;

pub use core::bridge::{
    BridgeFrameId, BridgeFramePlan, BridgeGpuTaskKind, BridgeMpsSubmission, BridgeSubmitReceipt,
    BridgeTaskDescriptor, BridgeTaskRouting, MpsGmsBridge, MpsGmsBridgeConfig, MpsGmsBridgeMetrics,
};
pub use core::mgs_bridge::{
    MgsBridgeFrameId, MgsBridgeFramePlan, MgsBridgeMpsSubmission, MgsBridgeSubmissionId,
    MgsBridgeSubmitReceipt, MgsBridgeTaskDescriptor, MpsMgsBridge, MpsMgsBridgeConfig,
    MpsMgsBridgeMetrics,
};
pub use gms::{AdaptiveBufferDecision, AdaptiveFrameTelemetry};
pub use graphics::multigpu::sync::{
    ComposeBarrierState, GpuQueueLane, GpuSubmissionHandle, GpuSubmissionWaitStatus,
    GpuSubmissionWaiter, MultiGpuFrameSyncConfig, MultiGpuFrameSynchronizer, MultiGpuSyncSnapshot,
    SharedPlacementPolicy, SyncBackendHint, WgpuSubmissionWaiter,
};
pub use graphics::vulkan_backend::{
    FrameInstanceTransform, FrameLightRecord, FrameMaterialRecord, FrameSubmissionTelemetry,
    FrameTextureRecord, LinuxWindowSystemIntegration, PresentModePreference, RenderStateSnapshot,
    VulkanBackend, VulkanBackendConfig, VulkanBackendError, VulkanDeviceExtensionSupport,
    VulkanFrameExecutionTelemetry, VulkanMultiGpuCapabilities, VulkanMultiGpuConfig,
    VulkanMultiGpuFramePlan, VulkanNativeMultiGpuSupport, VulkanPhysicalDeviceProfile,
    VulkanQueueSelection, VulkanSnapshotSlotState,
};
pub use graphics::vulkan_physics_compute::{
    VulkanPhysicsComputeBackend, VulkanPhysicsComputeCapabilities, VulkanPhysicsComputeConfig,
    VulkanPhysicsDispatchPlan,
};
pub use tlscript::{
    annotate_typed_ir_with_parallel_hooks, lower_to_typed_ir, lower_to_typed_ir_with_config,
    AssignStmt, BinaryOp, Block, BoundsCheckEnforcement, BoundsCheckPolicy, CodegenExportEntry,
    Decorator, DecoratorArg, DecoratorKind, DecoratorValue, ExportAbiPolicy, Expr, ExprKind,
    ExprStmt, ExternalCallReturnHint, ForRangeStmt, FunctionDef, FunctionSemanticSummary,
    FunctionSignature, HostImportSignature, IfBranch, IfStmt, IrBlockId, IrBlockKind, IrCallee,
    IrConstValue, IrEffectMask, IrExecutionPolicy, IrExternalCallFlavor, IrFunctionId, IrInstKind,
    IrInstMeta, IrLocalId, IrLocalKind, IrParallelDomain, IrReduceKind, IrScheduleHint,
    IrSimdAnnotation, IrTempId, IrTerminator, IrValue, Item, LetStmt, LexError, LexErrorKind,
    Lexer, LoweringExternalSignature, Module, NetBindingHook, NetDecoratorHookConfig,
    NetDeliveryMode, NetFunctionHook, NetHookAnalyzer, NetHookError, NetHookErrorKind,
    NetHookOutcome, NetHookWarning, NetHookWarningKind, NetSyncMode, OwnershipLifetimePolicy,
    ParallelAdvisor, ParallelAdvisorConfig, ParallelAdvisorReport, ParallelAdvisorSuggestion,
    ParallelAdvisoryStatus, ParallelContractTemplate, ParallelContractTemplateSource,
    ParallelDispatchDecision, ParallelDispatchMode, ParallelDispatchPlanner,
    ParallelDispatchPlannerConfig, ParallelDispatchPlannerMetrics, ParallelExecutionPolicy,
    ParallelFallbackReason, ParallelFunctionAdvice, ParallelFunctionHook, ParallelHookAnalyzer,
    ParallelHookError, ParallelHookErrorKind, ParallelHookOutcome, ParallelHookWarning,
    ParallelHookWarningKind, ParallelReduceKind, ParallelRuntimeFallbackReason,
    ParallelScheduleHint, Param, ParseError, ParseErrorKind, Parser, PointerPolicy, RangeSpec,
    SemanticAnalyzer, SemanticConfig, SemanticError, SemanticErrorKind, SemanticOutcome,
    SemanticReport, SemanticSafetyPolicy, SemanticSafetySummary, SemanticType, SemanticWarning,
    SemanticWarningKind, Span, Stmt, Token, TokenKind, TypeAnnotation, TypeName,
    TypedIrArenaLayout, TypedIrArenaStats, TypedIrBlock, TypedIrExecutionMeta, TypedIrFunction,
    TypedIrFunctionMeta, TypedIrInst, TypedIrLocal, TypedIrLowerer, TypedIrLoweringConfig,
    TypedIrLoweringError, TypedIrLoweringErrorKind, TypedIrModule, TypedIrModuleMeta,
    TypedIrOptimizationHooks, TypedIrTemp, UnaryOp, WasmCodegenConfig, WasmCodegenError,
    WasmCodegenErrorKind, WasmCodegenOutcome, WasmCodegenOutput, WasmCodegenWarning,
    WasmCodegenWarningKind, WasmGenerator, WasmSandboxPolicy, WhileStmt,
};
