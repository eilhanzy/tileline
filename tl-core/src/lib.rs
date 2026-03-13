//! Tileline core orchestration crate.
//!
//! `tl-core` is the integration boundary between:
//! - `mps` (CPU scheduling / WASM execution)
//! - `gms` (GPU discovery, planning, and multi-GPU heuristics)
//! - runtime/render-loop code that records real `wgpu` submissions and present synchronization
//!
//! The crate exposes:
//! - [`core::bridge`] for MPS<->GMS frame planning
//! - [`graphics::multigpu::sync`] for portable explicit multi-GPU synchronization and UMA hooks
//! - [`tlscript`] for the in-memory, zero-copy `.tlscript` frontend lexer/token layer

pub mod core;
pub mod graphics;
pub mod tlscript;

pub use core::bridge::{
    BridgeFrameId, BridgeFramePlan, BridgeGpuTaskKind, BridgeMpsSubmission, BridgeSubmitReceipt,
    BridgeTaskDescriptor, BridgeTaskRouting, MpsGmsBridge, MpsGmsBridgeConfig, MpsGmsBridgeMetrics,
};
pub use gms::{AdaptiveBufferDecision, AdaptiveFrameTelemetry};
pub use graphics::multigpu::sync::{
    ComposeBarrierState, GpuQueueLane, MultiGpuFrameSyncConfig, MultiGpuFrameSynchronizer,
    MultiGpuSyncSnapshot, SharedPlacementPolicy, SyncBackendHint,
};
pub use tlscript::{
    annotate_typed_ir_with_parallel_hooks, lower_to_typed_ir, lower_to_typed_ir_with_config,
    AssignStmt, BinaryOp, Block, BoundsCheckEnforcement, BoundsCheckPolicy, CodegenExportEntry,
    Decorator, DecoratorArg, DecoratorKind, DecoratorValue, ExportAbiPolicy, Expr, ExprKind,
    ExprStmt, ForRangeStmt, FunctionDef, FunctionSemanticSummary, FunctionSignature,
    HostImportSignature, IfBranch, IfStmt, IrBlockId, IrBlockKind, IrCallee, IrConstValue,
    IrExecutionPolicy, IrExternalCallFlavor, IrFunctionId, IrInstKind, IrInstMeta, IrLocalId,
    IrLocalKind, IrReduceKind, IrScheduleHint, IrSimdAnnotation, IrTempId, IrTerminator, IrValue,
    Item, LetStmt, LexError, LexErrorKind, Lexer, LoweringExternalSignature, Module,
    NetBindingHook, NetDecoratorHookConfig, NetDeliveryMode, NetFunctionHook, NetHookAnalyzer,
    NetHookError, NetHookErrorKind, NetHookOutcome, NetHookWarning, NetHookWarningKind,
    NetSyncMode, OwnershipLifetimePolicy, ParallelAdvisor, ParallelAdvisorConfig,
    ParallelAdvisorReport, ParallelAdvisorSuggestion, ParallelAdvisoryStatus,
    ParallelContractTemplate, ParallelContractTemplateSource, ParallelDispatchDecision,
    ParallelDispatchMode, ParallelDispatchPlanner, ParallelDispatchPlannerConfig,
    ParallelDispatchPlannerMetrics, ParallelExecutionPolicy, ParallelFallbackReason,
    ParallelFunctionAdvice, ParallelFunctionHook, ParallelHookAnalyzer, ParallelHookError,
    ParallelHookErrorKind, ParallelHookOutcome, ParallelHookWarning, ParallelHookWarningKind,
    ParallelReduceKind, ParallelRuntimeFallbackReason, ParallelScheduleHint, Param, ParseError,
    ParseErrorKind, Parser, PointerPolicy, RangeSpec, SemanticAnalyzer, SemanticConfig,
    SemanticError, SemanticErrorKind, SemanticOutcome, SemanticReport, SemanticSafetyPolicy,
    SemanticSafetySummary, SemanticType, SemanticWarning, SemanticWarningKind, Span, Stmt, Token,
    TokenKind, TypeAnnotation, TypeName, TypedIrArenaLayout, TypedIrArenaStats, TypedIrBlock,
    TypedIrExecutionMeta, TypedIrFunction, TypedIrFunctionMeta, TypedIrInst, TypedIrLocal,
    TypedIrLowerer, TypedIrLoweringConfig, TypedIrLoweringError, TypedIrLoweringErrorKind,
    TypedIrModule, TypedIrModuleMeta, TypedIrOptimizationHooks, TypedIrTemp, UnaryOp,
    WasmCodegenConfig, WasmCodegenError, WasmCodegenErrorKind, WasmCodegenOutcome,
    WasmCodegenOutput, WasmCodegenWarning, WasmCodegenWarningKind, WasmGenerator,
    WasmSandboxPolicy, WhileStmt,
};
