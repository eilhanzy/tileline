//! `.tlscript` frontend building blocks.
//!
//! This module hosts the zero-copy lexer/token layer for the Tileline script language.
//! It is intentionally RAM-only and accepts in-memory `&str` sources to avoid any disk I/O
//! coupling in the hot path.

pub mod ast;
pub mod codegen;
pub mod lexer;
pub mod lowering;
pub mod net_hook;
pub mod parallel_advisor;
pub mod parallel_hook;
pub mod parallel_runtime;
pub mod parser;
pub mod semantic;
pub mod token;
pub mod typed_ir;

pub use ast::{
    AssignStmt, BinaryOp, Block, Decorator, DecoratorArg, DecoratorKind, DecoratorValue, Expr,
    ExprKind, ExprStmt, ForRangeStmt, FunctionDef, IfBranch, IfStmt, Item, LetStmt, Module, Param,
    RangeSpec, Stmt, TypeAnnotation, TypeName, UnaryOp, WhileStmt,
};
pub use codegen::{
    CodegenExportEntry, HostImportSignature, WasmCodegenConfig, WasmCodegenError,
    WasmCodegenErrorKind, WasmCodegenOutcome, WasmCodegenOutput, WasmCodegenWarning,
    WasmCodegenWarningKind, WasmGenerator,
};
pub use lexer::{LexError, LexErrorKind, Lexer};
pub use lowering::{
    lower_to_typed_ir, lower_to_typed_ir_with_config, LoweringExternalSignature, TypedIrLowerer,
    TypedIrLoweringConfig, TypedIrLoweringError, TypedIrLoweringErrorKind,
};
pub use net_hook::{
    NetBindingHook, NetDecoratorConfig as NetDecoratorHookConfig, NetDeliveryMode, NetFunctionHook,
    NetHookAnalyzer, NetHookError, NetHookErrorKind, NetHookOutcome, NetHookWarning,
    NetHookWarningKind, NetSyncMode,
};
pub use parallel_advisor::{
    ParallelAdvisor, ParallelAdvisorConfig, ParallelAdvisorReport, ParallelAdvisorSuggestion,
    ParallelAdvisoryStatus, ParallelContractTemplate, ParallelContractTemplateSource,
    ParallelFallbackReason, ParallelFunctionAdvice,
};
pub use parallel_hook::{
    annotate_typed_ir_with_parallel_hooks, ParallelExecutionPolicy, ParallelFunctionHook,
    ParallelHookAnalyzer, ParallelHookError, ParallelHookErrorKind, ParallelHookOutcome,
    ParallelHookWarning, ParallelHookWarningKind, ParallelReduceKind, ParallelScheduleHint,
};
pub use parallel_runtime::{
    ParallelDispatchDecision, ParallelDispatchMode, ParallelDispatchPlanner,
    ParallelDispatchPlannerConfig, ParallelDispatchPlannerMetrics, ParallelRuntimeFallbackReason,
};
pub use parser::{ParseError, ParseErrorKind, Parser};
pub use semantic::{
    BoundsCheckEnforcement, BoundsCheckPolicy, ExportAbiPolicy, FunctionSemanticSummary,
    FunctionSignature, OwnershipLifetimePolicy, PointerPolicy, SemanticAnalyzer, SemanticConfig,
    SemanticError, SemanticErrorKind, SemanticOutcome, SemanticReport, SemanticSafetyPolicy,
    SemanticSafetySummary, SemanticType, SemanticWarning, SemanticWarningKind, WasmSandboxPolicy,
};
pub use token::{Span, Token, TokenKind};
pub use typed_ir::{
    IrBlockId, IrBlockKind, IrCallee, IrConstValue, IrEffectMask, IrExecutionPolicy,
    IrExternalCallFlavor, IrFunctionId, IrInstKind, IrInstMeta, IrLocalId, IrLocalKind,
    IrParallelDomain, IrReduceKind, IrScheduleHint, IrSimdAnnotation, IrTempId, IrTerminator,
    IrValue, TypedIrArenaLayout, TypedIrArenaStats, TypedIrBlock, TypedIrExecutionMeta,
    TypedIrFunction, TypedIrFunctionMeta, TypedIrInst, TypedIrLocal, TypedIrModule,
    TypedIrModuleMeta, TypedIrOptimizationHooks, TypedIrTemp,
};
