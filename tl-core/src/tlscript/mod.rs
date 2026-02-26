//! `.tlscript` frontend building blocks.
//!
//! This module hosts the zero-copy lexer/token layer for the Tileline script language.
//! It is intentionally RAM-only and accepts in-memory `&str` sources to avoid any disk I/O
//! coupling in the hot path.

pub mod ast;
pub mod codegen;
pub mod lexer;
pub mod lowering;
pub mod parser;
pub mod semantic;
pub mod token;
pub mod typed_ir;

pub use ast::{
    AssignStmt, BinaryOp, Block, Decorator, DecoratorKind, Expr, ExprKind, ExprStmt, ForRangeStmt,
    FunctionDef, IfBranch, IfStmt, Item, LetStmt, Module, Param, RangeSpec, Stmt, TypeAnnotation,
    TypeName, UnaryOp, WhileStmt,
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
pub use parser::{ParseError, ParseErrorKind, Parser};
pub use semantic::{
    BoundsCheckEnforcement, BoundsCheckPolicy, ExportAbiPolicy, FunctionSemanticSummary,
    FunctionSignature, OwnershipLifetimePolicy, PointerPolicy, SemanticAnalyzer, SemanticConfig,
    SemanticError, SemanticErrorKind, SemanticOutcome, SemanticReport, SemanticSafetyPolicy,
    SemanticSafetySummary, SemanticType, SemanticWarning, SemanticWarningKind, WasmSandboxPolicy,
};
pub use token::{Span, Token, TokenKind};
pub use typed_ir::{
    IrBlockId, IrBlockKind, IrCallee, IrConstValue, IrExternalCallFlavor, IrFunctionId, IrInstKind,
    IrInstMeta, IrLocalId, IrLocalKind, IrSimdAnnotation, IrTempId, IrTerminator, IrValue,
    TypedIrArenaLayout, TypedIrArenaStats, TypedIrBlock, TypedIrFunction, TypedIrFunctionMeta,
    TypedIrInst, TypedIrLocal, TypedIrModule, TypedIrModuleMeta, TypedIrOptimizationHooks,
    TypedIrTemp,
};
