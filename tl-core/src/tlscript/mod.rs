//! `.tlscript` frontend building blocks.
//!
//! This module hosts the zero-copy lexer/token layer for the Tileline script language.
//! It is intentionally RAM-only and accepts in-memory `&str` sources to avoid any disk I/O
//! coupling in the hot path.

pub mod ast;
pub mod lexer;
pub mod parser;
pub mod semantic;
pub mod token;

pub use ast::{
    AssignStmt, BinaryOp, Block, Decorator, DecoratorKind, Expr, ExprKind, ExprStmt, ForRangeStmt,
    FunctionDef, IfBranch, IfStmt, Item, LetStmt, Module, Param, RangeSpec, Stmt, TypeAnnotation,
    TypeName, UnaryOp, WhileStmt,
};
pub use lexer::{LexError, LexErrorKind, Lexer};
pub use parser::{ParseError, ParseErrorKind, Parser};
pub use semantic::{
    BoundsCheckEnforcement, BoundsCheckPolicy, ExportAbiPolicy, FunctionSemanticSummary,
    FunctionSignature, OwnershipLifetimePolicy, PointerPolicy, SemanticAnalyzer, SemanticConfig,
    SemanticError, SemanticErrorKind, SemanticOutcome, SemanticReport, SemanticSafetyPolicy,
    SemanticSafetySummary, SemanticType, SemanticWarning, SemanticWarningKind, WasmSandboxPolicy,
};
pub use token::{Span, Token, TokenKind};
