//! AST definitions for `.tlscript` (V1 parser frontend).
//!
//! The AST borrows identifier and literal slices from the original source where possible to avoid
//! unnecessary string allocations. This keeps the frontend compatible with the engine's
//! low-latency, memory-resident execution pipeline.

use super::token::Span;

/// Parsed `.tlscript` module (single source unit).
#[derive(Debug, Clone, PartialEq)]
pub struct Module<'src> {
    /// Top-level items in source order.
    pub items: Vec<Item<'src>>,
    /// Span covering the parsed module.
    pub span: Span,
}

/// Top-level item.
#[derive(Debug, Clone, PartialEq)]
pub enum Item<'src> {
    /// Function definition item.
    Function(FunctionDef<'src>),
}

impl<'src> Item<'src> {
    /// Span of the item.
    pub fn span(&self) -> Span {
        match self {
            Self::Function(f) => f.span,
        }
    }
}

/// Function decorator.
#[derive(Debug, Clone, PartialEq)]
pub struct Decorator<'src> {
    /// Decorator kind (`@export` or generic named decorator).
    pub kind: DecoratorKind<'src>,
    /// Optional decorator arguments (`@net(sync=\"on_change\", unreliable)`).
    pub args: Vec<DecoratorArg<'src>>,
    /// Decorator token span.
    pub span: Span,
}

/// Decorator classification.
#[derive(Debug, Clone, PartialEq)]
pub enum DecoratorKind<'src> {
    /// Engine ABI export marker (`@export`).
    Export,
    /// Other named decorator (validated later by semantic pass).
    Named(&'src str),
}

/// Decorator argument (`flag` or `key=value`).
#[derive(Debug, Clone, PartialEq)]
pub enum DecoratorArg<'src> {
    /// Positional flag (`@net(unreliable)`).
    Flag {
        /// Flag name.
        name: &'src str,
        /// Source span.
        span: Span,
    },
    /// Key-value pair (`@net(sync=\"on_change\")`).
    KeyValue {
        /// Key name.
        key: &'src str,
        /// Parsed literal/identifier value.
        value: DecoratorValue<'src>,
        /// Source span of the full argument.
        span: Span,
    },
}

impl<'src> DecoratorArg<'src> {
    /// Argument span.
    pub fn span(&self) -> Span {
        match self {
            Self::Flag { span, .. } | Self::KeyValue { span, .. } => *span,
        }
    }
}

/// Decorator argument value (zero-copy where possible).
#[derive(Debug, Clone, PartialEq)]
pub enum DecoratorValue<'src> {
    Identifier(&'src str),
    String(&'src str),
    Bool(bool),
    Integer(&'src str),
    Float(&'src str),
}

/// Function definition.
#[derive(Debug, Clone, PartialEq)]
pub struct FunctionDef<'src> {
    /// Attached decorators in source order.
    pub decorators: Vec<Decorator<'src>>,
    /// Function name.
    pub name: &'src str,
    /// Name token span.
    pub name_span: Span,
    /// Parameters.
    pub params: Vec<Param<'src>>,
    /// Optional return type annotation.
    pub return_type: Option<TypeAnnotation>,
    /// Function body block.
    pub body: Block<'src>,
    /// Span covering the full function definition (including decorators).
    pub span: Span,
}

/// Function parameter.
#[derive(Debug, Clone, PartialEq)]
pub struct Param<'src> {
    /// Parameter name.
    pub name: &'src str,
    /// Parameter name span.
    pub name_span: Span,
    /// Optional type annotation.
    pub ty: Option<TypeAnnotation>,
    /// Full parameter span.
    pub span: Span,
}

/// Block of statements.
#[derive(Debug, Clone, PartialEq)]
pub struct Block<'src> {
    /// Statements in source order.
    pub statements: Vec<Stmt<'src>>,
    /// Span covering the block contents and indentation boundary.
    pub span: Span,
}

/// Statement node.
#[derive(Debug, Clone, PartialEq)]
pub enum Stmt<'src> {
    Let(LetStmt<'src>),
    Assign(AssignStmt<'src>),
    If(IfStmt<'src>),
    While(WhileStmt<'src>),
    ForRange(ForRangeStmt<'src>),
    Expr(ExprStmt<'src>),
}

impl<'src> Stmt<'src> {
    /// Span of the statement.
    pub fn span(&self) -> Span {
        match self {
            Self::Let(v) => v.span,
            Self::Assign(v) => v.span,
            Self::If(v) => v.span,
            Self::While(v) => v.span,
            Self::ForRange(v) => v.span,
            Self::Expr(v) => v.span,
        }
    }
}

/// `let` declaration statement.
#[derive(Debug, Clone, PartialEq)]
pub struct LetStmt<'src> {
    /// Statement decorators (currently used for `@net(...)` replication hooks).
    pub decorators: Vec<Decorator<'src>>,
    /// Binding name.
    pub name: &'src str,
    /// Binding name span.
    pub name_span: Span,
    /// Optional explicit type annotation.
    pub ty: Option<TypeAnnotation>,
    /// Initializer expression.
    pub value: Expr<'src>,
    /// Full statement span.
    pub span: Span,
}

/// Assignment statement (V1 target restricted to identifiers).
#[derive(Debug, Clone, PartialEq)]
pub struct AssignStmt<'src> {
    /// Assignment target name.
    pub target: &'src str,
    /// Target span.
    pub target_span: Span,
    /// Assigned value expression.
    pub value: Expr<'src>,
    /// Full statement span.
    pub span: Span,
}

/// Expression statement.
#[derive(Debug, Clone, PartialEq)]
pub struct ExprStmt<'src> {
    /// Expression body.
    pub expr: Expr<'src>,
    /// Full statement span.
    pub span: Span,
}

/// `if / elif / else` statement.
#[derive(Debug, Clone, PartialEq)]
pub struct IfStmt<'src> {
    /// Ordered `if` + `elif` branches.
    pub branches: Vec<IfBranch<'src>>,
    /// Optional `else` block.
    pub else_block: Option<Block<'src>>,
    /// Full statement span.
    pub span: Span,
}

/// Single conditional branch in an `if` chain.
#[derive(Debug, Clone, PartialEq)]
pub struct IfBranch<'src> {
    /// Branch condition.
    pub condition: Expr<'src>,
    /// Branch body.
    pub body: Block<'src>,
    /// Full branch span.
    pub span: Span,
}

/// `while` loop statement.
#[derive(Debug, Clone, PartialEq)]
pub struct WhileStmt<'src> {
    /// Loop condition.
    pub condition: Expr<'src>,
    /// Loop body.
    pub body: Block<'src>,
    /// Full statement span.
    pub span: Span,
}

/// `for <ident> in range(...)` loop statement.
#[derive(Debug, Clone, PartialEq)]
pub struct ForRangeStmt<'src> {
    /// Loop variable name.
    pub binding: &'src str,
    /// Loop variable span.
    pub binding_span: Span,
    /// Parsed `range(...)` call arguments.
    pub range: RangeSpec<'src>,
    /// Loop body.
    pub body: Block<'src>,
    /// Full statement span.
    pub span: Span,
}

/// Parsed `range(...)` specification.
#[derive(Debug, Clone, PartialEq)]
pub struct RangeSpec<'src> {
    /// `range` callee token span.
    pub callee_span: Span,
    /// Raw arguments (`1..=3` arguments accepted by parser).
    pub args: Vec<Expr<'src>>,
    /// Full span of the `range(...)` expression.
    pub span: Span,
}

/// Type annotation wrapper.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TypeAnnotation {
    /// Primitive type identifier.
    pub kind: TypeName,
    /// Source span of the type token.
    pub span: Span,
}

/// V1 primitive type names.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TypeName {
    Int,
    Float,
    Bool,
    Str,
}

/// Expression node wrapper.
#[derive(Debug, Clone, PartialEq)]
pub struct Expr<'src> {
    /// Expression kind.
    pub kind: ExprKind<'src>,
    /// Source span.
    pub span: Span,
}

impl<'src> Expr<'src> {
    /// Construct an expression node.
    pub fn new(kind: ExprKind<'src>, span: Span) -> Self {
        Self { kind, span }
    }
}

/// Expression variants.
#[derive(Debug, Clone, PartialEq)]
pub enum ExprKind<'src> {
    Identifier(&'src str),
    IntegerLiteral(&'src str),
    FloatLiteral(&'src str),
    BoolLiteral(bool),
    StringLiteral(&'src str),
    Unary {
        op: UnaryOp,
        expr: Box<Expr<'src>>,
    },
    Binary {
        op: BinaryOp,
        left: Box<Expr<'src>>,
        right: Box<Expr<'src>>,
    },
    Call {
        callee: Box<Expr<'src>>,
        args: Vec<Expr<'src>>,
    },
    Grouping(Box<Expr<'src>>),
}

/// Unary operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    /// Arithmetic negation.
    Neg,
}

/// Binary operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    EqEq,
    NotEq,
    Lt,
    LtEq,
    Gt,
    GtEq,
    AndAnd,
    OrOr,
}
