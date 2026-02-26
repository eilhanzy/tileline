//! Typed intermediate representation (IR) for `.tlscript`.
//!
//! This layer sits between semantic validation and backend code generation (WASM, future native
//! backends). It provides a flattened, typed, block-based form with explicit temporaries (SSA-lite)
//! while still allowing mutable locals for script variables and loop state.
//!
//! Design goals:
//! - canonical basic-block control flow for branches/loops
//! - explicit temporaries for nested expression flattening
//! - vector-backed arenas for low overhead and cache-friendly traversal
//! - optimization metadata hooks (const-folding / SIMD annotation)

use super::ast::{BinaryOp, UnaryOp};
use super::semantic::SemanticType;
use super::token::Span;

/// Function identifier inside [`TypedIrModule`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct IrFunctionId(pub u32);

/// Basic-block identifier inside a function.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct IrBlockId(pub u32);

/// Local slot identifier inside a function.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct IrLocalId(pub u32);

/// Temporary SSA-lite value identifier inside a function.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct IrTempId(pub u32);

/// Storage strategy used for IR node allocation.
///
/// V1 uses vector-backed arenas to keep allocation overhead low while preserving stable indices.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TypedIrArenaLayout {
    /// `Vec<T>`-backed contiguous arenas with index newtypes.
    VecBacked,
}

/// Module-level optimization hook availability flags.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TypedIrOptimizationHooks {
    /// The IR carries enough metadata for a later constant-folding pass.
    pub const_folding_ready: bool,
    /// The IR carries enough metadata for a later SIMD annotation/rewrite pass.
    pub simd_annotation_ready: bool,
}

/// Aggregate IR arena statistics for diagnostics and tooling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct TypedIrArenaStats {
    /// Total function count.
    pub functions: usize,
    /// Total basic-block count.
    pub blocks: usize,
    /// Total local-slot count.
    pub locals: usize,
    /// Total temporary count.
    pub temps: usize,
    /// Total instruction count.
    pub instructions: usize,
}

/// Module-level metadata for the typed IR graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TypedIrModuleMeta {
    /// Arena storage layout.
    pub arena_layout: TypedIrArenaLayout,
    /// Optimization hook readiness.
    pub optimization_hooks: TypedIrOptimizationHooks,
    /// Aggregate node counts.
    pub arena_stats: TypedIrArenaStats,
}

/// A fully lowered, typed `.tlscript` compilation unit.
#[derive(Debug, Clone, PartialEq)]
pub struct TypedIrModule<'src> {
    /// Functions in source order.
    pub functions: Vec<TypedIrFunction<'src>>,
    /// Source span of the original AST module.
    pub span: Span,
    /// IR metadata and storage stats.
    pub meta: TypedIrModuleMeta,
}

impl<'src> TypedIrModule<'src> {
    /// Recompute aggregate module metadata from function arenas.
    pub fn recompute_metadata(&mut self) {
        let mut stats = TypedIrArenaStats {
            functions: self.functions.len(),
            ..TypedIrArenaStats::default()
        };
        for func in &self.functions {
            stats.blocks += func.blocks.len();
            stats.locals += func.locals.len();
            stats.temps += func.temps.len();
            stats.instructions += func
                .blocks
                .iter()
                .map(|b| b.instructions.len())
                .sum::<usize>();
        }
        self.meta.arena_stats = stats;
    }
}

/// Per-function IR optimization metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IrExecutionPolicy {
    /// Default execution mode: callable, but no verified parallel safety contract.
    Serial,
    /// Function has a validated parallel contract and may be partitioned across MPS tasks.
    ParallelSafe,
    /// Function must run on the main/game thread (UI, renderer state, etc).
    MainThreadOnly,
}

impl Default for IrExecutionPolicy {
    fn default() -> Self {
        Self::Serial
    }
}

/// Core scheduling hint for MPS task placement generated from script decorators.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IrScheduleHint {
    /// Let runtime/MPS choose based on phase and pressure.
    Auto,
    /// Prefer performance cores.
    Performance,
    /// Prefer efficient cores.
    Efficient,
}

impl Default for IrScheduleHint {
    fn default() -> Self {
        Self::Auto
    }
}

/// Deterministic merge/reduction strategy hint for parallel functions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IrReduceKind {
    Sum,
    Min,
    Max,
    BitOr,
    BitAnd,
}

/// Execution/scheduling metadata attached to a lowered function.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct TypedIrExecutionMeta {
    /// Execution policy derived from script decorators or defaults.
    pub policy: IrExecutionPolicy,
    /// Whether the function contract requests deterministic behavior across parallel partitions.
    pub deterministic: bool,
    /// MPS scheduling preference hint.
    pub schedule_hint: IrScheduleHint,
    /// Optional partition chunk hint (`None` -> runtime heuristic).
    pub chunk_hint: Option<u16>,
    /// Optional deterministic reduction kind for merged outputs.
    pub reduce: Option<IrReduceKind>,
    /// Number of declared read-effect domains in the parallel contract (for planner heuristics).
    pub read_effect_count: u16,
    /// Number of declared write-effect domains in the parallel contract (for planner heuristics).
    pub write_effect_count: u16,
}

/// Per-function IR optimization + execution metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct TypedIrFunctionMeta {
    /// Number of instructions that are candidates for constant folding.
    pub const_fold_candidate_insts: u32,
    /// Number of instructions marked as SIMD-friendly scalar arithmetic sites.
    pub simd_candidate_insts: u32,
    /// Execution/scheduling metadata used by runtime/MPS planning.
    pub execution: TypedIrExecutionMeta,
}

/// A single lowered function in block form.
#[derive(Debug, Clone, PartialEq)]
pub struct TypedIrFunction<'src> {
    /// Stable function id in the module arena.
    pub id: IrFunctionId,
    /// Script function name.
    pub name: &'src str,
    /// `true` if function is exported through the engine WASM ABI.
    pub exported: bool,
    /// Source span covering the function definition.
    pub span: Span,
    /// Return type resolved by semantic analysis.
    pub return_type: SemanticType,
    /// Parameter locals in ABI order.
    pub params: Vec<IrLocalId>,
    /// Vector-backed local-slot arena.
    pub locals: Vec<TypedIrLocal<'src>>,
    /// Vector-backed temporary arena.
    pub temps: Vec<TypedIrTemp>,
    /// Vector-backed basic-block arena.
    pub blocks: Vec<TypedIrBlock<'src>>,
    /// Entry block id.
    pub entry_block: IrBlockId,
    /// Per-function optimization metadata.
    pub meta: TypedIrFunctionMeta,
}

/// Origin/kind of a local slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IrLocalKind {
    /// ABI parameter slot.
    Param,
    /// User-declared `let` local.
    Let,
    /// `for ... in range(...)` loop binding.
    LoopBinding,
    /// Hidden loop `range.end` storage.
    HiddenLoopEnd,
    /// Hidden loop `range.step` storage.
    HiddenLoopStep,
}

/// Mutable local slot declaration.
#[derive(Debug, Clone, PartialEq)]
pub struct TypedIrLocal<'src> {
    /// Stable local id.
    pub id: IrLocalId,
    /// Optional script-visible name (hidden loop temporaries use `None`).
    pub name: Option<&'src str>,
    /// Static local type.
    pub ty: SemanticType,
    /// Local category.
    pub kind: IrLocalKind,
    /// Source span of declaration (if any).
    pub declared_span: Option<Span>,
    /// Whether writes are legal after initialization.
    pub mutable: bool,
}

/// Temporary SSA-lite value declaration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TypedIrTemp {
    /// Stable temp id.
    pub id: IrTempId,
    /// Static temp type.
    pub ty: SemanticType,
    /// Producing block.
    pub producer_block: IrBlockId,
    /// Source span of the producing expression.
    pub span: Option<Span>,
}

/// Canonical basic-block category.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IrBlockKind {
    Entry,
    Linear,
    IfCond,
    IfThen,
    IfElse,
    IfMerge,
    WhileCond,
    WhileBody,
    WhileExit,
    ForGuard,
    ForCond,
    ForPosCond,
    ForNegCond,
    ForBody,
    ForExit,
}

/// Basic block with typed instructions and a terminating control-flow edge.
#[derive(Debug, Clone, PartialEq)]
pub struct TypedIrBlock<'src> {
    /// Stable block id.
    pub id: IrBlockId,
    /// Block role for tooling/backends.
    pub kind: IrBlockKind,
    /// Optional source span associated with the block origin.
    pub span: Option<Span>,
    /// Flat instruction list (vector-backed arena segment).
    pub instructions: Vec<TypedIrInst<'src>>,
    /// Mandatory terminator.
    pub terminator: IrTerminator,
}

/// A typed value reference in IR.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IrValue {
    /// Value produced by a temporary instruction result.
    Temp(IrTempId),
}

/// Constant payload carried by `Const` instructions and const-fold metadata.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum IrConstValue<'src> {
    Int(i32),
    Float(f32),
    Bool(bool),
    /// Borrowed source slice; backend decides layout/encoding.
    Str(&'src str),
}

/// External call flavor for backend ABI specialization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IrExternalCallFlavor {
    /// Generic host/runtime callback.
    Generic,
    /// Host call returning a handle-like engine identifier.
    HandleAcquire,
    /// Host call releasing a previously acquired handle.
    HandleRelease,
}

/// Callee classification for direct calls in V1.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum IrCallee<'src> {
    /// Script-local function call.
    Internal(&'src str),
    /// External/host call (ABI-bound at backend/codegen stage).
    External {
        /// Script-visible callee name.
        name: &'src str,
        /// Flavor hint for backend/runtime ABI handling.
        flavor: IrExternalCallFlavor,
    },
}

/// SIMD annotation category for optimization passes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IrSimdAnnotation {
    /// No SIMD hint.
    None,
    /// Scalar arithmetic pattern that may be widened/vectorized later.
    ArithmeticLaneCandidate,
}

/// Per-instruction optimization metadata hooks.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IrInstMeta<'src> {
    /// `true` if a later pass can attempt constant folding on this instruction.
    pub const_folding_candidate: bool,
    /// Known compile-time constant value for this instruction result, if available.
    pub known_const_value: Option<IrConstValue<'src>>,
    /// SIMD annotation for future backend-specific rewrites.
    pub simd_annotation: IrSimdAnnotation,
}

impl<'src> Default for IrInstMeta<'src> {
    fn default() -> Self {
        Self {
            const_folding_candidate: false,
            known_const_value: None,
            simd_annotation: IrSimdAnnotation::None,
        }
    }
}

/// Flattened typed instruction.
#[derive(Debug, Clone, PartialEq)]
pub struct TypedIrInst<'src> {
    /// Destination temporary (if instruction produces a value).
    pub dest: Option<IrTempId>,
    /// Static result type (when `dest` is present) or operation type (`Unit` ops may omit).
    pub ty: Option<SemanticType>,
    /// Source span of the originating AST node.
    pub span: Option<Span>,
    /// Instruction opcode/payload.
    pub kind: IrInstKind<'src>,
    /// Optimization annotations for later passes.
    pub meta: IrInstMeta<'src>,
}

/// IR instruction opcode set (V1).
#[derive(Debug, Clone, PartialEq)]
pub enum IrInstKind<'src> {
    /// Constant literal materialization.
    Const(IrConstValue<'src>),
    /// Load from mutable local slot.
    LoadLocal { local: IrLocalId },
    /// Store into mutable local slot.
    StoreLocal { local: IrLocalId, value: IrValue },
    /// Unary operation (currently only arithmetic negation).
    Unary { op: UnaryOp, operand: IrValue },
    /// Binary operation on typed values.
    Binary {
        op: BinaryOp,
        left: IrValue,
        right: IrValue,
    },
    /// Direct call to script-local or external callee.
    Call {
        callee: IrCallee<'src>,
        args: Vec<IrValue>,
    },
}

/// Basic-block terminator.
#[derive(Debug, Clone, PartialEq)]
pub enum IrTerminator {
    /// Unconditional jump.
    Jump { target: IrBlockId },
    /// Conditional branch on a boolean (`i32`-compatible) value.
    Branch {
        cond: IrValue,
        then_target: IrBlockId,
        else_target: IrBlockId,
    },
    /// Function return. `None` means unit/void return.
    Return { value: Option<IrValue> },
}
