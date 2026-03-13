//! AST + semantic-report lowering into typed block IR.
//!
//! This module converts semantically validated `.tlscript` AST into [`super::typed_ir`] so backend
//! code generation (WASM now, native/JIT later) can operate on a flattened, canonical CFG without
//! re-implementing frontend-specific tree walking.
//!
//! The lowering pass uses vector-backed arenas (functions/blocks/locals/temps/instructions) for a
//! cache-friendly, low-overhead representation and attaches optimization metadata hooks for future
//! passes (`const_folding`, `simd_annotation`).

use std::collections::{HashMap, HashSet};
use std::fmt;

use paradoxpe::abi::{
    HOST_CALLS_HANDLE_ACQUIRE, HOST_CALLS_HANDLE_RELEASE, HOST_CALL_CONTACT_COUNT,
    HOST_CALL_STEP_WORLD,
};

use super::ast::*;
use super::semantic::{FunctionSignature, SemanticReport, SemanticType};
use super::token::Span;
use super::typed_ir::*;

/// External call return signature used by lowering when semantic signatures do not cover a host call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoweringExternalSignature {
    /// Script-visible callee name.
    pub name: String,
    /// Static return type used by typed IR.
    pub result: SemanticType,
}

/// Lowering configuration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TypedIrLoweringConfig {
    /// Attach const-folding metadata to relevant instructions.
    pub annotate_const_folding_candidates: bool,
    /// Attach SIMD candidate metadata to scalar arithmetic instructions.
    pub annotate_simd_candidates: bool,
    /// Optional exact return-type overrides for external calls.
    pub external_function_signatures: Vec<LoweringExternalSignature>,
    /// Names treated as handle-acquiring external calls.
    pub handle_acquire_call_allowlist: Vec<String>,
    /// Names treated as handle-releasing external calls.
    pub handle_release_call_allowlist: Vec<String>,
}

impl Default for TypedIrLoweringConfig {
    fn default() -> Self {
        Self {
            annotate_const_folding_candidates: true,
            annotate_simd_candidates: true,
            external_function_signatures: vec![
                LoweringExternalSignature {
                    name: HOST_CALL_CONTACT_COUNT.to_string(),
                    result: SemanticType::Int,
                },
                LoweringExternalSignature {
                    name: HOST_CALL_STEP_WORLD.to_string(),
                    result: SemanticType::Int,
                },
            ],
            handle_acquire_call_allowlist: {
                let mut names = vec![
                    "spawn_sprite".to_string(),
                    "create_sprite".to_string(),
                    "alloc_handle".to_string(),
                ];
                names.extend(
                    HOST_CALLS_HANDLE_ACQUIRE
                        .iter()
                        .map(|name| (*name).to_string()),
                );
                names
            },
            handle_release_call_allowlist: {
                let mut names = vec![
                    "release_handle".to_string(),
                    "destroy_handle".to_string(),
                    "free_handle".to_string(),
                ];
                names.extend(
                    HOST_CALLS_HANDLE_RELEASE
                        .iter()
                        .map(|name| (*name).to_string()),
                );
                names.sort();
                names.dedup();
                names
            },
        }
    }
}

/// Lowering error category.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TypedIrLoweringErrorKind {
    /// A parsed function has no matching semantic signature.
    MissingSemanticSignature,
    /// A function parameter has no resolved semantic type.
    MissingParameterType,
    /// Lowering expected a value-producing expression but got unit.
    ExpectedValueExpression,
    /// Local binding lookup failed (semantic/report mismatch or invalid AST after semantic pass).
    UnknownLocalBinding,
    /// Unsupported call target shape (V1 only supports direct identifier calls).
    UnsupportedCallTarget,
    /// Type mismatch found during lowering-time type reconstruction.
    TypeMismatch,
    /// Invalid `range(...)` arity.
    InvalidRangeArity,
    /// Invalid `range(...)` argument type.
    InvalidRangeArgType,
    /// Internal CFG/invariant error.
    InternalInvariant,
    /// Unsupported construct in V1 lowering.
    UnsupportedConstruct,
    /// Invalid numeric literal for typed constant lowering.
    InvalidLiteral,
}

/// Lowering error (non-panicking, engine-safe diagnostic payload).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TypedIrLoweringError {
    /// Error category.
    pub kind: TypedIrLoweringErrorKind,
    /// Source span, if available.
    pub span: Option<Span>,
    /// Human-readable detail for engine/editor logs.
    pub detail: String,
}

impl TypedIrLoweringError {
    fn new(kind: TypedIrLoweringErrorKind, span: Option<Span>, detail: impl Into<String>) -> Self {
        Self {
            kind,
            span,
            detail: detail.into(),
        }
    }
}

impl fmt::Display for TypedIrLoweringError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(span) = self.span {
            write!(
                f,
                "{:?} at line {}, column {}: {}",
                self.kind, span.line, span.column, self.detail
            )
        } else {
            write!(f, "{:?}: {}", self.kind, self.detail)
        }
    }
}

impl std::error::Error for TypedIrLoweringError {}

/// `.tlscript` AST -> typed IR lowering pass.
pub struct TypedIrLowerer {
    config: TypedIrLoweringConfig,
    external_returns: HashMap<String, SemanticType>,
    handle_acquire_calls: HashSet<String>,
    handle_release_calls: HashSet<String>,
}

impl TypedIrLowerer {
    /// Construct a lowering pass with the given configuration.
    pub fn new(config: TypedIrLoweringConfig) -> Self {
        let external_returns = config
            .external_function_signatures
            .iter()
            .map(|s| (s.name.clone(), s.result))
            .collect();
        let handle_acquire_calls = config
            .handle_acquire_call_allowlist
            .iter()
            .cloned()
            .collect::<HashSet<_>>();
        let handle_release_calls = config
            .handle_release_call_allowlist
            .iter()
            .cloned()
            .collect::<HashSet<_>>();
        Self {
            config,
            external_returns,
            handle_acquire_calls,
            handle_release_calls,
        }
    }

    /// Lower a semantically validated AST module into typed IR.
    pub fn lower_module<'src>(
        &self,
        module: &Module<'src>,
        semantic: &SemanticReport<'src>,
    ) -> Result<TypedIrModule<'src>, TypedIrLoweringError> {
        lower_to_typed_ir_with_config(module, semantic, self.config.clone())
    }
}

impl Default for TypedIrLowerer {
    fn default() -> Self {
        Self::new(TypedIrLoweringConfig::default())
    }
}

/// Convenience API for one-shot lowering.
pub fn lower_to_typed_ir<'src>(
    module: &Module<'src>,
    semantic: &SemanticReport<'src>,
) -> Result<TypedIrModule<'src>, TypedIrLoweringError> {
    lower_to_typed_ir_with_config(module, semantic, TypedIrLoweringConfig::default())
}

#[derive(Debug, Clone, Copy)]
struct LoweredExpr<'src> {
    ty: SemanticType,
    value: Option<IrValue>,
    known_const: Option<IrConstValue<'src>>,
}

impl<'src> LoweredExpr<'src> {
    fn require_value(
        self,
        span: Span,
    ) -> Result<(IrValue, SemanticType, Option<IrConstValue<'src>>), TypedIrLoweringError> {
        let Some(value) = self.value else {
            return Err(TypedIrLoweringError::new(
                TypedIrLoweringErrorKind::ExpectedValueExpression,
                Some(span),
                "expression does not produce a value",
            ));
        };
        Ok((value, self.ty, self.known_const))
    }
}

#[derive(Debug)]
struct BlockBuilder<'src> {
    kind: IrBlockKind,
    span: Option<Span>,
    instructions: Vec<TypedIrInst<'src>>,
    terminator: Option<IrTerminator>,
}

struct FunctionLoweringCx<'ctx, 'ast, 'src> {
    lowerer: &'ctx TypedIrLowerer,
    function_id: IrFunctionId,
    func: &'ast FunctionDef<'src>,
    sig: &'ctx FunctionSignature<'src>,
    locals: Vec<TypedIrLocal<'src>>,
    temps: Vec<TypedIrTemp>,
    blocks: Vec<BlockBuilder<'src>>,
    params: Vec<IrLocalId>,
    entry_block: IrBlockId,
    current_block: IrBlockId,
    scopes: Vec<HashMap<&'src str, IrLocalId>>,
    func_meta: TypedIrFunctionMeta,
    local_types: Vec<SemanticType>,
    temp_types: Vec<SemanticType>,
    sig_map: &'ctx HashMap<&'src str, &'ctx FunctionSignature<'src>>,
}

impl<'ctx, 'ast, 'src> FunctionLoweringCx<'ctx, 'ast, 'src> {
    fn new(
        lowerer: &'ctx TypedIrLowerer,
        id: IrFunctionId,
        func: &'ast FunctionDef<'src>,
        sig: &'ctx FunctionSignature<'src>,
        sig_map: &'ctx HashMap<&'src str, &'ctx FunctionSignature<'src>>,
    ) -> Result<Self, TypedIrLoweringError> {
        let mut scopes = vec![HashMap::new()];
        let mut locals = Vec::new();
        let mut local_types = Vec::new();
        let mut params = Vec::with_capacity(func.params.len());

        for (idx, param) in func.params.iter().enumerate() {
            let ty = sig.params.get(idx).copied().flatten().ok_or_else(|| {
                TypedIrLoweringError::new(
                    TypedIrLoweringErrorKind::MissingParameterType,
                    Some(param.name_span),
                    format!("missing semantic parameter type for `{}`", param.name),
                )
            })?;
            let local_id = IrLocalId(locals.len() as u32);
            locals.push(TypedIrLocal {
                id: local_id,
                name: Some(param.name),
                ty,
                kind: IrLocalKind::Param,
                declared_span: Some(param.span),
                mutable: false,
            });
            local_types.push(ty);
            scopes[0].insert(param.name, local_id);
            params.push(local_id);
        }

        let entry_block = IrBlockId(0);
        let blocks = vec![BlockBuilder {
            kind: IrBlockKind::Entry,
            span: Some(func.body.span),
            instructions: Vec::new(),
            terminator: None,
        }];

        Ok(Self {
            lowerer,
            function_id: id,
            func,
            sig,
            locals,
            temps: Vec::new(),
            blocks,
            params,
            entry_block,
            current_block: entry_block,
            scopes,
            func_meta: TypedIrFunctionMeta::default(),
            local_types,
            temp_types: Vec::new(),
            sig_map,
        })
    }

    fn lower_function_body(&mut self) -> Result<(), TypedIrLoweringError> {
        self.lower_ast_block(&self.func.body)?;
        if !self.current_block_is_terminated()? {
            let ret = if self.sig.return_type == SemanticType::Unit {
                None
            } else {
                let v = self.emit_default_value(self.sig.return_type, self.func.span)?;
                Some(v)
            };
            self.set_terminator(IrTerminator::Return { value: ret })?;
        }
        Ok(())
    }

    fn finish(self) -> Result<TypedIrFunction<'src>, TypedIrLoweringError> {
        let mut blocks = Vec::with_capacity(self.blocks.len());
        for (idx, b) in self.blocks.into_iter().enumerate() {
            let term = b.terminator.ok_or_else(|| {
                TypedIrLoweringError::new(
                    TypedIrLoweringErrorKind::InternalInvariant,
                    b.span,
                    format!("basic block {} missing terminator", idx),
                )
            })?;
            blocks.push(TypedIrBlock {
                id: IrBlockId(idx as u32),
                kind: b.kind,
                span: b.span,
                instructions: b.instructions,
                terminator: term,
            });
        }

        Ok(TypedIrFunction {
            id: self.function_id,
            name: self.func.name,
            exported: self.sig.exported,
            span: self.func.span,
            return_type: self.sig.return_type,
            params: self.params,
            locals: self.locals,
            temps: self.temps,
            blocks,
            entry_block: self.entry_block,
            meta: self.func_meta,
        })
    }

    fn lower_ast_block(&mut self, block: &Block<'src>) -> Result<(), TypedIrLoweringError> {
        self.push_scope();
        for stmt in &block.statements {
            self.lower_stmt(stmt)?;
            if self.current_block_is_terminated()? {
                break;
            }
        }
        self.pop_scope();
        Ok(())
    }

    fn lower_stmt(&mut self, stmt: &Stmt<'src>) -> Result<(), TypedIrLoweringError> {
        match stmt {
            Stmt::Let(s) => self.lower_let_stmt(s),
            Stmt::Assign(s) => self.lower_assign_stmt(s),
            Stmt::If(s) => self.lower_if_stmt(s),
            Stmt::While(s) => self.lower_while_stmt(s),
            Stmt::ForRange(s) => self.lower_for_range_stmt(s),
            Stmt::Expr(s) => {
                let _ = self.lower_expr_any(&s.expr)?;
                Ok(())
            }
        }
    }

    fn lower_let_stmt(&mut self, stmt: &LetStmt<'src>) -> Result<(), TypedIrLoweringError> {
        let lowered = self.lower_expr_any(&stmt.value)?;
        let (value, inferred_ty, _) = lowered.require_value(stmt.value.span)?;
        let annotated_ty = stmt.ty.map(|t| map_type_name(t.kind));
        let ty = annotated_ty.unwrap_or(inferred_ty);
        if ty != inferred_ty {
            return Err(TypedIrLoweringError::new(
                TypedIrLoweringErrorKind::TypeMismatch,
                Some(stmt.value.span),
                format!(
                    "let type mismatch: expected {:?}, found {:?}",
                    ty, inferred_ty
                ),
            ));
        }

        let local = self.alloc_local(
            Some(stmt.name),
            ty,
            IrLocalKind::Let,
            Some(stmt.name_span),
            true,
        );
        self.bind_local(stmt.name, stmt.name_span, local)?;
        self.emit_store_local(local, value, Some(stmt.span));
        Ok(())
    }

    fn lower_assign_stmt(&mut self, stmt: &AssignStmt<'src>) -> Result<(), TypedIrLoweringError> {
        let local = self.resolve_local(stmt.target).ok_or_else(|| {
            TypedIrLoweringError::new(
                TypedIrLoweringErrorKind::UnknownLocalBinding,
                Some(stmt.target_span),
                format!("unknown local `{}`", stmt.target),
            )
        })?;
        let lowered = self.lower_expr_any(&stmt.value)?;
        let (value, rhs_ty, _) = lowered.require_value(stmt.value.span)?;
        let lhs_ty = self.local_type(local)?;
        if lhs_ty != rhs_ty {
            return Err(TypedIrLoweringError::new(
                TypedIrLoweringErrorKind::TypeMismatch,
                Some(stmt.value.span),
                format!("assignment type mismatch: {:?} vs {:?}", lhs_ty, rhs_ty),
            ));
        }
        self.emit_store_local(local, value, Some(stmt.span));
        Ok(())
    }

    fn lower_if_stmt(&mut self, stmt: &IfStmt<'src>) -> Result<(), TypedIrLoweringError> {
        if stmt.branches.is_empty() {
            return Ok(());
        }

        let merge_bb = self.new_block(IrBlockKind::IfMerge, Some(stmt.span));
        let mut cond_bb = self.current_block;

        for (idx, branch) in stmt.branches.iter().enumerate() {
            self.switch_to(cond_bb)?;
            let cond_val = self.lower_expr_bool_value(&branch.condition)?;
            let then_bb = self.new_block(IrBlockKind::IfThen, Some(branch.body.span));
            let is_last = idx + 1 == stmt.branches.len();
            let next_else_bb = if is_last {
                if let Some(else_block) = &stmt.else_block {
                    self.new_block(IrBlockKind::IfElse, Some(else_block.span))
                } else {
                    merge_bb
                }
            } else {
                self.new_block(IrBlockKind::IfCond, Some(stmt.branches[idx + 1].span))
            };
            self.set_terminator(IrTerminator::Branch {
                cond: cond_val,
                then_target: then_bb,
                else_target: next_else_bb,
            })?;

            self.switch_to(then_bb)?;
            self.lower_ast_block(&branch.body)?;
            self.jump_if_open(merge_bb)?;

            if is_last {
                if let Some(else_block) = &stmt.else_block {
                    self.switch_to(next_else_bb)?;
                    self.lower_ast_block(else_block)?;
                    self.jump_if_open(merge_bb)?;
                }
            } else {
                cond_bb = next_else_bb;
            }
        }

        self.switch_to(merge_bb)
    }

    fn lower_while_stmt(&mut self, stmt: &WhileStmt<'src>) -> Result<(), TypedIrLoweringError> {
        let cond_bb = self.new_block(IrBlockKind::WhileCond, Some(stmt.condition.span));
        let body_bb = self.new_block(IrBlockKind::WhileBody, Some(stmt.body.span));
        let exit_bb = self.new_block(IrBlockKind::WhileExit, Some(stmt.span));

        self.set_terminator(IrTerminator::Jump { target: cond_bb })?;

        self.switch_to(cond_bb)?;
        let cond_val = self.lower_expr_bool_value(&stmt.condition)?;
        self.set_terminator(IrTerminator::Branch {
            cond: cond_val,
            then_target: body_bb,
            else_target: exit_bb,
        })?;

        self.switch_to(body_bb)?;
        self.lower_ast_block(&stmt.body)?;
        self.jump_if_open(cond_bb)?;

        self.switch_to(exit_bb)
    }

    fn lower_for_range_stmt(
        &mut self,
        stmt: &ForRangeStmt<'src>,
    ) -> Result<(), TypedIrLoweringError> {
        let (start_v, end_v, step_v) = self.lower_range_args(&stmt.range)?;

        self.push_scope();
        let binding_local = self.alloc_local(
            Some(stmt.binding),
            SemanticType::Int,
            IrLocalKind::LoopBinding,
            Some(stmt.binding_span),
            true,
        );
        self.bind_local(stmt.binding, stmt.binding_span, binding_local)?;
        let end_local = self.alloc_local(
            None,
            SemanticType::Int,
            IrLocalKind::HiddenLoopEnd,
            Some(stmt.range.span),
            true,
        );
        let step_local = self.alloc_local(
            None,
            SemanticType::Int,
            IrLocalKind::HiddenLoopStep,
            Some(stmt.range.span),
            true,
        );
        self.emit_store_local(binding_local, start_v, Some(stmt.span));
        self.emit_store_local(end_local, end_v, Some(stmt.span));
        self.emit_store_local(step_local, step_v, Some(stmt.span));

        let guard_bb = self.new_block(IrBlockKind::ForGuard, Some(stmt.range.span));
        let cond_bb = self.new_block(IrBlockKind::ForCond, Some(stmt.range.span));
        let pos_cond_bb = self.new_block(IrBlockKind::ForPosCond, Some(stmt.range.span));
        let neg_cond_bb = self.new_block(IrBlockKind::ForNegCond, Some(stmt.range.span));
        let body_bb = self.new_block(IrBlockKind::ForBody, Some(stmt.body.span));
        let exit_bb = self.new_block(IrBlockKind::ForExit, Some(stmt.span));

        self.set_terminator(IrTerminator::Jump { target: guard_bb })?;

        self.switch_to(guard_bb)?;
        let step_now = self.emit_load_local(step_local, Some(stmt.range.span))?;
        let zero = self.emit_const_int(0, stmt.range.span)?;
        let step_nonzero =
            self.emit_binary_value(BinaryOp::NotEq, step_now, zero, stmt.range.span)?;
        self.set_terminator(IrTerminator::Branch {
            cond: step_nonzero,
            then_target: cond_bb,
            else_target: exit_bb,
        })?;

        self.switch_to(cond_bb)?;
        let step_now = self.emit_load_local(step_local, Some(stmt.range.span))?;
        let zero = self.emit_const_int(0, stmt.range.span)?;
        let step_pos = self.emit_binary_value(BinaryOp::Gt, step_now, zero, stmt.range.span)?;
        self.set_terminator(IrTerminator::Branch {
            cond: step_pos,
            then_target: pos_cond_bb,
            else_target: neg_cond_bb,
        })?;

        self.switch_to(pos_cond_bb)?;
        let i_val = self.emit_load_local(binding_local, Some(stmt.binding_span))?;
        let end_val = self.emit_load_local(end_local, Some(stmt.range.span))?;
        let cont_pos = self.emit_binary_value(BinaryOp::Lt, i_val, end_val, stmt.range.span)?;
        self.set_terminator(IrTerminator::Branch {
            cond: cont_pos,
            then_target: body_bb,
            else_target: exit_bb,
        })?;

        self.switch_to(neg_cond_bb)?;
        let i_val = self.emit_load_local(binding_local, Some(stmt.binding_span))?;
        let end_val = self.emit_load_local(end_local, Some(stmt.range.span))?;
        let cont_neg = self.emit_binary_value(BinaryOp::Gt, i_val, end_val, stmt.range.span)?;
        self.set_terminator(IrTerminator::Branch {
            cond: cont_neg,
            then_target: body_bb,
            else_target: exit_bb,
        })?;

        self.switch_to(body_bb)?;
        self.lower_ast_block(&stmt.body)?;
        if !self.current_block_is_terminated()? {
            let cur_i = self.emit_load_local(binding_local, Some(stmt.binding_span))?;
            let step_cur = self.emit_load_local(step_local, Some(stmt.range.span))?;
            let next_i = self.emit_binary_value(BinaryOp::Add, cur_i, step_cur, stmt.range.span)?;
            self.emit_store_local(binding_local, next_i, Some(stmt.span));
            self.set_terminator(IrTerminator::Jump { target: guard_bb })?;
        }

        self.pop_scope();
        self.switch_to(exit_bb)
    }

    fn lower_range_args(
        &mut self,
        range: &RangeSpec<'src>,
    ) -> Result<(IrValue, IrValue, IrValue), TypedIrLoweringError> {
        let mut lowered = Vec::with_capacity(range.args.len());
        for arg in &range.args {
            let (value, ty, _) = self.lower_expr_any(arg)?.require_value(arg.span)?;
            if ty != SemanticType::Int {
                return Err(TypedIrLoweringError::new(
                    TypedIrLoweringErrorKind::InvalidRangeArgType,
                    Some(arg.span),
                    format!("range arg must be int, found {:?}", ty),
                ));
            }
            lowered.push(value);
        }

        match lowered.as_slice() {
            [end] => {
                let start = self.emit_const_int(0, range.span)?;
                let step = self.emit_const_int(1, range.span)?;
                Ok((start, *end, step))
            }
            [start, end] => {
                let step = self.emit_const_int(1, range.span)?;
                Ok((*start, *end, step))
            }
            [start, end, step] => Ok((*start, *end, *step)),
            _ => Err(TypedIrLoweringError::new(
                TypedIrLoweringErrorKind::InvalidRangeArity,
                Some(range.span),
                format!("range expects 1..=3 args, got {}", lowered.len()),
            )),
        }
    }

    fn lower_expr_bool_value(
        &mut self,
        expr: &Expr<'src>,
    ) -> Result<IrValue, TypedIrLoweringError> {
        let (value, ty, _) = self.lower_expr_any(expr)?.require_value(expr.span)?;
        if ty != SemanticType::Bool {
            return Err(TypedIrLoweringError::new(
                TypedIrLoweringErrorKind::TypeMismatch,
                Some(expr.span),
                format!("condition must be bool, found {:?}", ty),
            ));
        }
        Ok(value)
    }

    fn lower_expr_any(
        &mut self,
        expr: &Expr<'src>,
    ) -> Result<LoweredExpr<'src>, TypedIrLoweringError> {
        match &expr.kind {
            ExprKind::Identifier(name) => {
                let local = self.resolve_local(name).ok_or_else(|| {
                    TypedIrLoweringError::new(
                        TypedIrLoweringErrorKind::UnknownLocalBinding,
                        Some(expr.span),
                        format!("unknown local `{name}`"),
                    )
                })?;
                let ty = self.local_type(local)?;
                let value = self.emit_value_inst(
                    IrInstKind::LoadLocal { local },
                    ty,
                    Some(expr.span),
                    IrInstMeta::default(),
                )?;
                Ok(LoweredExpr {
                    ty,
                    value: Some(value),
                    known_const: None,
                })
            }
            ExprKind::IntegerLiteral(raw) => {
                let v = parse_i32_literal(raw).ok_or_else(|| {
                    TypedIrLoweringError::new(
                        TypedIrLoweringErrorKind::InvalidLiteral,
                        Some(expr.span),
                        format!("invalid int literal `{raw}`"),
                    )
                })?;
                let value = self.emit_const(IrConstValue::Int(v), Some(expr.span))?;
                Ok(LoweredExpr {
                    ty: SemanticType::Int,
                    value: Some(value),
                    known_const: Some(IrConstValue::Int(v)),
                })
            }
            ExprKind::FloatLiteral(raw) => {
                let v = parse_f32_literal(raw).ok_or_else(|| {
                    TypedIrLoweringError::new(
                        TypedIrLoweringErrorKind::InvalidLiteral,
                        Some(expr.span),
                        format!("invalid float literal `{raw}`"),
                    )
                })?;
                let value = self.emit_const(IrConstValue::Float(v), Some(expr.span))?;
                Ok(LoweredExpr {
                    ty: SemanticType::Float,
                    value: Some(value),
                    known_const: Some(IrConstValue::Float(v)),
                })
            }
            ExprKind::BoolLiteral(v) => {
                let value = self.emit_const(IrConstValue::Bool(*v), Some(expr.span))?;
                Ok(LoweredExpr {
                    ty: SemanticType::Bool,
                    value: Some(value),
                    known_const: Some(IrConstValue::Bool(*v)),
                })
            }
            ExprKind::StringLiteral(s) => {
                let value = self.emit_const(IrConstValue::Str(s), Some(expr.span))?;
                Ok(LoweredExpr {
                    ty: SemanticType::Str,
                    value: Some(value),
                    known_const: Some(IrConstValue::Str(s)),
                })
            }
            ExprKind::Grouping(inner) => self.lower_expr_any(inner),
            ExprKind::Unary {
                op: UnaryOp::Neg,
                expr: inner,
            } => {
                let lowered = self.lower_expr_any(inner)?;
                let (operand, ty, known) = lowered.require_value(inner.span)?;
                if ty != SemanticType::Int && ty != SemanticType::Float {
                    return Err(TypedIrLoweringError::new(
                        TypedIrLoweringErrorKind::TypeMismatch,
                        Some(expr.span),
                        format!("unary neg expects int/float, found {:?}", ty),
                    ));
                }
                let known_const = match known {
                    Some(IrConstValue::Int(v)) => Some(IrConstValue::Int(v.saturating_neg())),
                    Some(IrConstValue::Float(v)) => Some(IrConstValue::Float(-v)),
                    _ => None,
                };
                let meta = IrInstMeta {
                    const_folding_candidate: self.lowerer.config.annotate_const_folding_candidates,
                    known_const_value: known_const,
                    simd_annotation: IrSimdAnnotation::None,
                };
                let value = self.emit_value_inst(
                    IrInstKind::Unary {
                        op: UnaryOp::Neg,
                        operand,
                    },
                    ty,
                    Some(expr.span),
                    meta,
                )?;
                Ok(LoweredExpr {
                    ty,
                    value: Some(value),
                    known_const,
                })
            }
            ExprKind::Binary { op, left, right } => {
                let l = self.lower_expr_any(left)?;
                let r = self.lower_expr_any(right)?;
                let (lv, lt, lconst) = l.require_value(left.span)?;
                let (rv, rt, rconst) = r.require_value(right.span)?;
                let result_ty = infer_binary_type(*op, lt, rt, expr.span)?;
                let meta = self.binary_meta(*op, result_ty, lconst, rconst);
                let value = self.emit_value_inst(
                    IrInstKind::Binary {
                        op: *op,
                        left: lv,
                        right: rv,
                    },
                    result_ty,
                    Some(expr.span),
                    meta,
                )?;
                Ok(LoweredExpr {
                    ty: result_ty,
                    value: Some(value),
                    known_const: meta.known_const_value,
                })
            }
            ExprKind::Call { callee, args } => self.lower_call(expr.span, callee, args),
        }
    }

    fn lower_call(
        &mut self,
        call_span: Span,
        callee: &Expr<'src>,
        args: &[Expr<'src>],
    ) -> Result<LoweredExpr<'src>, TypedIrLoweringError> {
        let ExprKind::Identifier(name) = &callee.kind else {
            return Err(TypedIrLoweringError::new(
                TypedIrLoweringErrorKind::UnsupportedCallTarget,
                Some(callee.span),
                "only direct identifier calls are supported in V1 lowering",
            ));
        };

        let mut ir_args = Vec::with_capacity(args.len());
        for arg in args {
            let (value, _ty, _) = self.lower_expr_any(arg)?.require_value(arg.span)?;
            ir_args.push(value);
        }

        let (ir_callee, result_ty) = if let Some(sig) = self.sig_map.get(name).copied() {
            (IrCallee::Internal(name), sig.return_type)
        } else if self.lowerer.handle_acquire_calls.contains(*name) {
            (
                IrCallee::External {
                    name,
                    flavor: IrExternalCallFlavor::HandleAcquire,
                },
                SemanticType::Handle,
            )
        } else if self.lowerer.handle_release_calls.contains(*name) {
            (
                IrCallee::External {
                    name,
                    flavor: IrExternalCallFlavor::HandleRelease,
                },
                SemanticType::Unit,
            )
        } else {
            let result_ty = self
                .lowerer
                .external_returns
                .get(*name)
                .copied()
                .unwrap_or(SemanticType::Unit);
            (
                IrCallee::External {
                    name,
                    flavor: IrExternalCallFlavor::Generic,
                },
                result_ty,
            )
        };

        let meta = IrInstMeta::default();
        if result_ty == SemanticType::Unit {
            self.emit_unit_inst(
                IrInstKind::Call {
                    callee: ir_callee,
                    args: ir_args,
                },
                Some(call_span),
                meta,
            )?;
            Ok(LoweredExpr {
                ty: SemanticType::Unit,
                value: None,
                known_const: None,
            })
        } else {
            let value = self.emit_value_inst(
                IrInstKind::Call {
                    callee: ir_callee,
                    args: ir_args,
                },
                result_ty,
                Some(call_span),
                meta,
            )?;
            Ok(LoweredExpr {
                ty: result_ty,
                value: Some(value),
                known_const: None,
            })
        }
    }

    fn binary_meta(
        &mut self,
        op: BinaryOp,
        result_ty: SemanticType,
        left_const: Option<IrConstValue<'src>>,
        right_const: Option<IrConstValue<'src>>,
    ) -> IrInstMeta<'src> {
        let simd_annotation = if self.lowerer.config.annotate_simd_candidates
            && matches!(
                op,
                BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul | BinaryOp::Div
            )
            && matches!(result_ty, SemanticType::Int | SemanticType::Float)
        {
            IrSimdAnnotation::ArithmeticLaneCandidate
        } else {
            IrSimdAnnotation::None
        };

        let known_const = if self.lowerer.config.annotate_const_folding_candidates {
            fold_binary_const(op, left_const, right_const)
        } else {
            None
        };

        IrInstMeta {
            const_folding_candidate: self.lowerer.config.annotate_const_folding_candidates,
            known_const_value: known_const,
            simd_annotation,
        }
    }

    fn emit_default_value(
        &mut self,
        ty: SemanticType,
        span: Span,
    ) -> Result<IrValue, TypedIrLoweringError> {
        match ty {
            SemanticType::Int | SemanticType::Handle | SemanticType::Str => {
                self.emit_const_int(0, span)
            }
            SemanticType::Bool => self.emit_const_bool(false, span),
            SemanticType::Float => self.emit_const_float(0.0, span),
            SemanticType::Unit => Err(TypedIrLoweringError::new(
                TypedIrLoweringErrorKind::InternalInvariant,
                Some(span),
                "unit default does not produce a value",
            )),
        }
    }

    fn emit_const(
        &mut self,
        value: IrConstValue<'src>,
        span: Option<Span>,
    ) -> Result<IrValue, TypedIrLoweringError> {
        let ty = match value {
            IrConstValue::Int(_) => SemanticType::Int,
            IrConstValue::Float(_) => SemanticType::Float,
            IrConstValue::Bool(_) => SemanticType::Bool,
            IrConstValue::Str(_) => SemanticType::Str,
        };
        let meta = IrInstMeta {
            const_folding_candidate: self.lowerer.config.annotate_const_folding_candidates,
            known_const_value: Some(value),
            simd_annotation: IrSimdAnnotation::None,
        };
        self.emit_value_inst(IrInstKind::Const(value), ty, span, meta)
    }

    fn emit_const_int(&mut self, v: i32, span: Span) -> Result<IrValue, TypedIrLoweringError> {
        self.emit_const(IrConstValue::Int(v), Some(span))
    }

    fn emit_const_float(&mut self, v: f32, span: Span) -> Result<IrValue, TypedIrLoweringError> {
        self.emit_const(IrConstValue::Float(v), Some(span))
    }

    fn emit_const_bool(&mut self, v: bool, span: Span) -> Result<IrValue, TypedIrLoweringError> {
        self.emit_const(IrConstValue::Bool(v), Some(span))
    }

    fn emit_load_local(
        &mut self,
        local: IrLocalId,
        span: Option<Span>,
    ) -> Result<IrValue, TypedIrLoweringError> {
        let ty = self.local_type(local)?;
        self.emit_value_inst(
            IrInstKind::LoadLocal { local },
            ty,
            span,
            IrInstMeta::default(),
        )
    }

    fn emit_store_local(&mut self, local: IrLocalId, value: IrValue, span: Option<Span>) {
        let _ = self.emit_unit_inst(
            IrInstKind::StoreLocal { local, value },
            span,
            IrInstMeta::default(),
        );
    }

    fn emit_binary_value(
        &mut self,
        op: BinaryOp,
        left: IrValue,
        right: IrValue,
        span: Span,
    ) -> Result<IrValue, TypedIrLoweringError> {
        let lt = self.value_type(left)?;
        let rt = self.value_type(right)?;
        let result_ty = infer_binary_type(op, lt, rt, span)?;
        let meta = self.binary_meta(op, result_ty, None, None);
        self.emit_value_inst(
            IrInstKind::Binary { op, left, right },
            result_ty,
            Some(span),
            meta,
        )
    }

    fn emit_value_inst(
        &mut self,
        kind: IrInstKind<'src>,
        ty: SemanticType,
        span: Option<Span>,
        meta: IrInstMeta<'src>,
    ) -> Result<IrValue, TypedIrLoweringError> {
        let block = self.current_block;
        let temp = self.alloc_temp(ty, block, span);
        if meta.const_folding_candidate {
            self.func_meta.const_fold_candidate_insts =
                self.func_meta.const_fold_candidate_insts.saturating_add(1);
        }
        if meta.simd_annotation != IrSimdAnnotation::None {
            self.func_meta.simd_candidate_insts =
                self.func_meta.simd_candidate_insts.saturating_add(1);
        }
        let inst = TypedIrInst {
            dest: Some(temp),
            ty: Some(ty),
            span,
            kind,
            meta,
        };
        self.current_block_mut()?.instructions.push(inst);
        Ok(IrValue::Temp(temp))
    }

    fn emit_unit_inst(
        &mut self,
        kind: IrInstKind<'src>,
        span: Option<Span>,
        meta: IrInstMeta<'src>,
    ) -> Result<(), TypedIrLoweringError> {
        if meta.const_folding_candidate {
            self.func_meta.const_fold_candidate_insts =
                self.func_meta.const_fold_candidate_insts.saturating_add(1);
        }
        if meta.simd_annotation != IrSimdAnnotation::None {
            self.func_meta.simd_candidate_insts =
                self.func_meta.simd_candidate_insts.saturating_add(1);
        }
        let inst = TypedIrInst {
            dest: None,
            ty: Some(SemanticType::Unit),
            span,
            kind,
            meta,
        };
        self.current_block_mut()?.instructions.push(inst);
        Ok(())
    }

    fn alloc_local(
        &mut self,
        name: Option<&'src str>,
        ty: SemanticType,
        kind: IrLocalKind,
        declared_span: Option<Span>,
        mutable: bool,
    ) -> IrLocalId {
        let id = IrLocalId(self.locals.len() as u32);
        self.locals.push(TypedIrLocal {
            id,
            name,
            ty,
            kind,
            declared_span,
            mutable,
        });
        self.local_types.push(ty);
        id
    }

    fn alloc_temp(
        &mut self,
        ty: SemanticType,
        producer_block: IrBlockId,
        span: Option<Span>,
    ) -> IrTempId {
        let id = IrTempId(self.temps.len() as u32);
        self.temps.push(TypedIrTemp {
            id,
            ty,
            producer_block,
            span,
        });
        self.temp_types.push(ty);
        id
    }

    fn local_type(&self, local: IrLocalId) -> Result<SemanticType, TypedIrLoweringError> {
        self.local_types
            .get(local.0 as usize)
            .copied()
            .ok_or_else(|| {
                TypedIrLoweringError::new(
                    TypedIrLoweringErrorKind::InternalInvariant,
                    None,
                    format!("invalid local id {}", local.0),
                )
            })
    }

    fn value_type(&self, value: IrValue) -> Result<SemanticType, TypedIrLoweringError> {
        match value {
            IrValue::Temp(t) => self.temp_types.get(t.0 as usize).copied().ok_or_else(|| {
                TypedIrLoweringError::new(
                    TypedIrLoweringErrorKind::InternalInvariant,
                    None,
                    format!("invalid temp id {}", t.0),
                )
            }),
        }
    }

    fn new_block(&mut self, kind: IrBlockKind, span: Option<Span>) -> IrBlockId {
        let id = IrBlockId(self.blocks.len() as u32);
        self.blocks.push(BlockBuilder {
            kind,
            span,
            instructions: Vec::new(),
            terminator: None,
        });
        id
    }

    fn switch_to(&mut self, block: IrBlockId) -> Result<(), TypedIrLoweringError> {
        if self.blocks.get(block.0 as usize).is_none() {
            return Err(TypedIrLoweringError::new(
                TypedIrLoweringErrorKind::InternalInvariant,
                None,
                format!("invalid block id {}", block.0),
            ));
        }
        self.current_block = block;
        Ok(())
    }

    fn current_block_mut(&mut self) -> Result<&mut BlockBuilder<'src>, TypedIrLoweringError> {
        self.blocks
            .get_mut(self.current_block.0 as usize)
            .ok_or_else(|| {
                TypedIrLoweringError::new(
                    TypedIrLoweringErrorKind::InternalInvariant,
                    None,
                    format!("invalid current block {}", self.current_block.0),
                )
            })
    }

    fn current_block_is_terminated(&self) -> Result<bool, TypedIrLoweringError> {
        Ok(self
            .blocks
            .get(self.current_block.0 as usize)
            .ok_or_else(|| {
                TypedIrLoweringError::new(
                    TypedIrLoweringErrorKind::InternalInvariant,
                    None,
                    format!("invalid current block {}", self.current_block.0),
                )
            })?
            .terminator
            .is_some())
    }

    fn set_terminator(&mut self, term: IrTerminator) -> Result<(), TypedIrLoweringError> {
        let block = self.current_block_mut()?;
        if block.terminator.is_some() {
            return Err(TypedIrLoweringError::new(
                TypedIrLoweringErrorKind::InternalInvariant,
                block.span,
                format!("block {} already terminated", self.current_block.0),
            ));
        }
        block.terminator = Some(term);
        Ok(())
    }

    fn jump_if_open(&mut self, target: IrBlockId) -> Result<(), TypedIrLoweringError> {
        if !self.current_block_is_terminated()? {
            self.set_terminator(IrTerminator::Jump { target })?;
        }
        Ok(())
    }

    fn push_scope(&mut self) {
        self.scopes.push(HashMap::new());
    }

    fn pop_scope(&mut self) {
        let _ = self.scopes.pop();
    }

    fn bind_local(
        &mut self,
        name: &'src str,
        span: Span,
        local: IrLocalId,
    ) -> Result<(), TypedIrLoweringError> {
        let Some(scope) = self.scopes.last_mut() else {
            return Err(TypedIrLoweringError::new(
                TypedIrLoweringErrorKind::InternalInvariant,
                Some(span),
                "scope stack is empty",
            ));
        };
        if scope.contains_key(name) {
            return Err(TypedIrLoweringError::new(
                TypedIrLoweringErrorKind::InternalInvariant,
                Some(span),
                format!("duplicate local binding during lowering: `{}`", name),
            ));
        }
        scope.insert(name, local);
        Ok(())
    }

    fn resolve_local(&self, name: &str) -> Option<IrLocalId> {
        self.scopes
            .iter()
            .rev()
            .find_map(|scope| scope.get(name).copied())
    }
}

/// Lower using an explicit configuration.
pub fn lower_to_typed_ir_with_config<'src>(
    module: &Module<'src>,
    semantic: &SemanticReport<'src>,
    config: TypedIrLoweringConfig,
) -> Result<TypedIrModule<'src>, TypedIrLoweringError> {
    let lowerer = TypedIrLowerer::new(config);

    let mut sig_map = HashMap::<&'src str, &FunctionSignature<'src>>::new();
    for sig in &semantic.signatures {
        sig_map.insert(sig.name, sig);
    }

    let mut functions = Vec::with_capacity(module.items.len());
    for (idx, item) in module.items.iter().enumerate() {
        let Item::Function(func) = item;
        let sig = sig_map.get(func.name).copied().ok_or_else(|| {
            TypedIrLoweringError::new(
                TypedIrLoweringErrorKind::MissingSemanticSignature,
                Some(func.name_span),
                format!("missing semantic signature for `{}`", func.name),
            )
        })?;
        let mut fcx =
            FunctionLoweringCx::new(&lowerer, IrFunctionId(idx as u32), func, sig, &sig_map)?;
        fcx.lower_function_body()?;
        functions.push(fcx.finish()?);
    }

    let mut ir = TypedIrModule {
        functions,
        span: module.span,
        meta: TypedIrModuleMeta {
            arena_layout: TypedIrArenaLayout::VecBacked,
            optimization_hooks: TypedIrOptimizationHooks {
                const_folding_ready: lowerer.config.annotate_const_folding_candidates,
                simd_annotation_ready: lowerer.config.annotate_simd_candidates,
            },
            arena_stats: TypedIrArenaStats::default(),
        },
    };
    ir.recompute_metadata();
    Ok(ir)
}

fn map_type_name(kind: TypeName) -> SemanticType {
    match kind {
        TypeName::Int => SemanticType::Int,
        TypeName::Float => SemanticType::Float,
        TypeName::Bool => SemanticType::Bool,
        TypeName::Str => SemanticType::Str,
    }
}

fn parse_i32_literal(text: &str) -> Option<i32> {
    let clean: String = text.chars().filter(|&c| c != '_').collect();
    clean.parse::<i32>().ok()
}

fn parse_f32_literal(text: &str) -> Option<f32> {
    let clean: String = text.chars().filter(|&c| c != '_').collect();
    clean.parse::<f32>().ok()
}

fn infer_binary_type(
    op: BinaryOp,
    left: SemanticType,
    right: SemanticType,
    span: Span,
) -> Result<SemanticType, TypedIrLoweringError> {
    use BinaryOp as B;
    use SemanticType as T;

    match op {
        B::Add | B::Sub | B::Mul => match (left, right) {
            (T::Int, T::Int) => Ok(T::Int),
            (T::Float, T::Float) => Ok(T::Float),
            _ => Err(TypedIrLoweringError::new(
                TypedIrLoweringErrorKind::TypeMismatch,
                Some(span),
                format!("invalid operands {:?} and {:?} for {:?}", left, right, op),
            )),
        },
        B::Div => match (left, right) {
            (T::Int, T::Int) => Ok(T::Int),
            (T::Float, T::Float) => Ok(T::Float),
            _ => Err(TypedIrLoweringError::new(
                TypedIrLoweringErrorKind::TypeMismatch,
                Some(span),
                format!("invalid operands {:?} and {:?} for {:?}", left, right, op),
            )),
        },
        B::Mod => match (left, right) {
            (T::Int, T::Int) => Ok(T::Int),
            _ => Err(TypedIrLoweringError::new(
                TypedIrLoweringErrorKind::UnsupportedConstruct,
                Some(span),
                format!(
                    "modulo is only supported for int in V1 lowering, got {:?}/{:?}",
                    left, right
                ),
            )),
        },
        B::EqEq | B::NotEq => {
            if left == right {
                Ok(T::Bool)
            } else {
                Err(TypedIrLoweringError::new(
                    TypedIrLoweringErrorKind::TypeMismatch,
                    Some(span),
                    format!("comparison type mismatch {:?} vs {:?}", left, right),
                ))
            }
        }
        B::Lt | B::LtEq | B::Gt | B::GtEq => match (left, right) {
            (T::Int, T::Int) | (T::Float, T::Float) => Ok(T::Bool),
            _ => Err(TypedIrLoweringError::new(
                TypedIrLoweringErrorKind::TypeMismatch,
                Some(span),
                format!(
                    "ordered compare requires int/int or float/float, got {:?}/{:?}",
                    left, right
                ),
            )),
        },
        B::AndAnd | B::OrOr => match (left, right) {
            (T::Bool, T::Bool) => Ok(T::Bool),
            _ => Err(TypedIrLoweringError::new(
                TypedIrLoweringErrorKind::TypeMismatch,
                Some(span),
                format!("logical op requires bool/bool, got {:?}/{:?}", left, right),
            )),
        },
    }
}

fn fold_binary_const<'src>(
    op: BinaryOp,
    left: Option<IrConstValue<'src>>,
    right: Option<IrConstValue<'src>>,
) -> Option<IrConstValue<'src>> {
    use BinaryOp as B;
    match (op, left?, right?) {
        (B::Add, IrConstValue::Int(a), IrConstValue::Int(b)) => {
            Some(IrConstValue::Int(a.saturating_add(b)))
        }
        (B::Sub, IrConstValue::Int(a), IrConstValue::Int(b)) => {
            Some(IrConstValue::Int(a.saturating_sub(b)))
        }
        (B::Mul, IrConstValue::Int(a), IrConstValue::Int(b)) => {
            Some(IrConstValue::Int(a.saturating_mul(b)))
        }
        (B::Div, IrConstValue::Int(_), IrConstValue::Int(0)) => None,
        (B::Div, IrConstValue::Int(a), IrConstValue::Int(b)) => Some(IrConstValue::Int(a / b)),
        (B::Mod, IrConstValue::Int(_), IrConstValue::Int(0)) => None,
        (B::Mod, IrConstValue::Int(a), IrConstValue::Int(b)) => Some(IrConstValue::Int(a % b)),
        (B::Add, IrConstValue::Float(a), IrConstValue::Float(b)) => {
            Some(IrConstValue::Float(a + b))
        }
        (B::Sub, IrConstValue::Float(a), IrConstValue::Float(b)) => {
            Some(IrConstValue::Float(a - b))
        }
        (B::Mul, IrConstValue::Float(a), IrConstValue::Float(b)) => {
            Some(IrConstValue::Float(a * b))
        }
        (B::Div, IrConstValue::Float(a), IrConstValue::Float(b)) => {
            Some(IrConstValue::Float(a / b))
        }
        (B::EqEq, IrConstValue::Int(a), IrConstValue::Int(b)) => Some(IrConstValue::Bool(a == b)),
        (B::NotEq, IrConstValue::Int(a), IrConstValue::Int(b)) => Some(IrConstValue::Bool(a != b)),
        (B::Lt, IrConstValue::Int(a), IrConstValue::Int(b)) => Some(IrConstValue::Bool(a < b)),
        (B::LtEq, IrConstValue::Int(a), IrConstValue::Int(b)) => Some(IrConstValue::Bool(a <= b)),
        (B::Gt, IrConstValue::Int(a), IrConstValue::Int(b)) => Some(IrConstValue::Bool(a > b)),
        (B::GtEq, IrConstValue::Int(a), IrConstValue::Int(b)) => Some(IrConstValue::Bool(a >= b)),
        (B::EqEq, IrConstValue::Bool(a), IrConstValue::Bool(b)) => Some(IrConstValue::Bool(a == b)),
        (B::NotEq, IrConstValue::Bool(a), IrConstValue::Bool(b)) => {
            Some(IrConstValue::Bool(a != b))
        }
        (B::AndAnd, IrConstValue::Bool(a), IrConstValue::Bool(b)) => {
            Some(IrConstValue::Bool(a && b))
        }
        (B::OrOr, IrConstValue::Bool(a), IrConstValue::Bool(b)) => Some(IrConstValue::Bool(a || b)),
        (B::EqEq, IrConstValue::Float(a), IrConstValue::Float(b)) => {
            Some(IrConstValue::Bool(a == b))
        }
        (B::NotEq, IrConstValue::Float(a), IrConstValue::Float(b)) => {
            Some(IrConstValue::Bool(a != b))
        }
        (B::Lt, IrConstValue::Float(a), IrConstValue::Float(b)) => Some(IrConstValue::Bool(a < b)),
        (B::LtEq, IrConstValue::Float(a), IrConstValue::Float(b)) => {
            Some(IrConstValue::Bool(a <= b))
        }
        (B::Gt, IrConstValue::Float(a), IrConstValue::Float(b)) => Some(IrConstValue::Bool(a > b)),
        (B::GtEq, IrConstValue::Float(a), IrConstValue::Float(b)) => {
            Some(IrConstValue::Bool(a >= b))
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tlscript::{Lexer, Parser, SemanticAnalyzer};

    fn parse_semantic(src: &str) -> (Module<'_>, SemanticReport<'_>) {
        let mut parser = Parser::new(Lexer::new(src));
        let module = parser.parse_module().expect("parse ok");
        let report = SemanticAnalyzer::default()
            .analyze(&module)
            .expect("semantic ok");
        (module, report)
    }

    #[test]
    fn lowers_if_and_while_into_canonical_blocks() {
        let (module, report) = parse_semantic(concat!(
            "@export\n",
            "def update(dt: float):\n",
            "    let x = 1\n",
            "    if dt > 0.0:\n",
            "        x = x + 1\n",
            "    else:\n",
            "        x = x + 2\n",
            "    while x < 4:\n",
            "        x = x + 1\n",
        ));

        let ir = lower_to_typed_ir(&module, &report).expect("lower ok");
        assert_eq!(ir.functions.len(), 1);
        let f = &ir.functions[0];
        assert!(f.exported);
        assert!(f.blocks.len() >= 7);
        assert!(f.blocks.iter().any(|b| matches!(
            b.kind,
            IrBlockKind::IfCond | IrBlockKind::IfThen | IrBlockKind::IfElse
        )));
        assert!(f.blocks.iter().any(|b| matches!(
            b.kind,
            IrBlockKind::WhileCond | IrBlockKind::WhileBody | IrBlockKind::WhileExit
        )));
    }

    #[test]
    fn lowers_for_range_with_hidden_loop_locals() {
        let (module, report) = parse_semantic(concat!(
            "def tick():\n",
            "    let sum = 0\n",
            "    for i in range(0, 8, 2):\n",
            "        sum = sum + i\n",
        ));

        let ir = lower_to_typed_ir(&module, &report).expect("lower ok");
        let f = &ir.functions[0];
        let loop_binding_count = f
            .locals
            .iter()
            .filter(|l| l.kind == IrLocalKind::LoopBinding)
            .count();
        let hidden_end_count = f
            .locals
            .iter()
            .filter(|l| l.kind == IrLocalKind::HiddenLoopEnd)
            .count();
        let hidden_step_count = f
            .locals
            .iter()
            .filter(|l| l.kind == IrLocalKind::HiddenLoopStep)
            .count();
        assert_eq!(loop_binding_count, 1);
        assert_eq!(hidden_end_count, 1);
        assert_eq!(hidden_step_count, 1);
        assert!(f.blocks.iter().any(|b| b.kind == IrBlockKind::ForGuard));
        assert!(f.blocks.iter().any(|b| b.kind == IrBlockKind::ForCond));
    }

    #[test]
    fn records_const_and_simd_metadata_hooks() {
        let (module, report) = parse_semantic(concat!(
            "def f(a: int, b: int):\n",
            "    let x = 1 + 2\n",
            "    let y = a + b\n",
        ));

        let ir = lower_to_typed_ir(&module, &report).expect("lower ok");
        let f = &ir.functions[0];
        assert!(f.meta.const_fold_candidate_insts > 0);
        assert!(f.meta.simd_candidate_insts > 0);
        let has_folded_const_hint = f.blocks.iter().flat_map(|b| &b.instructions).any(|inst| {
            inst.meta.known_const_value == Some(IrConstValue::Int(3))
                && matches!(
                    inst.kind,
                    IrInstKind::Binary {
                        op: BinaryOp::Add,
                        ..
                    }
                )
        });
        assert!(has_folded_const_hint);
    }
}
