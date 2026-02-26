//! Semantic analysis and safety policy checks for `.tlscript`.
//!
//! This module implements a V1 semantic validator focused on correctness and engine-safety rules:
//! - owned-only values with lexical lifetimes (scope-based validation)
//! - mandatory bounds-check policy (currently guaranteed by the V1 surface syntax)
//! - raw pointer operations forbidden
//! - WASM sandbox restrictions for exported functions and forbidden intrinsics

use std::collections::{HashMap, HashSet};
use std::fmt;

use super::ast::*;
use super::token::Span;

/// Primitive/inferred value type produced by semantic analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SemanticType {
    Int,
    Float,
    Bool,
    Str,
    /// Engine-managed opaque handle (sprite/body/etc). No raw pointer access is exposed.
    Handle,
    /// Function returns no value (implicit unit).
    Unit,
}

impl SemanticType {
    fn from_type_name(name: TypeName) -> Self {
        match name {
            TypeName::Int => Self::Int,
            TypeName::Float => Self::Float,
            TypeName::Bool => Self::Bool,
            TypeName::Str => Self::Str,
        }
    }
}

/// Ownership model enforced by the `.tlscript` frontend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OwnershipLifetimePolicy {
    /// All values are owned and scoped lexically; no borrow/reference syntax is exposed.
    OwnedOnlyLexicalScopes,
}

/// Bounds-check policy for indexed memory/container access.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundsCheckPolicy {
    /// Bounds checks are mandatory for any index operation.
    Required,
}

/// Current enforcement status for bounds safety.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundsCheckEnforcement {
    /// V1 parser/AST does not expose indexing syntax yet, so unchecked indexing is impossible.
    GuaranteedBySurfaceSyntaxV1,
    /// Explicit checked indexing validation was performed (future AST support).
    ExplicitlyValidated,
}

/// Pointer safety policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PointerPolicy {
    /// Raw pointers and pointer intrinsics are disallowed.
    Forbidden,
}

/// WASM sandbox policy used by the semantic validator.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WasmSandboxPolicy {
    /// Forbid direct raw memory / table intrinsics from script code.
    pub forbid_raw_memory_intrinsics: bool,
    /// Forbid pointer-like helper intrinsics (`addr_of`, `raw_ptr`, etc).
    pub forbid_pointer_intrinsics: bool,
    /// Reject calls to functions not declared in the module or allowlisted.
    pub forbid_unknown_calls: bool,
}

impl Default for WasmSandboxPolicy {
    fn default() -> Self {
        Self {
            forbid_raw_memory_intrinsics: true,
            forbid_pointer_intrinsics: true,
            forbid_unknown_calls: true,
        }
    }
}

/// Aggregate safety policy for semantic validation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SemanticSafetyPolicy {
    /// Ownership/lifetime model.
    pub ownership_lifetimes: OwnershipLifetimePolicy,
    /// Bounds-check requirement.
    pub bounds_checks: BoundsCheckPolicy,
    /// Pointer policy.
    pub pointers: PointerPolicy,
    /// WASM sandbox restrictions.
    pub wasm_sandbox: WasmSandboxPolicy,
}

impl Default for SemanticSafetyPolicy {
    fn default() -> Self {
        Self {
            ownership_lifetimes: OwnershipLifetimePolicy::OwnedOnlyLexicalScopes,
            bounds_checks: BoundsCheckPolicy::Required,
            pointers: PointerPolicy::Forbidden,
            wasm_sandbox: WasmSandboxPolicy::default(),
        }
    }
}

/// ABI validation policy for `@export` functions exposed to the MPS/WASM bridge.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExportAbiPolicy {
    /// Require explicit type annotations on all exported function parameters.
    pub require_typed_params: bool,
    /// Allow `str` in exported parameters/returns.
    pub allow_string_abi: bool,
    /// Allow handle values in exported parameters/returns.
    pub allow_handle_abi: bool,
}

impl Default for ExportAbiPolicy {
    fn default() -> Self {
        Self {
            require_typed_params: true,
            allow_string_abi: false,
            allow_handle_abi: false,
        }
    }
}

/// Semantic analyzer configuration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SemanticConfig {
    /// Safety policy switches.
    pub safety: SemanticSafetyPolicy,
    /// Additional allowlisted external calls permitted under WASM sandboxing.
    pub external_call_allowlist: Vec<String>,
    /// Intrinsics/functions that allocate or return engine handles.
    pub handle_acquire_call_allowlist: Vec<String>,
    /// Intrinsics/functions that release engine handles.
    pub handle_release_call_allowlist: Vec<String>,
    /// `@export` ABI validation policy.
    pub export_abi: ExportAbiPolicy,
}

impl Default for SemanticConfig {
    fn default() -> Self {
        Self {
            safety: SemanticSafetyPolicy::default(),
            external_call_allowlist: Vec::new(),
            handle_acquire_call_allowlist: vec![
                "spawn_sprite".to_string(),
                "spawn_body".to_string(),
                "create_sprite".to_string(),
                "create_body".to_string(),
                "alloc_handle".to_string(),
            ],
            handle_release_call_allowlist: vec![
                "release_handle".to_string(),
                "destroy_handle".to_string(),
                "free_handle".to_string(),
            ],
            export_abi: ExportAbiPolicy::default(),
        }
    }
}

/// Resolved function signature used during call validation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FunctionSignature<'src> {
    /// Function name.
    pub name: &'src str,
    /// Parameter types (all params must be annotated for exact call checking).
    pub params: Vec<Option<SemanticType>>,
    /// Return type (`Unit` when omitted).
    pub return_type: SemanticType,
    /// `true` if function has `@export`.
    pub exported: bool,
    /// Name span.
    pub name_span: Span,
}

/// Per-function semantic summary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FunctionSemanticSummary<'src> {
    /// Function name.
    pub name: &'src str,
    /// Whether the function is exported to the WASM ABI layer.
    pub exported: bool,
    /// Number of parameters.
    pub param_count: usize,
    /// Number of local bindings introduced (including loop bindings).
    pub local_bindings: usize,
}

/// Safety posture summary emitted by the semantic analyzer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SemanticSafetySummary {
    /// Ownership/lifetime enforcement model.
    pub ownership_lifetimes: OwnershipLifetimePolicy,
    /// Bounds-check requirement.
    pub bounds_checks: BoundsCheckPolicy,
    /// How bounds safety is enforced in the validated program surface.
    pub bounds_enforcement: BoundsCheckEnforcement,
    /// Pointer policy.
    pub pointers: PointerPolicy,
    /// WASM sandbox policy in effect.
    pub wasm_sandbox: WasmSandboxPolicy,
}

/// Final semantic validation report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SemanticReport<'src> {
    /// Validated function signatures.
    pub signatures: Vec<FunctionSignature<'src>>,
    /// Per-function summaries.
    pub functions: Vec<FunctionSemanticSummary<'src>>,
    /// Exported function count.
    pub exported_functions: usize,
    /// Effective safety summary.
    pub safety: SemanticSafetySummary,
}

/// Semantic warning kind (non-fatal diagnostics).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SemanticWarningKind {
    /// `let` binding relied on inferred static type (`let x = 5` -> `int`).
    ImplicitTypeInference { inferred: SemanticType },
    /// Expression result is computed and discarded.
    DiscardedExpressionValue { ty: SemanticType },
    /// Non-exported function parameter omitted an explicit annotation.
    UnannotatedParameter,
    /// Semantic analyzer continued after a function-local error and skipped codegen for that item.
    FunctionValidationSkippedAfterError,
}

/// Semantic warning with source span.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SemanticWarning {
    /// Warning category.
    pub kind: SemanticWarningKind,
    /// Source span.
    pub span: Span,
}

/// Soft semantic-analysis outcome.
///
/// This is the engine-facing "safe fallback" API: it returns diagnostics and a partial report
/// instead of panicking or requiring callers to abort the game/tool loop immediately.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SemanticOutcome<'src> {
    /// Best-effort semantic report for validated items.
    pub report: SemanticReport<'src>,
    /// Fatal diagnostics that block codegen for one or more items.
    pub errors: Vec<SemanticError>,
    /// Non-fatal diagnostics.
    pub warnings: Vec<SemanticWarning>,
    /// `true` when no semantic errors were emitted and WASM codegen may proceed.
    pub can_codegen: bool,
}

/// Semantic error kind.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SemanticErrorKind {
    DuplicateFunction,
    DuplicateParameter,
    DuplicateLocalBinding,
    UnknownDecorator,
    DuplicateExportDecorator,
    ExportAbiMissingParamType,
    ExportAbiUnsupportedParamType,
    ExportAbiUnsupportedReturnType,
    UnknownVariable,
    UseAfterRelease,
    DoubleReleaseHandle,
    ReleaseNonHandle,
    InvalidHandleReleaseCall,
    InvalidHandleReleaseTarget,
    InvalidHandleReleaseContext,
    UnknownFunctionCall,
    UnsupportedCallTarget,
    TypeMismatch {
        expected: SemanticType,
        found: SemanticType,
    },
    InvalidConditionType {
        found: SemanticType,
    },
    InvalidUnaryOperand,
    InvalidBinaryOperands,
    InvalidRangeArgType {
        found: SemanticType,
    },
    InvalidRangeArity,
    UnsafeWasmIntrinsicForbidden,
    PointerIntrinsicForbidden,
}

/// Semantic error with source span.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SemanticError {
    /// Error category.
    pub kind: SemanticErrorKind,
    /// Source span.
    pub span: Span,
}

impl SemanticError {
    fn new(kind: SemanticErrorKind, span: Span) -> Self {
        Self { kind, span }
    }
}

impl fmt::Display for SemanticError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{:?} at line {}, column {}",
            self.kind, self.span.line, self.span.column
        )
    }
}

impl std::error::Error for SemanticError {}

/// Semantic analyzer for `.tlscript`.
pub struct SemanticAnalyzer {
    config: SemanticConfig,
    allowlisted_calls: HashSet<String>,
    handle_acquire_calls: HashSet<String>,
    handle_release_calls: HashSet<String>,
}

impl SemanticAnalyzer {
    /// Construct a semantic analyzer with a config.
    pub fn new(config: SemanticConfig) -> Self {
        let allowlisted_calls = config.external_call_allowlist.iter().cloned().collect();
        let handle_acquire_calls = config
            .handle_acquire_call_allowlist
            .iter()
            .cloned()
            .collect();
        let handle_release_calls = config
            .handle_release_call_allowlist
            .iter()
            .cloned()
            .collect();
        Self {
            config,
            allowlisted_calls,
            handle_acquire_calls,
            handle_release_calls,
        }
    }

    /// Validate a parsed module and produce a semantic report.
    pub fn analyze<'src>(
        &self,
        module: &Module<'src>,
    ) -> Result<SemanticReport<'src>, SemanticError> {
        let outcome = self.analyze_soft(module);
        if let Some(err) = outcome.errors.first() {
            return Err(err.clone());
        }
        Ok(outcome.report)
    }

    /// Soft semantic analysis entry point.
    ///
    /// Returns a best-effort report plus diagnostics, allowing the engine/editor to continue
    /// without hard aborts for non-critical script issues.
    pub fn analyze_soft<'src>(&self, module: &Module<'src>) -> SemanticOutcome<'src> {
        let mut errors = Vec::new();
        let mut warnings = Vec::new();

        let signatures = self.collect_signatures_soft(module, &mut errors, &mut warnings);
        let signature_map: HashMap<&'src str, &FunctionSignature<'src>> =
            signatures.iter().map(|sig| (sig.name, sig)).collect();

        let mut summaries = Vec::with_capacity(signatures.len());
        let mut exported_functions = 0usize;

        for item in &module.items {
            let Item::Function(func) = item;
            let Some(sig) = signature_map.get(func.name).copied() else {
                // Signature phase rejected this function already. Keep going for other functions.
                warnings.push(SemanticWarning {
                    kind: SemanticWarningKind::FunctionValidationSkippedAfterError,
                    span: func.span,
                });
                continue;
            };

            if sig.exported {
                exported_functions += 1;
            }

            match self.validate_function(func, sig, &signature_map, &mut warnings) {
                Ok(local_bindings) => summaries.push(FunctionSemanticSummary {
                    name: func.name,
                    exported: sig.exported,
                    param_count: func.params.len(),
                    local_bindings,
                }),
                Err(err) => {
                    errors.push(err);
                    warnings.push(SemanticWarning {
                        kind: SemanticWarningKind::FunctionValidationSkippedAfterError,
                        span: func.span,
                    });
                }
            }
        }

        let report = SemanticReport {
            signatures,
            functions: summaries,
            exported_functions,
            safety: SemanticSafetySummary {
                ownership_lifetimes: self.config.safety.ownership_lifetimes,
                bounds_checks: self.config.safety.bounds_checks,
                bounds_enforcement: BoundsCheckEnforcement::GuaranteedBySurfaceSyntaxV1,
                pointers: self.config.safety.pointers,
                wasm_sandbox: self.config.safety.wasm_sandbox.clone(),
            },
        };

        SemanticOutcome {
            can_codegen: errors.is_empty(),
            report,
            errors,
            warnings,
        }
    }

    fn collect_signatures_soft<'src>(
        &self,
        module: &Module<'src>,
        errors: &mut Vec<SemanticError>,
        warnings: &mut Vec<SemanticWarning>,
    ) -> Vec<FunctionSignature<'src>> {
        let mut seen = HashSet::<&'src str>::new();
        let mut signatures = Vec::with_capacity(module.items.len());

        for item in &module.items {
            let Item::Function(func) = item;
            if !seen.insert(func.name) {
                errors.push(SemanticError::new(
                    SemanticErrorKind::DuplicateFunction,
                    func.name_span,
                ));
                continue;
            }

            let mut saw_export = false;
            let mut signature_valid = true;
            for dec in &func.decorators {
                match &dec.kind {
                    DecoratorKind::Export => {
                        if saw_export {
                            errors.push(SemanticError::new(
                                SemanticErrorKind::DuplicateExportDecorator,
                                dec.span,
                            ));
                            signature_valid = false;
                        }
                        saw_export = true;
                    }
                    DecoratorKind::Named(name) => {
                        if is_compiler_hook_decorator(name) {
                            // `@net(...)` is validated by the dedicated compiler hook pass.
                        } else {
                            errors.push(SemanticError::new(
                                SemanticErrorKind::UnknownDecorator,
                                dec.span,
                            ));
                            signature_valid = false;
                        }
                    }
                }
            }

            let mut param_names = HashSet::<&'src str>::new();
            let mut params = Vec::with_capacity(func.params.len());
            for p in &func.params {
                if !param_names.insert(p.name) {
                    errors.push(SemanticError::new(
                        SemanticErrorKind::DuplicateParameter,
                        p.name_span,
                    ));
                    signature_valid = false;
                    continue;
                }

                let ty = p.ty.map(|t| SemanticType::from_type_name(t.kind));
                if ty.is_none() && !saw_export {
                    warnings.push(SemanticWarning {
                        kind: SemanticWarningKind::UnannotatedParameter,
                        span: p.name_span,
                    });
                }
                params.push(ty);
            }

            let return_type = func
                .return_type
                .map(|t| SemanticType::from_type_name(t.kind))
                .unwrap_or(SemanticType::Unit);

            if saw_export {
                if self.config.export_abi.require_typed_params && params.iter().any(|t| t.is_none())
                {
                    errors.push(SemanticError::new(
                        SemanticErrorKind::ExportAbiMissingParamType,
                        func.name_span,
                    ));
                    signature_valid = false;
                }
                for p in &func.params {
                    if let Some(ann) = p.ty {
                        let ty = SemanticType::from_type_name(ann.kind);
                        if !self.is_export_abi_type_allowed(ty) {
                            errors.push(SemanticError::new(
                                SemanticErrorKind::ExportAbiUnsupportedParamType,
                                ann.span,
                            ));
                            signature_valid = false;
                        }
                    }
                }
                if !self.is_export_abi_type_allowed(return_type)
                    && return_type != SemanticType::Unit
                {
                    let span = func.return_type.map(|t| t.span).unwrap_or(func.name_span);
                    errors.push(SemanticError::new(
                        SemanticErrorKind::ExportAbiUnsupportedReturnType,
                        span,
                    ));
                    signature_valid = false;
                }
            }

            if signature_valid {
                signatures.push(FunctionSignature {
                    name: func.name,
                    params,
                    return_type,
                    exported: saw_export,
                    name_span: func.name_span,
                });
            }
        }

        signatures
    }

    fn validate_function<'src>(
        &self,
        func: &FunctionDef<'src>,
        sig: &FunctionSignature<'src>,
        signatures: &HashMap<&'src str, &FunctionSignature<'src>>,
        warnings: &mut Vec<SemanticWarning>,
    ) -> Result<usize, SemanticError> {
        let mut scope = ScopeStack::default();
        let mut local_bindings = 0usize;

        for (param, param_ty) in func.params.iter().zip(sig.params.iter()) {
            let ty = param_ty.unwrap_or(SemanticType::Unit);
            if scope.insert(param.name, ty).is_err() {
                return Err(SemanticError::new(
                    SemanticErrorKind::DuplicateParameter,
                    param.name_span,
                ));
            }
            local_bindings += 1;
        }

        self.validate_block(
            &func.body,
            &mut scope,
            signatures,
            &mut local_bindings,
            warnings,
        )?;
        Ok(local_bindings)
    }

    fn validate_block<'src>(
        &self,
        block: &Block<'src>,
        scope: &mut ScopeStack<'src>,
        signatures: &HashMap<&'src str, &FunctionSignature<'src>>,
        local_bindings: &mut usize,
        warnings: &mut Vec<SemanticWarning>,
    ) -> Result<(), SemanticError> {
        scope.push();
        for stmt in &block.statements {
            self.validate_stmt(stmt, scope, signatures, local_bindings, warnings)?;
        }
        scope.pop();
        Ok(())
    }

    fn validate_stmt<'src>(
        &self,
        stmt: &Stmt<'src>,
        scope: &mut ScopeStack<'src>,
        signatures: &HashMap<&'src str, &FunctionSignature<'src>>,
        local_bindings: &mut usize,
        warnings: &mut Vec<SemanticWarning>,
    ) -> Result<(), SemanticError> {
        match stmt {
            Stmt::Let(s) => {
                let value_ty = self.infer_expr(&s.value, scope, signatures)?;
                if let Some(ann) = s.ty {
                    let expected = SemanticType::from_type_name(ann.kind);
                    if value_ty != expected {
                        return Err(SemanticError::new(
                            SemanticErrorKind::TypeMismatch {
                                expected,
                                found: value_ty,
                            },
                            s.value.span,
                        ));
                    }
                }
                let bind_ty =
                    s.ty.map(|t| SemanticType::from_type_name(t.kind))
                        .unwrap_or(value_ty);
                if s.ty.is_none() {
                    warnings.push(SemanticWarning {
                        kind: SemanticWarningKind::ImplicitTypeInference { inferred: bind_ty },
                        span: s.name_span,
                    });
                }
                if scope.insert_current(s.name, bind_ty).is_err() {
                    return Err(SemanticError::new(
                        SemanticErrorKind::DuplicateLocalBinding,
                        s.name_span,
                    ));
                }
                *local_bindings += 1;
            }
            Stmt::Assign(s) => {
                let Some(target) = scope.get(s.target) else {
                    return Err(SemanticError::new(
                        SemanticErrorKind::UnknownVariable,
                        s.target_span,
                    ));
                };
                if target.released {
                    return Err(SemanticError::new(
                        SemanticErrorKind::UseAfterRelease,
                        s.target_span,
                    ));
                }
                let value_ty = self.infer_expr(&s.value, scope, signatures)?;
                if target.ty != value_ty {
                    return Err(SemanticError::new(
                        SemanticErrorKind::TypeMismatch {
                            expected: target.ty,
                            found: value_ty,
                        },
                        s.value.span,
                    ));
                }
            }
            Stmt::If(s) => {
                for branch in &s.branches {
                    let cond_ty = self.infer_expr(&branch.condition, scope, signatures)?;
                    if cond_ty != SemanticType::Bool {
                        return Err(SemanticError::new(
                            SemanticErrorKind::InvalidConditionType { found: cond_ty },
                            branch.condition.span,
                        ));
                    }
                    self.validate_block(&branch.body, scope, signatures, local_bindings, warnings)?;
                }
                if let Some(else_block) = &s.else_block {
                    self.validate_block(else_block, scope, signatures, local_bindings, warnings)?;
                }
            }
            Stmt::While(s) => {
                let cond_ty = self.infer_expr(&s.condition, scope, signatures)?;
                if cond_ty != SemanticType::Bool {
                    return Err(SemanticError::new(
                        SemanticErrorKind::InvalidConditionType { found: cond_ty },
                        s.condition.span,
                    ));
                }
                self.validate_block(&s.body, scope, signatures, local_bindings, warnings)?;
            }
            Stmt::ForRange(s) => {
                let argc = s.range.args.len();
                if !(1..=3).contains(&argc) {
                    return Err(SemanticError::new(
                        SemanticErrorKind::InvalidRangeArity,
                        s.range.span,
                    ));
                }
                for arg in &s.range.args {
                    let ty = self.infer_expr(arg, scope, signatures)?;
                    if ty != SemanticType::Int {
                        return Err(SemanticError::new(
                            SemanticErrorKind::InvalidRangeArgType { found: ty },
                            arg.span,
                        ));
                    }
                }
                scope.push();
                if scope.insert_current(s.binding, SemanticType::Int).is_err() {
                    return Err(SemanticError::new(
                        SemanticErrorKind::DuplicateLocalBinding,
                        s.binding_span,
                    ));
                }
                *local_bindings += 1;
                for stmt in &s.body.statements {
                    self.validate_stmt(stmt, scope, signatures, local_bindings, warnings)?;
                }
                scope.pop();
            }
            Stmt::Expr(s) => {
                if self.try_validate_handle_release_stmt(&s.expr, scope, signatures)? {
                    return Ok(());
                }
                let ty = self.infer_expr(&s.expr, scope, signatures)?;
                if ty != SemanticType::Unit {
                    warnings.push(SemanticWarning {
                        kind: SemanticWarningKind::DiscardedExpressionValue { ty },
                        span: s.span,
                    });
                }
            }
        }
        Ok(())
    }

    fn infer_expr<'src>(
        &self,
        expr: &Expr<'src>,
        scope: &ScopeStack<'src>,
        signatures: &HashMap<&'src str, &FunctionSignature<'src>>,
    ) -> Result<SemanticType, SemanticError> {
        match &expr.kind {
            ExprKind::Identifier(name) => {
                let Some(binding) = scope.get(*name) else {
                    return Err(SemanticError::new(
                        SemanticErrorKind::UnknownVariable,
                        expr.span,
                    ));
                };
                if binding.released {
                    return Err(SemanticError::new(
                        SemanticErrorKind::UseAfterRelease,
                        expr.span,
                    ));
                }
                Ok(binding.ty)
            }
            ExprKind::IntegerLiteral(_) => Ok(SemanticType::Int),
            ExprKind::FloatLiteral(_) => Ok(SemanticType::Float),
            ExprKind::BoolLiteral(_) => Ok(SemanticType::Bool),
            ExprKind::StringLiteral(_) => Ok(SemanticType::Str),
            ExprKind::Grouping(inner) => self.infer_expr(inner, scope, signatures),
            ExprKind::Unary {
                op: UnaryOp::Neg,
                expr: inner,
            } => {
                let ty = self.infer_expr(inner, scope, signatures)?;
                match ty {
                    SemanticType::Int | SemanticType::Float => Ok(ty),
                    _ => Err(SemanticError::new(
                        SemanticErrorKind::InvalidUnaryOperand,
                        expr.span,
                    )),
                }
            }
            ExprKind::Binary { op, left, right } => {
                let lt = self.infer_expr(left, scope, signatures)?;
                let rt = self.infer_expr(right, scope, signatures)?;
                self.infer_binary(*op, lt, rt, expr.span)
            }
            ExprKind::Call { callee, args } => {
                self.infer_call(expr.span, callee, args, scope, signatures)
            }
        }
    }

    fn infer_call<'src>(
        &self,
        call_span: Span,
        callee: &Expr<'src>,
        args: &[Expr<'src>],
        scope: &ScopeStack<'src>,
        signatures: &HashMap<&'src str, &FunctionSignature<'src>>,
    ) -> Result<SemanticType, SemanticError> {
        let ExprKind::Identifier(name) = &callee.kind else {
            return Err(SemanticError::new(
                SemanticErrorKind::UnsupportedCallTarget,
                callee.span,
            ));
        };

        self.enforce_sandbox_call_rules(name, call_span)?;

        if self.is_handle_release_call(name) {
            // Handle releases must be standalone statements so the semantic pass can mark the
            // binding as released and prevent subsequent use-after-free.
            return Err(SemanticError::new(
                SemanticErrorKind::InvalidHandleReleaseContext,
                call_span,
            ));
        }

        for arg in args {
            let _ = self.infer_expr(arg, scope, signatures)?;
        }

        if let Some(sig) = signatures.get(name).copied() {
            for (idx, expected) in sig.params.iter().enumerate() {
                if let (Some(expected_ty), Some(arg)) = (expected, args.get(idx)) {
                    let found = self.infer_expr(arg, scope, signatures)?;
                    if found != *expected_ty {
                        return Err(SemanticError::new(
                            SemanticErrorKind::TypeMismatch {
                                expected: *expected_ty,
                                found,
                            },
                            arg.span,
                        ));
                    }
                }
            }
            return Ok(sig.return_type);
        }

        if self.is_handle_acquire_call(name) {
            return Ok(SemanticType::Handle);
        }

        if self.config.safety.wasm_sandbox.forbid_unknown_calls
            && !self.allowlisted_calls.contains(*name)
        {
            return Err(SemanticError::new(
                SemanticErrorKind::UnknownFunctionCall,
                callee.span,
            ));
        }

        Ok(SemanticType::Unit)
    }

    fn try_validate_handle_release_stmt<'src>(
        &self,
        expr: &Expr<'src>,
        scope: &mut ScopeStack<'src>,
        signatures: &HashMap<&'src str, &FunctionSignature<'src>>,
    ) -> Result<bool, SemanticError> {
        let ExprKind::Call { callee, args } = &expr.kind else {
            return Ok(false);
        };
        let ExprKind::Identifier(name) = &callee.kind else {
            return Ok(false);
        };
        if !self.is_handle_release_call(name) {
            return Ok(false);
        }

        self.enforce_sandbox_call_rules(name, expr.span)?;

        if args.len() != 1 {
            return Err(SemanticError::new(
                SemanticErrorKind::InvalidHandleReleaseCall,
                expr.span,
            ));
        }
        let arg = &args[0];
        let ExprKind::Identifier(binding_name) = arg.kind else {
            return Err(SemanticError::new(
                SemanticErrorKind::InvalidHandleReleaseTarget,
                arg.span,
            ));
        };

        let Some(binding) = scope.get(binding_name) else {
            return Err(SemanticError::new(
                SemanticErrorKind::UnknownVariable,
                arg.span,
            ));
        };
        if binding.released {
            return Err(SemanticError::new(
                SemanticErrorKind::DoubleReleaseHandle,
                arg.span,
            ));
        }
        if binding.ty != SemanticType::Handle {
            return Err(SemanticError::new(
                SemanticErrorKind::ReleaseNonHandle,
                arg.span,
            ));
        }

        // Validate additional nested expressions if this ever expands beyond identifier target.
        let _ = self.infer_expr(arg, scope, signatures)?;

        scope
            .mark_released(binding_name)
            .map_err(|kind| SemanticError::new(kind, arg.span))?;
        Ok(true)
    }

    #[inline]
    fn is_handle_acquire_call(&self, name: &str) -> bool {
        self.handle_acquire_calls.contains(name)
    }

    #[inline]
    fn is_handle_release_call(&self, name: &str) -> bool {
        self.handle_release_calls.contains(name)
    }

    #[inline]
    fn is_export_abi_type_allowed(&self, ty: SemanticType) -> bool {
        match ty {
            SemanticType::Int | SemanticType::Float | SemanticType::Bool | SemanticType::Unit => {
                true
            }
            SemanticType::Str => self.config.export_abi.allow_string_abi,
            SemanticType::Handle => self.config.export_abi.allow_handle_abi,
        }
    }

    fn enforce_sandbox_call_rules(&self, name: &str, span: Span) -> Result<(), SemanticError> {
        if self.config.safety.wasm_sandbox.forbid_pointer_intrinsics
            && is_pointer_like_intrinsic(name)
        {
            return Err(SemanticError::new(
                SemanticErrorKind::PointerIntrinsicForbidden,
                span,
            ));
        }

        if self.config.safety.wasm_sandbox.forbid_raw_memory_intrinsics
            && is_raw_memory_intrinsic(name)
        {
            return Err(SemanticError::new(
                SemanticErrorKind::UnsafeWasmIntrinsicForbidden,
                span,
            ));
        }

        Ok(())
    }

    fn infer_binary(
        &self,
        op: BinaryOp,
        left: SemanticType,
        right: SemanticType,
        span: Span,
    ) -> Result<SemanticType, SemanticError> {
        use BinaryOp as B;
        use SemanticType as T;

        match op {
            B::Add | B::Sub | B::Mul | B::Div | B::Mod => match (left, right) {
                (T::Int, T::Int) => Ok(T::Int),
                (T::Float, T::Float) => Ok(T::Float),
                _ => Err(SemanticError::new(
                    SemanticErrorKind::InvalidBinaryOperands,
                    span,
                )),
            },
            B::EqEq | B::NotEq => {
                if left == right {
                    Ok(T::Bool)
                } else {
                    Err(SemanticError::new(
                        SemanticErrorKind::InvalidBinaryOperands,
                        span,
                    ))
                }
            }
            B::Lt | B::LtEq | B::Gt | B::GtEq => match (left, right) {
                (T::Int, T::Int) | (T::Float, T::Float) => Ok(T::Bool),
                _ => Err(SemanticError::new(
                    SemanticErrorKind::InvalidBinaryOperands,
                    span,
                )),
            },
            B::AndAnd | B::OrOr => match (left, right) {
                (T::Bool, T::Bool) => Ok(T::Bool),
                _ => Err(SemanticError::new(
                    SemanticErrorKind::InvalidBinaryOperands,
                    span,
                )),
            },
        }
    }
}

impl Default for SemanticAnalyzer {
    fn default() -> Self {
        Self::new(SemanticConfig::default())
    }
}

#[derive(Debug, Clone, Default)]
struct ScopeStack<'src> {
    scopes: Vec<HashMap<&'src str, BindingInfo>>,
}

impl<'src> ScopeStack<'src> {
    fn push(&mut self) {
        self.scopes.push(HashMap::new());
    }

    fn pop(&mut self) {
        let _ = self.scopes.pop();
    }

    fn insert(&mut self, name: &'src str, ty: SemanticType) -> Result<(), ()> {
        if self.scopes.is_empty() {
            self.push();
        }
        if self
            .scopes
            .last()
            .is_some_and(|scope| scope.contains_key(name))
        {
            return Err(());
        }
        if let Some(scope) = self.scopes.last_mut() {
            scope.insert(
                name,
                BindingInfo {
                    ty,
                    released: false,
                },
            );
            return Ok(());
        }
        return Err(());
    }

    fn insert_current(&mut self, name: &'src str, ty: SemanticType) -> Result<(), ()> {
        self.insert(name, ty)
    }

    fn get(&self, name: &str) -> Option<BindingInfo> {
        self.scopes
            .iter()
            .rev()
            .find_map(|scope| scope.get(name).copied())
    }

    fn mark_released(&mut self, name: &str) -> Result<(), SemanticErrorKind> {
        for scope in self.scopes.iter_mut().rev() {
            if let Some(binding) = scope.get_mut(name) {
                if binding.released {
                    return Err(SemanticErrorKind::DoubleReleaseHandle);
                }
                if binding.ty != SemanticType::Handle {
                    return Err(SemanticErrorKind::ReleaseNonHandle);
                }
                binding.released = true;
                return Ok(());
            }
        }
        Err(SemanticErrorKind::UnknownVariable)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct BindingInfo {
    ty: SemanticType,
    released: bool,
}

fn is_pointer_like_intrinsic(name: &str) -> bool {
    matches!(
        name,
        "addr_of"
            | "deref"
            | "raw_ptr"
            | "unsafe_ptr"
            | "ptr_read"
            | "ptr_write"
            | "unchecked_index"
    ) || name.starts_with("ptr_")
}

#[inline]
fn is_compiler_hook_decorator(name: &str) -> bool {
    matches!(
        name,
        "net" | "parallel" | "main_thread" | "deterministic" | "reduce"
    )
}

fn is_raw_memory_intrinsic(name: &str) -> bool {
    matches!(
        name,
        "memory_grow"
            | "memory_copy"
            | "memory_fill"
            | "memory_init"
            | "table_grow"
            | "__wasm_memory_grow"
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tlscript::{Lexer, Parser};

    fn analyze_ok(src: &str) -> SemanticReport<'_> {
        let mut parser = Parser::new(Lexer::new(src));
        let module = parser.parse_module().expect("parse ok");
        SemanticAnalyzer::default()
            .analyze(&module)
            .expect("semantic ok")
    }

    fn analyze_err(src: &str) -> SemanticError {
        let mut parser = Parser::new(Lexer::new(src));
        let module = parser.parse_module().expect("parse ok");
        SemanticAnalyzer::default()
            .analyze(&module)
            .expect_err("semantic err")
    }

    fn analyze_soft(src: &str) -> SemanticOutcome<'_> {
        let mut parser = Parser::new(Lexer::new(src));
        let module = parser.parse_module().expect("parse ok");
        SemanticAnalyzer::default().analyze_soft(&module)
    }

    #[test]
    fn reports_required_safety_policies() {
        let report = analyze_ok(concat!("def update(dt: float):\n", "    let x: int = 1\n"));
        assert_eq!(
            report.safety.ownership_lifetimes,
            OwnershipLifetimePolicy::OwnedOnlyLexicalScopes
        );
        assert_eq!(report.safety.bounds_checks, BoundsCheckPolicy::Required);
        assert_eq!(
            report.safety.bounds_enforcement,
            BoundsCheckEnforcement::GuaranteedBySurfaceSyntaxV1
        );
        assert_eq!(report.safety.pointers, PointerPolicy::Forbidden);
        assert!(report.safety.wasm_sandbox.forbid_raw_memory_intrinsics);
    }

    #[test]
    fn lexical_scope_lifetime_is_enforced() {
        let err = analyze_err(concat!(
            "def f():\n",
            "    if true:\n",
            "        let x: int = 1\n",
            "    x = 2\n",
        ));
        assert_eq!(err.kind, SemanticErrorKind::UnknownVariable);
    }

    #[test]
    fn range_args_must_be_int_for_bounds_safe_iteration() {
        let err = analyze_err(concat!(
            "def f():\n",
            "    for i in range(0.0, 10):\n",
            "        noop()\n",
        ));
        assert!(matches!(
            err.kind,
            SemanticErrorKind::InvalidRangeArgType { .. }
        ));
    }

    #[test]
    fn pointer_intrinsics_are_forbidden() {
        let err = analyze_err(concat!("def f():\n", "    raw_ptr(1)\n"));
        assert_eq!(err.kind, SemanticErrorKind::PointerIntrinsicForbidden);
    }

    #[test]
    fn wasm_memory_intrinsics_are_forbidden() {
        let err = analyze_err(concat!("def f():\n", "    memory_grow(1)\n"));
        assert_eq!(err.kind, SemanticErrorKind::UnsafeWasmIntrinsicForbidden);
    }

    #[test]
    fn exported_function_is_wasm_sandbox_checked() {
        let err = analyze_err(concat!(
            "@export\n",
            "def update():\n",
            "    host_tick(1)\n",
        ));
        assert_eq!(err.kind, SemanticErrorKind::UnknownFunctionCall);
    }

    #[test]
    fn unknown_calls_can_be_allowlisted() {
        let mut parser = Parser::new(Lexer::new(concat!(
            "@export\n",
            "def update():\n",
            "    host_tick(1)\n",
        )));
        let module = parser.parse_module().expect("parse ok");
        let analyzer = SemanticAnalyzer::new(SemanticConfig {
            external_call_allowlist: vec!["host_tick".to_string()],
            ..SemanticConfig::default()
        });
        let report = analyzer.analyze(&module).expect("semantic ok");
        assert_eq!(report.exported_functions, 1);
    }

    #[test]
    fn export_abi_rejects_string_param_by_default() {
        let err = analyze_err(concat!(
            "@export\n",
            "def update(name: str):\n",
            "    let x = 1\n",
        ));
        assert_eq!(err.kind, SemanticErrorKind::ExportAbiUnsupportedParamType);
    }

    #[test]
    fn handle_release_prevents_use_after_free() {
        let err = analyze_err(concat!(
            "def f():\n",
            "    let h = spawn_sprite()\n",
            "    release_handle(h)\n",
            "    release_handle(h)\n",
        ));
        assert_eq!(err.kind, SemanticErrorKind::DoubleReleaseHandle);
    }

    #[test]
    fn handle_use_after_release_is_rejected() {
        let err = analyze_err(concat!(
            "def f():\n",
            "    let h = spawn_sprite()\n",
            "    release_handle(h)\n",
            "    let h2 = h\n",
        ));
        assert_eq!(err.kind, SemanticErrorKind::UseAfterRelease);
    }

    #[test]
    fn soft_analysis_returns_partial_report_and_diagnostics() {
        let outcome = analyze_soft(concat!(
            "@export\n",
            "def ok(dt: float):\n",
            "    let x = 1\n",
            "\n",
            "def bad():\n",
            "    raw_ptr(1)\n",
        ));
        assert!(!outcome.can_codegen);
        assert!(!outcome.errors.is_empty());
        assert_eq!(outcome.report.functions.len(), 1);
        assert!(outcome
            .warnings
            .iter()
            .any(|w| matches!(w.kind, SemanticWarningKind::ImplicitTypeInference { .. })));
    }
}
