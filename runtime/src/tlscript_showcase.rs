//! `.tlscript` showcase bootstrap runtime.
//!
//! This module wires parser/semantic/parallel-hook stages to a lightweight runtime evaluator for
//! show-scene parameter controls. It is intentionally constrained and soft-fail oriented:
//! - parses and validates script source in-memory (no disk I/O)
//! - extracts one exported entry function
//! - evaluates a safe subset of statements/expressions
//! - emits bounded `BounceTankRuntimePatch` values instead of mutating engine state directly
//! - surfaces parallel-dispatch decisions for `@parallel(domain=\"bodies\")` visibility

use std::collections::HashMap;

use tl_core::{
    annotate_typed_ir_with_parallel_hooks, lower_to_typed_ir, BinaryOp, Block, DecoratorKind, Expr,
    ExprKind, FunctionDef, Item, Lexer, Module, ParallelDispatchDecision, ParallelDispatchPlanner,
    ParallelDispatchPlannerConfig, ParallelExecutionPolicy, ParallelHookAnalyzer,
    ParallelHookOutcome, ParallelScheduleHint, Parser, SemanticAnalyzer, SemanticOutcome, Stmt,
    TypedIrModule, UnaryOp,
};

use crate::scene::BounceTankRuntimePatch;

const SHOWCASE_BUILTIN_CALLS: [&str; 9] = [
    "set_spawn_per_tick",
    "set_target_ball_count",
    "set_linear_damping",
    "set_ball_restitution",
    "set_wall_restitution",
    "set_initial_speed",
    "set_initial_speed_min",
    "set_initial_speed_max",
    "set_fbx_full_render",
];

/// Showcase script compiler settings.
#[derive(Debug, Clone)]
pub struct TlscriptShowcaseConfig {
    /// Exported function name to execute each frame.
    pub entry_function: String,
    /// Max statement/expression operations per frame evaluation.
    pub max_eval_steps: usize,
    /// Max loop iterations per `while` / `for range(...)`.
    pub max_loop_iterations: usize,
    /// Parallel dispatch planner config for `@parallel` telemetry.
    pub planner_config: ParallelDispatchPlannerConfig,
}

impl Default for TlscriptShowcaseConfig {
    fn default() -> Self {
        Self {
            entry_function: "showcase_tick".to_string(),
            max_eval_steps: 50_000,
            max_loop_iterations: 2_048,
            planner_config: ParallelDispatchPlannerConfig::default(),
        }
    }
}

/// Soft compile result for showcase scripts.
#[derive(Debug, Clone)]
pub struct TlscriptShowcaseCompileOutcome<'src> {
    pub program: Option<TlscriptShowcaseProgram<'src>>,
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
}

impl<'src> TlscriptShowcaseCompileOutcome<'src> {
    pub fn can_run(&self) -> bool {
        self.program.is_some() && self.errors.is_empty()
    }
}

/// Frame input bindings exposed to script evaluation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TlscriptShowcaseFrameInput {
    pub frame_index: u64,
    pub live_balls: usize,
    pub spawned_this_tick: usize,
}

/// Per-frame script evaluation output.
#[derive(Debug, Clone)]
pub struct TlscriptShowcaseFrameOutput {
    pub patch: BounceTankRuntimePatch,
    pub force_full_fbx_sphere: Option<bool>,
    pub dispatch_decision: Option<ParallelDispatchDecision>,
    pub warnings: Vec<String>,
    pub aborted_early: bool,
}

/// Compiled showcase program (borrowing source slices from the original script string).
#[derive(Debug, Clone)]
pub struct TlscriptShowcaseProgram<'src> {
    module: Module<'src>,
    typed_ir: TypedIrModule<'src>,
    hooks: ParallelHookOutcome<'src>,
    entry_item_index: usize,
    entry_ir_index: Option<usize>,
    entry_function_name: String,
    max_eval_steps: usize,
    max_loop_iterations: usize,
    planner: ParallelDispatchPlanner,
}

impl<'src> TlscriptShowcaseProgram<'src> {
    pub fn entry_function_name(&self) -> &str {
        &self.entry_function_name
    }

    /// Returns `true` when the entry function has a validated `@parallel` contract.
    pub fn has_parallel_contract(&self) -> bool {
        self.hooks.functions.iter().any(|hook| {
            hook.function == self.entry_function_name
                && matches!(hook.policy, ParallelExecutionPolicy::ParallelSafe)
        })
    }

    /// Returns the entry function's MPS schedule hint, if a parallel contract exists.
    pub fn parallel_schedule_hint(&self) -> Option<ParallelScheduleHint> {
        self.hooks
            .functions
            .iter()
            .find(|hook| {
                hook.function == self.entry_function_name
                    && matches!(hook.policy, ParallelExecutionPolicy::ParallelSafe)
            })
            .map(|hook| hook.schedule_hint)
    }

    /// Evaluate one script frame and emit a bounded runtime patch.
    pub fn evaluate_frame(&self, input: TlscriptShowcaseFrameInput) -> TlscriptShowcaseFrameOutput {
        let mut state = EvalState::new(self.max_eval_steps, self.max_loop_iterations);
        state.vars.insert(
            "frame".to_string(),
            DemoValue::Int(input.frame_index as i64),
        );
        state.vars.insert(
            "live_balls".to_string(),
            DemoValue::Int(input.live_balls as i64),
        );
        state.vars.insert(
            "spawned_this_tick".to_string(),
            DemoValue::Int(input.spawned_this_tick as i64),
        );

        let Item::Function(entry_fn) = &self.module.items[self.entry_item_index];
        self.exec_block(&entry_fn.body, &mut state);

        let dispatch_decision = self.entry_ir_index.map(|index| {
            self.planner
                .plan_for_function(&self.typed_ir.functions[index], input.live_balls)
        });

        TlscriptShowcaseFrameOutput {
            patch: state.patch,
            force_full_fbx_sphere: state.force_full_fbx_sphere,
            dispatch_decision,
            warnings: state.warnings,
            aborted_early: state.aborted_early,
        }
    }

    fn exec_block(&self, block: &Block<'src>, state: &mut EvalState) {
        for stmt in &block.statements {
            if !state.step() {
                state.warn("evaluation step budget exhausted; frame script aborted early");
                state.aborted_early = true;
                break;
            }
            self.exec_stmt(stmt, state);
            if state.aborted_early {
                break;
            }
        }
    }

    fn exec_stmt(&self, stmt: &Stmt<'src>, state: &mut EvalState) {
        match stmt {
            Stmt::Let(s) => {
                let value = self.eval_expr(&s.value, state);
                state.vars.insert(s.name.to_string(), value);
            }
            Stmt::Assign(s) => {
                let value = self.eval_expr(&s.value, state);
                state.vars.insert(s.target.to_string(), value);
            }
            Stmt::Expr(s) => {
                let _ = self.eval_expr(&s.expr, state);
            }
            Stmt::If(s) => {
                let mut executed = false;
                for branch in &s.branches {
                    if self.eval_expr(&branch.condition, state).to_bool() {
                        self.exec_block(&branch.body, state);
                        executed = true;
                        break;
                    }
                }
                if !executed {
                    if let Some(else_block) = &s.else_block {
                        self.exec_block(else_block, state);
                    }
                }
            }
            Stmt::While(s) => {
                let mut iterations = 0usize;
                while self.eval_expr(&s.condition, state).to_bool() {
                    if iterations >= state.max_loop_iterations {
                        state.warn("while loop exceeded max_loop_iterations; loop truncated");
                        break;
                    }
                    iterations += 1;
                    if !state.step() {
                        state.warn("evaluation step budget exhausted in while loop");
                        state.aborted_early = true;
                        break;
                    }
                    self.exec_block(&s.body, state);
                    if state.aborted_early {
                        break;
                    }
                }
            }
            Stmt::ForRange(s) => {
                let (start, end, step) = self.eval_range_args(&s.range.args, state);
                if step == 0 {
                    state.warn("range step evaluated to 0; loop skipped");
                    return;
                }

                let mut iterations = 0usize;
                let mut i = start;
                if step > 0 {
                    while i < end {
                        if iterations >= state.max_loop_iterations {
                            state.warn("for-range exceeded max_loop_iterations; loop truncated");
                            break;
                        }
                        iterations += 1;
                        if !state.step() {
                            state.warn("evaluation step budget exhausted in for-range");
                            state.aborted_early = true;
                            break;
                        }
                        state.vars.insert(s.binding.to_string(), DemoValue::Int(i));
                        self.exec_block(&s.body, state);
                        if state.aborted_early {
                            break;
                        }
                        i = i.saturating_add(step);
                    }
                } else {
                    while i > end {
                        if iterations >= state.max_loop_iterations {
                            state.warn("for-range exceeded max_loop_iterations; loop truncated");
                            break;
                        }
                        iterations += 1;
                        if !state.step() {
                            state.warn("evaluation step budget exhausted in for-range");
                            state.aborted_early = true;
                            break;
                        }
                        state.vars.insert(s.binding.to_string(), DemoValue::Int(i));
                        self.exec_block(&s.body, state);
                        if state.aborted_early {
                            break;
                        }
                        i = i.saturating_add(step);
                    }
                }
            }
        }
    }

    fn eval_expr(&self, expr: &Expr<'src>, state: &mut EvalState) -> DemoValue {
        if !state.step() {
            state.aborted_early = true;
            return DemoValue::Int(0);
        }

        match &expr.kind {
            ExprKind::Identifier(name) => state.vars.get(*name).cloned().unwrap_or_else(|| {
                state.warn(format!("unknown variable '{name}' -> default 0"));
                DemoValue::Int(0)
            }),
            ExprKind::IntegerLiteral(raw) => raw
                .parse::<i64>()
                .map(DemoValue::Int)
                .unwrap_or(DemoValue::Int(0)),
            ExprKind::FloatLiteral(raw) => raw
                .parse::<f64>()
                .map(DemoValue::Float)
                .unwrap_or(DemoValue::Float(0.0)),
            ExprKind::BoolLiteral(v) => DemoValue::Bool(*v),
            ExprKind::StringLiteral(raw) => DemoValue::Str((*raw).to_string()),
            ExprKind::Grouping(inner) => self.eval_expr(inner, state),
            ExprKind::Unary { op, expr } => {
                let value = self.eval_expr(expr, state);
                match op {
                    UnaryOp::Neg => match value {
                        DemoValue::Int(v) => DemoValue::Int(v.saturating_neg()),
                        DemoValue::Float(v) => DemoValue::Float(-v),
                        _ => {
                            state.warn("invalid unary '-' on non-numeric value");
                            DemoValue::Int(0)
                        }
                    },
                }
            }
            ExprKind::Binary { op, left, right } => match op {
                BinaryOp::AndAnd => {
                    if !self.eval_expr(left, state).to_bool() {
                        DemoValue::Bool(false)
                    } else {
                        DemoValue::Bool(self.eval_expr(right, state).to_bool())
                    }
                }
                BinaryOp::OrOr => {
                    if self.eval_expr(left, state).to_bool() {
                        DemoValue::Bool(true)
                    } else {
                        DemoValue::Bool(self.eval_expr(right, state).to_bool())
                    }
                }
                _ => {
                    let l = self.eval_expr(left, state);
                    let r = self.eval_expr(right, state);
                    eval_binary(*op, l, r, state)
                }
            },
            ExprKind::Call { callee, args } => {
                let ExprKind::Identifier(name) = &callee.kind else {
                    state.warn("unsupported non-identifier call target");
                    return DemoValue::Int(0);
                };
                let values = args
                    .iter()
                    .map(|arg| self.eval_expr(arg, state))
                    .collect::<Vec<_>>();
                apply_builtin_patch_call(name, &values, state);
                DemoValue::Int(0)
            }
        }
    }

    fn eval_range_args(&self, args: &[Expr<'src>], state: &mut EvalState) -> (i64, i64, i64) {
        match args.len() {
            1 => (0, self.eval_expr(&args[0], state).to_i64(), 1),
            2 => (
                self.eval_expr(&args[0], state).to_i64(),
                self.eval_expr(&args[1], state).to_i64(),
                1,
            ),
            3 => (
                self.eval_expr(&args[0], state).to_i64(),
                self.eval_expr(&args[1], state).to_i64(),
                self.eval_expr(&args[2], state).to_i64(),
            ),
            _ => {
                state.warn("invalid range arity; expected 1..=3 args");
                (0, 0, 1)
            }
        }
    }
}

#[derive(Debug, Clone)]
struct EvalState {
    vars: HashMap<String, DemoValue>,
    patch: BounceTankRuntimePatch,
    force_full_fbx_sphere: Option<bool>,
    warnings: Vec<String>,
    steps: usize,
    max_steps: usize,
    max_loop_iterations: usize,
    aborted_early: bool,
}

impl EvalState {
    fn new(max_steps: usize, max_loop_iterations: usize) -> Self {
        Self {
            vars: HashMap::new(),
            patch: BounceTankRuntimePatch::default(),
            force_full_fbx_sphere: None,
            warnings: Vec::new(),
            steps: 0,
            max_steps: max_steps.max(1),
            max_loop_iterations: max_loop_iterations.max(1),
            aborted_early: false,
        }
    }

    fn step(&mut self) -> bool {
        if self.steps >= self.max_steps {
            return false;
        }
        self.steps += 1;
        true
    }

    fn warn(&mut self, message: impl Into<String>) {
        self.warnings.push(message.into());
    }
}

#[derive(Debug, Clone, PartialEq)]
enum DemoValue {
    Int(i64),
    Float(f64),
    Bool(bool),
    Str(String),
}

impl DemoValue {
    fn to_i64(&self) -> i64 {
        match self {
            Self::Int(v) => *v,
            Self::Float(v) => *v as i64,
            Self::Bool(v) => i64::from(*v),
            Self::Str(_) => 0,
        }
    }

    fn to_f64(&self) -> f64 {
        match self {
            Self::Int(v) => *v as f64,
            Self::Float(v) => *v,
            Self::Bool(v) => f64::from(*v as u8),
            Self::Str(_) => 0.0,
        }
    }

    fn to_bool(&self) -> bool {
        match self {
            Self::Bool(v) => *v,
            Self::Int(v) => *v != 0,
            Self::Float(v) => v.abs() > f64::EPSILON,
            Self::Str(v) => !v.is_empty(),
        }
    }
}

fn eval_binary(
    op: BinaryOp,
    left: DemoValue,
    right: DemoValue,
    state: &mut EvalState,
) -> DemoValue {
    match op {
        BinaryOp::Add => numeric_binary(left, right, |a, b| a + b),
        BinaryOp::Sub => numeric_binary(left, right, |a, b| a - b),
        BinaryOp::Mul => numeric_binary(left, right, |a, b| a * b),
        BinaryOp::Div => {
            let denom = right.to_f64();
            if denom.abs() <= f64::EPSILON {
                state.warn("division by zero; defaulted to 0");
                DemoValue::Int(0)
            } else {
                DemoValue::Float(left.to_f64() / denom)
            }
        }
        BinaryOp::Mod => {
            let denom = right.to_i64();
            if denom == 0 {
                state.warn("modulo by zero; defaulted to 0");
                DemoValue::Int(0)
            } else {
                DemoValue::Int(left.to_i64() % denom)
            }
        }
        BinaryOp::EqEq => DemoValue::Bool(left.to_f64() == right.to_f64()),
        BinaryOp::NotEq => DemoValue::Bool(left.to_f64() != right.to_f64()),
        BinaryOp::Lt => DemoValue::Bool(left.to_f64() < right.to_f64()),
        BinaryOp::LtEq => DemoValue::Bool(left.to_f64() <= right.to_f64()),
        BinaryOp::Gt => DemoValue::Bool(left.to_f64() > right.to_f64()),
        BinaryOp::GtEq => DemoValue::Bool(left.to_f64() >= right.to_f64()),
        BinaryOp::AndAnd | BinaryOp::OrOr => DemoValue::Bool(false),
    }
}

fn numeric_binary(left: DemoValue, right: DemoValue, f: impl FnOnce(f64, f64) -> f64) -> DemoValue {
    match (left, right) {
        (DemoValue::Int(a), DemoValue::Int(b)) => DemoValue::Int(f(a as f64, b as f64) as i64),
        (l, r) => DemoValue::Float(f(l.to_f64(), r.to_f64())),
    }
}

fn apply_builtin_patch_call(name: &str, args: &[DemoValue], state: &mut EvalState) {
    match name {
        "set_spawn_per_tick" => match args {
            [v] => state.patch.spawn_per_tick = Some(v.to_i64().max(0) as usize),
            _ => state.warn("set_spawn_per_tick expects 1 arg"),
        },
        "set_target_ball_count" => match args {
            [v] => state.patch.target_ball_count = Some(v.to_i64().max(0) as usize),
            _ => state.warn("set_target_ball_count expects 1 arg"),
        },
        "set_linear_damping" => match args {
            [v] => state.patch.linear_damping = Some(v.to_f64() as f32),
            _ => state.warn("set_linear_damping expects 1 arg"),
        },
        "set_ball_restitution" => match args {
            [v] => state.patch.ball_restitution = Some(v.to_f64() as f32),
            _ => state.warn("set_ball_restitution expects 1 arg"),
        },
        "set_wall_restitution" => match args {
            [v] => state.patch.wall_restitution = Some(v.to_f64() as f32),
            _ => state.warn("set_wall_restitution expects 1 arg"),
        },
        "set_initial_speed" => match args {
            [min_v, max_v] => {
                state.patch.initial_speed_min = Some(min_v.to_f64() as f32);
                state.patch.initial_speed_max = Some(max_v.to_f64() as f32);
            }
            _ => state.warn("set_initial_speed expects 2 args"),
        },
        "set_initial_speed_min" => match args {
            [v] => state.patch.initial_speed_min = Some(v.to_f64() as f32),
            _ => state.warn("set_initial_speed_min expects 1 arg"),
        },
        "set_initial_speed_max" => match args {
            [v] => state.patch.initial_speed_max = Some(v.to_f64() as f32),
            _ => state.warn("set_initial_speed_max expects 1 arg"),
        },
        "set_fbx_full_render" => match args {
            [v] => state.force_full_fbx_sphere = Some(v.to_bool()),
            _ => state.warn("set_fbx_full_render expects 1 arg"),
        },
        _ => state.warn(format!("unknown showcase builtin '{name}'")),
    }
}

/// Compile a `.tlscript` source string into a showcase program.
pub fn compile_tlscript_showcase<'src>(
    source: &'src str,
    config: TlscriptShowcaseConfig,
) -> TlscriptShowcaseCompileOutcome<'src> {
    let mut warnings = Vec::new();
    let mut errors = Vec::new();

    let mut parser = Parser::new(Lexer::new(source));
    let module = match parser.parse_module() {
        Ok(module) => module,
        Err(err) => {
            errors.push(format!("parse error: {err}"));
            return TlscriptShowcaseCompileOutcome {
                program: None,
                errors,
                warnings,
            };
        }
    };

    let mut semantic_config = tl_core::SemanticConfig::default();
    semantic_config.external_call_allowlist.extend(
        SHOWCASE_BUILTIN_CALLS
            .iter()
            .map(|name| (*name).to_string()),
    );
    semantic_config.external_call_allowlist.sort();
    semantic_config.external_call_allowlist.dedup();

    let semantic_outcome: SemanticOutcome<'src> =
        SemanticAnalyzer::new(semantic_config).analyze_soft(&module);
    warnings.extend(
        semantic_outcome
            .warnings
            .iter()
            .map(|w| format!("semantic warning: {:?} at {:?}", w.kind, w.span)),
    );
    errors.extend(
        semantic_outcome
            .errors
            .iter()
            .map(|e| format!("semantic error: {:?} at {:?}", e.kind, e.span)),
    );
    if !semantic_outcome.can_codegen {
        return TlscriptShowcaseCompileOutcome {
            program: None,
            errors,
            warnings,
        };
    }

    let hooks = ParallelHookAnalyzer::new().analyze(&module, &semantic_outcome.report);
    warnings.extend(
        hooks
            .warnings
            .iter()
            .map(|w| format!("parallel warning: {:?} at {:?}", w.kind, w.span)),
    );
    errors.extend(
        hooks
            .errors
            .iter()
            .map(|e| format!("parallel error: {:?} at {:?}", e.kind, e.span)),
    );
    if !hooks.can_emit_parallel_metadata() {
        return TlscriptShowcaseCompileOutcome {
            program: None,
            errors,
            warnings,
        };
    }

    let entry_item_index = module.items.iter().position(|item| {
        let Item::Function(f) = item;
        f.name == config.entry_function
    });
    let Some(entry_item_index) = entry_item_index else {
        errors.push(format!(
            "entry function '{}' was not found",
            config.entry_function
        ));
        return TlscriptShowcaseCompileOutcome {
            program: None,
            errors,
            warnings,
        };
    };

    let Item::Function(entry_fn) = &module.items[entry_item_index];
    if !has_export_decorator(entry_fn) {
        warnings.push(format!(
            "entry function '{}' is not exported; runtime can still evaluate but WASM ABI dispatch is unavailable",
            entry_fn.name
        ));
    }
    let has_parallel_bodies = hooks.functions.iter().any(|hook| {
        hook.function == entry_fn.name
            && matches!(hook.policy, ParallelExecutionPolicy::ParallelSafe)
            && hook.domain == Some("bodies")
    });
    if !has_parallel_bodies {
        warnings.push(format!(
            "entry function '{}' has no @parallel(domain=\"bodies\") contract; planner may fall back to serial/main-thread",
            entry_fn.name
        ));
    }

    let mut typed_ir = match lower_to_typed_ir(&module, &semantic_outcome.report) {
        Ok(ir) => ir,
        Err(err) => {
            errors.push(format!("typed IR lowering error: {err:?}"));
            return TlscriptShowcaseCompileOutcome {
                program: None,
                errors,
                warnings,
            };
        }
    };
    annotate_typed_ir_with_parallel_hooks(&mut typed_ir, &hooks);
    let entry_ir_index = typed_ir
        .functions
        .iter()
        .position(|f| f.name == config.entry_function);

    let program = TlscriptShowcaseProgram {
        module,
        typed_ir,
        hooks,
        entry_item_index,
        entry_ir_index,
        entry_function_name: config.entry_function,
        max_eval_steps: config.max_eval_steps.max(1),
        max_loop_iterations: config.max_loop_iterations.max(1),
        planner: ParallelDispatchPlanner::new(config.planner_config),
    };

    TlscriptShowcaseCompileOutcome {
        program: Some(program),
        errors,
        warnings,
    }
}

fn has_export_decorator(func: &FunctionDef<'_>) -> bool {
    func.decorators
        .iter()
        .any(|dec| matches!(dec.kind, DecoratorKind::Export))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compiles_and_emits_parallel_dispatch_decision() {
        let src = concat!(
            "@export\n",
            "@parallel(domain=\"bodies\", read=\"transform,aabb\", write=\"velocity\", chunk=128)\n",
            "def showcase_tick(frame: int, live_balls: int):\n",
            "    if live_balls < 500:\n",
            "        set_spawn_per_tick(640)\n",
            "    else:\n",
            "        set_spawn_per_tick(64)\n",
            "    let d: float = 0.018\n",
            "    set_linear_damping(d)\n",
        );
        let outcome = compile_tlscript_showcase(src, Default::default());
        assert!(outcome.errors.is_empty());
        let program = outcome.program.as_ref().expect("program");
        assert!(program.has_parallel_contract());

        let frame_low = program.evaluate_frame(TlscriptShowcaseFrameInput {
            frame_index: 1,
            live_balls: 300,
            spawned_this_tick: 100,
        });
        assert_eq!(frame_low.patch.spawn_per_tick, Some(640));
        assert!(frame_low.patch.linear_damping.unwrap_or(0.0) > 0.0);
        assert!(frame_low.dispatch_decision.is_some());

        let frame_high = program.evaluate_frame(TlscriptShowcaseFrameInput {
            frame_index: 2,
            live_balls: 1_200,
            spawned_this_tick: 40,
        });
        assert_eq!(frame_high.patch.spawn_per_tick, Some(64));
    }

    #[test]
    fn reports_missing_entry_function() {
        let src = concat!(
            "@export\n",
            "def tick(dt: float):\n",
            "    set_spawn_per_tick(100)\n",
        );
        let outcome = compile_tlscript_showcase(src, Default::default());
        assert!(outcome.program.is_none());
        assert!(!outcome.errors.is_empty());
    }

    #[test]
    fn supports_full_fbx_render_builtin_toggle() {
        let src = concat!(
            "@export\n",
            "def showcase_tick(frame: int, live_balls: int, spawned_this_tick: int):\n",
            "    set_fbx_full_render(true)\n",
        );
        let outcome = compile_tlscript_showcase(src, Default::default());
        assert!(outcome.errors.is_empty());
        let program = outcome.program.as_ref().expect("program");
        let out = program.evaluate_frame(TlscriptShowcaseFrameInput {
            frame_index: 0,
            live_balls: 0,
            spawned_this_tick: 0,
        });
        assert_eq!(out.force_full_fbx_sphere, Some(true));
    }
}
