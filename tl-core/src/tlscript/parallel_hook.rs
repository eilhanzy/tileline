//! Parallel execution contract analyzer for `.tlscript`.
//!
//! This pass validates decorators such as `@parallel`, `@main_thread`, `@deterministic`, and
//! `@reduce(...)` and produces execution metadata that runtime/MPS planning can consume.
//! It intentionally runs as a separate compiler hook on top of parser + semantic output so the core
//! semantic pass stays focused on type and safety correctness.

use std::collections::{HashMap, HashSet};

use super::ast::{Decorator, DecoratorArg, DecoratorKind, DecoratorValue, Item, Module};
use super::semantic::SemanticReport;
use super::token::Span;
use super::typed_ir::{IrExecutionPolicy, IrReduceKind, IrScheduleHint, TypedIrModule};

/// Execution policy emitted by the parallel hook.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParallelExecutionPolicy {
    /// Function may be partitioned into MPS tasks under the declared effect constraints.
    ParallelSafe,
    /// Function must remain on the main/game thread.
    MainThreadOnly,
}

/// MPS scheduling preference hint derived from script decorators.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParallelScheduleHint {
    Auto,
    Performance,
    Efficient,
}

/// Deterministic reduction/merge hint for parallel function outputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParallelReduceKind {
    Sum,
    Min,
    Max,
    BitOr,
    BitAnd,
}

/// Validated parallel execution contract for one script function.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParallelFunctionHook<'src> {
    /// Script function name.
    pub function: &'src str,
    /// Whether the function is also exported to the WASM ABI/MPS entrypoint surface.
    pub exported: bool,
    /// Execution policy (parallel-safe vs main-thread-only).
    pub policy: ParallelExecutionPolicy,
    /// Optional partition domain (`bodies`, `particles`, `chunks`, etc).
    pub domain: Option<&'src str>,
    /// Declared read-effect domains.
    pub read_effects: Vec<&'src str>,
    /// Declared write-effect domains.
    pub write_effects: Vec<&'src str>,
    /// Deterministic execution contract flag.
    pub deterministic: bool,
    /// Optional static chunk-size hint for partitioning.
    pub chunk_hint: Option<u16>,
    /// Core preference hint for MPS lane routing.
    pub schedule_hint: ParallelScheduleHint,
    /// Optional deterministic reduction hint.
    pub reduce: Option<ParallelReduceKind>,
    /// Span covering the primary execution decorator.
    pub span: Span,
}

/// Non-fatal diagnostics for the parallel hook.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParallelHookWarningKind {
    /// Parallelized function is not exported; runtime cannot dispatch it directly via MPS yet.
    ParallelFunctionMissingExport,
    /// Parallel function did not declare any write effects (read-only tasks are legal but often
    /// indicate a missing `write=` list when mutating host state indirectly).
    ParallelFunctionHasNoWriteEffects,
    /// A deterministic reduction was requested but `@deterministic` was omitted. The hook keeps the
    /// reduction and upgrades determinism internally.
    DeterministicReductionImplied,
}

/// Parallel hook warning with source span.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParallelHookWarning {
    pub kind: ParallelHookWarningKind,
    pub span: Span,
}

/// Fatal diagnostics for parallel decorator validation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParallelHookErrorKind {
    DuplicateParallelDecorator,
    DuplicateMainThreadDecorator,
    DuplicateDeterministicDecorator,
    DuplicateReduceDecorator,
    InvalidParallelDecoratorUsage,
    InvalidMainThreadDecoratorUsage,
    InvalidDeterministicDecoratorUsage,
    InvalidReduceDecoratorUsage,
    MutuallyExclusiveExecutionDecorators,
    MissingParallelDomain,
    DuplicateParallelArgKey,
    DuplicateParallelArgFlag,
    UnsupportedParallelArgKey,
    UnsupportedParallelFlag,
    InvalidChunkHint,
    UnsupportedScheduleHint,
    UnsupportedReduceKind,
    ReduceWithoutParallel,
    MissingFunctionSemanticSignature,
}

/// Parallel hook error with source span.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParallelHookError {
    pub kind: ParallelHookErrorKind,
    pub span: Span,
}

/// Soft result of the parallel execution hook analysis.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ParallelHookOutcome<'src> {
    /// Validated per-function execution contracts.
    pub functions: Vec<ParallelFunctionHook<'src>>,
    /// Fatal decorator validation errors.
    pub errors: Vec<ParallelHookError>,
    /// Non-fatal diagnostics.
    pub warnings: Vec<ParallelHookWarning>,
}

impl<'src> ParallelHookOutcome<'src> {
    /// `true` when no fatal errors were emitted.
    pub fn can_emit_parallel_metadata(&self) -> bool {
        self.errors.is_empty()
    }
}

/// Compiler hook analyzer for `.tlscript` parallel execution decorators.
#[derive(Debug, Default)]
pub struct ParallelHookAnalyzer;

impl ParallelHookAnalyzer {
    /// Create a new analyzer instance.
    pub fn new() -> Self {
        Self
    }

    /// Validate decorators and extract per-function parallel contracts.
    pub fn analyze<'src>(
        &self,
        module: &Module<'src>,
        semantic: &SemanticReport<'src>,
    ) -> ParallelHookOutcome<'src> {
        let mut out = ParallelHookOutcome::default();
        let sig_map: HashMap<&'src str, _> =
            semantic.signatures.iter().map(|s| (s.name, s)).collect();

        for item in &module.items {
            let Item::Function(func) = item;

            let Some(sig) = sig_map.get(func.name).copied() else {
                if has_parallel_family_decorator(&func.decorators) {
                    out.errors.push(ParallelHookError {
                        kind: ParallelHookErrorKind::MissingFunctionSemanticSignature,
                        span: func.name_span,
                    });
                }
                continue;
            };

            let mut parallel_dec: Option<&Decorator<'src>> = None;
            let mut main_thread_dec: Option<&Decorator<'src>> = None;
            let mut deterministic_dec: Option<&Decorator<'src>> = None;
            let mut reduce_dec: Option<&Decorator<'src>> = None;

            for dec in &func.decorators {
                match &dec.kind {
                    DecoratorKind::Named(name) if *name == "parallel" => {
                        if parallel_dec.replace(dec).is_some() {
                            out.errors.push(ParallelHookError {
                                kind: ParallelHookErrorKind::DuplicateParallelDecorator,
                                span: dec.span,
                            });
                        }
                    }
                    DecoratorKind::Named(name) if *name == "main_thread" => {
                        if main_thread_dec.replace(dec).is_some() {
                            out.errors.push(ParallelHookError {
                                kind: ParallelHookErrorKind::DuplicateMainThreadDecorator,
                                span: dec.span,
                            });
                        }
                    }
                    DecoratorKind::Named(name) if *name == "deterministic" => {
                        if deterministic_dec.replace(dec).is_some() {
                            out.errors.push(ParallelHookError {
                                kind: ParallelHookErrorKind::DuplicateDeterministicDecorator,
                                span: dec.span,
                            });
                        }
                    }
                    DecoratorKind::Named(name) if *name == "reduce" => {
                        if reduce_dec.replace(dec).is_some() {
                            out.errors.push(ParallelHookError {
                                kind: ParallelHookErrorKind::DuplicateReduceDecorator,
                                span: dec.span,
                            });
                        }
                    }
                    _ => {}
                }
            }

            if parallel_dec.is_some() && main_thread_dec.is_some() {
                out.errors.push(ParallelHookError {
                    kind: ParallelHookErrorKind::MutuallyExclusiveExecutionDecorators,
                    span: join_spans(parallel_dec.unwrap().span, main_thread_dec.unwrap().span),
                });
                continue;
            }

            if let Some(dec) = main_thread_dec {
                if !dec.args.is_empty() {
                    out.errors.push(ParallelHookError {
                        kind: ParallelHookErrorKind::InvalidMainThreadDecoratorUsage,
                        span: dec.span,
                    });
                    continue;
                }
                if let Some(det) = deterministic_dec {
                    if !det.args.is_empty() {
                        out.errors.push(ParallelHookError {
                            kind: ParallelHookErrorKind::InvalidDeterministicDecoratorUsage,
                            span: det.span,
                        });
                        continue;
                    }
                }
                if let Some(red) = reduce_dec {
                    out.errors.push(ParallelHookError {
                        kind: ParallelHookErrorKind::ReduceWithoutParallel,
                        span: red.span,
                    });
                    continue;
                }
                out.functions.push(ParallelFunctionHook {
                    function: func.name,
                    exported: sig.exported,
                    policy: ParallelExecutionPolicy::MainThreadOnly,
                    domain: None,
                    read_effects: Vec::new(),
                    write_effects: Vec::new(),
                    deterministic: deterministic_dec.is_some(),
                    chunk_hint: None,
                    schedule_hint: ParallelScheduleHint::Auto,
                    reduce: None,
                    span: dec.span,
                });
                continue;
            }

            let Some(parallel_dec) = parallel_dec else {
                // `@deterministic` or `@reduce` alone is invalid usage for now.
                if let Some(det) = deterministic_dec {
                    out.errors.push(ParallelHookError {
                        kind: ParallelHookErrorKind::InvalidDeterministicDecoratorUsage,
                        span: det.span,
                    });
                }
                if let Some(red) = reduce_dec {
                    out.errors.push(ParallelHookError {
                        kind: ParallelHookErrorKind::ReduceWithoutParallel,
                        span: red.span,
                    });
                }
                continue;
            };

            if let Some(det) = deterministic_dec {
                if !det.args.is_empty() {
                    out.errors.push(ParallelHookError {
                        kind: ParallelHookErrorKind::InvalidDeterministicDecoratorUsage,
                        span: det.span,
                    });
                    continue;
                }
            }

            let mut contract = match parse_parallel_decorator(parallel_dec) {
                Ok(v) => v,
                Err(err) => {
                    out.errors.push(err);
                    continue;
                }
            };

            if contract.domain.is_none() {
                out.errors.push(ParallelHookError {
                    kind: ParallelHookErrorKind::MissingParallelDomain,
                    span: parallel_dec.span,
                });
                continue;
            }

            if let Some(red) = reduce_dec {
                match parse_reduce_decorator(red) {
                    Ok(kind) => contract.reduce = Some(kind),
                    Err(err) => {
                        out.errors.push(err);
                        continue;
                    }
                }
            }

            if contract.reduce.is_some() && !contract.deterministic {
                contract.deterministic = true;
                out.warnings.push(ParallelHookWarning {
                    kind: ParallelHookWarningKind::DeterministicReductionImplied,
                    span: parallel_dec.span,
                });
            }

            if !sig.exported {
                out.warnings.push(ParallelHookWarning {
                    kind: ParallelHookWarningKind::ParallelFunctionMissingExport,
                    span: parallel_dec.span,
                });
            }
            if contract.write_effects.is_empty() {
                out.warnings.push(ParallelHookWarning {
                    kind: ParallelHookWarningKind::ParallelFunctionHasNoWriteEffects,
                    span: parallel_dec.span,
                });
            }

            out.functions.push(ParallelFunctionHook {
                function: func.name,
                exported: sig.exported,
                policy: ParallelExecutionPolicy::ParallelSafe,
                domain: contract.domain,
                read_effects: contract.read_effects,
                write_effects: contract.write_effects,
                deterministic: contract.deterministic,
                chunk_hint: contract.chunk_hint,
                schedule_hint: contract.schedule_hint,
                reduce: contract.reduce,
                span: parallel_dec.span,
            });
        }

        out
    }
}

#[derive(Debug, Default)]
struct ParsedParallelContract<'src> {
    domain: Option<&'src str>,
    read_effects: Vec<&'src str>,
    write_effects: Vec<&'src str>,
    deterministic: bool,
    chunk_hint: Option<u16>,
    schedule_hint: ParallelScheduleHint,
    reduce: Option<ParallelReduceKind>,
}

impl Default for ParallelScheduleHint {
    fn default() -> Self {
        Self::Auto
    }
}

fn has_parallel_family_decorator(decorators: &[Decorator<'_>]) -> bool {
    decorators.iter().any(|d| {
        matches!(
            d.kind,
            DecoratorKind::Named("parallel")
                | DecoratorKind::Named("main_thread")
                | DecoratorKind::Named("deterministic")
                | DecoratorKind::Named("reduce")
        )
    })
}

fn parse_parallel_decorator<'src>(
    dec: &Decorator<'src>,
) -> Result<ParsedParallelContract<'src>, ParallelHookError> {
    match dec.kind {
        DecoratorKind::Named("parallel") => {}
        _ => {
            return Err(ParallelHookError {
                kind: ParallelHookErrorKind::InvalidParallelDecoratorUsage,
                span: dec.span,
            })
        }
    }

    let mut out = ParsedParallelContract {
        deterministic: false,
        schedule_hint: ParallelScheduleHint::Auto,
        ..ParsedParallelContract::default()
    };
    let mut seen_keys = HashSet::<&str>::new();
    let mut seen_flags = HashSet::<&str>::new();

    for arg in &dec.args {
        match arg {
            DecoratorArg::Flag { name, span } => {
                if !seen_flags.insert(name) {
                    return Err(ParallelHookError {
                        kind: ParallelHookErrorKind::DuplicateParallelArgFlag,
                        span: *span,
                    });
                }
                match *name {
                    "deterministic" => out.deterministic = true,
                    "performance" => out.schedule_hint = ParallelScheduleHint::Performance,
                    "efficient" => out.schedule_hint = ParallelScheduleHint::Efficient,
                    "auto" => out.schedule_hint = ParallelScheduleHint::Auto,
                    _ => {
                        return Err(ParallelHookError {
                            kind: ParallelHookErrorKind::UnsupportedParallelFlag,
                            span: *span,
                        })
                    }
                }
            }
            DecoratorArg::KeyValue { key, value, span } => {
                if !seen_keys.insert(key) {
                    return Err(ParallelHookError {
                        kind: ParallelHookErrorKind::DuplicateParallelArgKey,
                        span: *span,
                    });
                }
                match *key {
                    "domain" => {
                        out.domain = Some(expect_text(
                            value,
                            *span,
                            ParallelHookErrorKind::UnsupportedParallelArgKey,
                        )?);
                    }
                    "read" => {
                        out.read_effects = split_effect_list(value, *span)?;
                    }
                    "write" => {
                        out.write_effects = split_effect_list(value, *span)?;
                    }
                    "chunk" => {
                        out.chunk_hint = Some(parse_chunk_hint(value, *span)?);
                    }
                    "schedule" | "core" => {
                        out.schedule_hint = parse_schedule_hint(value, *span)?;
                    }
                    "deterministic" => {
                        out.deterministic = parse_boolish(value, *span)?;
                    }
                    _ => {
                        return Err(ParallelHookError {
                            kind: ParallelHookErrorKind::UnsupportedParallelArgKey,
                            span: *span,
                        })
                    }
                }
            }
        }
    }

    Ok(out)
}

fn parse_reduce_decorator(dec: &Decorator<'_>) -> Result<ParallelReduceKind, ParallelHookError> {
    match dec.kind {
        DecoratorKind::Named("reduce") => {}
        _ => {
            return Err(ParallelHookError {
                kind: ParallelHookErrorKind::InvalidReduceDecoratorUsage,
                span: dec.span,
            })
        }
    }

    if dec.args.is_empty() {
        return Err(ParallelHookError {
            kind: ParallelHookErrorKind::InvalidReduceDecoratorUsage,
            span: dec.span,
        });
    }

    let mut reduce: Option<ParallelReduceKind> = None;
    for arg in &dec.args {
        match arg {
            DecoratorArg::Flag { name, span } => {
                if reduce.is_some() {
                    return Err(ParallelHookError {
                        kind: ParallelHookErrorKind::InvalidReduceDecoratorUsage,
                        span: *span,
                    });
                }
                reduce = Some(parse_reduce_name(name, *span)?);
            }
            DecoratorArg::KeyValue { key, value, span } => match *key {
                "kind" => {
                    if reduce.is_some() {
                        return Err(ParallelHookError {
                            kind: ParallelHookErrorKind::InvalidReduceDecoratorUsage,
                            span: *span,
                        });
                    }
                    let name =
                        expect_text(value, *span, ParallelHookErrorKind::UnsupportedReduceKind)?;
                    reduce = Some(parse_reduce_name(name, *span)?);
                }
                _ => {
                    return Err(ParallelHookError {
                        kind: ParallelHookErrorKind::InvalidReduceDecoratorUsage,
                        span: *span,
                    })
                }
            },
        }
    }

    reduce.ok_or(ParallelHookError {
        kind: ParallelHookErrorKind::InvalidReduceDecoratorUsage,
        span: dec.span,
    })
}

fn parse_reduce_name(text: &str, span: Span) -> Result<ParallelReduceKind, ParallelHookError> {
    match text {
        "sum" => Ok(ParallelReduceKind::Sum),
        "min" => Ok(ParallelReduceKind::Min),
        "max" => Ok(ParallelReduceKind::Max),
        "bitor" | "or" => Ok(ParallelReduceKind::BitOr),
        "bitand" | "and" => Ok(ParallelReduceKind::BitAnd),
        _ => Err(ParallelHookError {
            kind: ParallelHookErrorKind::UnsupportedReduceKind,
            span,
        }),
    }
}

fn split_effect_list<'src>(
    value: &DecoratorValue<'src>,
    span: Span,
) -> Result<Vec<&'src str>, ParallelHookError> {
    match value {
        DecoratorValue::Identifier(v) => Ok(vec![*v]),
        DecoratorValue::String(v) => {
            let mut out = Vec::new();
            for part in v.split(',') {
                let trimmed = part.trim();
                if !trimmed.is_empty() {
                    out.push(trimmed);
                }
            }
            if out.is_empty() {
                return Err(ParallelHookError {
                    kind: ParallelHookErrorKind::UnsupportedParallelArgKey,
                    span,
                });
            }
            Ok(out)
        }
        _ => Err(ParallelHookError {
            kind: ParallelHookErrorKind::UnsupportedParallelArgKey,
            span,
        }),
    }
}

fn parse_chunk_hint(value: &DecoratorValue<'_>, span: Span) -> Result<u16, ParallelHookError> {
    let text = match value {
        DecoratorValue::Integer(v) => *v,
        _ => {
            return Err(ParallelHookError {
                kind: ParallelHookErrorKind::InvalidChunkHint,
                span,
            })
        }
    };
    let parsed: u32 = text.parse().map_err(|_| ParallelHookError {
        kind: ParallelHookErrorKind::InvalidChunkHint,
        span,
    })?;
    if parsed == 0 || parsed > u16::MAX as u32 {
        return Err(ParallelHookError {
            kind: ParallelHookErrorKind::InvalidChunkHint,
            span,
        });
    }
    Ok(parsed as u16)
}

fn parse_schedule_hint(
    value: &DecoratorValue<'_>,
    span: Span,
) -> Result<ParallelScheduleHint, ParallelHookError> {
    match expect_text(value, span, ParallelHookErrorKind::UnsupportedScheduleHint)? {
        "auto" => Ok(ParallelScheduleHint::Auto),
        "performance" | "perf" | "p" => Ok(ParallelScheduleHint::Performance),
        "efficient" | "eff" | "e" => Ok(ParallelScheduleHint::Efficient),
        _ => Err(ParallelHookError {
            kind: ParallelHookErrorKind::UnsupportedScheduleHint,
            span,
        }),
    }
}

fn parse_boolish(value: &DecoratorValue<'_>, span: Span) -> Result<bool, ParallelHookError> {
    match value {
        DecoratorValue::Bool(v) => Ok(*v),
        DecoratorValue::Identifier("true") | DecoratorValue::String("true") => Ok(true),
        DecoratorValue::Identifier("false") | DecoratorValue::String("false") => Ok(false),
        _ => Err(ParallelHookError {
            kind: ParallelHookErrorKind::UnsupportedParallelArgKey,
            span,
        }),
    }
}

fn expect_text<'src>(
    value: &DecoratorValue<'src>,
    span: Span,
    on_error: ParallelHookErrorKind,
) -> Result<&'src str, ParallelHookError> {
    match value {
        DecoratorValue::Identifier(v)
        | DecoratorValue::String(v)
        | DecoratorValue::Integer(v)
        | DecoratorValue::Float(v) => Ok(*v),
        DecoratorValue::Bool(_) => Err(ParallelHookError {
            kind: on_error,
            span,
        }),
    }
}

fn join_spans(a: Span, b: Span) -> Span {
    let (line, column) = if a.start <= b.start {
        (a.line, a.column)
    } else {
        (b.line, b.column)
    };
    Span::new(a.start.min(b.start), a.end.max(b.end), line, column)
}

/// Apply validated parallel contracts onto typed-IR function metadata.
///
/// Functions without a validated hook remain in `Serial` execution mode.
pub fn annotate_typed_ir_with_parallel_hooks<'src>(
    ir: &mut TypedIrModule<'src>,
    hooks: &ParallelHookOutcome<'src>,
) {
    let by_name: HashMap<&'src str, &ParallelFunctionHook<'src>> =
        hooks.functions.iter().map(|f| (f.function, f)).collect();

    for func in &mut ir.functions {
        let Some(hook) = by_name.get(func.name).copied() else {
            continue;
        };
        let exec = &mut func.meta.execution;
        exec.policy = match hook.policy {
            ParallelExecutionPolicy::ParallelSafe => IrExecutionPolicy::ParallelSafe,
            ParallelExecutionPolicy::MainThreadOnly => IrExecutionPolicy::MainThreadOnly,
        };
        exec.deterministic = hook.deterministic;
        exec.schedule_hint = match hook.schedule_hint {
            ParallelScheduleHint::Auto => IrScheduleHint::Auto,
            ParallelScheduleHint::Performance => IrScheduleHint::Performance,
            ParallelScheduleHint::Efficient => IrScheduleHint::Efficient,
        };
        exec.chunk_hint = hook.chunk_hint;
        exec.reduce = hook.reduce.map(|r| match r {
            ParallelReduceKind::Sum => IrReduceKind::Sum,
            ParallelReduceKind::Min => IrReduceKind::Min,
            ParallelReduceKind::Max => IrReduceKind::Max,
            ParallelReduceKind::BitOr => IrReduceKind::BitOr,
            ParallelReduceKind::BitAnd => IrReduceKind::BitAnd,
        });
        exec.read_effect_count = hook.read_effects.len().min(u16::MAX as usize) as u16;
        exec.write_effect_count = hook.write_effects.len().min(u16::MAX as usize) as u16;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tlscript::{lower_to_typed_ir, Lexer, Parser, SemanticAnalyzer};

    fn parse_semantic(src: &str) -> (Module<'_>, SemanticReport<'_>) {
        let mut parser = Parser::new(Lexer::new(src));
        let module = parser.parse_module().expect("parse ok");
        let semantic = SemanticAnalyzer::new(Default::default())
            .analyze(&module)
            .expect("semantic ok");
        (module, semantic)
    }

    #[test]
    fn extracts_parallel_contract_and_annotates_ir() {
        let (module, semantic) = parse_semantic(concat!(
            "@export\n",
            "@parallel(domain=\"bodies\", read=\"transform,force\", write=\"velocity\", chunk=256, schedule=\"performance\")\n",
            "@deterministic\n",
            "@reduce(sum)\n",
            "def solve(dt: float):\n",
            "    let x: int = 1\n",
        ));

        let hooks = ParallelHookAnalyzer::new().analyze(&module, &semantic);
        assert!(hooks.errors.is_empty(), "{:?}", hooks.errors);
        assert_eq!(hooks.functions.len(), 1);
        let hook = &hooks.functions[0];
        assert_eq!(hook.domain, Some("bodies"));
        assert_eq!(hook.read_effects, vec!["transform", "force"]);
        assert_eq!(hook.write_effects, vec!["velocity"]);
        assert_eq!(hook.chunk_hint, Some(256));
        assert_eq!(hook.schedule_hint, ParallelScheduleHint::Performance);
        assert_eq!(hook.reduce, Some(ParallelReduceKind::Sum));

        let mut ir = lower_to_typed_ir(&module, &semantic).expect("lower ok");
        annotate_typed_ir_with_parallel_hooks(&mut ir, &hooks);
        let func = &ir.functions[0];
        assert_eq!(func.meta.execution.policy, IrExecutionPolicy::ParallelSafe);
        assert!(func.meta.execution.deterministic);
        assert_eq!(func.meta.execution.chunk_hint, Some(256));
        assert_eq!(
            func.meta.execution.schedule_hint,
            IrScheduleHint::Performance
        );
        assert_eq!(func.meta.execution.reduce, Some(IrReduceKind::Sum));
        assert_eq!(func.meta.execution.read_effect_count, 2);
        assert_eq!(func.meta.execution.write_effect_count, 1);
    }

    #[test]
    fn rejects_parallel_and_main_thread_together() {
        let (module, semantic) = parse_semantic(concat!(
            "@parallel(domain=\"bodies\")\n",
            "@main_thread\n",
            "def f():\n",
            "    let x: int = 1\n",
        ));
        let hooks = ParallelHookAnalyzer::new().analyze(&module, &semantic);
        assert!(hooks.errors.iter().any(|e| matches!(
            e.kind,
            ParallelHookErrorKind::MutuallyExclusiveExecutionDecorators
        )));
    }

    #[test]
    fn main_thread_contract_is_emitted() {
        let (module, semantic) = parse_semantic(concat!(
            "@main_thread\n",
            "def draw_ui():\n",
            "    let x: int = 1\n",
        ));
        let hooks = ParallelHookAnalyzer::new().analyze(&module, &semantic);
        assert!(hooks.errors.is_empty());
        assert_eq!(hooks.functions.len(), 1);
        assert_eq!(
            hooks.functions[0].policy,
            ParallelExecutionPolicy::MainThreadOnly
        );
    }
}
