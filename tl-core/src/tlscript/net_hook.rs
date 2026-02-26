//! `@net(...)` compiler hook for `.tlscript`.
//!
//! This pass runs after parsing and semantic validation and extracts network synchronization
//! annotations for the Tileline NPS layer. It validates decorator syntax/arguments and produces a
//! soft diagnostic report that tooling/runtime code can consume without panicking.

use std::collections::{HashMap, HashSet};

use super::ast::{
    Block, Decorator, DecoratorArg, DecoratorKind, DecoratorValue, FunctionDef, Item, LetStmt,
    Module, Stmt,
};
use super::semantic::{SemanticReport, SemanticType};
use super::token::Span;

/// Replication cadence for a networked field/function.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NetSyncMode {
    /// Replicate only when value changes.
    OnChange,
    /// Replicate every network tick.
    Always,
    /// Replication is triggered manually by script/runtime code.
    Manual,
}

/// Delivery mode for the UDP transport layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NetDeliveryMode {
    /// Fast unreliable transport for high-rate physics/input updates.
    Unreliable,
    /// Lightweight reliability + ordering for lifecycle/stateful events.
    ReliableOrdered,
}

/// Parsed/validated `@net(...)` configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NetDecoratorConfig {
    /// Replication cadence.
    pub sync: NetSyncMode,
    /// Delivery/reliability mode.
    pub delivery: NetDeliveryMode,
}

impl Default for NetDecoratorConfig {
    fn default() -> Self {
        Self {
            sync: NetSyncMode::OnChange,
            delivery: NetDeliveryMode::ReliableOrdered,
        }
    }
}

/// Function-level network hook generated from `@net(...)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NetFunctionHook<'src> {
    /// Script function name.
    pub function: &'src str,
    /// `true` if the function is also `@export` and can be surfaced via the WASM ABI/MPS bridge.
    pub exported: bool,
    /// Network decorator settings.
    pub config: NetDecoratorConfig,
    /// Decorator span.
    pub span: Span,
}

/// Local binding-level network hook generated from `@net(...) let ...`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NetBindingHook<'src> {
    /// Parent function name.
    pub function: &'src str,
    /// Local binding name.
    pub binding: &'src str,
    /// Declared type if present in syntax (`None` means inferred in semantic pass; codegen/runtime
    /// can consult lowering/type metadata later).
    pub declared_type: Option<SemanticType>,
    /// Network decorator settings.
    pub config: NetDecoratorConfig,
    /// Decorator span.
    pub span: Span,
}

/// Non-fatal `@net` hook warning.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NetHookWarningKind {
    /// Function was marked `@net` but not `@export`; it cannot be MPS/WASM-invoked directly yet.
    NetFunctionMissingExport,
    /// Binding is networked but no explicit type annotation was provided.
    NetBindingUsesInferredType,
    /// Unreliable string sync is usually a poor fit for bandwidth/ordering guarantees.
    UnreliableStringSync,
}

/// `@net` hook warning with source span.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NetHookWarning {
    pub kind: NetHookWarningKind,
    pub span: Span,
}

/// Fatal `@net` hook error kind.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NetHookErrorKind {
    DuplicateNetDecorator,
    InvalidNetDecoratorUsage,
    InvalidNetDecoratorArgument,
    DuplicateNetFlag,
    DuplicateNetKey,
    UnsupportedNetFlag,
    UnsupportedNetKey,
    UnsupportedSyncMode,
    UnsupportedDeliveryMode,
    MissingFunctionSemanticSignature,
}

/// `@net` hook error with source span.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NetHookError {
    pub kind: NetHookErrorKind,
    pub span: Span,
}

/// Soft result of the `@net` compiler hook pass.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct NetHookOutcome<'src> {
    /// Valid function-level hooks.
    pub functions: Vec<NetFunctionHook<'src>>,
    /// Valid local-binding hooks.
    pub bindings: Vec<NetBindingHook<'src>>,
    /// Fatal decorator diagnostics.
    pub errors: Vec<NetHookError>,
    /// Non-fatal warnings.
    pub warnings: Vec<NetHookWarning>,
}

impl<'src> NetHookOutcome<'src> {
    /// `true` when hook extraction completed without fatal decorator errors.
    pub fn can_emit_network_metadata(&self) -> bool {
        self.errors.is_empty()
    }
}

/// Compiler hook analyzer that extracts and validates `@net(...)` annotations.
#[derive(Debug, Default)]
pub struct NetHookAnalyzer;

impl NetHookAnalyzer {
    /// Create a new analyzer.
    pub fn new() -> Self {
        Self
    }

    /// Analyze a parsed module using semantic signatures to validate function export linkage.
    pub fn analyze<'src>(
        &self,
        module: &Module<'src>,
        semantic: &SemanticReport<'src>,
    ) -> NetHookOutcome<'src> {
        let mut out = NetHookOutcome::default();
        let sig_map: HashMap<&'src str, _> =
            semantic.signatures.iter().map(|s| (s.name, s)).collect();

        for item in &module.items {
            let Item::Function(func) = item;
            let mut function_net: Option<(NetDecoratorConfig, Span)> = None;
            let mut seen_net_on_fn = false;
            let mut exported = false;

            for dec in &func.decorators {
                match &dec.kind {
                    DecoratorKind::Export => {
                        exported = true;
                    }
                    DecoratorKind::Named(name) if *name == "net" => {
                        if seen_net_on_fn {
                            out.errors.push(NetHookError {
                                kind: NetHookErrorKind::DuplicateNetDecorator,
                                span: dec.span,
                            });
                            continue;
                        }
                        seen_net_on_fn = true;
                        match parse_net_decorator(dec) {
                            Ok(cfg) => function_net = Some((cfg, dec.span)),
                            Err(err) => out.errors.push(err),
                        }
                    }
                    _ => {}
                }
            }

            if let Some((cfg, span)) = function_net {
                if !sig_map.contains_key(func.name) {
                    out.errors.push(NetHookError {
                        kind: NetHookErrorKind::MissingFunctionSemanticSignature,
                        span: func.name_span,
                    });
                } else {
                    if !exported {
                        out.warnings.push(NetHookWarning {
                            kind: NetHookWarningKind::NetFunctionMissingExport,
                            span,
                        });
                    }
                    out.functions.push(NetFunctionHook {
                        function: func.name,
                        exported,
                        config: cfg,
                        span,
                    });
                }
            }

            self.walk_block_for_net_bindings(func, &func.body, &mut out);
        }

        out
    }

    fn walk_block_for_net_bindings<'src>(
        &self,
        func: &FunctionDef<'src>,
        block: &Block<'src>,
        out: &mut NetHookOutcome<'src>,
    ) {
        for stmt in &block.statements {
            match stmt {
                Stmt::Let(let_stmt) => self.collect_net_let_hook(func, let_stmt, out),
                Stmt::If(s) => {
                    for branch in &s.branches {
                        self.walk_block_for_net_bindings(func, &branch.body, out);
                    }
                    if let Some(b) = &s.else_block {
                        self.walk_block_for_net_bindings(func, b, out);
                    }
                }
                Stmt::While(s) => self.walk_block_for_net_bindings(func, &s.body, out),
                Stmt::ForRange(s) => self.walk_block_for_net_bindings(func, &s.body, out),
                Stmt::Assign(_) | Stmt::Expr(_) => {}
            }
        }
    }

    fn collect_net_let_hook<'src>(
        &self,
        func: &FunctionDef<'src>,
        let_stmt: &LetStmt<'src>,
        out: &mut NetHookOutcome<'src>,
    ) {
        let mut seen_net = false;
        for dec in &let_stmt.decorators {
            match &dec.kind {
                DecoratorKind::Named(name) if *name == "net" => {
                    if seen_net {
                        out.errors.push(NetHookError {
                            kind: NetHookErrorKind::DuplicateNetDecorator,
                            span: dec.span,
                        });
                        continue;
                    }
                    seen_net = true;
                    match parse_net_decorator(dec) {
                        Ok(cfg) => {
                            let declared_type = let_stmt.ty.map(|t| match t.kind {
                                super::ast::TypeName::Int => SemanticType::Int,
                                super::ast::TypeName::Float => SemanticType::Float,
                                super::ast::TypeName::Bool => SemanticType::Bool,
                                super::ast::TypeName::Str => SemanticType::Str,
                            });
                            if declared_type.is_none() {
                                out.warnings.push(NetHookWarning {
                                    kind: NetHookWarningKind::NetBindingUsesInferredType,
                                    span: let_stmt.name_span,
                                });
                            }
                            if matches!(declared_type, Some(SemanticType::Str))
                                && matches!(cfg.delivery, NetDeliveryMode::Unreliable)
                            {
                                out.warnings.push(NetHookWarning {
                                    kind: NetHookWarningKind::UnreliableStringSync,
                                    span: dec.span,
                                });
                            }
                            out.bindings.push(NetBindingHook {
                                function: func.name,
                                binding: let_stmt.name,
                                declared_type,
                                config: cfg,
                                span: dec.span,
                            });
                        }
                        Err(err) => out.errors.push(err),
                    }
                }
                DecoratorKind::Export => out.errors.push(NetHookError {
                    kind: NetHookErrorKind::InvalidNetDecoratorUsage,
                    span: dec.span,
                }),
                DecoratorKind::Named(_) => {}
            }
        }
    }
}

fn parse_net_decorator(dec: &Decorator<'_>) -> Result<NetDecoratorConfig, NetHookError> {
    let DecoratorKind::Named(name) = dec.kind.clone() else {
        return Err(NetHookError {
            kind: NetHookErrorKind::InvalidNetDecoratorUsage,
            span: dec.span,
        });
    };
    if name != "net" {
        return Err(NetHookError {
            kind: NetHookErrorKind::InvalidNetDecoratorUsage,
            span: dec.span,
        });
    }

    let mut cfg = NetDecoratorConfig::default();
    let mut seen_flags = HashSet::<&str>::new();
    let mut seen_keys = HashSet::<&str>::new();

    for arg in &dec.args {
        match arg {
            DecoratorArg::Flag { name, span } => {
                if !seen_flags.insert(name) {
                    return Err(NetHookError {
                        kind: NetHookErrorKind::DuplicateNetFlag,
                        span: *span,
                    });
                }
                match *name {
                    "unreliable" => cfg.delivery = NetDeliveryMode::Unreliable,
                    "reliable" | "reliable_ordered" | "ordered" => {
                        cfg.delivery = NetDeliveryMode::ReliableOrdered
                    }
                    "on_change" => cfg.sync = NetSyncMode::OnChange,
                    "always" => cfg.sync = NetSyncMode::Always,
                    "manual" => cfg.sync = NetSyncMode::Manual,
                    _ => {
                        return Err(NetHookError {
                            kind: NetHookErrorKind::UnsupportedNetFlag,
                            span: *span,
                        })
                    }
                }
            }
            DecoratorArg::KeyValue { key, value, span } => {
                if !seen_keys.insert(key) {
                    return Err(NetHookError {
                        kind: NetHookErrorKind::DuplicateNetKey,
                        span: *span,
                    });
                }
                match *key {
                    "sync" => {
                        cfg.sync = parse_sync_value(value).ok_or(NetHookError {
                            kind: NetHookErrorKind::UnsupportedSyncMode,
                            span: *span,
                        })?;
                    }
                    "delivery" | "mode" => {
                        cfg.delivery = parse_delivery_value(value).ok_or(NetHookError {
                            kind: NetHookErrorKind::UnsupportedDeliveryMode,
                            span: *span,
                        })?;
                    }
                    _ => {
                        return Err(NetHookError {
                            kind: NetHookErrorKind::UnsupportedNetKey,
                            span: *span,
                        })
                    }
                }
            }
        }
    }

    Ok(cfg)
}

fn parse_sync_value(value: &DecoratorValue<'_>) -> Option<NetSyncMode> {
    match value_string(value) {
        "on_change" => Some(NetSyncMode::OnChange),
        "always" => Some(NetSyncMode::Always),
        "manual" => Some(NetSyncMode::Manual),
        _ => None,
    }
}

fn parse_delivery_value(value: &DecoratorValue<'_>) -> Option<NetDeliveryMode> {
    match value_string(value) {
        "unreliable" => Some(NetDeliveryMode::Unreliable),
        "reliable" | "reliable_ordered" | "ordered" => Some(NetDeliveryMode::ReliableOrdered),
        _ => None,
    }
}

fn value_string<'a>(value: &'a DecoratorValue<'a>) -> &'a str {
    match value {
        DecoratorValue::Identifier(v)
        | DecoratorValue::String(v)
        | DecoratorValue::Integer(v)
        | DecoratorValue::Float(v) => v,
        DecoratorValue::Bool(true) => "true",
        DecoratorValue::Bool(false) => "false",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tlscript::{Lexer, Parser, SemanticAnalyzer};

    fn parse_and_semantic(src: &str) -> (Module<'_>, SemanticReport<'_>) {
        let mut parser = Parser::new(Lexer::new(src));
        let module = parser.parse_module().expect("parse ok");
        let report = SemanticAnalyzer::new(Default::default())
            .analyze(&module)
            .expect("semantic ok");
        (module, report)
    }

    #[test]
    fn extracts_function_and_binding_net_hooks() {
        let (module, report) = parse_and_semantic(concat!(
            "@export\n",
            "@net(sync=\"on_change\")\n",
            "def tick(dt: float):\n",
            "    @net(unreliable)\n",
            "    let pos: int = 5\n",
        ));

        let outcome = NetHookAnalyzer::new().analyze(&module, &report);
        assert!(outcome.errors.is_empty(), "{:?}", outcome.errors);
        assert_eq!(outcome.functions.len(), 1);
        assert_eq!(outcome.functions[0].function, "tick");
        assert_eq!(outcome.functions[0].config.sync, NetSyncMode::OnChange);
        assert_eq!(outcome.bindings.len(), 1);
        assert_eq!(outcome.bindings[0].binding, "pos");
        assert_eq!(
            outcome.bindings[0].config.delivery,
            NetDeliveryMode::Unreliable
        );
    }

    #[test]
    fn warns_when_network_function_is_not_exported() {
        let (module, report) = parse_and_semantic(concat!(
            "@net(unreliable)\n",
            "def local_fx():\n",
            "    let x: int = 1\n",
        ));
        let outcome = NetHookAnalyzer::new().analyze(&module, &report);
        assert!(outcome
            .warnings
            .iter()
            .any(|w| matches!(w.kind, NetHookWarningKind::NetFunctionMissingExport)));
    }

    #[test]
    fn rejects_unknown_sync_mode() {
        let (module, report) = parse_and_semantic(concat!(
            "@export\n",
            "@net(sync=\"weird\")\n",
            "def tick():\n",
            "    let x: int = 1\n",
        ));
        let outcome = NetHookAnalyzer::new().analyze(&module, &report);
        assert!(outcome
            .errors
            .iter()
            .any(|e| matches!(e.kind, NetHookErrorKind::UnsupportedSyncMode)));
    }
}
