//! WASM bytecode generation for `.tlscript` (V1).
//!
//! This module lowers a semantically-validated `.tlscript` AST into an in-memory `.wasm` binary
//! using a lightweight emitter (`wasm-encoder`). The generated module is intended for Tileline's
//! MPS-hosted sandbox execution path and therefore emphasizes:
//! - predictable MVP-compatible code generation
//! - exported wrapper functions for async task invocation (`@export` bridge)
//! - soft-fallback guards for trap-prone integer operations (division/modulo by zero)
//! - handle/index-based value passing (no raw pointers)
//!
//! SIMD optimizations are not emitted yet, but the generator records hook sites so future codegen
//! passes can replace scalar arithmetic sequences with SIMD forms on supported targets.

use std::borrow::Cow;
use std::collections::HashMap;
use std::fmt;

use super::ast::*;
use super::semantic::{FunctionSignature, SemanticReport, SemanticType};
use super::token::Span;
use wasm_encoder::{
    BlockType, CodeSection, ConstExpr, CustomSection, DataSection, EntityType, ExportKind,
    ExportSection, Function as WasmFunction, FunctionSection, ImportSection, Instruction,
    MemorySection, MemoryType, Module as WasmModule, TypeSection, ValType,
};

const DEFAULT_STRING_DATA_BASE: u32 = 1024;
const SOFT_EXCEPTION_INT_DIV_ZERO: i32 = 1;
const SOFT_EXCEPTION_INT_MOD_ZERO: i32 = 2;

/// Host import signature used by code generation for external calls.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HostImportSignature {
    /// Script-visible callee name (`host_tick`, `spawn_sprite`, ...).
    pub script_name: String,
    /// WASM import module name.
    pub import_module: String,
    /// WASM import field name.
    pub import_name: String,
    /// Parameter types.
    pub params: Vec<SemanticType>,
    /// Optional single result type (WASM MVP-style).
    pub result: Option<SemanticType>,
}

/// WASM codegen configuration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WasmCodegenConfig {
    /// Initial linear memory page count (64KiB/page).
    pub initial_memory_pages: u32,
    /// Base offset where static string data starts.
    pub string_data_base: u32,
    /// Whether the linear memory is exported as `memory`.
    pub export_memory: bool,
    /// Export name used for the linear memory.
    pub memory_export_name: String,
    /// Prefix used for generated wrapper function symbols.
    pub export_wrapper_prefix: String,
    /// Import module for soft-exception logging.
    pub soft_exception_log_import_module: String,
    /// Import field for soft-exception logging.
    pub soft_exception_log_import_name: String,
    /// Host import signatures available to script code.
    pub host_imports: Vec<HostImportSignature>,
    /// Emit metadata section with Tileline-specific hints (export wrapper map, SIMD hooks).
    pub emit_metadata_section: bool,
    /// Record scalar arithmetic locations that are future SIMD optimization candidates.
    pub enable_simd_hook_markers: bool,
}

impl Default for WasmCodegenConfig {
    fn default() -> Self {
        Self {
            initial_memory_pages: 1,
            string_data_base: DEFAULT_STRING_DATA_BASE,
            export_memory: true,
            memory_export_name: "memory".to_string(),
            export_wrapper_prefix: "__tls_export_".to_string(),
            soft_exception_log_import_module: "tileline".to_string(),
            soft_exception_log_import_name: "soft_exception".to_string(),
            host_imports: vec![
                HostImportSignature {
                    script_name: "spawn_sprite".to_string(),
                    import_module: "tileline".to_string(),
                    import_name: "spawn_sprite".to_string(),
                    params: Vec::new(),
                    result: Some(SemanticType::Handle),
                },
                HostImportSignature {
                    script_name: "spawn_body".to_string(),
                    import_module: "tileline".to_string(),
                    import_name: "spawn_body".to_string(),
                    params: Vec::new(),
                    result: Some(SemanticType::Handle),
                },
                HostImportSignature {
                    script_name: "release_handle".to_string(),
                    import_module: "tileline".to_string(),
                    import_name: "release_handle".to_string(),
                    params: vec![SemanticType::Handle],
                    result: None,
                },
            ],
            emit_metadata_section: true,
            enable_simd_hook_markers: true,
        }
    }
}

/// Export metadata for an `@export` function wrapper.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CodegenExportEntry {
    /// Script function name.
    pub script_name: String,
    /// Internal (non-exported) implementation symbol.
    pub internal_symbol: String,
    /// Wrapper symbol generated for the MPS/WASM bridge.
    pub wrapper_symbol: String,
    /// Public export name in the final module (same as script name in V1).
    pub export_name: String,
    /// ABI parameter types.
    pub params: Vec<SemanticType>,
    /// ABI result type.
    pub result: SemanticType,
    /// Tileline runtime hint: exported wrappers are async task entrypoints.
    pub mps_async_task_entry: bool,
}

/// Non-fatal codegen warning categories.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WasmCodegenWarningKind {
    /// A non-unit function had no explicit return and a default return value was synthesized.
    ImplicitDefaultReturnInserted,
    /// A division/modulo-by-zero guard was inserted with a soft fallback path.
    SoftFallbackGuardInserted,
    /// A scalar arithmetic site was recorded as a future SIMD optimization hook.
    SimdHookSiteRecorded,
}

/// Non-fatal codegen warning.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WasmCodegenWarning {
    /// Warning category.
    pub kind: WasmCodegenWarningKind,
    /// Optional source span.
    pub span: Option<Span>,
    /// Optional function name.
    pub function: Option<String>,
}

/// Fatal codegen error categories.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WasmCodegenErrorKind {
    /// No semantic signature found for a parsed function.
    MissingSemanticSignature,
    /// Unsupported or unknown type during WASM lowering.
    UnsupportedType,
    /// Unsupported expression/statement shape for V1 codegen.
    UnsupportedConstruct,
    /// Unknown local binding during emission.
    UnknownLocalBinding,
    /// Host import signature is required but was not provided.
    MissingHostImportSignature,
    /// Numeric literal could not be parsed or fit the target type.
    InvalidLiteral,
    /// `@export` function wrapper could not be generated due to ABI mismatch.
    ExportAbiMismatch,
}

/// Fatal codegen error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WasmCodegenError {
    /// Error category.
    pub kind: WasmCodegenErrorKind,
    /// Optional source span.
    pub span: Option<Span>,
    /// Human-readable detail string (safe for engine console logging).
    pub detail: String,
}

impl WasmCodegenError {
    fn new(kind: WasmCodegenErrorKind, span: Option<Span>, detail: impl Into<String>) -> Self {
        Self {
            kind,
            span,
            detail: detail.into(),
        }
    }
}

impl fmt::Display for WasmCodegenError {
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

impl std::error::Error for WasmCodegenError {}

/// Successful bytecode generation result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WasmCodegenOutput {
    /// In-memory `.wasm` module bytes.
    pub wasm_bytes: Vec<u8>,
    /// Export wrapper metadata for MPS async task integration.
    pub exports: Vec<CodegenExportEntry>,
    /// Total bytes reserved for static string data in linear memory.
    pub static_string_data_bytes: u32,
    /// Minimum memory page count in the generated module.
    pub memory_min_pages: u32,
    /// Number of scalar sites recorded as SIMD hook candidates.
    pub simd_hook_sites: u32,
}

/// Soft codegen result (engine-friendly, non-panicking).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WasmCodegenOutcome {
    /// Output bytes if generation succeeded.
    pub output: Option<WasmCodegenOutput>,
    /// Fatal diagnostics.
    pub errors: Vec<WasmCodegenError>,
    /// Non-fatal diagnostics.
    pub warnings: Vec<WasmCodegenWarning>,
    /// `true` when codegen succeeded and `output` is populated.
    pub can_instantiate: bool,
}

/// `.tlscript` -> WASM generator.
pub struct WasmGenerator {
    config: WasmCodegenConfig,
}

impl WasmGenerator {
    /// Construct a generator with the given configuration.
    pub fn new(config: WasmCodegenConfig) -> Self {
        Self { config }
    }

    /// Strict generation API.
    pub fn generate<'src>(
        &self,
        module: &Module<'src>,
        semantic: &SemanticReport<'src>,
    ) -> Result<WasmCodegenOutput, WasmCodegenError> {
        let outcome = self.generate_soft(module, semantic);
        if let Some(err) = outcome.errors.into_iter().next() {
            return Err(err);
        }
        outcome.output.ok_or_else(|| {
            WasmCodegenError::new(
                WasmCodegenErrorKind::UnsupportedConstruct,
                Some(module.span),
                "codegen failed without a concrete error payload",
            )
        })
    }

    /// Soft generation API suitable for engine/editor integration loops.
    pub fn generate_soft<'src>(
        &self,
        module: &Module<'src>,
        semantic: &SemanticReport<'src>,
    ) -> WasmCodegenOutcome {
        let mut warnings = Vec::new();
        match self.generate_inner(module, semantic, &mut warnings) {
            Ok(output) => WasmCodegenOutcome {
                output: Some(output),
                errors: Vec::new(),
                warnings,
                can_instantiate: true,
            },
            Err(err) => WasmCodegenOutcome {
                output: None,
                errors: vec![err],
                warnings,
                can_instantiate: false,
            },
        }
    }

    fn generate_inner<'src>(
        &self,
        module: &Module<'src>,
        semantic: &SemanticReport<'src>,
        warnings: &mut Vec<WasmCodegenWarning>,
    ) -> Result<WasmCodegenOutput, WasmCodegenError> {
        let mut sig_map = HashMap::<&'src str, &FunctionSignature<'src>>::new();
        for sig in &semantic.signatures {
            sig_map.insert(sig.name, sig);
        }

        let mut host_imports = HostImportRegistry::from_config(&self.config);
        let mut type_registry = TypeRegistry::default();

        // Logger import is always present so soft fallback guards can call it cheaply.
        let logger_sig = FuncAbiSig {
            params: vec![SemanticType::Int, SemanticType::Int],
            result: SemanticType::Unit,
        };
        let logger_type = type_registry.get_or_insert(&logger_sig);
        host_imports.set_logger(
            self.config.soft_exception_log_import_module.clone(),
            self.config.soft_exception_log_import_name.clone(),
            logger_type,
        );
        host_imports.assign_all_type_indices(&mut type_registry);

        let mut planned_functions = Vec::<PlannedFunction<'src>>::new();
        let mut string_pool = StringPool::new(self.config.string_data_base);

        // First pass: plan function locals and reserve types.
        for item in &module.items {
            let Item::Function(func) = item;
            let Some(sig) = sig_map.get(func.name).copied() else {
                return Err(WasmCodegenError::new(
                    WasmCodegenErrorKind::MissingSemanticSignature,
                    Some(func.name_span),
                    format!("missing semantic signature for function `{}`", func.name),
                ));
            };

            let param_tys = self.resolve_param_types(func, sig)?;
            let mut planner = LocalPlanner::new(
                func.name,
                &param_tys,
                sig.return_type,
                &sig_map,
                &host_imports,
                &mut string_pool,
            );
            planner.seed_params(func)?;
            planner.plan_block(&func.body)?;

            let internal_sig = FuncAbiSig {
                params: param_tys.clone(),
                result: sig.return_type,
            };
            let internal_type_idx = type_registry.get_or_insert(&internal_sig);

            let wrapper_type_idx = if sig.exported {
                Some(type_registry.get_or_insert(&internal_sig))
            } else {
                None
            };

            planned_functions.push(PlannedFunction {
                func,
                sig,
                param_tys,
                internal_type_idx,
                wrapper_type_idx,
                local_plan: planner.finish(),
            });
        }

        // Second pass: assign import/function indices.
        let logger_index = host_imports.finalize_logger_index();
        let import_entries = host_imports.entries();
        let imported_count = import_entries.len() as u32;

        let mut internal_fn_indices = HashMap::<&'src str, u32>::new();
        let mut wrapper_fn_indices = HashMap::<&'src str, u32>::new();
        let mut next_function_index = imported_count;
        for pf in &planned_functions {
            internal_fn_indices.insert(pf.func.name, next_function_index);
            next_function_index += 1;
        }
        for pf in &planned_functions {
            if pf.sig.exported {
                wrapper_fn_indices.insert(pf.func.name, next_function_index);
                next_function_index += 1;
            }
        }

        // Emit sections.
        let mut types = TypeSection::new();
        type_registry.emit_into(&mut types);

        let mut imports = ImportSection::new();
        for import in import_entries {
            imports.import(
                &import.import_module,
                &import.import_name,
                EntityType::Function(import.type_index),
            );
        }

        let mut functions = FunctionSection::new();
        for pf in &planned_functions {
            functions.function(pf.internal_type_idx);
        }
        for pf in &planned_functions {
            if let Some(type_idx) = pf.wrapper_type_idx {
                functions.function(type_idx);
            }
        }

        let mut memories = MemorySection::new();
        memories.memory(MemoryType {
            minimum: self.config.initial_memory_pages as u64,
            maximum: None,
            memory64: false,
            shared: false,
            page_size_log2: None,
        });

        let mut exports = ExportSection::new();
        if self.config.export_memory {
            exports.export(&self.config.memory_export_name, ExportKind::Memory, 0);
        }

        let mut code = CodeSection::new();
        let mut export_entries = Vec::new();
        let mut simd_hook_sites = 0u32;

        for pf in &planned_functions {
            let internal_index = *internal_fn_indices
                .get(pf.func.name)
                .expect("internal index");
            let fctx = EmitFunctionContext::new(
                pf,
                &self.config,
                &host_imports,
                &sig_map,
                &internal_fn_indices,
                logger_index,
                &string_pool,
                warnings,
                &mut simd_hook_sites,
            )?;
            let body = fctx.emit_internal_body()?;
            code.function(&body);

            let _ = internal_index; // kept for clarity; wrappers use the map below.
        }

        for pf in &planned_functions {
            if !pf.sig.exported {
                continue;
            }
            let internal_index = *internal_fn_indices
                .get(pf.func.name)
                .expect("internal index");
            let wrapper_index = *wrapper_fn_indices.get(pf.func.name).expect("wrapper index");

            let wrapper_body = emit_export_wrapper_body(pf, internal_index)?;
            code.function(&wrapper_body);

            // Export public name -> wrapper index.
            exports.export(pf.func.name, ExportKind::Func, wrapper_index);

            export_entries.push(CodegenExportEntry {
                script_name: pf.func.name.to_string(),
                internal_symbol: format!("__tls_fn_{}", pf.func.name),
                wrapper_symbol: format!("{}{}", self.config.export_wrapper_prefix, pf.func.name),
                export_name: pf.func.name.to_string(),
                params: pf.param_tys.clone(),
                result: pf.sig.return_type,
                mps_async_task_entry: true,
            });
        }

        let mut data = DataSection::new();
        for segment in string_pool.segments() {
            data.active(
                0,
                &ConstExpr::i32_const(segment.offset as i32),
                segment.bytes.iter().copied(),
            );
        }

        let mut module_bytes = WasmModule::new();
        module_bytes.section(&types);
        module_bytes.section(&imports);
        module_bytes.section(&functions);
        module_bytes.section(&memories);
        module_bytes.section(&exports);
        module_bytes.section(&code);
        if !string_pool.segments().is_empty() {
            module_bytes.section(&data);
        }

        if self.config.emit_metadata_section {
            let metadata = build_metadata_section(&export_entries, simd_hook_sites, &self.config);
            let custom = CustomSection {
                name: Cow::Borrowed("tileline.codegen"),
                data: Cow::Owned(metadata),
            };
            module_bytes.section(&custom);
        }

        Ok(WasmCodegenOutput {
            wasm_bytes: module_bytes.finish(),
            exports: export_entries,
            static_string_data_bytes: string_pool.total_data_bytes(),
            memory_min_pages: self.config.initial_memory_pages,
            simd_hook_sites,
        })
    }

    fn resolve_param_types<'src>(
        &self,
        func: &FunctionDef<'src>,
        sig: &FunctionSignature<'src>,
    ) -> Result<Vec<SemanticType>, WasmCodegenError> {
        let mut out = Vec::with_capacity(func.params.len());
        for (idx, param) in func.params.iter().enumerate() {
            let ty = if let Some(ty) = sig.params.get(idx).and_then(|v| *v) {
                ty
            } else if let Some(ann) = param.ty {
                map_type_name_codegen(ann.kind)
            } else {
                return Err(WasmCodegenError::new(
                    WasmCodegenErrorKind::ExportAbiMismatch,
                    Some(param.name_span),
                    format!("parameter `{}` has no resolved semantic type", param.name),
                ));
            };
            out.push(ty);
        }
        Ok(out)
    }
}

impl Default for WasmGenerator {
    fn default() -> Self {
        Self::new(WasmCodegenConfig::default())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct FuncAbiSig {
    params: Vec<SemanticType>,
    result: SemanticType,
}

#[derive(Default)]
struct TypeRegistry {
    order: Vec<FuncAbiSig>,
    map: HashMap<FuncAbiSig, u32>,
}

impl TypeRegistry {
    fn get_or_insert(&mut self, sig: &FuncAbiSig) -> u32 {
        if let Some(idx) = self.map.get(sig).copied() {
            return idx;
        }
        let idx = self.order.len() as u32;
        self.order.push(sig.clone());
        self.map.insert(sig.clone(), idx);
        idx
    }

    fn emit_into(&self, section: &mut TypeSection) {
        for sig in &self.order {
            let params: Vec<ValType> = sig
                .params
                .iter()
                .map(|t| map_semantic_type_to_valtype(*t))
                .collect();
            let results: Vec<ValType> = if sig.result == SemanticType::Unit {
                Vec::new()
            } else {
                vec![map_semantic_type_to_valtype(sig.result)]
            };
            section.ty().function(params, results);
        }
    }
}

#[derive(Debug, Clone)]
struct ImportFuncEntry {
    import_module: String,
    import_name: String,
    params: Vec<SemanticType>,
    result: SemanticType,
    type_index: u32,
    function_index: u32,
}

#[derive(Default)]
struct HostImportRegistry {
    entries_by_script: HashMap<String, ImportFuncEntry>,
    logger_script_name: String,
    logger_module: String,
    logger_name: String,
    logger_type_index: u32,
    logger_function_index: Option<u32>,
}

impl HostImportRegistry {
    fn from_config(config: &WasmCodegenConfig) -> Self {
        let mut reg = Self {
            entries_by_script: HashMap::new(),
            logger_script_name: "__tileline_soft_exception_log".to_string(),
            logger_module: String::new(),
            logger_name: String::new(),
            logger_type_index: 0,
            logger_function_index: None,
        };

        for host in &config.host_imports {
            reg.entries_by_script.insert(
                host.script_name.clone(),
                ImportFuncEntry {
                    import_module: host.import_module.clone(),
                    import_name: host.import_name.clone(),
                    params: host.params.clone(),
                    result: host.result.unwrap_or(SemanticType::Unit),
                    type_index: 0,
                    function_index: 0,
                },
            );
        }
        reg
    }

    fn set_logger(&mut self, module: String, name: String, type_index: u32) {
        self.logger_module = module.clone();
        self.logger_name = name.clone();
        self.logger_type_index = type_index;
        self.entries_by_script.insert(
            self.logger_script_name.clone(),
            ImportFuncEntry {
                import_module: module,
                import_name: name,
                params: vec![SemanticType::Int, SemanticType::Int],
                result: SemanticType::Unit,
                type_index,
                function_index: 0,
            },
        );
    }

    fn finalize_logger_index(&mut self) -> u32 {
        self.assign_function_indices();
        let entry = self
            .entries_by_script
            .get(&self.logger_script_name)
            .expect("logger entry");
        entry.function_index
    }

    fn entries(&mut self) -> Vec<ImportFuncEntry> {
        self.assign_function_indices();
        let mut entries: Vec<_> = self.entries_by_script.values().cloned().collect();
        entries.sort_by(|a, b| a.function_index.cmp(&b.function_index));
        entries
    }

    fn assign_all_type_indices(&mut self, type_registry: &mut TypeRegistry) {
        let keys: Vec<String> = self.entries_by_script.keys().cloned().collect();
        for key in keys {
            let Some(entry) = self.entries_by_script.get(&key).cloned() else {
                continue;
            };
            let sig = FuncAbiSig {
                params: entry.params,
                result: entry.result,
            };
            let type_index = type_registry.get_or_insert(&sig);
            if let Some(dst) = self.entries_by_script.get_mut(&key) {
                dst.type_index = type_index;
            }
        }
    }

    fn get(&self, script_name: &str) -> Option<&ImportFuncEntry> {
        self.entries_by_script.get(script_name)
    }

    fn assign_function_indices(&mut self) {
        if self.logger_function_index.is_some() {
            return;
        }
        let mut keys: Vec<_> = self.entries_by_script.keys().cloned().collect();
        keys.sort();
        for (idx, key) in keys.iter().enumerate() {
            if let Some(entry) = self.entries_by_script.get_mut(key) {
                entry.function_index = idx as u32;
            }
        }
        self.logger_function_index = self
            .entries_by_script
            .get(&self.logger_script_name)
            .map(|e| e.function_index);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct BindingKey {
    start: usize,
    end: usize,
}

impl BindingKey {
    fn from_span(span: Span) -> Self {
        Self {
            start: span.start,
            end: span.end,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct LocalBindingSlot {
    local_index: u32,
    ty: SemanticType,
}

#[derive(Debug, Clone, Copy)]
struct ForLoopTemps {
    end_local: u32,
    step_local: u32,
}

#[derive(Debug, Clone)]
struct FunctionLocalPlan {
    /// Non-param local declarations in declaration order.
    local_decls: Vec<ValType>,
    /// Slots for `let` and loop bindings (keyed by binding span).
    binding_slots: HashMap<BindingKey, LocalBindingSlot>,
    /// Hidden loop temporaries keyed by `range` span.
    for_loop_temps: HashMap<BindingKey, ForLoopTemps>,
    /// Shared scratch locals for guarded integer div/mod paths.
    scratch_i32_left: u32,
    scratch_i32_right: u32,
}

struct LocalPlanner<'a, 'src> {
    function_name: &'a str,
    sig_map: &'a HashMap<&'src str, &'a FunctionSignature<'src>>,
    host_imports: &'a HostImportRegistry,
    string_pool: &'a mut StringPool,
    scopes: Vec<HashMap<&'src str, LocalBindingSlot>>,
    local_decls: Vec<ValType>,
    binding_slots: HashMap<BindingKey, LocalBindingSlot>,
    for_loop_temps: HashMap<BindingKey, ForLoopTemps>,
    next_local_index: u32,
    param_types: Vec<SemanticType>,
    scratch_i32_left: Option<u32>,
    scratch_i32_right: Option<u32>,
}

impl<'a, 'src> LocalPlanner<'a, 'src> {
    fn new(
        function_name: &'a str,
        param_types: &[SemanticType],
        _sig_return_type: SemanticType,
        sig_map: &'a HashMap<&'src str, &'a FunctionSignature<'src>>,
        host_imports: &'a HostImportRegistry,
        string_pool: &'a mut StringPool,
    ) -> Self {
        let scopes = vec![HashMap::new()];
        for (idx, ty) in param_types.iter().copied().enumerate() {
            // Param names are inserted later in `seed_params` because this planner also needs names.
            let _ = idx;
            let _ = ty;
        }
        Self {
            function_name,
            sig_map,
            host_imports,
            string_pool,
            scopes,
            local_decls: Vec::new(),
            binding_slots: HashMap::new(),
            for_loop_temps: HashMap::new(),
            next_local_index: param_types.len() as u32,
            param_types: param_types.to_vec(),
            scratch_i32_left: None,
            scratch_i32_right: None,
        }
    }

    fn finish(mut self) -> FunctionLocalPlan {
        let left = self.ensure_scratch_i32();
        let right = self.ensure_scratch_i32();
        FunctionLocalPlan {
            local_decls: self.local_decls,
            binding_slots: self.binding_slots,
            for_loop_temps: self.for_loop_temps,
            scratch_i32_left: left,
            scratch_i32_right: right,
        }
    }

    fn seed_params(&mut self, func: &FunctionDef<'src>) -> Result<(), WasmCodegenError> {
        for (idx, param) in func.params.iter().enumerate() {
            let ty = self.param_types.get(idx).copied().ok_or_else(|| {
                WasmCodegenError::new(
                    WasmCodegenErrorKind::MissingSemanticSignature,
                    Some(param.name_span),
                    format!("missing parameter type for `{}`", param.name),
                )
            })?;
            self.insert_binding(param.name, param.name_span, idx as u32, ty)?;
        }
        Ok(())
    }

    fn plan_block(&mut self, block: &Block<'src>) -> Result<(), WasmCodegenError> {
        self.push_scope();
        for stmt in &block.statements {
            self.plan_stmt(stmt)?;
        }
        self.pop_scope();
        Ok(())
    }

    fn plan_stmt(&mut self, stmt: &Stmt<'src>) -> Result<(), WasmCodegenError> {
        match stmt {
            Stmt::Let(s) => {
                let ty = if let Some(ann) = s.ty {
                    map_type_name_codegen(ann.kind)
                } else {
                    self.infer_expr_type(&s.value)?
                };
                let local_index = self.alloc_local(ty);
                self.insert_binding(s.name, s.name_span, local_index, ty)?;
                self.infer_expr_type(&s.value)?;
            }
            Stmt::Assign(s) => {
                let target = self.resolve_binding(s.target).ok_or_else(|| {
                    WasmCodegenError::new(
                        WasmCodegenErrorKind::UnknownLocalBinding,
                        Some(s.target_span),
                        format!(
                            "unknown local `{}` in function `{}`",
                            s.target, self.function_name
                        ),
                    )
                })?;
                let rhs_ty = self.infer_expr_type(&s.value)?;
                if target.ty != rhs_ty {
                    return Err(WasmCodegenError::new(
                        WasmCodegenErrorKind::UnsupportedType,
                        Some(s.value.span),
                        format!(
                            "assignment type mismatch during codegen planning: {:?} vs {:?}",
                            target.ty, rhs_ty
                        ),
                    ));
                }
            }
            Stmt::If(s) => {
                for branch in &s.branches {
                    let _ = self.infer_expr_type(&branch.condition)?;
                    self.plan_block(&branch.body)?;
                }
                if let Some(else_block) = &s.else_block {
                    self.plan_block(else_block)?;
                }
            }
            Stmt::While(s) => {
                let _ = self.infer_expr_type(&s.condition)?;
                self.plan_block(&s.body)?;
            }
            Stmt::ForRange(s) => {
                // Loop binding local (i32) + hidden end/step locals.
                let binding_idx = self.alloc_local(SemanticType::Int);
                self.insert_binding(s.binding, s.binding_span, binding_idx, SemanticType::Int)?;
                let temps = ForLoopTemps {
                    end_local: self.alloc_local(SemanticType::Int),
                    step_local: self.alloc_local(SemanticType::Int),
                };
                self.for_loop_temps
                    .insert(BindingKey::from_span(s.range.span), temps);
                for arg in &s.range.args {
                    let arg_ty = self.infer_expr_type(arg)?;
                    if arg_ty != SemanticType::Int {
                        return Err(WasmCodegenError::new(
                            WasmCodegenErrorKind::UnsupportedType,
                            Some(arg.span),
                            "for-range args must lower to i32",
                        ));
                    }
                }
                self.plan_block(&s.body)?;
            }
            Stmt::Expr(s) => {
                let _ = self.infer_expr_type(&s.expr)?;
            }
        }
        Ok(())
    }

    fn infer_expr_type(&mut self, expr: &Expr<'src>) -> Result<SemanticType, WasmCodegenError> {
        match &expr.kind {
            ExprKind::Identifier(name) => {
                self.resolve_binding(name).map(|b| b.ty).ok_or_else(|| {
                    WasmCodegenError::new(
                        WasmCodegenErrorKind::UnknownLocalBinding,
                        Some(expr.span),
                        format!("unknown local `{name}`"),
                    )
                })
            }
            ExprKind::IntegerLiteral(_) => Ok(SemanticType::Int),
            ExprKind::FloatLiteral(_) => Ok(SemanticType::Float),
            ExprKind::BoolLiteral(_) => Ok(SemanticType::Bool),
            ExprKind::StringLiteral(s) => {
                let _ = self.string_pool.intern(s);
                Ok(SemanticType::Str)
            }
            ExprKind::Grouping(inner) => self.infer_expr_type(inner),
            ExprKind::Unary {
                op: UnaryOp::Neg,
                expr: inner,
            } => {
                let ty = self.infer_expr_type(inner)?;
                match ty {
                    SemanticType::Int | SemanticType::Float => Ok(ty),
                    _ => Err(WasmCodegenError::new(
                        WasmCodegenErrorKind::UnsupportedType,
                        Some(expr.span),
                        "unary negation requires int or float",
                    )),
                }
            }
            ExprKind::Binary { op, left, right } => {
                let lt = self.infer_expr_type(left)?;
                let rt = self.infer_expr_type(right)?;
                if matches!(op, BinaryOp::Div | BinaryOp::Mod)
                    && lt == SemanticType::Int
                    && rt == SemanticType::Int
                {
                    let _ = self.ensure_scratch_i32();
                    let _ = self.ensure_scratch_i32();
                }
                infer_binary_type_codegen(*op, lt, rt, expr.span)
            }
            ExprKind::Call { callee, args } => {
                let ExprKind::Identifier(name) = &callee.kind else {
                    return Err(WasmCodegenError::new(
                        WasmCodegenErrorKind::UnsupportedConstruct,
                        Some(callee.span),
                        "only direct identifier calls are supported in V1 codegen",
                    ));
                };

                for arg in args {
                    let _ = self.infer_expr_type(arg)?;
                }

                if let Some(sig) = self.sig_map.get(name).copied() {
                    return Ok(sig.return_type);
                }
                if let Some(host) = self.host_imports.get(name) {
                    return Ok(host.result);
                }
                Err(WasmCodegenError::new(
                    WasmCodegenErrorKind::MissingHostImportSignature,
                    Some(callee.span),
                    format!("missing host import signature for `{name}`"),
                ))
            }
        }
    }

    fn push_scope(&mut self) {
        self.scopes.push(HashMap::new());
    }

    fn pop_scope(&mut self) {
        let _ = self.scopes.pop();
    }

    fn insert_binding(
        &mut self,
        name: &'src str,
        span: Span,
        local_index: u32,
        ty: SemanticType,
    ) -> Result<(), WasmCodegenError> {
        let key = BindingKey::from_span(span);
        let slot = LocalBindingSlot { local_index, ty };
        self.binding_slots.insert(key, slot);
        let Some(scope) = self.scopes.last_mut() else {
            return Err(WasmCodegenError::new(
                WasmCodegenErrorKind::UnsupportedConstruct,
                Some(span),
                "planner scope stack is empty",
            ));
        };
        scope.insert(name, slot);
        Ok(())
    }

    fn resolve_binding(&self, name: &str) -> Option<LocalBindingSlot> {
        self.scopes.iter().rev().find_map(|s| s.get(name).copied())
    }

    fn alloc_local(&mut self, ty: SemanticType) -> u32 {
        let idx = self.next_local_index;
        self.next_local_index += 1;
        self.local_decls.push(map_semantic_type_to_valtype(ty));
        idx
    }

    fn ensure_scratch_i32(&mut self) -> u32 {
        if self.scratch_i32_left.is_none() {
            let idx = self.alloc_local(SemanticType::Int);
            self.scratch_i32_left = Some(idx);
            return idx;
        }
        if self.scratch_i32_right.is_none() {
            let idx = self.alloc_local(SemanticType::Int);
            self.scratch_i32_right = Some(idx);
            return idx;
        }
        self.scratch_i32_right.expect("scratch exists")
    }
}

#[derive(Debug)]
struct PlannedFunction<'src> {
    func: &'src FunctionDef<'src>,
    sig: &'src FunctionSignature<'src>,
    param_tys: Vec<SemanticType>,
    internal_type_idx: u32,
    wrapper_type_idx: Option<u32>,
    local_plan: FunctionLocalPlan,
}

struct EmitFunctionContext<'a, 'src> {
    planned: &'a PlannedFunction<'src>,
    config: &'a WasmCodegenConfig,
    host_imports: &'a HostImportRegistry,
    sig_map: &'a HashMap<&'src str, &'a FunctionSignature<'src>>,
    internal_fn_indices: &'a HashMap<&'src str, u32>,
    logger_func_index: u32,
    string_pool: &'a StringPool,
    warnings: &'a mut Vec<WasmCodegenWarning>,
    simd_hook_sites: &'a mut u32,
    func: WasmFunction,
    scopes: Vec<HashMap<&'src str, LocalBindingSlot>>,
}

impl<'a, 'src> EmitFunctionContext<'a, 'src> {
    fn new(
        planned: &'a PlannedFunction<'src>,
        config: &'a WasmCodegenConfig,
        host_imports: &'a HostImportRegistry,
        sig_map: &'a HashMap<&'src str, &'a FunctionSignature<'src>>,
        internal_fn_indices: &'a HashMap<&'src str, u32>,
        logger_func_index: u32,
        string_pool: &'a StringPool,
        warnings: &'a mut Vec<WasmCodegenWarning>,
        simd_hook_sites: &'a mut u32,
    ) -> Result<Self, WasmCodegenError> {
        let mut grouped = Vec::<(u32, ValType)>::new();
        for vt in &planned.local_plan.local_decls {
            if let Some((count, prev)) = grouped.last_mut() {
                if *prev == *vt {
                    *count += 1;
                    continue;
                }
            }
            grouped.push((1, *vt));
        }

        let mut scopes = vec![HashMap::new()];
        for (idx, param) in planned.func.params.iter().enumerate() {
            let ty = planned.param_tys.get(idx).copied().ok_or_else(|| {
                WasmCodegenError::new(
                    WasmCodegenErrorKind::MissingSemanticSignature,
                    Some(param.name_span),
                    "missing parameter semantic type during emission",
                )
            })?;
            scopes[0].insert(
                param.name,
                LocalBindingSlot {
                    local_index: idx as u32,
                    ty,
                },
            );
        }

        Ok(Self {
            planned,
            config,
            host_imports,
            sig_map,
            internal_fn_indices,
            logger_func_index,
            string_pool,
            warnings,
            simd_hook_sites,
            func: WasmFunction::new(grouped),
            scopes,
        })
    }

    fn emit_internal_body(mut self) -> Result<WasmFunction, WasmCodegenError> {
        self.emit_block(&self.planned.func.body)?;
        self.emit_default_return_if_needed(
            self.planned.sig.return_type,
            Some(self.planned.func.span),
        );
        self.func.instruction(&Instruction::End);
        Ok(self.func)
    }

    fn emit_block(&mut self, block: &Block<'src>) -> Result<(), WasmCodegenError> {
        self.push_scope();
        for stmt in &block.statements {
            self.emit_stmt(stmt)?;
        }
        self.pop_scope();
        Ok(())
    }

    fn emit_stmt(&mut self, stmt: &Stmt<'src>) -> Result<(), WasmCodegenError> {
        match stmt {
            Stmt::Let(s) => {
                let ty = self.expr_type(&s.value)?;
                self.emit_expr(&s.value)?;
                let slot = *self
                    .planned
                    .local_plan
                    .binding_slots
                    .get(&BindingKey::from_span(s.name_span))
                    .ok_or_else(|| {
                        WasmCodegenError::new(
                            WasmCodegenErrorKind::UnknownLocalBinding,
                            Some(s.name_span),
                            format!("missing local slot for `let {}`", s.name),
                        )
                    })?;
                self.func
                    .instruction(&Instruction::LocalSet(slot.local_index));
                self.current_scope_mut().insert(
                    s.name,
                    LocalBindingSlot {
                        local_index: slot.local_index,
                        ty,
                    },
                );
            }
            Stmt::Assign(s) => {
                let slot = self.resolve_local(s.target).ok_or_else(|| {
                    WasmCodegenError::new(
                        WasmCodegenErrorKind::UnknownLocalBinding,
                        Some(s.target_span),
                        format!("unknown local `{}`", s.target),
                    )
                })?;
                self.emit_expr(&s.value)?;
                self.func
                    .instruction(&Instruction::LocalSet(slot.local_index));
            }
            Stmt::Expr(s) => {
                if self.try_emit_release_handle_stmt(&s.expr)? {
                    return Ok(());
                }
                let ty = self.expr_type(&s.expr)?;
                self.emit_expr(&s.expr)?;
                if ty != SemanticType::Unit {
                    self.func.instruction(&Instruction::Drop);
                }
            }
            Stmt::If(s) => {
                self.emit_if_stmt(s)?;
            }
            Stmt::While(s) => {
                self.emit_while_stmt(s)?;
            }
            Stmt::ForRange(s) => {
                self.emit_for_range_stmt(s)?;
            }
        }
        Ok(())
    }

    fn emit_if_stmt(&mut self, s: &IfStmt<'src>) -> Result<(), WasmCodegenError> {
        if s.branches.is_empty() {
            return Ok(());
        }
        self.emit_if_chain(&s.branches, s.else_block.as_ref())
    }

    fn emit_if_chain(
        &mut self,
        branches: &[IfBranch<'src>],
        else_block: Option<&Block<'src>>,
    ) -> Result<(), WasmCodegenError> {
        let first = &branches[0];
        self.emit_expr(&first.condition)?;
        self.func.instruction(&Instruction::If(BlockType::Empty));
        self.emit_block(&first.body)?;
        if branches.len() > 1 || else_block.is_some() {
            self.func.instruction(&Instruction::Else);
            if branches.len() > 1 {
                self.emit_if_chain(&branches[1..], else_block)?;
            } else if let Some(else_block) = else_block {
                self.emit_block(else_block)?;
            }
        }
        self.func.instruction(&Instruction::End);
        Ok(())
    }

    fn emit_while_stmt(&mut self, s: &WhileStmt<'src>) -> Result<(), WasmCodegenError> {
        self.func.instruction(&Instruction::Block(BlockType::Empty));
        self.func.instruction(&Instruction::Loop(BlockType::Empty));
        self.emit_expr(&s.condition)?;
        self.func.instruction(&Instruction::I32Eqz);
        self.func.instruction(&Instruction::BrIf(1));
        self.emit_block(&s.body)?;
        self.func.instruction(&Instruction::Br(0));
        self.func.instruction(&Instruction::End); // loop
        self.func.instruction(&Instruction::End); // block
        Ok(())
    }

    fn emit_for_range_stmt(&mut self, s: &ForRangeStmt<'src>) -> Result<(), WasmCodegenError> {
        let binding_slot = *self
            .planned
            .local_plan
            .binding_slots
            .get(&BindingKey::from_span(s.binding_span))
            .ok_or_else(|| {
                WasmCodegenError::new(
                    WasmCodegenErrorKind::UnknownLocalBinding,
                    Some(s.binding_span),
                    "missing for-range binding local",
                )
            })?;
        let temps = *self
            .planned
            .local_plan
            .for_loop_temps
            .get(&BindingKey::from_span(s.range.span))
            .ok_or_else(|| {
                WasmCodegenError::new(
                    WasmCodegenErrorKind::UnsupportedConstruct,
                    Some(s.range.span),
                    "missing for-range temp locals",
                )
            })?;

        // Initialize loop state.
        match s.range.args.len() {
            1 => {
                self.func.instruction(&Instruction::I32Const(0));
                self.func
                    .instruction(&Instruction::LocalSet(binding_slot.local_index));
                self.emit_expr(&s.range.args[0])?;
                self.func
                    .instruction(&Instruction::LocalSet(temps.end_local));
                self.func.instruction(&Instruction::I32Const(1));
                self.func
                    .instruction(&Instruction::LocalSet(temps.step_local));
            }
            2 => {
                self.emit_expr(&s.range.args[0])?;
                self.func
                    .instruction(&Instruction::LocalSet(binding_slot.local_index));
                self.emit_expr(&s.range.args[1])?;
                self.func
                    .instruction(&Instruction::LocalSet(temps.end_local));
                self.func.instruction(&Instruction::I32Const(1));
                self.func
                    .instruction(&Instruction::LocalSet(temps.step_local));
            }
            3 => {
                self.emit_expr(&s.range.args[0])?;
                self.func
                    .instruction(&Instruction::LocalSet(binding_slot.local_index));
                self.emit_expr(&s.range.args[1])?;
                self.func
                    .instruction(&Instruction::LocalSet(temps.end_local));
                self.emit_expr(&s.range.args[2])?;
                self.func
                    .instruction(&Instruction::LocalSet(temps.step_local));
            }
            _ => {
                return Err(WasmCodegenError::new(
                    WasmCodegenErrorKind::UnsupportedConstruct,
                    Some(s.range.span),
                    "range() arity must be 1..=3 in V1 codegen",
                ));
            }
        }

        self.push_scope();
        self.current_scope_mut().insert(s.binding, binding_slot);

        // Soft fallback: step==0 logs and skips the loop.
        self.func
            .instruction(&Instruction::LocalGet(temps.step_local));
        self.func.instruction(&Instruction::I32Eqz);
        self.func.instruction(&Instruction::If(BlockType::Empty));
        self.emit_soft_exception_log(SOFT_EXCEPTION_INT_DIV_ZERO, s.range.span.line as i32);
        self.func.instruction(&Instruction::Else);

        self.func.instruction(&Instruction::Block(BlockType::Empty));
        self.func.instruction(&Instruction::Loop(BlockType::Empty));

        // cond = step > 0 ? i < end : i > end
        self.func
            .instruction(&Instruction::LocalGet(temps.step_local));
        self.func.instruction(&Instruction::I32Const(0));
        self.func.instruction(&Instruction::I32GtS);
        self.func
            .instruction(&Instruction::If(BlockType::Result(ValType::I32)));
        self.func
            .instruction(&Instruction::LocalGet(binding_slot.local_index));
        self.func
            .instruction(&Instruction::LocalGet(temps.end_local));
        self.func.instruction(&Instruction::I32LtS);
        self.func.instruction(&Instruction::Else);
        self.func
            .instruction(&Instruction::LocalGet(binding_slot.local_index));
        self.func
            .instruction(&Instruction::LocalGet(temps.end_local));
        self.func.instruction(&Instruction::I32GtS);
        self.func.instruction(&Instruction::End);

        self.func.instruction(&Instruction::I32Eqz);
        self.func.instruction(&Instruction::BrIf(1));

        self.emit_block(&s.body)?;

        self.func
            .instruction(&Instruction::LocalGet(binding_slot.local_index));
        self.func
            .instruction(&Instruction::LocalGet(temps.step_local));
        self.func.instruction(&Instruction::I32Add);
        self.func
            .instruction(&Instruction::LocalSet(binding_slot.local_index));
        self.func.instruction(&Instruction::Br(0));

        self.func.instruction(&Instruction::End); // loop
        self.func.instruction(&Instruction::End); // block
        self.func.instruction(&Instruction::End); // else branch of step==0 guard
        self.func.instruction(&Instruction::End); // if step==0

        self.pop_scope();
        Ok(())
    }

    fn try_emit_release_handle_stmt(
        &mut self,
        expr: &Expr<'src>,
    ) -> Result<bool, WasmCodegenError> {
        let ExprKind::Call { callee, args } = &expr.kind else {
            return Ok(false);
        };
        let ExprKind::Identifier(name) = &callee.kind else {
            return Ok(false);
        };
        if !is_handle_release_name(name) {
            return Ok(false);
        }
        let Some(host) = self.host_imports.get(name) else {
            return Err(WasmCodegenError::new(
                WasmCodegenErrorKind::MissingHostImportSignature,
                Some(callee.span),
                format!("missing host import signature for handle release call `{name}`"),
            ));
        };
        if args.len() != 1 {
            return Err(WasmCodegenError::new(
                WasmCodegenErrorKind::UnsupportedConstruct,
                Some(expr.span),
                "handle release call requires exactly one argument",
            ));
        }
        self.emit_expr(&args[0])?;
        self.func
            .instruction(&Instruction::Call(host.function_index));
        Ok(true)
    }

    fn emit_expr(&mut self, expr: &Expr<'src>) -> Result<(), WasmCodegenError> {
        match &expr.kind {
            ExprKind::Identifier(name) => {
                let slot = self.resolve_local(name).ok_or_else(|| {
                    WasmCodegenError::new(
                        WasmCodegenErrorKind::UnknownLocalBinding,
                        Some(expr.span),
                        format!("unknown local `{name}`"),
                    )
                })?;
                self.func
                    .instruction(&Instruction::LocalGet(slot.local_index));
            }
            ExprKind::IntegerLiteral(v) => {
                let value = parse_i32_literal(v).ok_or_else(|| {
                    WasmCodegenError::new(
                        WasmCodegenErrorKind::InvalidLiteral,
                        Some(expr.span),
                        format!("invalid int literal `{v}`"),
                    )
                })?;
                self.func.instruction(&Instruction::I32Const(value));
            }
            ExprKind::FloatLiteral(v) => {
                let value = parse_f32_literal(v).ok_or_else(|| {
                    WasmCodegenError::new(
                        WasmCodegenErrorKind::InvalidLiteral,
                        Some(expr.span),
                        format!("invalid float literal `{v}`"),
                    )
                })?;
                self.func.instruction(&Instruction::F32Const(value));
            }
            ExprKind::BoolLiteral(v) => {
                self.func
                    .instruction(&Instruction::I32Const(if *v { 1 } else { 0 }));
            }
            ExprKind::StringLiteral(s) => {
                let ptr = self.string_pool.lookup(s).ok_or_else(|| {
                    WasmCodegenError::new(
                        WasmCodegenErrorKind::UnsupportedConstruct,
                        Some(expr.span),
                        format!("string literal `{s}` missing from planner string pool"),
                    )
                })?;
                self.func.instruction(&Instruction::I32Const(ptr as i32));
            }
            ExprKind::Grouping(inner) => self.emit_expr(inner)?,
            ExprKind::Unary {
                op: UnaryOp::Neg,
                expr: inner,
            } => {
                let ty = self.expr_type(expr)?;
                self.emit_expr(inner)?;
                match ty {
                    SemanticType::Int => {
                        self.func.instruction(&Instruction::I32Const(-1));
                        self.func.instruction(&Instruction::I32Mul);
                    }
                    SemanticType::Float => {
                        self.func.instruction(&Instruction::F32Neg);
                    }
                    _ => {
                        return Err(WasmCodegenError::new(
                            WasmCodegenErrorKind::UnsupportedType,
                            Some(expr.span),
                            "unary negation supports only int/float",
                        ));
                    }
                }
            }
            ExprKind::Binary { op, left, right } => {
                let lt = self.expr_type(left)?;
                let rt = self.expr_type(right)?;
                self.emit_binary(expr.span, *op, left, right, lt, rt)?;
            }
            ExprKind::Call { callee, args } => {
                let ExprKind::Identifier(name) = &callee.kind else {
                    return Err(WasmCodegenError::new(
                        WasmCodegenErrorKind::UnsupportedConstruct,
                        Some(callee.span),
                        "only direct calls are supported in V1 codegen",
                    ));
                };
                if is_handle_release_name(name) {
                    return Err(WasmCodegenError::new(
                        WasmCodegenErrorKind::UnsupportedConstruct,
                        Some(expr.span),
                        "handle release calls are only allowed as standalone statements",
                    ));
                }
                for arg in args {
                    self.emit_expr(arg)?;
                }
                if let Some(index) = self.internal_fn_indices.get(name).copied() {
                    self.func.instruction(&Instruction::Call(index));
                } else if let Some(host) = self.host_imports.get(name) {
                    self.func
                        .instruction(&Instruction::Call(host.function_index));
                } else {
                    return Err(WasmCodegenError::new(
                        WasmCodegenErrorKind::MissingHostImportSignature,
                        Some(callee.span),
                        format!("missing host import signature for `{name}`"),
                    ));
                }
            }
        }
        Ok(())
    }

    fn emit_binary(
        &mut self,
        span: Span,
        op: BinaryOp,
        left: &Expr<'src>,
        right: &Expr<'src>,
        lt: SemanticType,
        rt: SemanticType,
    ) -> Result<(), WasmCodegenError> {
        use BinaryOp as B;
        use SemanticType as T;

        match op {
            B::AndAnd => {
                self.emit_expr(left)?;
                self.func
                    .instruction(&Instruction::If(BlockType::Result(ValType::I32)));
                self.emit_expr(right)?;
                self.func.instruction(&Instruction::Else);
                self.func.instruction(&Instruction::I32Const(0));
                self.func.instruction(&Instruction::End);
                return Ok(());
            }
            B::OrOr => {
                self.emit_expr(left)?;
                self.func
                    .instruction(&Instruction::If(BlockType::Result(ValType::I32)));
                self.func.instruction(&Instruction::I32Const(1));
                self.func.instruction(&Instruction::Else);
                self.emit_expr(right)?;
                self.func.instruction(&Instruction::End);
                return Ok(());
            }
            _ => {}
        }

        if self.config.enable_simd_hook_markers && matches!(op, B::Add | B::Sub | B::Mul | B::Div) {
            *self.simd_hook_sites = self.simd_hook_sites.saturating_add(1);
            self.warnings.push(WasmCodegenWarning {
                kind: WasmCodegenWarningKind::SimdHookSiteRecorded,
                span: Some(span),
                function: Some(self.planned.func.name.to_string()),
            });
        }

        match (op, lt, rt) {
            (B::Div, T::Int, T::Int) | (B::Mod, T::Int, T::Int) => {
                let l = self.planned.local_plan.scratch_i32_left;
                let r = self.planned.local_plan.scratch_i32_right;
                self.emit_expr(left)?;
                self.func.instruction(&Instruction::LocalSet(l));
                self.emit_expr(right)?;
                self.func.instruction(&Instruction::LocalSet(r));

                self.func.instruction(&Instruction::LocalGet(r));
                self.func.instruction(&Instruction::I32Eqz);
                self.func
                    .instruction(&Instruction::If(BlockType::Result(ValType::I32)));
                self.emit_soft_exception_log(
                    if matches!(op, B::Div) {
                        SOFT_EXCEPTION_INT_DIV_ZERO
                    } else {
                        SOFT_EXCEPTION_INT_MOD_ZERO
                    },
                    span.line as i32,
                );
                self.func.instruction(&Instruction::I32Const(0));
                self.func.instruction(&Instruction::Else);
                self.func.instruction(&Instruction::LocalGet(l));
                self.func.instruction(&Instruction::LocalGet(r));
                match op {
                    B::Div => {
                        self.func.instruction(&Instruction::I32DivS);
                    }
                    B::Mod => {
                        self.func.instruction(&Instruction::I32RemS);
                    }
                    _ => unreachable!(),
                };
                self.func.instruction(&Instruction::End);

                self.warnings.push(WasmCodegenWarning {
                    kind: WasmCodegenWarningKind::SoftFallbackGuardInserted,
                    span: Some(span),
                    function: Some(self.planned.func.name.to_string()),
                });
                return Ok(());
            }
            _ => {}
        }

        self.emit_expr(left)?;
        self.emit_expr(right)?;

        match (op, lt, rt) {
            (B::Add, T::Int, T::Int) => {
                self.func.instruction(&Instruction::I32Add);
            }
            (B::Sub, T::Int, T::Int) => {
                self.func.instruction(&Instruction::I32Sub);
            }
            (B::Mul, T::Int, T::Int) => {
                self.func.instruction(&Instruction::I32Mul);
            }
            (B::Div, T::Float, T::Float) => {
                self.func.instruction(&Instruction::F32Div);
            }
            (B::Add, T::Float, T::Float) => {
                self.func.instruction(&Instruction::F32Add);
            }
            (B::Sub, T::Float, T::Float) => {
                self.func.instruction(&Instruction::F32Sub);
            }
            (B::Mul, T::Float, T::Float) => {
                self.func.instruction(&Instruction::F32Mul);
            }
            (B::EqEq, T::Int, T::Int)
            | (B::EqEq, T::Bool, T::Bool)
            | (B::EqEq, T::Handle, T::Handle)
            | (B::EqEq, T::Str, T::Str) => {
                self.func.instruction(&Instruction::I32Eq);
            }
            (B::NotEq, T::Int, T::Int)
            | (B::NotEq, T::Bool, T::Bool)
            | (B::NotEq, T::Handle, T::Handle)
            | (B::NotEq, T::Str, T::Str) => {
                self.func.instruction(&Instruction::I32Ne);
            }
            (B::EqEq, T::Float, T::Float) => {
                self.func.instruction(&Instruction::F32Eq);
            }
            (B::NotEq, T::Float, T::Float) => {
                self.func.instruction(&Instruction::F32Ne);
            }
            (B::Lt, T::Int, T::Int) => {
                self.func.instruction(&Instruction::I32LtS);
            }
            (B::LtEq, T::Int, T::Int) => {
                self.func.instruction(&Instruction::I32LeS);
            }
            (B::Gt, T::Int, T::Int) => {
                self.func.instruction(&Instruction::I32GtS);
            }
            (B::GtEq, T::Int, T::Int) => {
                self.func.instruction(&Instruction::I32GeS);
            }
            (B::Lt, T::Float, T::Float) => {
                self.func.instruction(&Instruction::F32Lt);
            }
            (B::LtEq, T::Float, T::Float) => {
                self.func.instruction(&Instruction::F32Le);
            }
            (B::Gt, T::Float, T::Float) => {
                self.func.instruction(&Instruction::F32Gt);
            }
            (B::GtEq, T::Float, T::Float) => {
                self.func.instruction(&Instruction::F32Ge);
            }
            (B::Mod, T::Int, T::Int) => {
                self.func.instruction(&Instruction::I32RemS);
            }
            _ => {
                return Err(WasmCodegenError::new(
                    WasmCodegenErrorKind::UnsupportedConstruct,
                    Some(span),
                    format!(
                        "unsupported binary op/type combination: {:?} {:?} {:?}",
                        op, lt, rt
                    ),
                ))
            }
        }
        Ok(())
    }

    fn expr_type(&self, expr: &Expr<'src>) -> Result<SemanticType, WasmCodegenError> {
        match &expr.kind {
            ExprKind::Identifier(name) => {
                self.resolve_local(name).map(|slot| slot.ty).ok_or_else(|| {
                    WasmCodegenError::new(
                        WasmCodegenErrorKind::UnknownLocalBinding,
                        Some(expr.span),
                        format!("unknown local `{name}`"),
                    )
                })
            }
            ExprKind::IntegerLiteral(_) => Ok(SemanticType::Int),
            ExprKind::FloatLiteral(_) => Ok(SemanticType::Float),
            ExprKind::BoolLiteral(_) => Ok(SemanticType::Bool),
            ExprKind::StringLiteral(_) => Ok(SemanticType::Str),
            ExprKind::Grouping(inner) => self.expr_type(inner),
            ExprKind::Unary {
                op: UnaryOp::Neg,
                expr: inner,
            } => {
                let ty = self.expr_type(inner)?;
                match ty {
                    SemanticType::Int | SemanticType::Float => Ok(ty),
                    _ => Err(WasmCodegenError::new(
                        WasmCodegenErrorKind::UnsupportedType,
                        Some(expr.span),
                        "unary negation requires int or float",
                    )),
                }
            }
            ExprKind::Binary { op, left, right } => {
                let lt = self.expr_type(left)?;
                let rt = self.expr_type(right)?;
                infer_binary_type_codegen(*op, lt, rt, expr.span)
            }
            ExprKind::Call { callee, .. } => {
                let ExprKind::Identifier(name) = &callee.kind else {
                    return Err(WasmCodegenError::new(
                        WasmCodegenErrorKind::UnsupportedConstruct,
                        Some(callee.span),
                        "only direct calls are supported",
                    ));
                };
                if let Some(sig) = self.sig_map.get(name).copied() {
                    Ok(sig.return_type)
                } else if let Some(host) = self.host_imports.get(name) {
                    Ok(host.result)
                } else {
                    Err(WasmCodegenError::new(
                        WasmCodegenErrorKind::MissingHostImportSignature,
                        Some(callee.span),
                        format!("missing host import signature for `{name}`"),
                    ))
                }
            }
        }
    }

    fn emit_default_return_if_needed(&mut self, ty: SemanticType, span: Option<Span>) {
        match ty {
            SemanticType::Unit => {}
            SemanticType::Int | SemanticType::Bool | SemanticType::Handle | SemanticType::Str => {
                self.func.instruction(&Instruction::I32Const(0));
                self.warnings.push(WasmCodegenWarning {
                    kind: WasmCodegenWarningKind::ImplicitDefaultReturnInserted,
                    span,
                    function: Some(self.planned.func.name.to_string()),
                });
            }
            SemanticType::Float => {
                self.func.instruction(&Instruction::F32Const(0.0));
                self.warnings.push(WasmCodegenWarning {
                    kind: WasmCodegenWarningKind::ImplicitDefaultReturnInserted,
                    span,
                    function: Some(self.planned.func.name.to_string()),
                });
            }
        }
    }

    fn emit_soft_exception_log(&mut self, code: i32, line: i32) {
        self.func.instruction(&Instruction::I32Const(code));
        self.func.instruction(&Instruction::I32Const(line));
        self.func
            .instruction(&Instruction::Call(self.logger_func_index));
    }

    fn resolve_local(&self, name: &str) -> Option<LocalBindingSlot> {
        self.scopes.iter().rev().find_map(|s| s.get(name).copied())
    }

    fn push_scope(&mut self) {
        self.scopes.push(HashMap::new());
    }

    fn pop_scope(&mut self) {
        let _ = self.scopes.pop();
    }

    fn current_scope_mut(&mut self) -> &mut HashMap<&'src str, LocalBindingSlot> {
        self.scopes.last_mut().expect("scope exists")
    }
}

fn emit_export_wrapper_body<'src>(
    pf: &PlannedFunction<'src>,
    internal_index: u32,
) -> Result<WasmFunction, WasmCodegenError> {
    let mut func = WasmFunction::new(Vec::new());
    for idx in 0..pf.param_tys.len() {
        func.instruction(&Instruction::LocalGet(idx as u32));
    }
    func.instruction(&Instruction::Call(internal_index));
    if pf.sig.return_type == SemanticType::Unit {
        // nothing on stack
    }
    func.instruction(&Instruction::End);
    Ok(func)
}

#[derive(Default)]
struct StringPool {
    next_offset: u32,
    interned: HashMap<String, u32>,
    segments: Vec<StringSegment>,
    total_bytes: u32,
}

#[derive(Debug, Clone)]
struct StringSegment {
    offset: u32,
    bytes: Vec<u8>,
}

impl StringPool {
    fn new(base: u32) -> Self {
        Self {
            next_offset: base,
            interned: HashMap::new(),
            segments: Vec::new(),
            total_bytes: 0,
        }
    }

    fn intern(&mut self, s: &str) -> u32 {
        if let Some(offset) = self.interned.get(s).copied() {
            return offset;
        }
        let offset = self.next_offset;
        let mut bytes = Vec::with_capacity(4 + s.len() + 1);
        bytes.extend_from_slice(&(s.len() as u32).to_le_bytes());
        bytes.extend_from_slice(s.as_bytes());
        bytes.push(0);

        self.total_bytes = self.total_bytes.saturating_add(bytes.len() as u32);
        self.next_offset = align4(offset.saturating_add(bytes.len() as u32));
        self.segments.push(StringSegment { offset, bytes });
        self.interned.insert(s.to_string(), offset);
        offset
    }

    fn lookup(&self, s: &str) -> Option<u32> {
        self.interned.get(s).copied()
    }

    fn segments(&self) -> &[StringSegment] {
        &self.segments
    }

    fn total_data_bytes(&self) -> u32 {
        self.total_bytes
    }
}

fn build_metadata_section(
    exports: &[CodegenExportEntry],
    simd_hook_sites: u32,
    config: &WasmCodegenConfig,
) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(b"tileline-codegen-v1\n");
    out.extend_from_slice(format!("simd_hook_sites={simd_hook_sites}\n").as_bytes());
    out.extend_from_slice(format!("memory_export={}\n", config.memory_export_name).as_bytes());
    for e in exports {
        out.extend_from_slice(
            format!(
                "export:{} wrapper:{} async={}\n",
                e.export_name, e.wrapper_symbol, e.mps_async_task_entry
            )
            .as_bytes(),
        );
    }
    out
}

fn infer_binary_type_codegen(
    op: BinaryOp,
    left: SemanticType,
    right: SemanticType,
    span: Span,
) -> Result<SemanticType, WasmCodegenError> {
    use BinaryOp as B;
    use SemanticType as T;
    let bad = || {
        WasmCodegenError::new(
            WasmCodegenErrorKind::UnsupportedType,
            Some(span),
            format!(
                "unsupported binary operand types {:?} and {:?} for {:?}",
                left, right, op
            ),
        )
    };

    match op {
        B::Add | B::Sub | B::Mul | B::Div => match (left, right) {
            (T::Int, T::Int) => Ok(T::Int),
            (T::Float, T::Float) => Ok(T::Float),
            _ => Err(bad()),
        },
        B::Mod => match (left, right) {
            (T::Int, T::Int) => Ok(T::Int),
            _ => Err(bad()),
        },
        B::EqEq | B::NotEq => {
            if left == right {
                Ok(T::Bool)
            } else {
                Err(bad())
            }
        }
        B::Lt | B::LtEq | B::Gt | B::GtEq => match (left, right) {
            (T::Int, T::Int) | (T::Float, T::Float) => Ok(T::Bool),
            _ => Err(bad()),
        },
        B::AndAnd | B::OrOr => match (left, right) {
            (T::Bool, T::Bool) => Ok(T::Bool),
            _ => Err(bad()),
        },
    }
}

#[inline]
fn map_type_name_codegen(ty: TypeName) -> SemanticType {
    match ty {
        TypeName::Int => SemanticType::Int,
        TypeName::Float => SemanticType::Float,
        TypeName::Bool => SemanticType::Bool,
        TypeName::Str => SemanticType::Str,
    }
}

#[inline]
fn map_semantic_type_to_valtype(ty: SemanticType) -> ValType {
    match ty {
        SemanticType::Int
        | SemanticType::Bool
        | SemanticType::Handle
        | SemanticType::Str
        | SemanticType::Unit => ValType::I32,
        SemanticType::Float => ValType::F32,
    }
}

#[inline]
fn parse_i32_literal(text: &str) -> Option<i32> {
    let clean: String = text.chars().filter(|&c| c != '_').collect();
    clean.parse::<i32>().ok()
}

#[inline]
fn parse_f32_literal(text: &str) -> Option<f32> {
    let clean: String = text.chars().filter(|&c| c != '_').collect();
    clean.parse::<f32>().ok()
}

#[inline]
fn align4(v: u32) -> u32 {
    (v + 3) & !3
}

#[inline]
fn is_handle_release_name(name: &str) -> bool {
    matches!(name, "release_handle" | "destroy_handle" | "free_handle")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tlscript::{Lexer, Parser, SemanticAnalyzer, SemanticConfig};

    fn parse_and_semantic(src: &str) -> (Module<'_>, SemanticReport<'_>) {
        let mut parser = Parser::new(Lexer::new(src));
        let module = parser.parse_module().expect("parse ok");
        let report = SemanticAnalyzer::default()
            .analyze(&module)
            .expect("semantic ok");
        (module, report)
    }

    #[test]
    fn generates_wasm_bytes_and_export_wrapper() {
        let (module, report) = parse_and_semantic(concat!(
            "@export\n",
            "def update(dt: float):\n",
            "    let x = 1\n",
            "    if dt > 0.0:\n",
            "        x = x + 1\n",
            "    while x < 3:\n",
            "        x = x + 1\n",
        ));
        let out = WasmGenerator::default()
            .generate(&module, &report)
            .expect("codegen ok");
        assert!(out.wasm_bytes.len() > 8);
        assert_eq!(&out.wasm_bytes[0..4], b"\0asm");
        assert_eq!(out.exports.len(), 1);
        assert_eq!(out.exports[0].export_name, "update");
    }

    #[test]
    fn soft_fallback_guard_warning_for_int_division() {
        let (module, report) = parse_and_semantic(concat!("def f():\n", "    let x = 10 / 0\n",));
        let outcome = WasmGenerator::default().generate_soft(&module, &report);
        assert!(outcome.can_instantiate);
        assert!(outcome
            .warnings
            .iter()
            .any(|w| matches!(w.kind, WasmCodegenWarningKind::SoftFallbackGuardInserted)));
    }

    #[test]
    fn codegen_fails_for_missing_host_signature() {
        let src = concat!("def f():\n", "    host_tick(1)\n",);
        let mut parser = Parser::new(Lexer::new(src));
        let module = parser.parse_module().expect("parse ok");
        let report = SemanticAnalyzer::new(SemanticConfig {
            external_call_allowlist: vec!["host_tick".to_string()],
            ..SemanticConfig::default()
        })
        .analyze(&module)
        .expect("semantic ok");
        let err = WasmGenerator::default()
            .generate(&module, &report)
            .expect_err("codegen err");
        assert_eq!(err.kind, WasmCodegenErrorKind::MissingHostImportSignature);
    }
}
