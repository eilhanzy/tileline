use tl_core::{
    annotate_typed_ir_with_parallel_hooks, lower_to_typed_ir, IrEffectMask, IrParallelDomain,
    IrScheduleHint, Lexer, ParallelDispatchMode, ParallelDispatchPlanner, ParallelHookAnalyzer,
    Parser, SemanticAnalyzer, WasmGenerator,
};

const SCENE_BOOTSTRAP: &str =
    include_str!("../../docs/examples/tlscript/paradoxpe/scene_bootstrap.tlscript");
const FORCE_PULSE: &str =
    include_str!("../../docs/examples/tlscript/paradoxpe/force_pulse.tlscript");
const TICK_WORLD: &str = include_str!("../../docs/examples/tlscript/paradoxpe/tick_world.tlscript");
const SOLVE_BODIES_PARALLEL: &str =
    include_str!("../../docs/examples/tlscript/paradoxpe/solve_bodies_parallel.tlscript");

fn compile_pipeline(
    src: &str,
) -> (
    tl_core::Module<'_>,
    tl_core::SemanticReport<'_>,
    tl_core::ParallelHookOutcome<'_>,
    tl_core::TypedIrModule<'_>,
    tl_core::WasmCodegenOutput,
) {
    let mut parser = Parser::new(Lexer::new(src));
    let module = parser.parse_module().expect("parse ok");
    let semantic = SemanticAnalyzer::default()
        .analyze(&module)
        .expect("semantic ok");
    let hooks = ParallelHookAnalyzer::new().analyze(&module, &semantic);
    assert!(hooks.errors.is_empty(), "{:?}", hooks.errors);
    let mut ir = lower_to_typed_ir(&module, &semantic).expect("lower ok");
    annotate_typed_ir_with_parallel_hooks(&mut ir, &hooks);
    let wasm = WasmGenerator::default()
        .generate(&module, &semantic)
        .expect("codegen ok");
    (module, semantic, hooks, ir, wasm)
}

#[test]
fn paradoxpe_examples_compile_through_full_tlscript_pipeline() {
    let cases = [
        ("bootstrap_scene", SCENE_BOOTSTRAP),
        ("force_pulse", FORCE_PULSE),
        ("tick_world", TICK_WORLD),
        ("solve_bodies", SOLVE_BODIES_PARALLEL),
    ];

    for (expected_export, src) in cases {
        let (_module, _semantic, _hooks, _ir, wasm) = compile_pipeline(src);
        assert!(wasm.wasm_bytes.len() > 8, "{expected_export}");
        assert_eq!(&wasm.wasm_bytes[0..4], b"\0asm", "{expected_export}");
        assert_eq!(wasm.exports.len(), 1, "{expected_export}");
        assert_eq!(wasm.exports[0].export_name, expected_export);
    }
}

#[test]
fn body_parallel_example_preserves_domain_metadata_and_parallel_planning() {
    let (_module, _semantic, _hooks, ir, _wasm) = compile_pipeline(SOLVE_BODIES_PARALLEL);
    let func = &ir.functions[0];

    assert_eq!(func.meta.execution.domain, IrParallelDomain::Bodies);
    assert!(func
        .meta
        .execution
        .read_effect_mask
        .contains(IrEffectMask::TRANSFORM));
    assert!(func
        .meta
        .execution
        .read_effect_mask
        .contains(IrEffectMask::FORCE));
    assert!(func
        .meta
        .execution
        .read_effect_mask
        .contains(IrEffectMask::AABB));
    assert!(func
        .meta
        .execution
        .write_effect_mask
        .contains(IrEffectMask::VELOCITY));

    let planner = ParallelDispatchPlanner::default();
    let decision = planner.plan_for_function(func, 2048);
    assert_eq!(decision.mode, ParallelDispatchMode::ParallelChunked);
    assert_eq!(decision.schedule_hint, IrScheduleHint::Performance);
}
