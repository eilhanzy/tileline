# ParadoxPE `.tlscript` Examples

This document collects the first real `.tlscript` examples that target the current ParadoxPE host
ABI and runtime planning surface.

The goal is not to show a finished gameplay language yet. The goal is to provide source files that:

- use the current ParadoxPE host ABI names
- compile through the real parser/semantic/lowering/codegen pipeline
- demonstrate the current `@parallel(domain="bodies")` contract shape

The example sources live in `docs/examples/tlscript/paradoxpe/` and are compiled in tests through
`include_str!(...)`, so they remain RAM-fed examples instead of runtime disk I/O dependencies.

## Included Scripts

### `scene_bootstrap.tlscript`

Path: `docs/examples/tlscript/paradoxpe/scene_bootstrap.tlscript`

Purpose:

- spawn a floor body
- attach colliders
- spawn a dynamic crate
- apply an initial force

This is the simplest "bring up a physics scene" example against the current handle-based ABI.

### `force_pulse.tlscript`

Path: `docs/examples/tlscript/paradoxpe/force_pulse.tlscript`

Purpose:

- spawn a body
- apply force and direct velocity
- query contact snapshot handles
- consume `contact_count(...)`

This example is useful for validating handle-acquire calls and contact snapshot flow.

### `tick_world.tlscript`

Path: `docs/examples/tlscript/paradoxpe/tick_world.tlscript`

Purpose:

- drive `step_world(dt)` from an exported script function

This is the smallest fixed-step host-call example.

### `solve_bodies_parallel.tlscript`

Path: `docs/examples/tlscript/paradoxpe/solve_bodies_parallel.tlscript`

Purpose:

- show the canonical `@parallel(domain="bodies")` contract
- keep read/write effects aligned with current ParadoxPE SoA safety shape
- verify runtime planning preserves body-domain specialization

Current contract:

```tlscript
@export
@parallel(domain="bodies", read="transform,force,aabb", write="velocity", chunk=128, schedule="auto")
@deterministic
def solve_bodies(dt: float):
    let inv_dt: float = 1.0 / dt
    let damping: float = inv_dt * 0.25
```

Runtime meaning today:

- `domain="bodies"` is preserved into typed IR
- `schedule="auto"` is promoted to `performance`
- velocity writes are accepted
- unsupported body write sets fall back to serial

## Verification Path

These example scripts are compiled end-to-end in:

- `tl-core/tests/paradoxpe_tlscript_examples.rs`

The test pipeline covers:

1. lexer
2. parser
3. semantic analysis
4. parallel hook analysis
5. typed IR lowering
6. typed IR annotation
7. WASM code generation

This keeps the examples implementation-adjacent instead of becoming unverified documentation.

## Current Limitations

These examples reflect the current language/runtime limits:

- no explicit `handle` type syntax in source yet
- no script-level indexed body iteration primitive yet
- `@parallel(domain="bodies")` is currently a contract + planner path, not a full per-body DSL
- ParadoxPE host ABI is still scalarized and intentionally minimal

Those limits are acceptable for now because the examples are meant to stabilize the ABI and
compiler/runtime integration before the scripting surface grows.
