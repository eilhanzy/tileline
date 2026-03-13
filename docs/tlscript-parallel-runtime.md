# `.tlscript` Parallel Contracts, Advisor, and Runtime Planning

This document describes the `.tlscript` parallelization workflow added to make Tileline's
parallel-first architecture accessible without hiding performance controls.

The goal is a two-layer model:

- safe defaults (serial/main-thread fallback when uncertain)
- explicit opt-in contracts for high performance (`@parallel(...)`)

## Why This Layer Exists

Tileline is designed around MPS/GMS parallel scaling, but full manual parallel contracts are hard
for end users to write correctly from day one.

The compiler/runtime stack now includes:

1. strict parallel-contract validation (`parallel_hook`)
2. developer-facing explanation and suggestions (`parallel_advisor`)
3. runtime dispatch planner + fallback reason counters (`parallel_runtime`)

This lets the engine stay performance-oriented while remaining approachable.

## Decorator Surface (Current)

### Execution decorators

Supported function decorators:

- `@parallel(...)`
- `@main_thread`
- `@deterministic`
- `@reduce(...)`

Examples:

```tlscript
@export
@parallel(domain="bodies", read="transform,force", write="velocity", chunk=256, schedule="performance")
@deterministic
@reduce(sum)
def solve_forces(dt: float):
    ...
```

```tlscript
@main_thread
def draw_ui():
    ...
```

### Decorator argument support

The parser/AST supports generic decorator arguments (flags + key/value) and statement decorators
(e.g. `@net(...) let ...`) via shared decorator parsing infrastructure.

Implementation: `tl-core/src/tlscript/ast.rs`, `tl-core/src/tlscript/parser.rs`

## `parallel_hook` (Strict Validation)

`parallel_hook` validates execution decorators and emits a strict per-function contract.

### Validated rules (examples)

- `@parallel` and `@main_thread` are mutually exclusive
- `@reduce(...)` requires `@parallel`
- `@parallel` requires `domain=...`
- `chunk` must be a positive integer (`u16` range)
- `schedule` must be `auto|performance|efficient`
- reductions require deterministic semantics (or trigger an implied-deterministic warning path)

### Output

`ParallelFunctionHook` includes:

- execution policy
- domain
- read/write effect lists
- deterministic flag
- chunk hint
- MPS scheduling hint
- reduction strategy

Implementation: `tl-core/src/tlscript/parallel_hook.rs`

## Typed IR Execution Metadata

Validated contracts are projected onto typed IR function metadata so runtime/MPS planning can work
without re-reading decorators.

Key types:

- `IrExecutionPolicy` (`Serial`, `ParallelSafe`, `MainThreadOnly`)
- `TypedIrExecutionMeta`
- `TypedIrFunctionMeta` (now includes `execution`)

Helper:

- `annotate_typed_ir_with_parallel_hooks(...)`

Implementation: `tl-core/src/tlscript/typed_ir.rs`, `tl-core/src/tlscript/parallel_hook.rs`

## `parallel_advisor` (Accessibility Layer)

The advisor is a non-fatal, developer-facing layer that explains serial fallback and generates
copy/paste-ready suggestions.

### What it reports

Per function:

- `ParallelReady`
- `MainThreadOnly`
- `SerialFallback`

If serial fallback occurs, the advisor records a reason such as:

- `MissingParallelDecorator`
- `MissingExportDecorator`
- `InvalidParallelContract`
- `AutoParallelNeedsContract` (for `@parallel(auto)`)
- `MainThreadHostCalls`
- `UnknownHostCalls`

### Suggestions

The advisor can suggest:

- `@export`
- `@main_thread`
- a starter `@parallel(...)` contract
- a replacement template for `@parallel(auto)`
- host call classification hints (`main_thread_only` or explicit annotation)

### `@parallel(auto)` workflow support

`ParallelAdvisorReport` now exposes:

- `summary_line()` for compact logs/tooling panes
- `suggested_contract_templates()` for quick-fix style workflows

This creates a path where users start with `@parallel(auto)` and later accept/generated templates as
explicit contracts.

Implementation: `tl-core/src/tlscript/parallel_advisor.rs`

## `parallel_runtime` (Runtime Dispatch Planner + Fallback Metrics)

`parallel_runtime` is the runtime-facing planning layer used before MPS submission.

### Responsibilities

- choose `MainThread`, `Serial`, or `ParallelChunked`
- respect typed-IR execution metadata and workload size
- track fallback reasons with counters for observability

### Example fallback reasons

- `MainThreadOnlyPolicy`
- `MissingParallelContract`
- `WorkloadTooSmall`
- `SingleChunkOnly`
- `NonDeterministicReduction`

This gives engine developers concrete telemetry for why functions did not use full parallel chunking.

Implementation: `tl-core/src/tlscript/parallel_runtime.rs`

## `runtime` Crate Integration (Engine-Side Glue)

`runtime/src/tlscript_parallel.rs` wraps the advisor + dispatch planner so application/runtime code
can use a single coordinator for script parallel planning, diagnostics, and MPS dispatch routing.

### Coordinator responsibilities

- run `ParallelAdvisor` on parsed/validated modules
- cache compact advisory summary lines for logs/tooling
- count generated contract templates
- plan typed-IR function dispatch via `ParallelDispatchPlanner`
- route native work chunks to MPS (`main-thread`, `serial`, or `parallel batch`)
- expose aggregated metrics (`TlscriptParallelRuntimeMetrics`) and overlay/log helper lines

This keeps the ergonomics/performance planning stack in `src/` instead of examples.

Implementation: `runtime/src/tlscript_parallel.rs`

### MPS routing helper (runtime integration)

`TlscriptParallelRuntimeCoordinator::dispatch_native_chunks_for_function(...)` consumes a
`ParallelDispatchDecision` and a chunk builder closure, then performs the appropriate routing:

- `MainThread`: no MPS submission; runtime keeps execution on the calling thread
- `Serial`: single MPS native task submission
- `ParallelChunked`: batched MPS submission using chunk ranges derived from planner output

The helper maps script scheduling hints to MPS core preferences (`auto`, `performance`,
`efficient`) so `.tlscript` contracts feed directly into CPU core-aware scheduling.

This is intentionally a native/chunked routing primitive so engine code can reuse it for script
execution, staging transforms, or host callback fan-out without duplicating planner policy logic.

### Runtime metrics and tooling overlay support

`TlscriptParallelRuntimeMetrics` now tracks:

- advisor summaries seen
- generated contract templates
- planner fallback counters (mirrored from `ParallelDispatchPlannerMetrics`)
- dispatch route calls
- main-thread-required dispatches
- serial MPS submissions
- parallel batch submissions
- total parallel tasks submitted

Helper methods:

- `planner_fallbacks_line()`: compact planner fallback summary for logs
- `overlay_lines()`: small list of human-readable lines for debug overlays/console panes

This makes the "why did this run serial?" path visible without requiring deep profiling tools.

## Recommended Engine Workflow

1. Parse + semantic analyze script module
2. Run `ParallelHookAnalyzer` (strict contract validation)
3. Lower to typed IR
4. Annotate IR with `annotate_typed_ir_with_parallel_hooks(...)`
5. Run `ParallelAdvisor` (or runtime coordinator wrapper) for diagnostics/templates
6. Before invocation, call `ParallelDispatchPlanner::plan_*` (or coordinator wrapper)
7. Route work through `TlscriptParallelRuntimeCoordinator::dispatch_native_chunks_for_function(...)`
8. Submit to MPS as:
   - main-thread call
   - serial task
   - chunked parallel tasks

## Trade-offs (Intentional)

- Defaults favor safety and accessibility over peak throughput.
- Full performance requires explicit contracts and host API classification.
- This adds compiler/runtime metadata complexity, but greatly improves usability and observability.

This trade-off aligns with Tileline's goals:

- high ceiling for expert users
- approachable defaults for everyone else
