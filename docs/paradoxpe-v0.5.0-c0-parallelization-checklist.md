# ParadoxPE v0.5.0 C0 Parallelization Checklist

This checklist tracks the `C0. Full Parallelization Hard Gate` in `docs/tileline-v0.5.0-roadmap.md`.

## Goal

Make `broadphase | narrowphase | solver | integrate` parallel-by-default in shipping presets, with explicit and telemetry-visible serial fallback reasons.

## Current Audit (2026-04-10)

- `broadphase`: parallel helper path exists, but serial fallback could be implicit.
- `narrowphase`: parallel filter-map path exists, but serial fallback could be implicit.
- `solver`: parallel Jacobi path exists, but switches to sequential under threshold/single-worker.
- `integrate`: deterministic shard plan exists, but execution path is still effectively serial in the current implementation.

## Work Items

- [x] C0-S1: Add phase-level parallel execution telemetry primitives.
  - Added `ParallelExecutionMode` and `serial_fallback_reason()` in `paradoxpe/src/parallel.rs`.
  - Helper APIs now report execution mode:
    - `for_each_mut_indexed`
    - `for_each_index`
    - `collect_filter_map`

- [x] C0-S2: Thread execution mode through phase stats.
  - `BroadphaseStats` now reports pair/sweep execution mode + fallback reason.
  - `NarrowphaseStats` now reports manifold-build execution mode + fallback reason.
  - `ContactSolverStats` now reports solve/projection execution mode + fallback reason.

- [x] C0-S3: Expose integrate fallback mode explicitly.
  - `BodyRegistry::integrate_with_shards` now returns `ParallelExecutionMode`.
  - Current fast path reports `SerialUnimplemented` when shard plan is parallel-safe but parallel body integration is not yet wired.

- [x] C0-S4: Surface per-step phase mode in world timings.
  - `PhysicsStepTimings` now carries per-phase mode + serial fallback reason:
    - integrate
    - broadphase
    - narrowphase
    - solver

- [x] C0-S5: Implement true parallel shard integration for `integrate`.
  - `BodyRegistry::integrate_with_shards` now runs deterministic disjoint chunk execution in parallel
    when shard plan + workload thresholds are satisfied.
  - Serial fallback remains explicit (`SerialUnsupportedPlan`, `SerialSingleWorker`,
    `SerialSmallWorkload`) and no longer reports `SerialUnimplemented`.

- [x] C0-S6: Add runtime/console telemetry lines for C0 gate fields.
  - TLApp title + FPS log now print per-phase mode/reason/serial-us for
    `integrate|broadphase|narrowphase|solver`.
  - Console `status` / `perf.report` now include the same C0 gate telemetry fields.

- [ ] C0-S7: Add C0 gate tests.
  - Unit tests for mode transitions (`Parallel`, `SerialSmallWorkload`, `SerialSingleWorker`, etc.).
  - Stress test (`30k`) that fails if hidden serial hot-phase ownership appears without telemetry reason.

## Exit Criteria

- No hot phase silently executes sequentially.
- Serial fallback, when used, always has an explicit reason in telemetry.
- `integrate` phase no longer reports `SerialUnimplemented` on shipping preset workloads.
