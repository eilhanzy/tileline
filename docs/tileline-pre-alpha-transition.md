# Tileline Pre-Alpha Transition Plan

This document defines the immediate transition from foundation work to a coherent pre-alpha state.
It is intentionally narrower than the full beta roadmap.

## Pre-Alpha Definition

Tileline is considered pre-alpha when all of these hold:

- core runtime path exists in `src/` crates, not benchmark-only glue
- `.tlscript -> semantic -> typed IR -> WASM -> runtime dispatch` runs in one path
- ParadoxPE fixed-step simulation is driven by runtime ownership rules
- GMS/MGS rendering path consumes real runtime scene state
- NPS transport loop can send/receive input + snapshot packets without stalling present
- unified telemetry explains regressions (CPU scheduler, GPU planner/runtime, physics, network)

## Current Status Snapshot

- `MPS`: foundation ready, needs stronger runtime-facing telemetry/fallback reporting
- `GMS`: multi-GPU planner/runtime is active with adaptive secondary governor
- `MGS`: mobile fallback path exists, still needs runtime parity checks vs GMS
- `.tlscript`: compiler pipeline exists, runtime compile/cache/submit integration still partial
- `ParadoxPE`: SoA + broadphase + solver base in place, gameplay-facing control layer incomplete
- `NPS`: packet format/protocol groundwork done, runtime tick/channel integration incomplete
- `runtime` / `tl-core`: bridge path exists, canonical game-loop ownership not fully frozen

## Pre-Alpha Exit Gates

- one canonical scene loop updates physics and renders every frame from runtime-owned state
- `.tlscript` hooks can mutate/query ParadoxPE via stable host ABI calls
- NPS input/snapshot path runs in same loop with bounded queue/latency behavior
- desktop Linux primary path + one secondary target path pass smoke and regression checks
- docs reflect actual implementation and known limitations

## Sprint Plan (Recommended)

### Sprint 1: Runtime Ownership Freeze

- freeze frame ownership boundaries in `runtime` and `tl-core`
- finalize canonical update order: net -> script -> physics -> render plan -> present
- publish telemetry surface contract for frame-level diagnostics

### Sprint 2: `.tlscript` Runtime Integration

- complete compile/cache/submit path in runtime
- connect `@parallel(domain=\"bodies\")` to MPS chunk execution in live loop
- surface parallel fallback reasons in runtime HUD/log channel

### Sprint 3: Physics + Network Loop Closure

- wire NPS tick/channel loop into fixed-step ParadoxPE updates
- finalize snapshot/input packet boundaries and authority handoff semantics
- validate rollback/resim policy constraints for pre-alpha scope

### Sprint 4: Pre-Alpha Vertical Slice

- run one playable sandbox slice without benchmark-only dependencies
- validate MGPU/MGS fallback behavior under scene load
- capture baseline telemetry bundle and freeze pre-alpha support matrix

## Validation Commands

Use these as baseline checks during transition:

```bash
cargo check
cargo test -p tl-core
cargo test -p runtime
cargo test -p gms
cargo test -p paradoxpe
cargo test -p nps
```

## Non-Goals In Pre-Alpha

- full editor workflow
- broad asset import pipeline
- advanced renderer feature matrix expansion
- broad gameplay framework abstractions

These remain beta-or-later scope.
