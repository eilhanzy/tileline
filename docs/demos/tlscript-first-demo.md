# .tlscript First Demo Prep

Status: Draft (preparation phase)

## Goal

Deliver the first `.tlscript`-driven demo after the current runtime scene/GMS integration pass.

## Baseline Expectations

- Use runtime-owned scene orchestration (not benchmark-only logic).
- Use ParadoxPE for physics stepping.
- Use GMS workload planning path derived from runtime scene payloads.
- Store all demo flow/spec notes in `docs/demos/*`.

## Proposed Demo Slice

1. Spawn phase:
   transparent 3D container + progressive ball spawn.
2. Scripted control:
   `.tlscript` function adjusts spawn burst and damping at runtime.
3. Stress phase:
   increased body count, bounce intensity, and scripted impulses.
4. Telemetry overlay:
   FPS, physics substeps, and GMS lane estimates.

## Ready Checklist

- [x] Runtime-to-render path consumes `SceneFrameInstances` end-to-end.
- [x] `.tlscript` call path can mutate scene/physics parameters safely.
- [ ] Telemetry export format is stable enough for show capture.

## Current Bootstrap (Implemented)

- `runtime/src/tlscript_showcase.rs`:
  - in-memory `.tlscript` compile path (`Lexer` -> `Parser` -> `SemanticAnalyzer` -> `ParallelHookAnalyzer`)
  - safe subset evaluator producing `BounceTankRuntimePatch`
  - per-frame parallel dispatch decision export for `@parallel(domain="bodies")`
- `runtime/examples/tlscript_first_demo.rs`:
  - scripted spawn-burst + damping control
  - dynamic patch application into bounce scene + ParadoxPE world
