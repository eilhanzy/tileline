# Tileline v0.5.0 Roadmap

This document defines the target scope for `v0.5.0`.

`v0.5.0` is not intended to be a cosmetic release. It is the release where Tileline should stop
behaving like a partially parallel prototype and start behaving like a genuinely coherent
runtime-first engine core.

The release theme is:

- keep the rendering stack feature-complete enough to build real scenes
- optimize the existing ray tracing / shader / lighting path instead of endlessly re-authoring it
- add the missing asset-facing pieces (`effects`, `textures`)
- perform a serious ParadoxPE + MPS revision
- remove the remaining practical dependence on `rayon` and `bevy` from shipping runtime paths

## Current Snapshot

The engine already has these major pieces in place:

- shaders: present and functional
- lighting: present and functional
- ray tracing: present in hybrid form, but still needs optimization and stronger fallback behavior
- runtime scene path: working
- `.tlscript` + `.tlsprite` authoring path: working
- ParadoxPE: real physics core, but still not fully exploiting MPS
- MPS: custom dispatcher foundation exists, but runtime integration is not yet the final form

That means `v0.5.0` should focus on **optimization, dependency cleanup, and missing production
features**, not on inventing entirely new subsystems.

## Release Goals

`v0.5.0` should deliver all of the following:

- an optimized forward + hybrid RT render path that is stable in the runtime
- a usable effects layer on top of the current shader/light stack
- first-class texture support in the runtime scene/material path
- a more aggressively parallel ParadoxPE execution model
- a stricter MPS-driven runtime execution path with fewer serial bottlenecks
- no shipping runtime dependence on `Bevy` scheduling/tasks
- no hot-path runtime/physics dependence on `rayon`

## Non-Goals

These are intentionally out of scope for `v0.5.0`:

- full cinematic renderer or GI/denoiser stack
- high-level gameplay framework abstractions
- complete editor productization
- broad asset import coverage beyond what is needed for the runtime scene path
- fully finished multiplayer gameplay loop

## Definition Of Done

`v0.5.0` should only be considered complete when all of these are true:

- RT can be enabled without destabilizing the main runtime path
- lights, shaders, textures, and effects all work together in the same scene path
- ParadoxPE no longer depends on coarse serial stepping in its main hot path
- MPS is the primary CPU execution path for physics/runtime jobs
- no `Bevy App/System/bevy_tasks` dependence exists in shipping runtime execution
- no `rayon` dependence remains in shipping hot paths for physics/runtime scheduling
- the runtime can explain performance regressions through telemetry instead of guesswork

## Workstream A: Render Stack Optimization

This workstream treats rendering as "feature present, now make it production-usable."

### A1. Ray Tracing Optimization

Current status:

- implemented
- optional
- requires better cost control and clearer fallback behavior

Target work:

- reduce RT frame-time spikes during dynamic scene updates
- improve acceleration-structure build/update budgeting
- tighten `Auto` fallback so unsupported or overloaded paths return to forward rendering cleanly
- reduce RT shadow/specular instability across frame-to-frame scene changes
- expose stronger runtime telemetry:
  - `rt_active`
  - `rt_dynamic_count`
  - `rt_fallback_reason`
  - RT build/update timing

Acceptance gates:

- RT off must not regress current stable runtime path
- RT on must degrade gracefully instead of collapsing frame pacing
- forward fallback must be automatic, visible, and non-fatal

### A2. Shader Pipeline Hardening

Current status:

- functional
- not yet fully optimized around real runtime scene pressure

Target work:

- reduce redundant pipeline/state churn
- harden material/shader parameter flow from `.tlsprite` / runtime scene data
- formalize shader feature flags instead of ad-hoc branching
- ensure the runtime scene path does not require showcase-only glue

Acceptance gates:

- no per-frame shader-state thrash for common scene updates
- one stable default material path for opaque + transparent scene content

### A3. Lighting Optimization

Current status:

- working
- likely needs culling/budget tuning and tighter integration with scene workload

Target work:

- light budget prioritization for runtime scenes
- lower-cost camera-relative light pruning
- more predictable shadow-casting cost
- better telemetry for light count vs render cost

Acceptance gates:

- lighting remains visually stable under scene motion
- light-heavy scenes do not explode frame variance without explanation

## Workstream B: Effects And Texture Support

This is the main content-facing feature gap for `v0.5.0`.

### B1. Effects Layer

Target scope:

- bloom or glow refinement
- haze / distance effects integration cleanup
- optional lightweight post-FX chain
- sprite-friendly effect hooks for HUD / emissive content

Rules:

- effects must be optional
- effects must fail soft on weaker paths
- effect toggles must be controllable from runtime/CLI/console

Acceptance gates:

- at least one stable post-FX chain works in TLApp runtime
- effects can be disabled cleanly without breaking scene output

### B2. Texture Support

Target scope:

- texture-backed materials in runtime scene rendering
- texture slot binding from `.tlsprite` and scene assets
- texture lifetime/cache policy suitable for runtime use
- fallback behavior when assets are missing or unsupported

Rules:

- no texture-only behavior hidden in examples
- runtime `src/` remains the source of truth

Acceptance gates:

- textured scene objects render through the canonical runtime path
- missing textures fall back safely and visibly

## Workstream C: ParadoxPE Revision

This is one of the most important `v0.5.0` goals.

Current status:

- physically working
- better than before
- still too coarse in its hot path
- still not fully exploiting available CPU parallelism

### C1. Step Pipeline Refactor

Target work:

- break `PhysicsWorld::step` into stronger explicit phases
- reduce world-lock scope across those phases
- make broadphase, narrowphase, solver, and integration more dispatch-friendly
- eliminate remaining "one heavy thread does too much" behavior

### C2. Contact / Solver Scalability

Target work:

- reduce contact explosion under dense ball stacks
- add better budget control for narrowphase/manifold generation
- improve solver cost scaling under heavy contact sets
- keep correctness ahead of reckless parallelism

### C3. Runtime Stability

Target work:

- reduce tick runaway / governor oscillation further
- reduce major drop clusters caused by physics overload
- improve telemetry:
  - phase timings
  - manifold growth
  - per-step candidate pair pressure
  - effective physics ceiling

Acceptance gates:

- dense scenes degrade in a bounded way
- heavy physics load is diagnosable from logs/telemetry
- no hidden serial bottleneck dominates the entire simulation

## Workstream D: MPS Revision

This workstream makes MPS the real scheduler of the engine instead of a partial helper.

### D1. Bare-Metal Runtime Control

Target work:

- continue moving runtime and physics work onto the custom MPS dispatcher
- reduce reliance on generic scheduler behavior
- improve affinity/priority handling on Linux
- tighten double-buffered overlap between render and physics

### D2. Runtime Integration

Target work:

- make MPS the canonical CPU orchestration path for TLApp runtime
- reduce phase-to-phase latency
- improve overlap between Render N and Physics N+1
- make telemetry expose queue depth, lag frames, in-flight work, and fallback usage

Acceptance gates:

- runtime performance no longer depends on general-purpose task stealing behavior
- overlap path is not just present but measurably useful

## Workstream E: Rayon / Bevy Independence

This is a hard release theme, not a stretch goal.

### E1. Bevy Independence

Target work:

- remove any remaining shipping runtime dependence on:
  - `App`
  - `System`
  - `bevy_tasks`
  - stage/barrier scheduling assumptions

Acceptance gate:

- shipping runtime path has zero Bevy scheduler/task dependence

### E2. Rayon Independence

Target work:

- remove remaining `rayon` use from shipping hot paths
- replace it with MPS-native or explicitly managed parallel execution
- keep serial fallback only where correctness requires it temporarily

Acceptance gate:

- shipping runtime/physics hot paths are no longer backed by `rayon`

Note:

- temporary use in tools, offline processing, or non-hot-path utilities can be tolerated during
  migration, but the release goal is to stop depending on it for the actual runtime-critical path.

## Milestones

### Milestone 1: Rendering Core Consolidation

Target contents:

- RT optimization pass 1
- shader/light cleanup
- first stable effect hooks
- initial texture pipeline

Exit criteria:

- one canonical runtime scene path supports shaders + lights + textures + optional RT/effects

### Milestone 2: Physics + MPS Restructure

Target contents:

- ParadoxPE phase refactor
- stronger MPS dispatch ownership
- lower serial fraction in world step
- better physics telemetry

Exit criteria:

- physics step is measurably more parallel and less bottlenecked

### Milestone 3: Dependency Exit + Release Hardening

Target contents:

- `rayon` hot-path exit
- `Bevy` runtime-path exit
- regression validation
- final docs and release notes

Exit criteria:

- `v0.5.0` can be shipped without pretending the core still depends on generic schedulers

## Validation Gates

The release should be validated against real workloads, not just compile success.

### Desktop Gate

- target class: Ryzen 9 7900 / similar high-core desktop CPU
- TLApp showcase remains stable under normal runtime usage
- dense physics scenes show lower frame-time variance than current `v0.4.x`/early `v0.5.0` state

### ARM / Mobile-Class Gate

- Orange Pi 5 / Mali path remains functional
- mobile-safe presets still behave predictably
- fallback quality controls remain bounded and understandable

### Regression Gate

- `cargo check` stays green across active crates
- targeted runtime + physics tests stay green
- new telemetry does not significantly distort runtime cost

## Recommended Execution Order

The most productive order for `v0.5.0` is:

1. render optimization and texture/effects foundation
2. ParadoxPE phase refactor
3. MPS ownership expansion
4. `rayon` hot-path removal
5. final Bevy independence cleanup
6. release hardening + docs + validation

This order matters because it avoids doing dependency cleanup before the replacement execution path
is strong enough.

## Release Summary

If `v0.3.0` was foundation and `v0.4.x` is stabilization/continuity, then `v0.5.0` should be the
release where Tileline starts to look like a real parallel engine core instead of a promising
prototype.
