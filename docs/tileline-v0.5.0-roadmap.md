# Tileline v0.5.0 Roadmap

This document defines the target scope for `v0.5.0`.

`v0.5.0` is not intended to be a cosmetic release. It is the release where Tileline should stop
behaving like a partially parallel prototype and start behaving like a genuinely coherent
runtime-first engine core.

The release theme is:

- keep the rendering stack feature-complete enough to build real scenes
- add a first-class 2D side-view foundation for a Terraria-like survival sandbox path
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
- sprite-oriented runtime authoring already exists, so 2D should extend the real runtime path instead
  of creating a separate mini-engine
- ParadoxPE: real physics core, but still not fully exploiting MPS
- MPS: custom dispatcher foundation exists, but runtime integration is not yet the final form

That means `v0.5.0` should focus on **optimization, dependency cleanup, and missing production
features**, not on inventing entirely new subsystems.

## Release Goals

`v0.5.0` should deliver all of the following:

- an optimized forward + hybrid RT render path that is stable in the runtime
- a first-class orthographic / side-view 2D runtime path
- a chunked tile-world foundation suitable for Terraria-like digging/building scenes
- a usable effects layer on top of the current shader/light stack
- first-class texture support in the runtime scene/material path
- a first-class SPIR-V shader path for the raw Vulkan backend
- a more aggressively parallel ParadoxPE execution model
- a flat-2D ParadoxPE mode for side-view terrain/body collision
- a stricter MPS-driven runtime execution path with fewer serial bottlenecks
- a Linux-first raw Vulkan backend inside `tl-core`
- no shipping runtime dependence on `Bevy` scheduling/tasks
- no hot-path runtime/physics dependence on `rayon`
- no shipping runtime render-path dependence on `wgpu`

## Non-Goals

These are intentionally out of scope for `v0.5.0`:

- full cinematic renderer or GI/denoiser stack
- high-level gameplay framework abstractions
- complete editor productization
- broad asset import coverage beyond what is needed for the runtime scene path
- fully finished multiplayer gameplay loop
- full survival gameplay systems such as inventory, crafting, AI, fluids, biomes, or day-night loops

## Definition Of Done

`v0.5.0` should only be considered complete when all of these are true:

- RT can be enabled without destabilizing the main runtime path
- lights, shaders, textures, and effects all work together in the same scene path
- one canonical side-view 2D scene path works in runtime without example-only hacks
- chunked tile worlds can render, scroll, and mutate locally through dig/place updates
- ParadoxPE flat-2D mode exists and is the canonical 2D terrain/body collision path
- 2D camera, animation, and VFX can coexist on the same runtime scene path
- SPIR-V is a supported, documented shader/runtime artifact path for Vulkan scene rendering
- the Vulkan backend migration path inside `tl-core` is real and no longer speculative
- ParadoxPE no longer depends on coarse serial stepping in its main hot path
- MPS is the primary CPU execution path for physics/runtime jobs
- no `Bevy App/System/bevy_tasks` dependence exists in shipping runtime execution
- no `rayon` dependence remains in shipping hot paths for physics/runtime scheduling
- no `wgpu`-backed render path remains in shipping runtime execution
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
- canonical SPIR-V shader artifacts now live in-repo under `tl-core/assets/shaders/spv/`
- the active Vulkan path no longer depends on runtime GLSL compilation or `shaderc` build glue

Target work:

- reduce redundant pipeline/state churn
- harden material/shader parameter flow from `.tlsprite` / runtime scene data
- formalize shader feature flags instead of ad-hoc branching
- formalize a canonical SPIR-V path:
  - keep precompiled `.spv` modules as the official shader artifact for shipping/runtime-facing Vulkan paths
  - move any optional shader authoring/transpile tools outside the hot build/runtime path
  - pipeline-layout compatibility rules that are explicit instead of implicit
- add shader/pipeline cache policy around SPIR-V modules so common scene boots do not constantly
  rebuild the same pipelines
- ensure the runtime scene path does not require showcase-only glue

Acceptance gates:

- no per-frame shader-state thrash for common scene updates
- one stable default material path for opaque + transparent scene content
- SPIR-V-backed Vulkan pipelines can be built repeatably without ad-hoc shader glue
- in-repo `.spv` artifacts are consumed by the Vulkan backend without runtime GLSL compilation
  or `shaderc` in the active path

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

### A4. Raw Vulkan Backend Transition

Current status:

- a Linux-first `ash` backend skeleton exists in `tl-core`
- the broader runtime still contains `wgpu`-based paths that must be migrated deliberately
- native MultiGPU bootstrap/probing now exists in the Vulkan backend as an execution skeleton
- runtime now has an explicit draw-frame -> Vulkan snapshot handoff helper
- runtime now also has a first dedicated Vulkan scene renderer adapter that can turn
  `RuntimeDrawFrame` into a Vulkan snapshot + explicit MGPU frame plan
- TLApp runtime now carries a shared renderer surface (`wgpu` + experimental `TILELINE_RENDERER=vulkan`)
  so the cutover can proceed incrementally without forking the app loop
- bridge/sync submission tracking has started moving from raw `wgpu` types toward backend-neutral handles

Target work:

- move `tl-core` graphics ownership toward raw Vulkan objects and frame resources
- define the canonical `Render N` state-snapshot handoff from MPS/ParadoxPE into Vulkan-visible
  memory
- treat SPIR-V as the official shader artifact for the Vulkan path, not a side detail
- preserve explicit multi-GPU planning so the Vulkan path does not regress from current GMS/MGS
  topology work
- keep present mode policy Linux-first with `MAILBOX` preference and explicit fallback to
  `IMMEDIATE` / `FIFO`
- formalize command pool / command buffer / persistently mapped buffer ownership
- migrate runtime-facing renderer layers toward the new Vulkan backend in stages instead of
  breaking the engine in one patch

Acceptance gates:

- `tl-core` contains a real Vulkan backend implementation instead of a placeholder document
- double-buffered snapshot upload is explicit and renderer-owned
- SPIR-V module creation / pipeline creation is part of the canonical Vulkan backend path
- secondary GPU planning remains a first-class migration concern instead of being deferred away
- the migration path away from `wgpu` is documented, incremental, and testable

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

## Workstream E: Rayon / Bevy / WGPU Independence

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

### E3. WGPU Independence

Target work:

- complete migration from `runtime` rendering hot paths to the canonical raw Vulkan backend
- remove remaining `wgpu` ownership from shipping TLApp runtime rendering flow
- keep migration incremental during development, but lock shipping target to Vulkan-native ownership
- ensure renderer telemetry and console status report Vulkan-native path details instead of `wgpu` terms

Acceptance gate:

- shipping runtime render path is no longer backed by `wgpu`

Note:

- temporary use in tools, offline processing, or non-hot-path utilities can be tolerated during
  migration, but the release goal is to stop depending on it for the actual runtime-critical path.

## Workstream F: 2D Foundation

This workstream turns the existing sprite-oriented runtime into a real 2D side-view engine path
instead of treating 2D as a toy layer on top of 3D.

Chosen defaults:

- first 2D target: Terraria-like side-view survival sandbox
- world model: chunked tile world
- physics strategy: ParadoxPE flat-2D
- content priority: tilemap -> animation -> VFX
- `v0.5.0` scope: engine foundation + digging/building, not full survival gameplay systems

### F1. 2D Runtime Rendering

Target work:

- add orthographic / side-view camera mode as a first-class runtime scene mode
- add a tilemap draw path using atlas-backed quads or batches through the canonical renderer
- keep HUD and sprite overlays on the same 2D-friendly render path instead of splitting them away
- support deterministic layer ordering with room for future parallax expansion
- keep 2D as an extension of the canonical renderer migration, including the Vulkan path, instead of
  creating another temporary render fork

Acceptance gates:

- one 2D scene can render tile layers, animated sprites, and overlays together
- camera motion is stable and does not rely on 3D fallback behavior
- 2D batching stays central so the path does not regress into per-sprite draw overhead

### F2. Chunked Tile World

Target work:

- define chunked tile storage as the canonical world model for 2D sandbox scenes
- support tile mutation operations needed for dig, place, and visible update flows
- keep tile updates local to changed chunks instead of assuming whole-world rebuilds
- add chunk visibility and chunk update telemetry so larger worlds remain diagnosable

Acceptance gates:

- chunked worlds load, scroll, render, and mutate without rebuilding the whole map
- tile changes are reflected within bounded frame cost
- runtime telemetry can explain chunk visibility/update pressure

### F3. ParadoxPE Flat-2D

Target work:

- add a flat-2D execution mode in ParadoxPE instead of a separate lightweight physics engine
- constrain motion and collision semantics to side-view 2D
- support tile terrain collision plus simple 2D-friendly actor shapes first (AABB/capsule-style)
- align broadphase and narrowphase data flow with chunked tile terrain updates
- keep the implementation deterministic-first and avoid overcommitting to full rigid-body sandbox
  behavior in `v0.5.0`

Acceptance gates:

- player and actor movement collides correctly with tile terrain
- dig/place updates can invalidate or rebuild only the relevant collision region
- no global collision rebuild is required for local terrain edits

### F4. 2D Content And Authoring Path

Target work:

- extend `.tlsprite` with 2D atlas, tile, and animation-facing metadata
- extend `.tlscript` with 2D camera movement, tile query, dig/place, and 2D spawn helpers
- keep `.tljoint` and runtime loading as the composition point for 2D scenes too
- prioritize engine-owned runtime loading instead of example-owned loaders
- keep 2D `.tlsprite` additions additive so current light and 3D bindings remain valid

Acceptance gates:

- a 2D scene can be authored without hardcoding engine internals into examples
- script-driven dig/place and sprite animation work through supported APIs
- scripts do not need to manage low-level chunk internals directly

### F5. 2D Validation Scenarios

Representative acceptance scenes:

- a small static side-view map
- a medium chunked map with repeated dig/place activity
- a stress scene with many visible tiles, animated actors, and HUD overlays

Validation rules:

- a side-view chunked tile scene loads and scrolls correctly
- tiles can be dug and placed repeatedly without full-world rebuild behavior
- 2D actor collision against terrain remains stable after tile edits
- animated sprite characters render correctly on top of the tile world
- 2D scene frame pacing remains diagnosable from telemetry
- the 2D path works through runtime content loading, not only through bespoke examples

## Milestones

### Milestone 1: 2D Foundation First

Target contents:

- Workstream F baseline delivery (`F1` + `F2` core)
- initial 2D runtime rendering path (orthographic camera + tile/sprite layer composition)
- chunked tile-world scene representation and local mutation loop (dig/place)
- 2D authoring flow anchored in `.tlsprite` + `.tlscript` + `.tljoint`

Exit criteria:

- one canonical 2D side-view scene path supports tiles + sprites + overlays
- chunked tile dig/place path is functional without full-world rebuild behavior

### Milestone 2: Render + Effects + Texture Consolidation

Target contents:

- RT optimization pass 1
- shader/light cleanup
- SPIR-V shader artifact path and pipeline-cache groundwork
- first stable effect hooks
- initial texture pipeline
- raw Vulkan backend backbone in `tl-core`

Exit criteria:

- one canonical runtime scene path supports shaders + lights + textures + optional RT/effects
- 2D content path reuses the canonical renderer instead of a side renderer

### Milestone 3: MPS + Independence Hardening

Target contents:

- stronger MPS dispatch ownership in runtime
- ParadoxPE phase refactor and lower serial world-step fraction
- `rayon` hot-path exit
- `Bevy` runtime-path exit
- `wgpu` shipping render-path exit
- regression validation
- final docs and release notes

Exit criteria:

- `v0.5.0` can be shipped without depending on generic schedulers or `wgpu` in shipping runtime

## Validation Gates

The release should be validated against real workloads, not just compile success.

## Performance Contract

`v0.5.0` needs a measurable performance contract, not just a generic promise to "optimize later."

The contract below assumes Linux-first desktop validation on a Ryzen 9 7900-class CPU for the
primary gate, with Orange Pi 5 / Mali remaining the mobile-class sanity gate.

### Scenario Matrix

| Scenario | Current rough baseline | `v0.5.0` ship target | Stretch target |
| --- | --- | --- | --- |
| `8k` balls / normal showcase density | `58-60 FPS`, generally good behavior, some variance under load spikes | lock `60 FPS` with lower frame variance, `240-360 Hz` effective tick, no sustained governor hunting | `60+ FPS` uncapped, `360+ Hz` effective tick with stable frametime pacing |
| `30k` balls / dense contact stress | about `16 FPS`, about `40 Hz` tick, major drops still possible | `30-45 FPS`, `90-140 Hz` effective tick, bounded degradation instead of collapse | `45-60 FPS`, `140+ Hz` tick in lighter-contact windows, no catastrophic drop clusters |
| `60k` objects / extreme stress scene | not a stable production path yet | keep simulation alive with predictable quality reduction, `15-25 FPS`, `60-90 Hz` effective tick, no runaway oscillation | `25-35 FPS`, `90+ Hz` tick with island/contact budgeting and stronger scene partitioning |

Notes:

- `8k` is the "must feel polished" gate.
- `30k` is the main "serious engine credibility" gate.
- `60k` is a stress/diagnostic gate, not a requirement for full visual parity.

### Frametiming And Stability Rules

These rules matter as much as average FPS:

- no long governor oscillation where tick rate repeatedly overshoots and then panics downward
- no unexplained frame-time spikes that cannot be diagnosed from telemetry
- `P95` frame time on the `8k` desktop gate should stay near the `60 FPS` band
- `30k` scenes may degrade, but degradation must be smooth, bounded, and explainable
- overload should prefer:
  - render quality reduction
  - light/effect budget reduction
  - RT fallback
  instead of physics collapse or random drop clusters

### Expected Gain Budget

This is the rough optimization budget for the `v0.5.0` independence work:

| Work item | Expected gain | Main reason |
| --- | --- | --- |
| `wgpu -> raw Vulkan` | about `5%-25%` typical, occasionally higher | lower render overhead, explicit present/upload control, tighter pipeline ownership |
| `rayon/bevy_tasks -> MPS-native scheduler` | about `20%-80%` depending on scene | lower scheduling overhead, better overlap, better core ownership |
| `ParadoxPE phase refactor + solver/contact scaling` | about `1.3x-2x` on heavy dense scenes | less serial hot-path work, lower contact explosion cost |
| Combined `v0.5.0` payoff | roughly `1.3x-2x` realistic, `2x-3x` stretch on the worst current bottlenecks | render + scheduler + physics scaling improvements stacking together |

This budget is intentionally conservative. The biggest wins should come from removing serial
bottlenecks in ParadoxPE and making MPS the real hot-path scheduler, with raw Vulkan then
removing renderer overhead and giving us better control over pacing, uploads, and MultiGPU
execution.

### Desktop Gate

- target class: Ryzen 9 7900 / similar high-core desktop CPU
- TLApp showcase remains stable under normal runtime usage
- dense physics scenes show lower frame-time variance than current `v0.4.x`/early `v0.5.0` state
- `8k` scene satisfies the ship-target row in the scenario matrix
- `30k` scene satisfies the ship-target row in the scenario matrix

### ARM / Mobile-Class Gate

- Orange Pi 5 / Mali path remains functional
- mobile-safe presets still behave predictably
- fallback quality controls remain bounded and understandable
- mobile-safe scenes should keep `30+ FPS` behavior without severe chopping

### Regression Gate

- `cargo check` stays green across active crates
- targeted runtime + physics tests stay green
- new telemetry does not significantly distort runtime cost

## Recommended Execution Order

The most productive order for `v0.5.0` is:

1. Workstream F first: 2D foundation (`side-view` + `chunked tile world` + dig/place basics)
2. Workstream A: render stack optimization
3. Workstream B: effects + texture support
4. Workstream D with `F3` bridge: MPS ownership expansion and 2D-ready runtime overlap
5. Workstream E: Rayon / Bevy / WGPU independence cleanup
6. final release hardening + docs + validation

This order matters because it avoids doing dependency cleanup before the replacement execution path
is strong enough.

## Release Summary

If `v0.3.0` was foundation and `v0.4.x` is stabilization/continuity, then `v0.5.0` should be the
release where Tileline starts to look like a real parallel engine core instead of a promising
prototype.
