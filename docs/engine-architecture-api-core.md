# Tileline Engine Architecture (API Core)

Version target: `v0.4.5` baseline  
Audience: engine/runtime contributors, integrators, and technical onboarding

This document is a system-level map of the Tileline engine. It focuses on crate boundaries,
runtime lifecycle, and public integration surfaces. It is intentionally not a symbol-by-symbol
reference.

## 1. Workspace Responsibility Map

Tileline is a multi-crate workspace with explicit subsystem separation.

| Crate | Responsibility | Notes |
|---|---|---|
| `tl-core` | Core engine interfaces and graphics/script frontend glue | Owns `.tlscript` frontend modules and platform graphics backends (`Vulkan` on Linux, `Metal` on macOS) |
| `runtime` | App/runtime orchestration (`tlapp`, project GUI, scene/bootstrap flow) | Integrates render loop, scene state, script evaluation, input, telemetry, and content loading |
| `paradoxpe` | Physics engine core | Handle-based SoA bodies, broadphase/narrowphase/solver/integrate pipeline, snapshots |
| `mps` | CPU scheduler/dispatcher foundation | Worker pools, runtime-dispatched compute orchestration used by engine subsystems |
| `mgs` | Mobile-first graphics scheduler path | Lower-overhead/mobile-oriented scheduler and hints |
| `gms` | Graphics multi-lane scheduling/planning | GPU workload planning and dispatch-oriented policy surface |
| `nps` | Network packet scaler/runtime protocol layer | Packet model, reliability surface, and ParadoxPE-oriented state transport |

For roadmap context, see:
- `docs/tileline-v0.4.5-roadmap.md`
- `docs/tileline-v0.5.0-roadmap.md`

## 2. End-to-End Runtime Flows

### 2.1 Startup and Runtime Bootstrap

The runtime bootstrap path initializes:
- adapter/surface policy
- scheduler path (`GMS` vs `MGS`)
- scene/runtime controller
- script runtime compilation
- content bindings (`.tlpfile` / `.tljoint` / `.tlsprite` / `.tlscript`)

Related docs:
- `docs/runtime-scheduler-path.md`
- `docs/runtime-scene-showcase.md`
- `docs/runtime-tlpfile-gui.md`
- `docs/runtime-tljoint.md`

### 2.2 Render and Scheduler Path

Render work is generated from runtime scene payloads, then routed through scheduler/planner paths.

Conceptual flow:
1. Scene frame state is built (`runtime` scene/controller path)
2. Workload is estimated and shaped for scheduler lanes
3. Runtime bridge coordinates CPU/GPU sync and submission
4. Present path completes frame output

Related docs:
- `docs/runtime-bridge-flow.md`
- `docs/runtime-scene-workload.md`
- `docs/gms-dispatch-planner.md`
- `docs/mgs-scene-workload.md`

### 2.3 Physics Path (ParadoxPE + MPS)

Physics is managed by ParadoxPE with MPS-backed execution planning.

Conceptual flow:
1. Runtime applies bounded patch/state updates
2. ParadoxPE executes substeps (integrate, broadphase, narrowphase, solver)
3. Contact/snapshot telemetry is published back to runtime
4. Runtime consumes results for script/render/diagnostics

Related docs:
- `docs/paradoxpe-foundation.md`
- `docs/paradoxpe-v0.5.0-c0-parallelization-checklist.md`

### 2.4 Scripting Path (`.tlscript`)

The script path is memory-resident and engine-validated before runtime evaluation.

Conceptual compile pipeline:
1. Lexer
2. Parser
3. Semantic analyzer
4. Parallel-hook/planner analysis
5. Lowering/codegen hooks

Conceptual frame pipeline:
1. Evaluate script entry with runtime bindings
2. Produce bounded runtime patch and optional control outputs
3. Merge with runtime state and telemetry

Related docs:
- `docs/tlscript-lexer.md`
- `docs/tlscript-parser-plan.md`
- `docs/tlscript-semantic.md`
- `docs/tlscript-parallel-runtime.md`
- `docs/runtime-tlscript-showcase.md`

### 2.5 Networking Path (`NPS`)

Networking uses NPS protocol/runtime planning for deterministic and bandwidth-conscious state flow.

Conceptual flow:
1. Tick/channel ownership is planned
2. Quantized payloads are encoded/decoded
3. Reliability and authority handoff are enforced
4. Physics-facing state is synchronized with runtime boundaries

Related docs:
- `docs/nps-protocol.md`
- `docs/nps-runtime-plan.md`

## 3. Public Integration Surfaces

### 3.1 Content and Project Surfaces

Tileline runtime content composition surfaces:
- `.tlpfile` project manifest
- `.tljoint` scene composition manifest
- `.tlscript` logic/control layer
- `.tlsprite` visual/light bindings
- `.pak` packaged runtime content

Related docs:
- `docs/runtime-tlpfile-gui.md`
- `docs/runtime-tljoint.md`
- `docs/runtime-tlsprite.md`
- `docs/runtime-pak.md`
- `docs/demos/tlapp/README.md`

### 3.2 Runtime Lifecycle Hooks

Contributor-visible lifecycle checkpoints:
- startup initialization and scheduler selection
- per-frame update/compile/submit/present
- runtime console/telemetry status surfaces
- reload paths for scene/script/sprite assets

Related docs:
- `docs/runtime-tlapp-console.md`
- `docs/runtime-draw-hud.md`
- `docs/runtime-upscaler.md`

### 3.3 Telemetry and Diagnostics

The runtime is designed to expose enough signals to diagnose performance regressions:
- FPS/frame pacing
- physics substep and phase timing
- scheduler path and queue depth indicators
- scene density and tile/chunk pressure

Related docs:
- `docs/runtime-scene-workload.md`
- `docs/runtime-bridge-flow.md`
- `docs/paradoxpe-v0.5.0-c0-parallelization-checklist.md`

## 4. v0.4.5 Operations Baseline

`v0.4.5` is the 2D foundation consolidation baseline.

Canonical release docs:
- `docs/releases/v0.4.5.md`
- `docs/tileline-v0.4.5-roadmap.md`

Suggested onboarding run path:
1. Use `.tlpfile` project boot (`docs/demos/tlapp/tlapp_project.tlpfile`)
2. Validate scene dimension separation (`2d` / `3d`)
3. Validate script + tile mutation flow on side-view demo scenes
4. Validate packaging/mount path with `.pak` tooling

## 5. Deep-Dive Index

Use this map to jump from architecture view to subsystem detail:
- Render bridge and scheduling: `docs/runtime-bridge-flow.md`
- Runtime scene workload shaping: `docs/runtime-scene-workload.md`
- Mobile scheduler policy: `docs/runtime-scheduler-path.md`, `docs/mgs-scene-workload.md`
- Physics internals: `docs/paradoxpe-foundation.md`
- Script frontend and runtime contracts: `docs/tlscript-lexer.md`, `docs/tlscript-parser-plan.md`, `docs/tlscript-semantic.md`, `docs/tlscript-parallel-runtime.md`, `docs/runtime-tlscript-showcase.md`
- Network protocol/runtime plan: `docs/nps-protocol.md`, `docs/nps-runtime-plan.md`
- Packaging/content path: `docs/runtime-pak.md`, `docs/runtime-tljoint.md`, `docs/runtime-tlpfile-gui.md`
