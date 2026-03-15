# Tileline Beta Roadmap

This document defines the practical path from the current Tileline foundation state to a usable
beta release.

The goal is not to finish every subsystem to "engine complete" level. The goal is to ship a
coherent beta that validates Tileline's core differentiator:

- explicit CPU parallelism through MPS
- explicit GPU scaling through GMS
- `.tlscript` as a WASM-targeted scripting layer
- ParadoxPE as the fixed-step physics core
- NPS as the network/packet layer that does not stall the frame loop

## Current Assessment

The repository is no longer in the "empty architecture" stage. It already contains real
implementation across CPU scheduling, GPU planning, scripting, networking foundations, and
physics.

A blunt assessment:

- low-level architecture foundation: roughly one-third complete
- beta-ready engine product: still early

This is normal. The first milestones solved high-risk core architecture. The remaining work is
mostly productization, tooling, integration hardening, and real workload validation.

## Beta Definition

For Tileline, "beta" should mean all of the following:

- a small but real playable sample can run on the engine
- MPS, GMS, ParadoxPE, NPS, and `.tlscript` are integrated into one runtime path
- one supported rendering path is stable on desktop Linux and one secondary target is stable
- scripting, physics, and networking can be exercised without benchmark-only glue
- profiling and diagnostics are strong enough to explain why performance regressed
- the engine can fail softly instead of collapsing under common script/runtime errors

Anything below that is still a foundation or prototype milestone, not beta.

## Roadmap Principles

- Keep `src/` crates as the source of truth. Avoid pushing core behavior into examples.
- Preserve zero-copy and bounded-wait design where it already exists.
- Prefer safe serial fallback over incorrect parallel execution.
- Add observability before aggressive optimization where possible.
- Define a narrow beta scope. Do not try to ship a general-purpose AAA engine in beta one.

## Phase A: Pre-Alpha Transition (Current Focus)

This phase bridges foundation work into one runtime-owned vertical path. It should be treated as
the active phase before broad beta feature expansion.

### Exit Criteria

- canonical runtime update order is frozen: net -> script -> physics -> render plan -> present
- `.tlscript` compile/cache/submit path is live in runtime
- ParadoxPE state mutation/query path is exercised from script host ABI in runtime loop
- NPS input + snapshot loops run without blocking render/present
- telemetry surface is stable enough to explain cross-subsystem regressions

### Work Items

- implement and validate pre-alpha gates listed in `docs/tileline-pre-alpha-transition.md`
- keep multi-GPU and mobile fallback behavior in `gms/src` and `mgs/src` aligned with runtime flow
- keep ARM mobile validation repeatable with the Orange Pi 5 MGS gate:
  - runner: `scripts/test_mgs_orangepi5.sh`
  - guide: `docs/mgs-orangepi5-validation.md`
- prioritize integration bugs over new subsystem scope

## Phase 0: Foundation Hardening

This phase is partially complete. The remaining work is mostly stabilization.

### Exit Criteria

- `cargo check` and targeted tests stay green across current crates
- public APIs stop churning for core integration points
- subsystem docs describe actual behavior, not intended behavior

### Work Items

#### MPS

- add scheduler tracing for:
  - queue depth
  - task steal/spill counts
  - task wait/backoff duration
  - per-core-class load
- expose stable frame/task metrics to runtime
- validate WASM task cache path and cold-start overhead

#### GMS

- move more benchmark-only reporting logic into reusable `src/` utilities
- add stronger adapter diagnostics for backend/device filtering
- add timestamp-query-backed timing where supported
- stabilize UMA and Panthor fallback behaviors under real workloads

#### `.tlscript`

- complete typed-IR-driven codegen refactor
- reduce remaining AST-direct lowering/codegen paths
- add diagnostics that explain:
  - why `@parallel` fell back to serial
  - which host calls force main-thread execution
  - which effects prevent safe body-domain chunking

#### ParadoxPE

- improve solver quality for larger contact stacks
- add friction/restitution combine validation tests
- harden snapshot restore/interpolation semantics

#### NPS

- finalize packet/channel/tick model before transport runtime work expands
- define authoritative snapshot and input packet boundaries clearly

## Phase 1: Vertical Slice Integration

This is the first truly important beta milestone.

The engine needs one canonical end-to-end path:

- `.tlscript` compiles to WASM
- runtime loads/caches it
- MPS schedules it
- ParadoxPE host calls mutate/query physics state
- GMS renders the result
- NPS can serialize the relevant state

### Exit Criteria

- one sample scene runs without benchmark glue
- one fixed-step update loop is authoritative
- one render loop consumes real runtime state
- script invocation, physics stepping, and rendering are observable in the same metrics surface

### Work Items

#### Runtime

- build a canonical game loop runner instead of subsystem-local smoke tests
- connect `.tlscript` invocation to the existing runtime parallel coordinator
- connect ParadoxPE snapshots to runtime interpolation/presentation flow

#### `.tlscript`

- implement compile/cache/submit path:
  - source -> semantic -> IR -> WASM
  - module cache
  - export metadata
  - runtime dispatch metadata
- connect `@parallel(domain="bodies")` to real MPS chunk execution

#### ParadoxPE

- define fixed-step world ownership in runtime
- add a small gameplay-facing control layer over raw host ABI calls
- validate body-domain chunked scripting against SoA read/write views

#### GMS

- render one real scene path instead of synthetic clear-only throughput work
- support one stable forward render path for beta
- define minimal UI/Post-FX path that cooperates with the planner

#### NPS

- define transport runtime loop using `tokio::UdpSocket`
- connect input packet path and snapshot packet path
- verify MPS-backed encode/decode jobs do not stall render/present

## Phase 2: Playable Sandbox Beta Core

This is where Tileline becomes a real engine beta instead of a systems prototype.

### Exit Criteria

- one playable sandbox demo
- 8-player heavy interaction target has bounded degradation instead of catastrophic collapse
- script, physics, networking, and rendering can all be exercised simultaneously

### Beta Scope for the First Playable Demo

Keep the scope narrow:

- simple world loading
- rigid bodies
- scripted forces/interactions
- UI overlay/debug HUD
- authoritative host or listen-server networking
- snapshot/interpolation path
- one material/lighting path

Do not try to ship advanced editor, cinematic pipeline, asset store features, or full gameplay
framework abstractions in this phase.

### Work Items

#### ParadoxPE

- add constraint quality improvements:
  - contact manifold persistence refinement
  - stronger warm starting
  - better island scheduling
- add at least one more joint class beyond distance/fixed if required by the demo
- add rollback/resimulation policy suitable for NPS integration

#### NPS

- reliable ordered lifecycle channel
- unreliable sequenced physics/input channel
- snapshot delta path
- bandwidth budget controls
- RTT/jitter/loss reporting in runtime metrics

#### `.tlscript`

- gameplay-facing standard library for engine calls
- better compile diagnostics and effect guidance
- stable ABI contract for beta scripts

#### Runtime / Tooling

- in-engine diagnostics overlay:
  - MPS fallback reasons
  - GMS adapter/planner state
  - ParadoxPE active bodies / contacts / solver cost
  - NPS RTT/loss/packet rate
- capture/export frame telemetry for regression checks

## Phase 3: Beta Stabilization

This phase is not about adding major systems. It is about making the chosen beta scope reliable.

### Exit Criteria

- reproducible sample scenes
- stable regression benchmarks
- known platform support matrix
- documented limitations instead of hidden failure cases

### Work Items

#### Performance and Stability

- set budget targets per subsystem:
  - MPS scheduling overhead
  - ParadoxPE fixed-step cost
  - GMS submit/present costs
  - NPS encode/decode budget
- establish regression suites for:
  - desktop dGPU
  - Apple Silicon UMA
  - ARM/Panthor class systems

#### API Freeze (Beta-Level)

- freeze core script ABI
- freeze physics handle semantics
- freeze network packet header and channel model
- freeze runtime telemetry surface shape

#### Documentation

- publish "how to build a sandbox game" docs
- publish `.tlscript` parallel contract guide
- publish platform support caveats

## Phase 4: Deferred Until After Beta

These are valid goals, but they should not block beta one.

- full editor workflow
- generalized asset pipeline/importer suite
- advanced renderer feature matrix
- large joint/constraint catalog
- production rollback netcode polish
- hot reload across all runtime layers
- deep ECS/gameplay framework abstractions
- broad market platform packaging

## Critical Risks

These are the risks most likely to delay beta if left vague.

### 1. Parallelism Ergonomics

Risk:

- the engine is architecturally parallel, but hard to use safely

Mitigation:

- keep serial-safe defaults
- strengthen advisor diagnostics
- show runtime fallback reasons in metrics/UI

### 2. Observability Gap

Risk:

- regressions cannot be explained because subsystem timings are disconnected

Mitigation:

- unify runtime telemetry surface
- treat tracing/profiling as core work, not a late extra

### 3. Platform Variance

Risk:

- driver/backend behavior differs sharply across NVIDIA, AMD, Apple, and ARM/Panthor

Mitigation:

- define explicit support tiers
- keep layered probe/fallback design
- maintain real hardware regression coverage

### 4. Integration Lag

Risk:

- subsystems look strong in isolation but are not exercised together

Mitigation:

- prioritize vertical-slice runtime work over more isolated subsystem features

## Recommended Immediate Sequence

This is the practical order to follow next.

1. Finish `.tlscript` IR-driven codegen and runtime compile/cache/submit path.
2. Finalize NPS packet/tick/channel model and connect real UDP transport.
3. Build the first canonical runtime vertical slice:
   - script
   - ParadoxPE
   - render
   - snapshot export
4. Add unified runtime telemetry/overlay.
5. Build one playable sandbox sample and iterate from real workload measurements.

## Short Version

Tileline should not aim for "feature complete." It should aim for:

- one coherent runtime path
- one playable sample
- one explainable performance story
- one stable scripting/physics/network/render integration surface

If that is achieved, beta is realistic. If not, more subsystem features will not fix the real gap.
