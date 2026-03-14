# Tileline Engine (Prototype Workspace)

Tileline is a parallel-first game engine architecture prototype focused on explicit CPU/GPU scaling:

- `MPS` (Multi Processing Scaler): CPU-side task scheduling and WASM execution
- `GMS` (Graphics Multi Scaler): GPU discovery, scoring, and asymmetric multi-GPU planning
- `MGS` (Mobile Graphics Scheduler): mobile TBDR-aware serial fallback scheduler and tile planning
- `NPS` (Network Packet Scaler): bit-packed UDP protocol, reliability, and MPS-offloaded packet processing
- `ParadoxPE`: fixed-step physics foundation with SoA storage, broadphase, solver, joints, sleep, snapshot/interpolation support, and script/WASM host ABI
- `tl-core`: engine bridge layer that synchronizes MPS and GMS
- `runtime`: render-loop integration glue for `wgpu` submit/present flows and NPS UDP transport/telemetry pumping

This workspace is currently in pre-alpha transition phase (Foundation -> Pre-Alpha). The main goals are:

- lock-free task flow between CPU and GPU planning stages
- explicit multi-GPU orchestration (primary throughput GPU + secondary latency/helper GPU)
- Apple Silicon / UMA-aware stability controls
- WASM-based scripting/runtime integration path (MPS-targeted)
- one canonical runtime-owned scene loop (script + physics + render + network)
- telemetry-first regression visibility across MPS/GMS/ParadoxPE/NPS

## Workspace Layout

- `mps/`: CPU topology detection, priority balancer, lock-free scheduler, WASM dispatch (Wasmer)
- `gms/`: GPU inventory/scoring, multi-GPU planner, adaptive UMA buffer control, benchmark tooling
- `mgs/`: mobile GPU family detection, TBDR tile planner, serial fallback chain, and mobile render benchmark
- `nps/`: low-level UDP packet protocol, bit packing, reliability, authority handoff, MPS-integrated packet manager
- `paradoxpe/`: fixed-step physics core with packed handles, SoA body storage, parallel broadphase, narrowphase/solver passes, starter joints/sleep, snapshot/interpolation buffering, and `.tlscript`-friendly host ABI
- `tl-core/`: `MpsGmsBridge`, portable multi-GPU sync abstractions, and `.tlscript` compiler/runtime metadata layers
- `runtime/`: frame-loop coordinators, scene/sprite runtime payload management, `.tlscript` parallel planning glue, and NPS UDP transport runtime integration

## Documentation

- `README.md`: workspace overview and quick start
- `docs/README.md`: documentation index
- `docs/tlscript-lexer.md`: `.tlscript` zero-copy lexer/token design
- `docs/tlscript-parser-plan.md`: `.tlscript` parser/AST roadmap and V1 grammar
- `docs/tlscript-semantic.md`: `.tlscript` semantic analyzer (types, handles, WASM sandboxing)
- `docs/tlscript-parallel-runtime.md`: `.tlscript` parallel contracts, advisor, and runtime dispatch planning
- `docs/tileline-pre-alpha-transition.md`: immediate Foundation -> Pre-Alpha transition gates and sprint plan
- `docs/tileline-beta-roadmap.md`: phased plan from foundation state to a usable beta
- `docs/nps-protocol.md`: NPS packet format, reliability, authority handoff, and MPS integration
- `docs/nps-runtime-plan.md`: NPS channel/tick/snapshot runtime plan for the beta transport path
- `docs/paradoxpe-foundation.md`: ParadoxPE handle model, SoA storage, broadphase/solver pipeline, snapshot base, and script ABI
- `docs/paradoxpe-tlscript-examples.md`: verified `.tlscript` examples targeting the current ParadoxPE ABI
- `docs/gms-dispatch-planner.md`: GMS workload planning and multi-GPU dispatch notes
- `docs/runtime-bridge-flow.md`: canonical MPS -> GMS -> runtime synchronization flow
- `docs/runtime-scene-showcase.md`: runtime scene/sprite payload model and bounce-tank showcase scaffolding
- `docs/runtime-scene-workload.md`: runtime scene/sprite density to GMS workload mapping
- `docs/mgs-scene-workload.md`: runtime/mobile scene/sprite density to MGS hint and tile planning mapping
- `docs/demos/README.md`: show/demo documentation area (`docs/demos/*`)

## Current Architecture

### 1. CPU Production (MPS)

CPU-side preprocessing / simulation tasks are submitted to `mps::MpsScheduler` with:

- priority (`Critical`, `High`, `Normal`, `Background`)
- core preference (`Performance`, `Efficient`, `Auto`)
- native Rust closures or WASM tasks

The scheduler is topology-aware and designed to saturate logical cores in a `make -j$(nproc)` style.

### 2. CPU -> GPU Bridge (`tl-core`)

`tl_core::MpsGmsBridge` converts completed MPS tasks into frame-scoped GPU workload plans:

- lock-free completion queue (`crossbeam::queue::SegQueue`)
- frame sealing (`seal_frame`) to avoid partial planning
- data-oriented mapping from bridge tasks to GMS workload classes
- single-GPU and explicit multi-GPU plans produced per frame

### 3. GPU Planning and Sync (GMS + `tl-core::graphics::multigpu::sync`)

GMS provides:

- adapter discovery and scoring
- SM/CU/CoreCluster-aware distribution heuristics (with native-probe -> table -> heuristic fallback)
- asymmetric multi-GPU planning (heavy lanes to primary, UI/Post-FX to secondary)
- aggressive secondary-lane floor + heavy spill redistribution so helper GPU utilization does not
  collapse under primary saturation
- portable sync plan metadata (queue timelines + bounded waits)

`tl-core` adds a portable runtime synchronizer:

- `SubmissionIndex` as timeline markers
- bounded `Device::poll(PollType::Wait { .. })` compose waits (default `0.8ms`)
- Apple UMA integration via `gms::AdaptiveBuffer`

### 4. Render Loop Integration (`runtime`)

`runtime` contains canonical non-benchmark integration glue:

- bridge pumping and frame-plan draining
- queue submission recording (primary / secondary / transfer)
- present reconcile calls before `SurfaceTexture::present`
- optional Apple UMA telemetry feedback

## Quick Start

### Build / Check

```bash
cargo check
```

### Run GMS Render Benchmark

```bash
cargo run -p gms --example render_benchmark -- --mode max --vsync off --warmup 2 --duration 10 --resolution 1280x720
```

Stable mode (recommended for Apple Silicon UMA tests):

```bash
cargo run -p gms --example render_benchmark -- --mode stable --vsync on --warmup 2 --duration 10 --resolution 1920x1080
```

### Run MGS Render Benchmark

```bash
cargo run -p mgs --example render_benchmark -- --mode stable --vsync on --warmup 2 --duration 10 --resolution 1280x720
```

### Test Core Runtime Integration

```bash
cargo test -p tl-core
cargo test -p runtime
```

## Canonical Runtime Flow (Non-Benchmark)

The intended engine-side `wgpu` frame flow is:

1. `runtime::WgpuRenderLoopCoordinator::tick_bridge()`
2. `begin_next_frame_plan()`
3. Submit primary GPU work and call `record_primary_submission(...)`
4. Optionally submit secondary helper work with `submit_secondary_helper_for_frame(...)`
5. Optionally submit transfer work and call `record_transfer_submission(...)`
6. Call `reconcile_present(...)` before `frame.present()`
7. Call `report_frame_telemetry(...)` (especially for Apple UMA paths)

This keeps synchronization policy inside `src/` crates instead of benchmark/example code.

For pre-alpha integration freeze, prefer:

1. `runtime::WgpuRenderLoopCoordinator::run_pre_alpha_frame(...)`
2. with canonical phase order: `network -> script -> physics -> render_plan -> present`
3. when wiring real systems, use `run_pre_alpha_frame_with_systems(...)`

## Platform Notes

### NVIDIA / AMD / Apple GPU Unit Counts

GMS uses a layered detection strategy for SM/CU/CoreCluster counts:

1. Native probe (best effort):
   - NVIDIA: `nvidia-smi`
   - AMD: `rocminfo`
   - Apple: `system_profiler`
2. Device-name lookup table
3. `wgpu`-limits-based heuristic fallback

The benchmark and planner diagnostics show the active source (`native`, `table`, or `heuristic`).

### Apple Silicon (Metal / UMA)

UMA-specific tuning and adaptive buffer controls are implemented in `gms` and wired into
`tl-core`/`runtime` for stability recovery and encoder-window management.

## Current `.tlscript` Status

`.tlscript` now includes a substantial compiler/runtime planning pipeline:

- zero-copy lexer/token model (`&str` slices)
- indentation-aware parser + AST
- semantic analysis (types, handles, WASM sandbox policy)
- typed IR + lowering
- WASM codegen (MVP-oriented)
- `@net(...)` compiler hook for sync metadata extraction
- `@parallel(...)` / `@main_thread` / `@reduce(...)` contract validation
- parallel advisor + runtime dispatch planner/fallback metrics
- ParadoxPE-aware `domain="bodies"` runtime planning and MPS chunk routing helpers
- verified ParadoxPE `.tlscript` examples compiled through lexer -> semantic -> IR -> WASM tests
- initial ParadoxPE host ABI names wired into semantic/lowering/codegen defaults
- fixed joint support, material combine rules, and snapshot/interpolation primitives in the physics core
- NPS-side direct `PhysicsSnapshot` to quantized transform-batch export path

Next pipeline steps:

- typed IR-driven WASM codegen refactor (replace remaining AST-direct paths)
- `.tlscript` -> MPS compile/cache/submit runtime path
- richer host ABI for gameplay systems (ParadoxPE solver/control, networking, engine handles)
- runtime profiling/diagnostics surfaces for script parallel dispatch decisions

## License

This repository is structured to be MIT-license ready (code comments and docs use FOSS-friendly
terminology and style). Add/update the root `LICENSE` file as the project license source of truth.
