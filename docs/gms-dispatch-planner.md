# GMS Dispatch Planner (GPU Workload Distribution)

This document summarizes the planning-side behavior of Tileline's `GMS` module. The planner
produces workload assignments and synchronization metadata; it does not directly execute GPU
pipelines.

## Implementation Location

- `gms/src/hardware.rs`
- `gms/src/bridge.rs`
- `gms/src/multi_gpu_runtime.rs`
- `gms/src/tuning.rs`

## Core Responsibilities

- Enumerate adapters with `wgpu`
- Build a scored adapter inventory (discrete/integrated/virtual aware)
- Estimate SM/CU/CoreCluster parallelism (`native -> table -> heuristic`)
- Split work proportionally across GPUs
- Produce multi-GPU bridge/sync plans for runtime integration

## Scoring Inputs (Asymmetric Weighting)

The planner combines multiple signals:

- Compute unit count (SM/CU/Core Clusters): primary throughput factor
- Memory bandwidth estimate: heavy texture / sampled workloads
- VRAM/unified memory characteristics
- Topology bias (discrete VRAM vs unified/system memory)
- `wgpu` limits/features and mappability hints

This allows heavy throughput tasks to prefer the strongest GPU while still assigning UI/Post-FX or
latency-sensitive lanes to secondary adapters when beneficial.

## Dispatch Outputs

### Single-GPU / General Dispatch

`DispatchPlan` and `GpuWorkAssignment` describe:

- object update jobs
- physics jobs
- workgroup sizing
- estimated dispatch workgroups
- zero-copy upload/storage buffer strategy (`MAP_WRITE`-oriented)

### Explicit Multi-GPU Dispatch

`MultiGpuDispatcher` emits:

- lane roles (primary present / secondary latency / auxiliary compute)
- shared transfer strategy (portable host-bridge model in `wgpu`)
- sync plan metadata (timeline-like queue submissions + bounded waits)
- projected score gain estimates and `%20` target checks
- Vulkan API compatibility gate for explicit multi-GPU startup

## Vulkan Version Gate (Explicit Multi-GPU Safety)

`gms/src/multi_gpu_runtime.rs` now validates Vulkan API major/minor compatibility before opening
the secondary helper lane:

- Both adapters must report Vulkan API versions with the same `major.minor` (for example `1.3` +
  `1.3`).
- A mismatch (`1.3` + `1.2`) hard-stops explicit multi-GPU startup.
- If a Vulkan version cannot be parsed for either adapter, explicit multi-GPU startup is refused
  to avoid undefined cross-adapter behavior.
- Version extraction first checks adapter text fields for Vulkan/API-version markers, then falls
  back to `vulkaninfo --summary` device rows (`apiVersion`) when needed.

Policy behavior:

- `MultiGpuInitPolicy::Auto`: returns `Ok(None)` (multi-GPU disabled for the session)
- `MultiGpuInitPolicy::Force`: returns an initialization error with adapter/version diagnostics

## Portable Synchronization Model

`wgpu` does not expose native cross-adapter semaphores/fences directly. Tileline therefore models
sync with:

- queue `SubmissionIndex` values as timeline markers
- bounded `Device::poll(...)` waits
- explicit compose budget policy (sub-millisecond target in `tl-core`)
- in-flight backpressure: if secondary queue depth is saturated and wait times out, helper submit is
  skipped for that frame instead of over-queuing bursts

## Apple Silicon / UMA Interaction

GMS tuning and `AdaptiveBuffer` support provide UMA-aware behavior for Metal/Apple Silicon paths:

- command encoder window regulation
- shared buffer arbitration
- stability recovery heuristics

These policies are consumed by `tl-core` and `runtime` rather than being confined to benchmarks.
