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
- dual-dGPU-ready helper selection:
  if primary is discrete and another discrete adapter exists, helper lane prefers the secondary
  discrete adapter before falling back to integrated-latency selection
- adaptive throughput stress:
  benchmark burst work units scale with resolution/device type, and secondary helper workload uses
  `passes_per_work_unit` intensity to reduce low-utilization spikes
- non-multi-GPU parity:
  primary (single-GPU) path also applies `passes_per_work_unit` intensity so utilization tuning is
  consistent whether helper lanes are enabled or disabled
- primary stress kernel:
  single-GPU path can route synthetic workload through a compute shader storage-buffer stress pass
  (not only clear passes) to improve sustained GPU occupancy and reduce command-burst overhead
- shared transfer strategy (portable host-bridge model in `wgpu`)
- sync plan metadata (timeline-like queue submissions + bounded waits)
- projected score gain estimates and `%20` target checks
- Vulkan API compatibility gate for explicit multi-GPU startup

### Aggressive Secondary Utilization (Core Path)

To avoid the helper GPU staying near-idle while the primary is saturated, the core planner/runtime
now apply an aggressive secondary-lane policy directly in `gms/src`:

- `gms/src/bridge.rs` enforces a minimum secondary total-job share after initial lane assignment.
- If secondary load is below target, heavy lanes are moved from primary -> secondary.
- Lane move order is topology-aware:
  - secondary dGPU: sampled/physics first
  - secondary UMA/iGPU: object/physics first
- Target share is bounded by topology and relative score, with dual-dGPU bias for stronger helper
  scaling.
- `gms/src/multi_gpu_runtime.rs` derives secondary WU/present with stronger floors so helper
  submission remains meaningful even when planner ratios are conservative.
- Secondary pass intensity (`passes_per_work_unit`) is expanded to a wider range so helper-side
  GPU occupancy is less bursty.
- Runtime intensity governor:
  secondary WU/present + passes/WU start from a conservative fraction of cap, ramp up on stable
  frames, and back off automatically on queue backpressure/timeout events to protect frame-time.

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
- in-flight backpressure: if secondary queue depth is saturated and wait times out, runtime drops
  one oldest tracked submission slot, applies governor backoff, and continues helper submission

## Runtime Telemetry To Watch

When tuning explicit multi-GPU behavior, track these summary fields:

- `secondary WU/present`
- `passes/WU`
- `total secondary WU`
- `queue waits/polls/timeouts/skips`

These values are emitted by `gms/src/render_benchmark.rs` from `MultiGpuExecutorSummary` and map
directly to the core planner/runtime behavior.

## Apple Silicon / UMA Interaction

GMS tuning and `AdaptiveBuffer` support provide UMA-aware behavior for Metal/Apple Silicon paths:

- command encoder window regulation
- shared buffer arbitration
- stability recovery heuristics

These policies are consumed by `tl-core` and `runtime` rather than being confined to benchmarks.

## Linux Panthor Path (Mali / Immortalis)

`gms/src/tuning.rs` now includes a dedicated Linux Panthor-oriented runtime profile:

- activation heuristic:
  - Vulkan backend
  - ARM/Mali/Immortalis adapter name
  - Panthor/Panfrost hint from driver text or `/sys/module/panthor`
- tuning behavior:
  - deeper frame-latency queue (`desired_maximum_frame_latency`)
  - lower synthetic burst caps for integrated paths
  - startup ramp + prewarm tuned for embedded SoC stability
  - tighter pass-per-work-unit cap in benchmark stress mode

Goal: reduce bursty frame-time spikes on RK3588-class systems while keeping throughput scaling
predictable.
