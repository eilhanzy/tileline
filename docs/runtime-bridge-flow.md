# Runtime Bridge Flow (MPS -> GMS -> Present Sync)

This document describes the canonical engine-side flow for integrating Tileline's CPU/GPU scaling
stack into a `wgpu` render loop.

## Relevant Modules

- `tl-core/src/core/bridge.rs` (`MpsGmsBridge`)
- `tl-core/src/graphics/multigpu/sync.rs` (`MultiGpuFrameSynchronizer`)
- `runtime/src/frame_loop.rs` (`FrameLoopRuntime`)
- `runtime/src/network_transport.rs` (`NetworkTransportRuntime`)
- `runtime/src/pre_alpha_loop.rs` (`RuntimePhaseOrderTracker`)
- `runtime/src/scene.rs` (`BounceTankSceneController`, scene/sprite payloads, tick-rate policy)
- `runtime/src/scene_dispatch.rs` (scene workload estimate -> bridge task submission + frame seal)
- `runtime/src/scene_workload.rs` (scene density -> GMS planner request synthesis)
- `runtime/src/mobile_scene_workload.rs` (scene density -> MGS bridge hint synthesis)
- `runtime/src/scheduler_path.rs` (auto select `GMS` vs `MGS` from adapter metadata)
- `runtime/src/wgpu_render_loop.rs` (`WgpuRenderLoopCoordinator`)
- `runtime/src/tlscript_parallel.rs` (`TlscriptParallelRuntimeCoordinator`)

## Design Constraints

- CPU workers must not block waiting for GPU frame completion
- Communication between MPS and GMS planning should remain lock-free
- Present synchronization may block, but only within a strict bounded budget (default `0.8ms`)
- Runtime code should consume policies from `src/` crates, not duplicate logic in examples

## Canonical Per-Frame Flow

1. Submit CPU-side preprocessing/simulation work through `MpsGmsBridge`
2. Seal the frame (`seal_frame`) once all expected CPU tasks for that frame are queued
3. Runtime executes pre-alpha phase order in `runtime/src`:
   `network -> script -> physics -> render_plan -> present`
4. RenderPlan phase pumps the bridge (`pump`) to drain MPS completions and publish frame plans
5. Runtime drains published `BridgeFramePlan` objects into a render-thread local queue
6. Render loop submits primary GPU work and records `SubmissionIndex`
7. Optional secondary/helper GPU work is submitted and recorded
8. Optional transfer/copy queue work is submitted and recorded
9. Present reconcile checks queue completion state with bounded waits
10. Present proceeds when ready, or times out/spillback policy applies
11. Apple UMA telemetry feeds adaptive buffer decisions (when active)
12. Script-side workloads (when present) can be planned/routed through
    `TlscriptParallelRuntimeCoordinator` before MPS submission so `.tlscript` parallel contracts
    and fallback telemetry are preserved in the runtime path

`WgpuRenderLoopCoordinator::run_pre_alpha_frame(...)` is the generic canonical entrypoint for this
order.

For integrated runtime systems, use
`WgpuRenderLoopCoordinator::run_pre_alpha_frame_with_systems(...)`, which wires:

- `NetworkTransportRuntime::pump_nonblocking(...)`
- `TlscriptParallelRuntimeCoordinator` script phase callback
- `PhysicsWorld::step(...)` + snapshot cadence queueing
- render-plan callback + present reconcile

## Why This Is Split Across Crates

- `mps`: CPU execution and WASM scheduling
- `gms`: GPU discovery/planning/runtime helper primitives
- `tl-core`: bridge + portable multi-GPU synchronization policy
- `runtime`: actual frame-loop glue to real `wgpu::Queue::submit` and `present`

This keeps the engine runtime path reusable and avoids benchmark-only logic leaking into the core.

## NPS Runtime Path

Networking follows the same rule: runtime-owned glue stays in `src/`, not examples.

`NetworkTransportRuntime` currently provides:

- explicit peer/address mapping for UDP
- non-blocking socket pumping with `try_recv_from` / `try_send_to`
- NPS manager encode/decode integration
- runtime-owned send queue for encoded and retransmit datagrams
- ParadoxPE snapshot cadence emission

This gives NPS a canonical runtime entry point parallel to the existing MPS/GMS coordinators.
