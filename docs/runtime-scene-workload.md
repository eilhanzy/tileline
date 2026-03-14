# Runtime Scene -> GMS Workload Mapping

This document describes the runtime-side path that turns scene/sprite payload density into GMS
planner requests without relying on benchmark-only code.

## Source Files

- `gms/src/scene_workload.rs`
- `runtime/src/scene_dispatch.rs`
- `runtime/src/scene_workload.rs`
- `runtime/examples/bounce_tank_showcase.rs`

## Why This Exists

- Keep workload synthesis inside `src/` crates.
- Reuse one estimator for runtime and demos.
- Preserve decoupled physics tick vs render FPS behavior while still feeding GMS meaningful load.

## Data Flow

1. `BounceTankSceneController` emits `SceneFrameInstances`.
2. `runtime::build_scene_workload_snapshot(...)` derives:
   `opaque/transparent/sprite counts`, `shadow flags`, `dynamic body count`, estimated contacts.
3. `gms::estimate_scene_workload(...)` outputs:
   - `WorkloadRequest` (single-GPU planner input)
   - `MultiGpuWorkloadRequest` (multi-GPU planner input)
4. Runtime can submit these estimates into the bridge via
   `runtime::submit_scene_estimate_to_bridge(...)` or
   `WgpuRenderLoopCoordinator::submit_scene_workload_for_frame(...)`.

## Tick/FPS Decoupling

Use:

- `TickRatePolicy::resolve_fixed_dt_seconds(...)` for physics fixed-step.
- `PhysicsWorld::step(render_dt)` for substep catch-up.
- `PhysicsWorld::interpolation_alpha()` for render interpolation.

This keeps simulation cadence deterministic while render remains synced to V-Sync/FPS policy.

## Showcase Example

Run:

```bash
cargo run -p runtime --example bounce_tank_showcase
```

The example prints frame-level spawn progress and estimated GMS job lanes
(`sampled/object/physics/ui/postfx`) from runtime scene payloads.
