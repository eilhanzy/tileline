# Runtime Scene + Sprite Showcase Design

This note documents the pre-alpha scene orchestration foundation added for the upcoming show demo.

## Implementation Location

- `runtime/src/scene.rs`
- `runtime/src/scene_workload.rs`
- `runtime/src/wgpu_render_loop.rs`
- `paradoxpe/src/world.rs` (interpolation alpha helper)

## Goals

- Transparent 3D tank volume (visual only) containing many bouncing balls
- Progressive spawn of thousands of balls (not all in one frame)
- Per-ball color variation + shading-ready material payload
- Scene payloads in `src/` (not benchmark-only code)
- Decoupled render FPS and physics tick rate

## Scene Data Model

`runtime::scene` exports:

- `SceneInstance3d`: primitive + transform + material + shadow flags
- `SpriteInstance`: overlay/sprite payload (HUD/progress/etc.)
- `SceneFrameInstances`: `opaque_3d`, `transparent_3d`, `sprites`
- `ScenePrimitive3d`: `Sphere` / `Box` / `Mesh { slot }`
- `SceneMaterial` + `ShadingModel`
- `set_sprite_program(...)` hook for compiled `.tlsprite` overlays

## Bounce Tank Controller

`BounceTankSceneController` manages:

- static wall collider setup (6 AABB walls in ParadoxPE)
- progressive ball spawning (`spawn_per_tick` up to `target_ball_count`)
- sphere collider assignment per ball
- deterministic color/material variation per ball
- render payload emission (`build_frame_instances`)

The transparent tank is emitted as one `transparent_3d` box instance plus thin edge prisms for
silhouette clarity; balls are emitted as `opaque_3d` spheres by default.

When `.tlscript` sets `set_ball_mesh_slot(...)` or `set_container_mesh_slot(...)`, the same scene
path can emit `ScenePrimitive3d::Mesh { slot }` and consume FBX bindings loaded from `.tlsprite`.

## Tick/FPS Decoupling

`TickRatePolicy` resolves physics tick rate from render pacing:

- `RenderSyncMode::Vsync { display_hz }`
- `RenderSyncMode::FpsCap { fps }`
- `RenderSyncMode::Uncapped`

Policy resolves:

- `resolve_tick_hz(...)`
- `resolve_fixed_dt_seconds(...)`

ParadoxPE now exposes `PhysicsWorld::interpolation_alpha()` to support render interpolation when
physics tick and render FPS differ.

## Runtime Integration Path

Use `WgpuRenderLoopCoordinator::run_pre_alpha_frame_with_systems(...)` for canonical order:

1. Network
2. Script
3. Physics
4. RenderPlan
5. Present

The render phase can consume `BounceTankSceneController::build_frame_instances(...)` and forward
instance payloads into the active renderer backend.

For deterministic backend batching, runtime now exposes `DrawPathCompiler` and `RuntimeDrawFrame`
(`runtime/src/draw_path.rs`), plus `TelemetryHudComposer` for overlay meters
(`runtime/src/telemetry_hud.rs`) and `WgpuSceneRenderer` for real render-pass encoding
(`runtime/src/wgpu_scene_renderer.rs`).

`runtime::estimate_scene_workload_requests(...)` can be used to translate scene/sprite density into
`gms::WorkloadRequest` and `gms::MultiGpuWorkloadRequest` for planner-driven dispatch.

For mobile/serial fallback paths, `runtime::estimate_mobile_workload_hint(...)` converts the same
scene payload into `mgs::MpsWorkloadHint`.
