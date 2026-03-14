# Runtime `.tlscript` Showcase Bootstrap

This note documents the first runtime-integrated `.tlscript` control path for the bounce-tank demo.

## Source Files

- `runtime/src/tlscript_showcase.rs`
- `runtime/examples/tlscript_first_demo.rs`
- `runtime/src/scene.rs` (`BounceTankRuntimePatch`, `apply_runtime_patch`)

## Pipeline

The compile path is fully memory-resident:

1. `Lexer`
2. `Parser`
3. `SemanticAnalyzer` (soft diagnostics)
4. `ParallelHookAnalyzer`
5. Typed-IR lowering + parallel metadata annotation

The frame path:

1. evaluate entry function (`showcase_tick`) with runtime bindings
2. emit bounded `BounceTankRuntimePatch`
3. apply patch to scene/world (`apply_runtime_patch`)
4. expose planner decision (`ParallelDispatchDecision`) for telemetry

## Supported Builtins (V1)

- `set_spawn_per_tick(v)`
- `set_target_ball_count(v)`
- `set_linear_damping(v)`
- `set_ball_restitution(v)`
- `set_wall_restitution(v)`
- `set_initial_speed(min, max)`
- `set_initial_speed_min(v)`
- `set_initial_speed_max(v)`
- `set_ball_mesh_slot(slot)`
- `set_container_mesh_slot(slot)`
- `set_fbx_full_render(bool)`
- `set_camera_move_speed(v)`
- `set_camera_look_sensitivity(v)`
- `set_camera_pose(ex, ey, ez, tx, ty, tz)`

All values are soft-validated and clamped during patch application to keep simulation and runtime
controls safe.
