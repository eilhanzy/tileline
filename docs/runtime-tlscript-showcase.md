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
5. emit light overrides + RT mode hints for renderer merge

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
- `set_light_enabled(id, bool)`
- `set_light_position(id, x, y, z)`
- `set_light_direction(id, x, y, z)`
- `set_light_intensity(id, v)`
- `set_light_range(id, v)`
- `set_light_cone(id, inner_deg, outer_deg)`
- `set_light_color(id, r, g, b)`
- `set_light_softness(id, v)`
- `set_rt_mode(mode)` (`off|auto|on`)

All values are soft-validated and clamped during patch application to keep simulation and runtime
controls safe.

## Light Merge Rule

Final frame lights are merged deterministically:

1. Static `.tlsprite` light entries build the base list.
2. `.tlscript` applies id-based partial overrides.
3. Unknown `light_id` emits soft warning (no hard fail).
4. Renderer clamps to `MAX_LIGHTS` with deterministic priority.
