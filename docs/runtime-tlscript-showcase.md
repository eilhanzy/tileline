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

## Supported Builtins (V1+)

- `set_spawn_per_tick(v)`
- `set_target_ball_count(v)`
- `set_scene_mode(mode)` (`2d|3d`)
- `set_linear_damping(v)`
- `set_ball_restitution(v)`
- `set_wall_restitution(v)`
- `set_initial_speed(min, max)`
- `set_initial_speed_min(v)`
- `set_initial_speed_max(v)`
- `set_spawn_profile_2d(target_balls, spawn_per_tick, speed_min, speed_max)`
- `set_side_view_plane_z(z)`
- `set_side_view_center(x, y)`
- `set_side_view_zoom(zoom)`
- `set_side_view_camera(center_x, center_y, zoom)`
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
- `set_render_distance(v)` (`v <= 0` => `off`)
- `set_adaptive_distance(mode)` (`auto|on|off` or bool/int)
- `set_distance_blur(mode)` (`auto|on|off` or bool/int)
- `set_msaa(samples)` (`off|1|2|4`)
- `contact_any()` / `touch_any()`
- `contact_pairs()` / `touch_pairs()`
- `contact_manifolds()` / `touch_manifolds()`
- `tile_set(x, y, tile_id)` / `tile_place(x, y, tile_id)`
- `tile_dig(x, y)`
- `tile_fill(x0, y0, x1, y1, tile_id)`
- `tile_get(x, y)` (runtime tile-world query)

All values are soft-validated and clamped during patch application to keep simulation and runtime
controls safe.

Tile query resolution order is deterministic:

1. latest same-frame mutation (`tile_set` / `tile_place` / `tile_dig`)
2. latest same-frame `tile_fill`
3. runtime tile-world lookup
4. empty tile (`0`)

Global contact variables are auto-injected each frame from the latest completed physics step:

- `contact_any`, `contact_pairs`, `contact_manifolds`
- aliases: `touch_any`, `touch_pairs`, `touch_manifolds`

Source mapping:

- `contact_pairs = broadphase.candidate_pairs`
- `contact_manifolds = narrowphase.manifolds`
- `contact_any = (contact_manifolds > 0)`

## Light Merge Rule

Final frame lights are merged deterministically:

1. Static `.tlsprite` light entries build the base list.
2. `.tlscript` applies id-based partial overrides.
3. Unknown `light_id` emits soft warning (no hard fail).
4. Renderer clamps to `MAX_LIGHTS` with deterministic priority.

## Runtime Merge Precedence

For frame-level script controls, precedence is deterministic:

1. CLI script overlay
2. Scene `.tlscript` (single / multi / `.tljoint`)
3. Existing runtime state

This applies to the new effect controls (`render_distance`, `adaptive_distance`, `distance_blur`,
`msaa`) as well.
