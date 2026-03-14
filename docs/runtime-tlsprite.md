# Runtime `.tlsprite` Sprite Management

This note documents the first `.tlsprite` integration path for runtime-managed sprite overlays.

## Implementation Location

- `runtime/src/tlsprite.rs`
- `runtime/src/scene.rs` (`BounceTankSceneController::set_sprite_program`)
- `runtime/examples/assets/bounce_hud.tlsprite`

## Format (`tlsprite_v1`)

`.tlsprite` is a lightweight text format with section blocks:

```text
tlsprite_v1
[name]
sprite_id = 1
texture_slot = 0
layer = 100
position = x, y, z
size = w, h
rotation_rad = 0.0
color = r, g, b, a
scale_axis = x | y
scale_source = spawn_progress | spawn_remaining | live_ball_ratio
scale_min = 0.02
scale_max = 1.0
```

## Runtime Flow

1. Load source as in-memory string (`include_str!` or engine asset loader).
2. Compile with `runtime::compile_tlsprite(...)`.
3. Inspect diagnostics.
4. Pass resulting `TlspriteProgram` to scene controller:
   - `scene.set_sprite_program(program)`
5. Per-frame emission is done during `build_frame_instances(...)`.

If no program is installed (or emitted list is empty), runtime keeps the built-in fallback progress sprite.

## Hot Reload (Phase 1 + Phase 2)

`runtime::TlspriteHotReloader` provides hash-based polling reload:

1. Read file from disk.
2. Hash content.
3. Recompile only when hash changes.
4. If compile fails, keep last valid program (`keep_last_good_program = true` by default).

`runtime::TlspriteWatchReloader` adds event-driven watch behavior:

1. Tries `notify` backend first (OS file events).
2. Falls back to polling automatically when watcher init fails.
3. Performs safety polling at `poll_interval_ms` to avoid missed-edge cases.
4. Reuses `TlspriteHotReloader` semantics for diagnostics and last-good fallback.

Examples wired:

- `runtime/examples/bounce_tank_showcase.rs`
- `runtime/examples/auto_scene_scheduler.rs`

## 3-Phase Plan

- Phase 1 (implemented): polling hot reload + safe fallback.
- Phase 2 (implemented): `notify` file-watch backend + polling fallback.
- Phase 3: asset pipeline integration (precompiled sprite packs + runtime cache invalidation).

## Asset Pipeline (Phase 3)

Runtime now exposes precompiled pack + cache primitives:

- `compile_tlsprite_pack(source)` -> `TlspritePack`
- `load_tlsprite_pack(bytes)` -> `TlspriteProgram`
- `TlspriteProgramCache`:
  - deduplicates programs by source hash
  - binds cache entries by asset path
  - supports `invalidate_path(...)` / `invalidate_all()`
  - provides telemetry via `stats()`

`TlspriteWatchReloader::reload_into_cache(...)` bridges hot reload events into cache bindings.

## Current Scope

- Deterministic parser and soft diagnostics.
- Dynamic axis scaling from runtime signals for HUD bars.
- No disk I/O required in runtime core.
- Renderer-agnostic output (`Vec<SpriteInstance>`).
