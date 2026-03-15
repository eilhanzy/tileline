# Runtime `.tlsprite` Sprite Management

This note documents the first `.tlsprite` integration path for runtime-managed sprite overlays.

## Implementation Location

- `runtime/src/tlsprite.rs`
- `runtime/src/tlsprite_editor.rs` (Alpha list-mode editor model)
- `runtime/src/scene.rs` (`BounceTankSceneController::set_sprite_program`)
- `docs/demos/tlapp/bounce_hud.tlsprite`

## Format (`tlsprite_v1`)

`.tlsprite` is a lightweight text format with section blocks:

```text
tlsprite_v1
[name]
sprite_id = 1
kind = generic | hud | camera | terrain
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

`kind` defaults:

- explicit `kind` wins
- if omitted and `layer >= 100`, runtime infers `hud`
- otherwise runtime uses `generic`

`camera` and `terrain` kinds provide sensible defaults (position/size/layer/color) when fields are
not specified, so minimal sections can be authored quickly.

### Light Authoring (`entry_kind = light`)

`.tlsprite` now supports static light definitions in the same file. Use `entry_kind` to switch an
entry from sprite payload to light payload:

```text
[light.flashlight]
entry_kind = light
light_id = 1001
light_type = spot
position = 0.0, 3.0, 6.0
direction = 0.0, -0.4, -1.0
color = 1.0, 0.95, 0.85
intensity = 14.0
range = 40.0
inner_cone_deg = 12.0
outer_cone_deg = 24.0
softness = 0.35
casts_shadow = true
specular_strength = 1.1
layer = 20
```

Supported light keys:

- `light_id`
- `light_type = spot | point`
- `position`
- `direction`
- `color`
- `intensity`
- `range`
- `inner_cone_deg`
- `outer_cone_deg`
- `softness`
- `casts_shadow`
- `specular_strength`
- `layer`

Light entries do not emit sprite draw commands. They are routed into
`SceneFrameInstances::lights` and consumed by runtime lighting in `src/`.

## Runtime Flow

1. Load source as in-memory string (engine asset loader or `std::fs` at app layer).
2. Compile with `runtime::compile_tlsprite(...)`.
3. Inspect diagnostics.
4. Pass resulting `TlspriteProgram` to scene controller:
   - `scene.set_sprite_program(program)`
5. Per-frame emission is done during `build_frame_instances(...)`.

If no program is installed (or emitted list is empty), runtime keeps the built-in fallback progress sprite.

Reference starter file for sprite kinds:

- `docs/examples/tlsprite/runtime_basic_types.tlsprite`

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
- `runtime/examples/tlapp.rs`
- `runtime/examples/tlsprite_list_editor.rs` (Alpha list inspector/editor scaffold)
- `runtime/src/bin/tlsprite_editor.rs` (runtime-owned CLI entrypoint)

## 3-Phase Plan

- Phase 1 (implemented): polling hot reload + safe fallback.
- Phase 2 (implemented): `notify` file-watch backend + polling fallback.
- Phase 3: asset pipeline integration (precompiled sprite packs + runtime cache invalidation).

## Asset Pipeline (Phase 3)

Runtime now exposes precompiled pack + cache primitives:

- `compile_tlsprite_pack(source)` -> `TlspritePack` (`TLSPK003`)
- `load_tlsprite_pack(bytes)` -> `TlspriteProgram`
- `TlspriteProgramCache`:
  - deduplicates programs by source hash
  - binds cache entries by asset path
  - supports `invalidate_path(...)` / `invalidate_all()`
  - provides telemetry via `stats()`
- `TlspriteProgram::merge_programs(...)`:
  - appends multiple compiled `.tlsprite` programs in deterministic order
  - used by `.tljoint` scene bundles for multi-file composition

`load_tlsprite_pack(...)` keeps backward compatibility with old pack versions (`TLSPK001`,
`TLSPK002`) while writing new light-aware packs as `TLSPK003`.

`TlspriteWatchReloader::reload_into_cache(...)` bridges hot reload events into cache bindings.

## Current Scope

- Deterministic parser and soft diagnostics.
- Dynamic axis scaling from runtime signals for HUD bars.
- Base sprite kinds (`generic`, `hud`, `camera`, `terrain`) with runtime defaults.
- No disk I/O required in runtime core.
- Renderer-agnostic output (`Vec<SpriteInstance>`).

## Editor CLI (List Mode)

Run the editor directly as a runtime binary:

```bash
cargo run -p runtime --bin tlsprite_editor -- --file docs/demos/tlapp/bounce_hud.tlsprite
```

Useful options:

- `--watch-ms 250 --clear`: hot-reload mode with clean redraw
- `--markdown`: print markdown table output
- `--write-markdown docs/demos/tlapp/bounce_hud.table.md`: export table file
- `--strict` / `--strict-warnings`: CI-friendly non-zero exit behavior
- `--init-if-missing`: create a starter `.tlsprite` file if missing

## Renderer Mapping Notes

Current `runtime::WgpuSceneRenderer` path consumes sprite kind metadata and applies:

- kind-aware virtual atlas rect selection (by `kind` + `texture_slot`)
- camera-style shader treatment (lens/ring emphasis)
- terrain-style shader treatment (banding/gradient emphasis)

This keeps type behavior in `src/` runtime code rather than benchmark-only scripts.

## FBX Mesh Slot Binding

`TlspriteProgram::mesh_fbx_bindings()` exports unique `(slot, path)` pairs inferred from sprite
sections that define `fbx = ...`:

- `texture_slot` is reused as mesh slot id (0..255)
- first declaration per slot wins

`TLApp` uses this to call `WgpuSceneRenderer::bind_fbx_mesh_slot_from_path(...)`, which enables
runtime `ScenePrimitive3d::Mesh { slot }` rendering via external FBX assets.
