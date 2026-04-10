# TLScript Guide for Tileline v0.4.5

Release scope: **strict v0.4.5**  
Language: English

This document is the canonical `.tlscript` guide for the `v0.4.5` release line.

## Scope Lock

This guide is intentionally locked to:
- `docs/releases/v0.4.5.md`
- `docs/tileline-v0.4.5-roadmap.md`

It does **not** document patch/newer script surfaces added after `v0.4.5`.
A dedicated exclusion table is provided in this guide.

## 1. What TLScript Is in v0.4.5

In `v0.4.5`, `.tlscript` is the runtime control layer for scene behavior and 2D foundation flows.
It is used to:
- tune spawn/physics behavior at runtime
- drive side-view scene configuration
- mutate/query tile-world state
- coordinate camera and mesh-slot controls

Primary runtime references:
- `docs/runtime-tlscript-showcase.md`
- `docs/demos/tlapp/README.md`

## 2. Compile and Execution Pipeline

### 2.1 Compile Pipeline

The compile pipeline is memory-resident and follows this order:
1. `Lexer`
2. `Parser`
3. `SemanticAnalyzer` (soft diagnostics)
4. Parallel contract analysis hooks
5. Lowering/codegen integration points

Deep references:
- `docs/tlscript-lexer.md`
- `docs/tlscript-parser-plan.md`
- `docs/tlscript-semantic.md`
- `docs/tlscript-parallel-runtime.md`

### 2.2 Runtime Frame Model

Per frame, runtime:
1. evaluates script entry function
2. emits bounded runtime patch data
3. merges script output with runtime state
4. applies patch to scene/physics systems

Merge precedence is deterministic:
1. CLI script overlay
2. scene script (`.tlscript` / multi-script / `.tljoint`)
3. existing runtime state

## 3. v0.4.5 Built-ins (In Scope)

The built-ins below are the v0.4.5 guide surface for the 2D foundation/runtime path.

| Area | Built-ins | Purpose |
|---|---|---|
| Spawn/control | `set_spawn_per_tick`, `set_target_ball_count`, `set_scene_mode` | Runtime scene pressure and mode control |
| Physics tuning | `set_linear_damping`, `set_ball_restitution`, `set_wall_restitution`, `set_initial_speed`, `set_initial_speed_min`, `set_initial_speed_max`, `set_spawn_profile_2d` | Core motion and bounce profile tuning |
| Side-view camera plane | `set_side_view_plane_z`, `set_side_view_center`, `set_side_view_zoom`, `set_side_view_camera` | 2D side-view framing and plane control |
| Mesh/camera controls | `set_ball_mesh_slot`, `set_container_mesh_slot`, `set_fbx_full_render`, `set_camera_move_speed`, `set_camera_look_sensitivity`, `set_camera_pose` | Visual/control adjustments used by runtime demos |
| Tile-world mutation/query | `tile_set`, `tile_place`, `tile_dig`, `tile_fill`, `tile_get` | Chunked tile-world scripting operations |

### Tile Query Resolution Order

`tile_get(x, y)` resolves deterministically as:
1. latest same-frame mutation (`tile_set` / `tile_place` / `tile_dig`)
2. latest same-frame `tile_fill`
3. runtime tile-world lookup
4. empty tile (`0`)

## 4. Excluded (Post-v0.4.5)

These are intentionally **out of scope** for this guide because they belong to patch/newer tracks.

| Built-in / Surface | Reason excluded |
|---|---|
| `contact_any`, `contact_pairs`, `contact_manifolds` | Added in `v0.4.5.1` patch track |
| `touch_any`, `touch_pairs`, `touch_manifolds` | Added in `v0.4.5.1` patch track |
| `set_render_distance` | Added in `v0.4.5.1` patch track |
| `set_adaptive_distance` | Added in `v0.4.5.1` patch track |
| `set_distance_blur` | Added in `v0.4.5.1` patch track |
| `set_msaa` | Added in `v0.4.5.1` patch track |

Reference: `docs/tileline-v0.4.5-roadmap.md` (`Patch Track: v0.4.5.1`).

## 5. Example (v0.4.5-Style)

```tlscript
@export
def showcase_tick(frame: int):
    set_scene_mode("2d")
    set_target_ball_count(8000)
    set_spawn_per_tick(96)
    set_linear_damping(0.015)

    set_side_view_camera(0.0, 0.0, 1.0)

    if frame % 120 == 0:
        tile_place(4, 1, 3)
    if frame % 240 == 0:
        tile_dig(4, 1)
```

## 6. Build and Run (v0.4.5 Baseline)

Run TLApp with project-scoped content:

```bash
cargo run -p runtime --bin tlapp -- \
  --project docs/demos/tlapp/tlapp_project.tlpfile \
  --scene sideview_static
```

Run with explicit script/sprite path:

```bash
cargo run -p runtime --bin tlapp -- \
  --script docs/demos/tlapp/bounce_showcase.tlscript \
  --sprite docs/demos/tlapp/bounce_hud.tlsprite
```

## 7. Troubleshooting and Compatibility Notes

- If a function is recognized in `runtime/src/tlscript_showcase.rs` but listed as excluded here,
  treat it as post-v0.4.5 and keep it out of v0.4.5 release docs.
- If scene behavior appears inconsistent, verify merge precedence (CLI overlay > scene script > runtime state).
- For 2D validation, prefer side-view scenes in `docs/demos/tlapp/` before using 3D showcase presets.

## 8. Related Docs

- `docs/releases/v0.4.5.md`
- `docs/tileline-v0.4.5-roadmap.md`
- `docs/runtime-tlscript-showcase.md`
- `docs/paradoxpe-tlscript-examples.md`
- `docs/runtime-tljoint.md`

## 9. PDF Generation

Generate release PDF from this guide source:

```bash
./scripts/build_tlscript_pdf.sh \
  --source docs/tlscript-v0.4.5-guide.md \
  --output dist/docs/tlscript-v0.4.5-guide.pdf
```
