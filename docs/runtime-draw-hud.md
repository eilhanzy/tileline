# Runtime Draw Path + Telemetry HUD

This note documents the runtime-owned path from `SceneFrameInstances` to backend draw batches and
telemetry HUD overlay sprites.

## Implementation

- `runtime/src/draw_path.rs`
- `runtime/src/telemetry_hud.rs`
- `runtime/examples/bounce_tank_showcase.rs`
- `runtime/examples/auto_scene_scheduler.rs`

## Draw Path

`DrawPathCompiler` converts:

- `opaque_3d` -> deterministic opaque batches
- `transparent_3d` -> deterministic transparent batches
- `sprites` -> layer-sorted sprite draw list

Output container:

- `RuntimeDrawFrame`
  - `opaque_batches`
  - `transparent_batches`
  - `sprites`
  - `stats` (`total_draw_calls`, batch counts, instance counts)

The compiler emits backend-friendly instance payloads (`DrawInstance3d`) with model matrix columns
and material parameters.

## Telemetry HUD

`TelemetryHudComposer` appends panel+meter sprites to the same scene sprite list:

- FPS meter
- frame-time budget meter
- physics substep pressure meter
- scene load pressure meter (live bodies + draw calls)

The HUD remains renderer-agnostic and works with both GMS and MGS runtime paths.

## Runtime Status

The runtime examples now execute:

1. scene build (`SceneFrameInstances`)
2. `.tlsprite` hot-reload overlay integration
3. telemetry HUD append
4. draw path compile
5. GMS/MGS workload estimation + bridge submission
