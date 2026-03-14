# Runtime `.tlpfile` Project GUI

This document describes the first general-purpose GUI shell that reads a project from `.tlpfile`.

## Implementation

- `runtime/src/tlpfile.rs`
- `runtime/src/tlpfile_gui.rs`
- `runtime/src/bin/tlproject_gui.rs`
- `docs/demos/tlapp/tlapp_project.tlpfile`

## Manifest Format (`tlpfile_v1`)

Minimal example:

```text
tlpfile_v1
[project]
name = TLApp Showcase Project
scheduler = gms
default_scene = main

[scene.main]
tljoint = main.tljoint
tljoint_scene = main
tlscripts = extra_rules.tlscript
tlsprites = overlay.tlsprite
```

Supported scene keys:

- `tljoint` / `joint`
- `tljoint_scene` / `joint_scene`
- `tlscript` / `script`
- `tlscripts` / `scripts` (CSV)
- `tlsprite` / `sprite`
- `tlsprites` / `sprites` (CSV)

## GUI Usage

```bash
cargo run -p runtime --bin tlproject_gui -- --project docs/demos/tlapp/tlapp_project.tlpfile
```

Optional:

- `--scene <name>`: initial scene
- `--resolution <WxH>`: default `1280x820`
- `--vsync on|off`

## Behavior

- Left panel: scene list + project file explorer + quick scene/file creation actions.
- Center panel: merged compile summary (scripts/sprites/joint) + runtime-linked lightweight scene view.
- Right panel: parse + compile diagnostics.
- Top bar: icon buttons for `Reload`, `Compile`, `Start`, `Pause`, `Stop`, scheduler apply, and TLApp `Run/Stop`.
- Bottom panel: mini `.tlpfile` text editor (`Load`, `Save`, `Parse`, `Save+Compile`).

## Scene View (Light Mode)

- A lightweight animated scene preview is rendered directly in the GUI.
- Preview compile path enforces `scheduler = gms` (MGS scenes are shown as compile-blocked in this viewer).
- `Light Mode` toggle keeps preview simulation cheap for editor responsiveness.
- Playback controls:
  - `Start` runs preview simulation
  - `Pause` freezes simulation state
  - `Stop` resets and re-seeds preview bodies

## Transform Tool (Pre-Beta)

- Editor has a transform lane with:
  - `Coordinate Space`: `world` / `local`
  - `Move Δ`: X/Y/Z deltas
  - `Rotate Δ`: yaw/pitch deltas
  - one-click nudge buttons (`±X`, `±Y`, `±Yaw`)
- `Apply Preview` updates the embedded scene-view tool marker.
- `Append .tlscript` writes a transform snippet into selected `.tlscript`:

```text
set_coordinate_space("world"|"local")
move_camera(dx, dy, dz)
rotate_camera(yaw_deg, pitch_deg)
```

The compiler path merges:

1. `.tljoint` scene bundle (optional),
2. direct `.tlscript` paths from `.tlpfile`,
3. direct `.tlsprite` paths from `.tlpfile`.

Relative `fbx = ...` hints inside `.tlsprite` are resolved against sprite file directory and
project directory roots.
