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
default_scene = main

[scene.main]
tljoint = bounce_showcase.tljoint
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

- Left panel: scene selection from `.tlpfile`.
- Center panel: merged compile summary (scripts/sprites/joint) + lightweight scene view.
- Right panel: parse + compile diagnostics.
- Top bar: icon buttons for `Reload`, `Compile`, `Start`, `Pause`, `Stop`.
- Bottom panel: mini `.tlpfile` text editor (`Load`, `Save`, `Parse`, `Save+Compile`).

## Scene View (Light Mode)

- A lightweight animated scene preview is rendered directly in the GUI.
- `Light Mode` toggle keeps preview simulation cheap for editor responsiveness.
- Playback controls:
  - `Start` runs preview simulation
  - `Pause` freezes simulation state
  - `Stop` resets and re-seeds preview bodies

The compiler path merges:

1. `.tljoint` scene bundle (optional),
2. direct `.tlscript` paths from `.tlpfile`,
3. direct `.tlsprite` paths from `.tlpfile`.

Relative `fbx = ...` hints inside `.tlsprite` are resolved against sprite file directory and
project directory roots.
