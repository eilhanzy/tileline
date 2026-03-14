# Runtime `.tljoint` Multi-Scene Binding

`.tljoint` is a small scene manifest format for composing multiple `.tlscript` and `.tlsprite`
files per scene.

## Implementation

- `runtime/src/tljoint.rs`
- `runtime/examples/tljoint_runner.rs`
- demo manifests:
  - `docs/demos/tlapp/bounce_showcase.tljoint`
  - `runtime/examples/assets/bounce_showcase.tljoint`

## Format

```text
tljoint_v1

[scene.main]
tlscripts = bounce_showcase.tlscript, rules_extra.tlscript
tlsprites = bounce_hud.tlsprite, overlay.tlsprite
```

Accepted aliases:

- `tlscript` / `script`
- `tlscripts` / `scripts`
- `tlsprite` / `sprite`
- `tlsprites` / `sprites`

## Runtime Behavior

1. Parse `.tljoint` with `parse_tljoint(...)` or `load_tljoint(...)`.
2. Compile selected scene with `compile_tljoint_scene_from_path(...)`.
3. Evaluate frame via `TljointSceneBundle::evaluate_frame(...)`.

Frame merge policy (deterministic):

- scripts execute in declaration order
- runtime patch fields are last-writer-wins
- camera overrides are last-writer-wins
- `camera_reset_pose` is OR-merged
- `aborted_early` is OR-merged
- warnings are accumulated with script index prefix

Sprite merge policy:

- all compiled `.tlsprite` programs are merged by append order using
  `TlspriteProgram::merge_programs(...)`

## Example

```bash
cargo run -p runtime --example tljoint_runner -- \
  --joint docs/demos/tlapp/bounce_showcase.tljoint \
  --scene main \
  --frames 16
```

