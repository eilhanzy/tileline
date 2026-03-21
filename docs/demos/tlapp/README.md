# TLApp Demo Assets

- `bounce_showcase.tlscript`: runtime control logic (spawn, damping, camera, mesh-slot mapping)
- `bounce_showcase_mobile_safe.tlscript`: mobile-safe preset tuned for lower-end/UMA class GPUs
- `sideview_static_map.tlscript`: small static side-view map validation script
- `sideview_chunk_mutation.tlscript`: medium chunked map with dig/place mutation loop
- `sideview_stress_map.tlscript`: stress side-view map script for heavy tile/actor visibility
- `bounce_hud.tlsprite`: HUD sprites + FBX mesh slot bindings
- `main.tljoint`: primary scene manifest binding script/sprite groups
- `bounce_showcase_mobile_safe.tljoint`: mobile-safe scene manifest
- `sideview_static_map.tljoint`: static side-view 2D validation scene
- `sideview_chunk_mutation.tljoint`: chunk mutation (dig/place) side-view 2D validation scene
- `sideview_stress_map.tljoint`: stress side-view 2D validation scene
- `tlapp_project.tlpfile`: project-level manifest that unifies scene bindings for GUI/tools

Run with:

```bash
cargo run -p runtime --example tlapp -- \
  --script docs/demos/tlapp/bounce_showcase.tlscript \
  --sprite docs/demos/tlapp/bounce_hud.tlsprite
```

Run TLApp directly from `.tljoint`:

```bash
cargo run -p runtime --bin tlapp -- \
  --joint docs/demos/tlapp/main.tljoint \
  --scene main
```

Run mobile-safe preset:

```bash
cargo run -p runtime --bin tlapp -- \
  --joint docs/demos/tlapp/bounce_showcase_mobile_safe.tljoint \
  --scene main
```

Run TLApp directly from `.tlpfile`:

```bash
cargo run -p runtime --bin tlapp -- \
  --project docs/demos/tlapp/tlapp_project.tlpfile \
  --scene main
```

Run mobile-safe scene from `.tlpfile`:

```bash
cargo run -p runtime --bin tlapp -- \
  --project docs/demos/tlapp/tlapp_project.tlpfile \
  --scene mobile_safe
```

Run side-view static validation scene from `.tlpfile`:

```bash
cargo run -p runtime --bin tlapp -- \
  --project docs/demos/tlapp/tlapp_project.tlpfile \
  --scene sideview_static
```

Run side-view chunk mutation validation scene from `.tlpfile`:

```bash
cargo run -p runtime --bin tlapp -- \
  --project docs/demos/tlapp/tlapp_project.tlpfile \
  --scene sideview_chunk_mutation
```

Run side-view stress validation scene from `.tlpfile`:

```bash
cargo run -p runtime --bin tlapp -- \
  --project docs/demos/tlapp/tlapp_project.tlpfile \
  --scene sideview_stress
```

Preview `.tljoint` scene composition:

```bash
cargo run -p runtime --example tljoint_runner -- \
  --joint docs/demos/tlapp/main.tljoint \
  --scene main
```

Open project GUI from `.tlpfile`:

```bash
cargo run -p runtime --bin tlproject_gui -- \
  --project docs/demos/tlapp/tlapp_project.tlpfile
```
