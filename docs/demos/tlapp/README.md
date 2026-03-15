# TLApp Demo Assets

- `bounce_showcase.tlscript`: runtime control logic (spawn, damping, camera, mesh-slot mapping)
- `bounce_showcase_mobile_safe.tlscript`: mobile-safe preset tuned for lower-end/UMA class GPUs
- `bounce_hud.tlsprite`: HUD sprites + FBX mesh slot bindings
- `main.tljoint`: primary scene manifest binding script/sprite groups
- `bounce_showcase_mobile_safe.tljoint`: mobile-safe scene manifest
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
