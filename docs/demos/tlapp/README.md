# TLApp Demo Assets

- `bounce_showcase.tlscript`: runtime control logic (spawn, damping, camera, mesh-slot mapping)
- `bounce_hud.tlsprite`: HUD sprites + FBX mesh slot bindings
- `bounce_showcase.tljoint`: scene manifest binding script/sprite groups

Run with:

```bash
cargo run -p runtime --example tlapp -- \
  --script docs/demos/tlapp/bounce_showcase.tlscript \
  --sprite docs/demos/tlapp/bounce_hud.tlsprite
```

Preview `.tljoint` scene composition:

```bash
cargo run -p runtime --example tljoint_runner -- \
  --joint docs/demos/tlapp/bounce_showcase.tljoint \
  --scene main
```
