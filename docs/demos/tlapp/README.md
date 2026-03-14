# TLApp Demo Assets

- `bounce_showcase.tlscript`: runtime control logic (spawn, damping, camera, mesh-slot mapping)
- `bounce_hud.tlsprite`: HUD sprites + FBX mesh slot bindings

Run with:

```bash
cargo run -p runtime --example tlapp -- \
  --script docs/demos/tlapp/bounce_showcase.tlscript \
  --sprite docs/demos/tlapp/bounce_hud.tlsprite
```
