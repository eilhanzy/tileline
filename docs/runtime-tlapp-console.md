# TLApp In-App CLI Console

This document covers the runtime console available inside TLApp.

## Toggle

- Open/close: `Ctrl + F1`
- Alternate open/close: `F1`
- Submit command: `Enter`
- Edit line: `Backspace`
- Navigate history: `ArrowUp` / `ArrowDown`
- Switch editable box: `Tab`
- Close console: `Esc` or `Ctrl + F1`
- Mouse select: left-click editable boxes

When the console is open, camera/mouse-look input is intentionally paused so command entry is safe.

## Visual Layout

- Full-screen semi-transparent console shell (green terminal style).
- Left panel: built-in command list.
- Right panel: editable runtime boxes (`fps_cap`, `render_distance`, `fsr_sharpness`).
- Bottom panel: live command input line and output stream.
- Error entries blink red; normal entries stay green.

`Tab` cycles the active editable box in this order:

1. `command`
2. `fps_cap`
3. `render_distance`
4. `fsr_sharpness`

Pressing `Enter` applies the active box.

Mouse usage:

1. Click `command` input bar to focus command typing.
2. Click `fps_cap`, `render_distance`, or `fsr_sharpness` rows to switch active box.
3. Click `APPLY ALL BOXES` to apply all 3 quick-box values in one action.

## Command Surface

### Generic

- `help`
- `help <file|gfx|sim|script|cam|log>`
- `status`
- `clear`
- `exit` / `quit`

### Simulation / Scene

- `sim.status`
- `sim.pause`
- `sim.resume`
- `sim.step <n>`
- `sim.reset`
- `scene.reload`
- `script.reload`
- `sprite.reload`
- `perf.snapshot`

### Physics / Camera / Logs

- `phys.gravity <x y z>`
- `phys.substeps <auto|n>`
- `cam.speed <v>`
- `cam.sens <v>`
- `cam.reset`
- `log.clear`
- `log.tail <off|n>`
- `log.level <all|info|error>`

### File I/O (`file.*`)

- `file.exists <path>`
- `file.head <path> [lines]`
- `file.tail <path> [lines] [max_bytes]`
- `file.find <path> <pattern> [max_matches] [max_bytes]`
- `file.findi <path> <pattern> [max_matches] [max_bytes]`
- `file.findr <path> <regex> [max_matches] [max_bytes]`
- `file.grep <path> <pattern> [context] [max_matches] [max_bytes]`
- `file.tailf <path>|stop [poll_ms] [max_lines]`
- `file.watch <path>|stop [poll_ms]`
- `file.read <path> [max_bytes]`
- `file.list <dir> [limit]`

Safety rules:

- Paths are read-only and resolved under TLApp workspace root.
- Existing files/directories outside workspace root are denied.
- `file.read` is preview-limited (default `4096` bytes, max `131072`).
- `file.find` scan window defaults to `131072` bytes (max `524288`).
- `file.tail` tail window defaults to `65536` bytes (max `524288`).
- `file.head` / `file.tail` / `file.list` outputs stay line/entry limited.
- `file.tailf` follows appended lines from current EOF (fail-soft; no panic on file rotation).
- `file.watch` reports size/mtime changes (metadata watch, no content read).
- `file.grep` prints match lines and optional context (`0..8` lines around each match).

### Graphics (`gfx.*`)

- `gfx.status`
- `gfx.vsync <auto|on|off>`
- `gfx.fps_cap <off|N>`
- `gfx.rt <off|auto|on>`
- `gfx.fsr <off|auto|on>`
- `gfx.fsr_quality <native|ultra|quality|balanced|performance>`
- `gfx.fsr_sharpness <0..1>`
- `gfx.fsr_scale <auto|0.5..1>`
- `gfx.render_distance <off|N>`
- `gfx.adaptive_distance <auto|on|off>`
- `gfx.distance_blur <auto|on|off>`

These commands update live runtime state (no restart required).

### `.tlscript` Control (`script.*`)

- `script.var <name> <expr>`: set a reusable variable (`$name`) for CLI script statements
- `script.unset <name>`
- `script.vars`
- `script.call <fn(args)>`: append a persistent TLScript builtin call
- `script.exec <statement>`: append a persistent generic TLScript statement
- `script.uncall <index|all>`
- `script.list`
- `script.clear`

`script.call` and `script.exec` statements are rebuilt into an overlay and merged after normal
scene script evaluation. This means runtime CLI statements can override gameplay-script values
without editing source files.

## Examples

```text
gfx.vsync off
gfx.fps_cap 60
gfx.fsr auto
gfx.fsr_quality balanced
gfx.render_distance 72
```

```text
sim.pause
sim.step 3
perf.snapshot
```

```text
file.exists docs/runtime-tlapp-console.md
file.head docs/runtime-tlapp-console.md 12
file.tail runtime/src/tlapp_app.rs 20
file.find runtime/src/tlapp_app.rs gfx.profile 10
file.findi runtime/src/tlapp_app.rs GFX.PROFILE 10
file.findr runtime/src/tlapp_app.rs \"file\\.(find|tail)\"
file.grep runtime/src/tlapp_app.rs file.watch 2 12
file.tailf logs/tlapp.log 220 24
file.watch runtime/src/tlapp_app.rs 400
file.list docs 20
file.read README.md 2048
```

```text
script.var gravity_y -12.5
script.call set_gravity(0.0, $gravity_y, 0.0)
script.call set_spawn_per_tick(96)
script.call set_bounce(0.82, 0.78)
script.list
```

## Notes

- CLI script overlay strips transient camera-delta commands from persistence to avoid accidental
  continuous drift.
- Any statement compile/eval warnings are reported in the console output (`[tlapp console] ...`).
- Unknown variables in `$var` expansion are fail-soft errors for that command, not hard panics.
