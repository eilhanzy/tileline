# Tileline v0.4.5 Roadmap (2D Foundation Track)

This document defines the 2D scope moved from `v0.5.0` into `v0.4.5`.

`v0.4.5` is now the release line for first-class side-view 2D engine foundations.
`v0.5.0` continues with render optimization + independence work on top of this baseline.

## Patch Track: v0.4.5.1 (Strict Gate)

`v0.4.5.1` is a patch-stability release on top of this track with two runtime/script deliveries:

- global 2D/3D contact query surface in `.tlscript`:
  - built-ins: `contact_any`, `contact_pairs`, `contact_manifolds`
  - aliases: `touch_any`, `touch_pairs`, `touch_manifolds`
  - frame vars auto-injected from latest completed physics step
- scripted effect control without adding a new render pass:
  - `set_render_distance(v)`
  - `set_adaptive_distance(mode)`
  - `set_distance_blur(mode)`
  - `set_msaa(samples)`

Merge precedence lock for patch behavior:

1. CLI script overlay
2. scene script (`.tlscript` / multi-script / `.tljoint`)
3. existing runtime state

Strict gate for `v0.4.5.1`:

- unit + integration tests for touch/effect script surface
- benchmark comparison vs `v0.4.5` baseline (FPS/p95/tick jitter)
- 20 minute soak (script reload + scene reload) without panic/segfault/leak trend

## Release Goals

`v0.4.5` should deliver all of the following:

- a first-class orthographic / side-view 2D runtime scene mode
- a chunked tile-world foundation suitable for digging/building loops
- a canonical 2D coordinate + rotation system shared by runtime, script, and physics
- runtime-backed `.tlscript` tile query/write semantics
- a flat-2D ParadoxPE mode for side-view terrain/body collision
- 2D content authoring through `.tlsprite` + `.tlscript` + `.tljoint` without example-only glue
- `.pak` packaging + mount/read path for 2D runtime content
- startup `.ini` boot settings surface for game/runtime defaults
- telemetry for tile/chunk visibility and mutation pressure

Chosen defaults:

- first 2D target: Terraria-like side-view survival sandbox foundation
- world model: chunked tile world
- physics strategy: ParadoxPE flat-2D
- content priority: tilemap -> animation -> VFX
- scope lock: engine foundation + digging/building only

## Non-Goals

These are intentionally out of scope for `v0.4.5`:

- full survival gameplay systems (inventory, crafting, AI, fluids, biome simulation, day-night loop)
- full editor productization
- broad asset import coverage outside runtime path needs
- multiplayer gameplay loop completion

## Definition Of Done

`v0.4.5` should only be considered complete when all of these are true:

- one canonical side-view 2D scene path works in runtime without example-only hacks
- chunked tile worlds render, scroll, and mutate locally through dig/place updates
- 2D coordinate space and rotation semantics are deterministic across render/script/physics
- `.tlscript` tile reads/writes resolve against canonical runtime tile state
- ParadoxPE flat-2D mode is the canonical 2D terrain/body collision path
- 2D camera, animation, and VFX coexist on the same runtime scene path
- runtime can load 2D scenes/assets from `.pak` without bespoke example loaders
- startup `.ini` can define initial graphics/audio/scene settings deterministically
- 2D-heavy scenes remain diagnosable through runtime telemetry

## Workstream F: 2D Foundation

### F1. 2D Runtime Rendering

Target work:

- orthographic/side-view camera mode as a first-class runtime scene mode
- tilemap draw path using atlas-backed quads/batches through canonical renderer
- keep HUD + sprite overlays on same 2D-friendly path
- deterministic layer ordering with parallax-ready structure

Acceptance gates:

- one 2D scene renders tile layers + animated sprites + overlays together
- camera motion remains stable without 3D fallback behavior
- batching stays central (no per-sprite draw-path regression)

### F2. Chunked Tile World

Target work:

- define chunked tile storage as canonical 2D world model
- support dig/place/update mutations
- keep updates local to changed chunks (no full-world rebuild assumption)
- add chunk visibility/update telemetry

Acceptance gates:

- chunked worlds load, scroll, render, and mutate without full rebuilds
- tile updates are visible within bounded frame cost
- telemetry explains chunk pressure in real scenes

### F2.5. Coordinate + Rotation System

Target work:

- lock canonical coordinate conventions (X=right, Y=up, Z=depth lane, world-unit scale, tile origin policy)
- define authoritative transform contract for 2D entities (`position`, `rotation_deg`, `scale`)
- unify rotation pivot semantics (center/pivot offset) across sprite, tile, and actor presentation
- expose script-facing helpers for 2D move/rotate with deterministic clamping and precision rules
- align physics-body orientation mapping with runtime/script transform orientation
- define side-view Z plane semantics (default plane, layer offsets, and clamp behavior)

Acceptance gates:

- the same object transform resolves identically in renderer, script runtime, and ParadoxPE flat-2D
- camera movement + object rotation does not drift or flip under long-running updates
- side-view Z plane and depth layering remain deterministic across runtime reloads
- pivot-origin behavior is documented and stable for sprite/tile composition
- transform serialization/reload round-trips without coordinate or angle corruption

### F3. ParadoxPE Flat-2D

Target work:

- add flat-2D execution mode inside ParadoxPE
- side-view constrained motion/collision semantics
- tile-terrain collision + simple actor shapes first (AABB/capsule-style)
- align broadphase/narrowphase with chunked terrain mutation flow

Acceptance gates:

- actor movement collides correctly with tile terrain
- dig/place updates invalidate only relevant collision regions
- no global collision rebuild for local terrain edits

### F4. 2D Content And Authoring Path

Target work:

- extend `.tlsprite` with 2D atlas/tile/animation metadata
- extend `.tlscript` with 2D camera, tile query, dig/place, and 2D spawn helpers
- keep `.tljoint` + runtime loading as the composition point
- keep 2D additions additive so existing 3D/light bindings stay valid

Acceptance gates:

- 2D scenes are authored without hardcoded engine internals in examples
- script-driven dig/place + animation works through supported runtime APIs
- scripts do not need low-level chunk-internal management

### F5. Validation Scenarios

Representative acceptance scenes:

- small static side-view map
- medium chunked map with repeated dig/place mutations
- stress scene: high visible-tile count + animated actors + HUD overlays

Validation rules:

- side-view chunked tile scene loads and scrolls correctly
- repeated dig/place does not trigger full-world rebuild behavior
- 2D terrain collision remains stable after tile edits
- frame pacing remains diagnosable from telemetry

### F6. Script-Runtime Bridge Hardening

Target work:

- `tile_get` resolves against canonical runtime chunked tile state (without local override)
- same-frame `tile_set`/`tile_fill`/`tile_dig` overlays remain deterministic
- single-script / multi-script / `.tljoint` paths share one tile lookup contract
- `script.call` console semantics align with runtime scene script semantics

Acceptance gates:

- `tile_get` returns live runtime tile value without example-only glue
- local same-frame script mutations still override deterministically
- `.tljoint` compositions remain consistent with single-script behavior

## Workstream G: Packaging + Boot Config

### G1. `.pak` Runtime Content Path

Target work:

- finalize canonical 2D asset packing layout (`scripts`, `sprites`, `tiles`, `textures`, `audio`)
- wire runtime loader to resolve scene content from mounted `.pak` first, then filesystem fallback
- keep path resolution deterministic and diagnostics-friendly

Acceptance gates:

- side-view demo scenes load through `.pak` without custom glue
- missing/corrupt `.pak` entries fail-soft with clear diagnostics
- pack/list/unpack toolchain remains compatible with current runtime `.pak` format

### G2. Startup `.ini` Settings Surface

Target work:

- add a simple `.ini` profile file for initial game/runtime options
- include startup scene/joint, graphics baseline, audio baseline, and input profile toggles
- support optional "open settings panel on boot" flag for first-run flow
- define precedence: CLI > env > `.ini` > built-in defaults

Acceptance gates:

- runtime starts with deterministic settings when `.ini` is present
- invalid `.ini` keys fail-soft and are reported in startup diagnostics
- settings panel auto-open behavior is controllable by `.ini` and does not break headless/CLI flow

## Immediate Execution Plan (Now)

Current order we follow:

1. `F2.5` (coordinate + rotation): lock canonical transform semantics before physics expansion.
2. `F3` (ParadoxPE flat-2D): stabilize terrain/actor collision + chunk-local collision rebuild.
3. `F4` (2D authoring path): complete `.tlsprite` + `.tlscript` 2D-facing API surface.
4. `F5` (validation): run static/medium/stress scenes and lock telemetry baselines.
5. `G1` + `G2` packaging/bootstrap: `.pak` runtime content flow + startup `.ini` settings.

Sprint-level deliverables:

- one canonical transform contract doc + implementation checklist for 2D (`position/rotation/scale`)
- one reproducible flat-2D collision demo with repeated dig/place
- one runtime-loaded 2D scene launched from `.pak`
- one `game.ini` profile controlling initial scene + gfx/audio + settings auto-open behavior
- release checklist entries for `.pak` and `.ini` smoke tests

## Milestones

### Milestone 1: 2D Runtime + Chunk Core

Target contents:

- F1 baseline delivery
- F2 chunk storage + local mutation loop
- first canonical 2D side-view demo path

Exit criteria:

- side-view scene supports tiles + sprites + overlays
- local dig/place updates are visible and stable

### Milestone 2: Physics + Script Bridge

Target contents:

- F2.5 coordinate + rotation contract
- F3 flat-2D ParadoxPE path
- F4 content path wiring
- F6 script/runtime bridge consistency
- G2 startup `.ini` baseline

Exit criteria:

- runtime/script/physics agree on 2D transform + rotation semantics
- terrain/actor collision remains stable after chunk edits
- `.tlscript` tile ops are runtime-backed and deterministic across composition modes
- startup `.ini` settings apply deterministically at launch

### Milestone 3: Validation + Release Hardening

Target contents:

- F5 stress and regression validation
- G1 `.pak` runtime content validation
- telemetry finalization
- docs and release notes prep

Exit criteria:

- all Definition Of Done gates are met
- `.pak` and `.ini` startup flows pass release smoke checks
- `v0.4.5` is ready as the 2D baseline for `v0.5.0`

## Handoff To v0.5.0

After `v0.4.5` closes:

- `v0.5.0` focuses on render stack optimization, effects + textures, MPS/ParadoxPE revision, and
  `rayon`/`bevy`/`wgpu` independence
- only incremental 2D polish continues in `v0.5.x`; foundational 2D gates stay anchored to
  `v0.4.5`
