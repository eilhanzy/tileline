# ParadoxPE Foundation (Handles, World Skeleton, Script ABI)

This document describes the first ParadoxPE foundation layer added to the Tileline workspace.

The current goal is not a finished solver. The goal is to lock down the resource model and the
script/runtime ABI before the full physics pipeline is implemented.

## Relevant Modules

- `paradoxpe/src/handle.rs`
- `paradoxpe/src/body.rs`
- `paradoxpe/src/world.rs`
- `paradoxpe/src/abi.rs`

## Design Goals

- No raw pointers across script/WASM/runtime boundaries
- Stable opaque handles for physics resources
- Deterministic fixed-step scheduling
- Engine-facing host calls that map cleanly into `.tlscript`
- Minimal collision/contact pipeline to validate ABI shape early

## Handle Model

ParadoxPE uses packed 32-bit generational handles.

Encoded fields:

- resource kind (`Body`, `Collider`, `ContactSnapshot`)
- slot index
- generation counter

This keeps handles compatible with:

- `.tlscript` opaque handle semantics
- WASM MVP `i32` transport
- MPS/NPS-friendly low-overhead copying

The generic runtime handle is `PhysicsHandle`, with typed wrappers:

- `BodyHandle`
- `ColliderHandle`
- `ContactHandle`

## World Skeleton

`PhysicsWorld` currently provides:

- body storage with generational invalidation
- collider storage with body attachment
- generic `release_handle(...)`
- fixed-step stepping via `FixedStepClock`
- starter integration path (`gravity`, accumulated force, velocity, damping)
- starter contact rebuild using simple overlap tests
- immutable contact snapshots addressable by handle

The current collision/contact logic is intentionally conservative and exists to validate data flow.
It is not the final ParadoxPE solver architecture.

## `.tlscript` / WASM Host ABI

The first script-facing ABI includes:

- `spawn_body(kind, x, y, mass) -> handle`
- `spawn_collider(body, shape, a, b) -> handle`
- `release_handle(handle)`
- `apply_force(body, x, y)`
- `set_velocity(body, x, y)`
- `query_contacts(body) -> handle`
- `contact_count(contact_snapshot) -> int`
- `step_world(dt) -> int`

This ABI is pointer-free and scalarized so it fits current `.tlscript` + WASM MVP constraints.

## Compiler Integration

`tl-core` defaults now recognize ParadoxPE ABI names in the `.tlscript` pipeline:

- semantic allowlists for safe external calls
- handle acquire/release allowlists
- lowering-time return-type hints
- codegen host import signatures

This means ParadoxPE-oriented scripts can start targeting stable names without custom config in the
common path.

## What Is Still Missing

The following are not implemented yet:

- broadphase acceleration structure
- narrowphase manifold generation
- impulse/contact solver
- constraints/joints
- sleep/island management
- rollback/interpolation hooks
- high-level gameplay physics API

Those layers should now build on top of a fixed handle/ABI surface instead of inventing it later.
