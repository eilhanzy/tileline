# ParadoxPE Foundation (Handles, SoA Storage, Broadphase, Solver, Script ABI)

This document describes the current ParadoxPE foundation layer in the Tileline workspace.

The goal is still not a finished production solver. The current target is a stable, data-oriented
physics pipeline that already covers the hot path shape:

- packed generational handles
- SoA body storage
- allocation-free hot stepping after capacity sync
- parallel broadphase candidate generation
- narrowphase manifold generation
- sequential impulse contact solving
- starter joint and sleep/island systems
- pointer-free `.tlscript` / WASM host ABI

## Relevant Modules

- `paradoxpe/src/handle.rs`
- `paradoxpe/src/body.rs`
- `paradoxpe/src/storage.rs`
- `paradoxpe/src/broadphase.rs`
- `paradoxpe/src/narrowphase.rs`
- `paradoxpe/src/solver.rs`
- `paradoxpe/src/joint.rs`
- `paradoxpe/src/sleep.rs`
- `paradoxpe/src/world.rs`
- `paradoxpe/src/abi.rs`

## Design Goals

- No raw pointers across script/WASM/runtime boundaries
- Stable opaque handles for physics resources
- Deterministic fixed-step scheduling
- Engine-facing host calls that map cleanly into `.tlscript`
- Cache-friendly dense body storage
- No runtime allocations in the `step_world` hot loop once buffers are synchronized
- A shape that can be consumed safely by `.tlscript @parallel(domain="bodies")`

## Handle Model

ParadoxPE uses packed 32-bit generational handles.

Encoded fields:

- resource kind (`Body`, `Collider`, `ContactSnapshot`, `Joint`)
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
- `JointHandle`

## SoA Body Storage

`BodyRegistry` stores hot body data in separate dense vectors instead of AoS records.

Current dense arrays include:

- positions
- rotations
- linear velocities
- accumulated forces
- inverse masses
- damping
- sleeping flags and sleep timers
- broadphase AABBs

Sparse handle metadata maps stable generational handles to dense indices. Removal uses
swap-remove semantics so the hot arrays stay contiguous. The registry updates sparse metadata
accordingly, which keeps:

- iteration cache-friendly
- hot loops dense
- handle validation stable

Two access views are already prepared for future `.tlscript` parallel domains:

- read-oriented body domain access
- write-oriented velocity domain access

This is the key prerequisite for `@parallel(domain="bodies")` style script execution without
coarse locks.

## World Pipeline

`PhysicsWorld` currently provides:

- body/collider/joint storage with generational invalidation
- collider storage with body attachment
- distance-joint storage
- generic `release_handle(...)`
- fixed-step stepping via `FixedStepClock`
- SoA integration path (`gravity`, accumulated force, velocity, damping)
- parallel broadphase rebuild
- narrowphase manifold rebuild
- contact solver pass
- joint solver pass
- sleep/island update pass
- immutable contact snapshots addressable by handle

The current fixed-step pass order is:

1. integrate active bodies
2. rebuild broadphase candidate pairs
3. rebuild narrowphase manifolds
4. solve contact impulses
5. solve joints
6. rebuild immutable contact snapshots
7. update sleep/island state

The architecture is still conservative, but it is no longer just a placeholder skeleton. It now
exercises the actual physics hot path end-to-end.

## Broadphase

ParadoxPE currently uses a shard-parallel sweep-and-prune broadphase.

Properties:

- pair buffers are reused
- hot-path pair generation avoids per-step allocation
- the output is a dense `Vec<(BodyHandle, BodyHandle)>` style candidate buffer
- the implementation is designed to scale across Tileline's CPU task execution model

This is the first stage that prepares ParadoxPE for real MPS-backed CPU parallelism.

## Narrowphase and Contact Solver

The narrowphase currently supports conservative manifold generation for the starter collider set.

The solver currently includes:

- positional correction
- normal impulse solving
- tangent/friction impulse solving
- pair-based warm-start caching

This is still a first-pass solver. It is good enough to validate world stepping, contact snapshot
rebuild, and data flow into higher layers.

## Joints and Sleep/Islands

ParadoxPE now includes two more foundation systems that sit after contact solving:

- distance joint solving
- sleep/island management

The distance-joint path currently provides a first constraint type that validates:

- handle-based joint lifetime
- constraint iteration ordering
- joint participation in world stepping

The sleep manager groups dynamic bodies into islands and can transition sufficiently calm islands
into sleeping state. This reduces wasted work in the fixed-step loop and prepares the system for
larger sandbox scenes.

The current sleep logic is intentionally simple, but the architecture is already reusable and
hot-loop friendly.

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

## Performance Notes

Current implementation constraints:

- capacity synchronization happens outside the hot loop
- body/joint/contact buffers are reused
- hot stepping avoids fresh allocation once the world has been synchronized for its current load
- broadphase and body storage both favor cache-line-friendly dense traversal

This is the baseline needed before pushing ParadoxPE harder through MPS and `.tlscript` parallel
contracts.

## What Is Still Missing

The following are not implemented yet:

- manifold persistence / contact IDs
- richer joint set beyond distance constraints
- stronger island scheduling heuristics
- sleep heuristics refinement and wake propagation tuning
- rollback/interpolation hooks
- broadphase variants beyond the current parallel SAP path
- ECS/runtime-facing higher-level gameplay physics API

Those layers now build on top of a real data-oriented physics path instead of a placeholder handle
surface.
