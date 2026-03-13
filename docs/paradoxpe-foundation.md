# ParadoxPE Foundation (Handles, SoA Storage, Broadphase, Solver, Snapshot ABI)

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
- snapshot capture / restore / interpolation buffering
- NPS-friendly quantized snapshot export path
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
- `paradoxpe/src/snapshot.rs`
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

`BodyRegistry` also exposes `active_parallel_body_count()` so runtime/MPS planners can scale script
chunking to active simulation load instead of total handle count.

## World Pipeline

`PhysicsWorld` currently provides:

- body/collider/joint storage with generational invalidation
- collider storage with body attachment
- collider materials with friction/restitution combine rules
- distance-joint storage
- fixed-joint storage
- generic `release_handle(...)`
- fixed-step stepping via `FixedStepClock`
- SoA integration path (`gravity`, accumulated force, velocity, damping)
- parallel broadphase rebuild
- narrowphase manifold rebuild
- contact solver pass
- joint solver pass
- sleep/island update pass
- immutable contact snapshots addressable by handle
- full-world snapshot capture / restore
- interpolation buffer maintenance for render/network smoothing

The current fixed-step pass order is:

1. integrate active bodies
2. rebuild broadphase candidate pairs
3. rebuild narrowphase manifolds
4. solve contact impulses
5. solve joints
6. rebuild immutable contact snapshots
7. update sleep/island state
8. push world snapshot into the interpolation ring

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

Each manifold now carries:

- a stable `ContactId`
- a `persisted_frames` counter

This lets ParadoxPE preserve contact identity across frames and gives the solver/snapshot layer a
stable notion of "the same contact" instead of treating every overlap as a fresh event.

Each collider also carries a `ColliderMaterial` with:

- restitution
- friction
- restitution combine rule
- friction combine rule

The narrowphase resolves coefficients per contact pair before solver execution. This moves the
material policy out of the hot impulse loop and makes the resulting behavior deterministic at the
manifold level.

The solver currently includes:

- positional correction
- normal impulse solving
- tangent/friction impulse solving
- `ContactId`-based warm-start caching

This is still a first-pass solver. It is good enough to validate world stepping, contact snapshot
rebuild, and data flow into higher layers.

## Joints and Sleep/Islands

ParadoxPE now includes two more foundation systems that sit after contact solving:

- distance joint solving
- fixed joint solving
- sleep/island management

The current joint path validates:

- handle-based joint lifetime
- constraint iteration ordering
- joint participation in world stepping
- preserved body offsets for fixed-body pair locking

The sleep manager groups dynamic bodies into islands and can transition sufficiently calm islands
into sleeping state. This reduces wasted work in the fixed-step loop and prepares the system for
larger sandbox scenes.

The current sleep logic is intentionally simple, but the architecture is already reusable and
hot-loop friendly.

## Snapshot, Rollback, and Interpolation Base

ParadoxPE now includes a first snapshot layer in `paradoxpe/src/snapshot.rs`.

Current pieces:

- `PhysicsSnapshot`: dense frame-state capture for active bodies
- `PhysicsWorld::capture_snapshot()`
- `PhysicsWorld::restore_snapshot(...)`
- `PhysicsInterpolationBuffer`: bounded ring of recent snapshots
- `PhysicsWorld::interpolate_body_pose(...)`

This is not yet a full rollback netcode implementation, but it establishes the data shape needed
for:

- deterministic state rewinds
- render-time pose interpolation
- NPS snapshot packet generation

The world now pushes snapshots into the interpolation ring after each fixed-step update.

## NPS Snapshot Bridge

`nps::NetworkPacketManager` can now consume a `PhysicsSnapshot` directly and queue it as a
quantized transform batch.

The current export path:

- keeps handle transport in packed `u32` form
- quantizes positions to Tileline's grid-oriented packet format
- normalizes velocity before packet encoding
- reuses the existing MPS-backed outbound encode path

This gives ParadoxPE a direct bridge into NPS without introducing a second physics-specific
network packet representation.

## `.tlscript` Parallel Runtime Alignment

The `.tlscript` runtime planner now recognizes `domain="bodies"` as a first-class IR/runtime
domain instead of a generic parallel flag.

Current behavior:

- `schedule="auto"` is upgraded to performance-core preference for body-domain work
- body-domain chunk planning uses `PhysicsWorld::active_parallel_body_count()`
- unsupported body-domain write sets fall back to serial with explicit planner telemetry

This keeps the current script/runtime safety surface aligned with ParadoxPE's actual SoA access
model instead of relying on generic heuristics.

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
- snapshot interpolation uses a bounded ring buffer instead of per-frame snapshot churn

This is the baseline needed before pushing ParadoxPE harder through MPS and `.tlscript` parallel
contracts.

## What Is Still Missing

The following are not implemented yet:

- richer joint set beyond distance/fixed constraints
- stronger island scheduling heuristics
- sleep heuristics refinement and wake propagation tuning
- solver quality upgrades for larger stack/contact scenarios
- full rollback reconciliation and resimulation policy
- broadphase variants beyond the current parallel SAP path
- ECS/runtime-facing higher-level gameplay physics API

Those layers now build on top of a real data-oriented physics path instead of a placeholder handle
surface.
