# NPS Protocol and MPS Integration

This document describes the first `NPS` (Network Packet Scaler) implementation for Tileline.

Scope of this version:

- low-level UDP packet header + bit-packed payloads
- quantized physics/input payload encoding (bandwidth-oriented)
- lightweight reliability over UDP for lifecycle events
- deterministic physics-object authority handoff (physgun-style)
- MPS-integrated async encode/decode task flow

## Goals

- Keep packet processing off the render/GMS hot path.
- Use dense packet formats (bit packing + quantization) for sandbox workloads.
- Preserve deterministic ownership semantics for authoritative physics interactions.
- Maintain a conservative, engine-safe path that can evolve into a full transport runtime.

## Crate Layout

`nps/src/`

- `bitpack.rs`: zero-allocation bit writer/reader over caller-provided buffers
- `model.rs`: canonical channel/tick/snapshot semantics used for runtime planning
- `packet.rs`: header layout, payload codecs, quantization helpers
- `reliability.rs`: ACK window, retransmit bookkeeping, authority table
- `manager.rs`: `NetworkPacketManager` + MPS offload integration

## Wire Format (V1)

### Header

`PacketHeader` is a fixed-size, byte-aligned header (`20` bytes) for fast validation and ACK handling.

Core fields:

- `magic`, `version`
- `channel` (`Physics`, `Input`, `Lifecycle`, `Ui`, `Script`)
- `kind` (`TransformBatch`, `InputFrame`, `LifecycleEvent`, `AuthorityTransfer`, `AckOnly`)
- `sequence`, `ack`, `ack_bits`
- `tick`
- `payload_bits`

Implementation: `nps/src/packet.rs`

## Bit-Packed Payloads and Quantization

### Zero-copy-friendly packing

`BitWriter` writes into a caller-owned mutable byte slice.
`BitReader` borrows the datagram payload bytes directly and decodes bit fields without allocation.

This keeps encode/decode hot paths cache-friendly and avoids `String`/buffer churn.

Implementation: `nps/src/bitpack.rs`

### Physics transforms (`TransformBatch`)

Transform snapshots are optimized for network size:

- positions are quantized to `u16` on a normalized grid (default `2048x2048`)
- signed normalized velocity components are quantized to `u16`
- object ids and owner peer ids remain integer ids/handles

`GridQuantization` provides encode/decode helpers.

Implementation: `nps/src/packet.rs`

### Input frames

`InputFrame` uses compact fields for deterministic tick-based simulation:

- `tick`, `player_peer`
- `buttons` bitmask
- quantized movement axes
- quantized aim coordinates

Implementation: `nps/src/packet.rs`

## Reliability Layer (UDP)

`NPS` keeps physics/input traffic fast and mostly unreliable while adding a lightweight reliability path
for long-lived gameplay/lifecycle events.

### ACK window

`AckWindow` stores:

- latest received sequence (`ack`)
- previous 32 packets in `ack_bits`

This supports compact retransmit and ack pruning without TCP-style head-of-line blocking.

### Per-peer reliability state

`PeerReliabilityState` handles:

- outbound sequence allocation
- inbound ACK observation
- reliable in-flight queue
- retransmit timeout scanning (non-blocking collection)

Implementation: `nps/src/reliability.rs`

## Deterministic Ownership / Physgun Authority Model

`PhysAuthorityTable` implements a master/slave ownership model for physics objects.

Behavior:

- each object has a baseline `master_owner`
- `begin_physgun_grab` transfers temporary authority to grabbing peer
- `end_physgun_grab` restores authority to the master owner

This models GMOD-style "physgun" authority transfer while staying explicit and deterministic.

Implementation: `nps/src/reliability.rs`

## MPS Integration (`NetworkPacketManager`)

`NetworkPacketManager` is the engine-facing packet manager for NPS V1.

### Responsibilities

- queue inbound UDP datagrams
- submit decode jobs to `mps::MpsScheduler` (or run inline fallback)
- queue outbound payload jobs and submit encode jobs to MPS
- maintain peer reliability state and ACK/retransmit bookkeeping
- expose lock-free decoded/encoded result queues to runtime transport code
- maintain authority handoff helpers (`begin_physgun_grab`, `end_physgun_grab`)

### Design notes

- encode/decode jobs are CPU tasks and never require blocking the render loop
- transport I/O is intentionally external (runtime-specific `tokio::UdpSocket` or platform backend)
- manager is safe to use without MPS; it falls back to inline processing for tests/tools

Implementation: `nps/src/manager.rs`

## Reliability Policy Mapping (Current)

Default policy mapping:

- `Physics`, `Input` -> `UnreliableSequenced`
- `Lifecycle` -> `ReliableOrdered`
- `Ui`, `Script` -> `ReliableOrdered` (conservative default)

This can be made script-driven later via `.tlscript` `@net(...)` metadata.

## Canonical Tick and Channel Model

NPS now carries an explicit semantic classification layer in `nps/src/model.rs`.

This avoids scattering transport policy across manager/reliability/runtime code.

Current semantic lanes:

- `PhysicsState`
- `PlayerInput`
- `LifecycleEvent`
- `UiSync`
- `ScriptSync`

Current tick scopes:

- `Simulation`
- `Control`
- `Heartbeat`

Current snapshot modes:

- `QuantizedTransformBatch`
- `InputFrame`
- `LifecycleEvent`
- `None`

This model is already used by `SendPolicy::for_payload(...)` so reliability/sequencing policy now
derives from the same canonical mapping that future runtime transport code will use.

## Link Telemetry

NPS now exposes best-effort per-peer link telemetry derived from reliable ACK flow.

Current fields:

- smoothed RTT estimate
- jitter estimate (EWMA of RTT deltas)
- reliable sent count
- reliable acked count
- retransmit count
- coarse retransmit-based loss estimate

This is intentionally conservative. It is meant to give the runtime a stable diagnostics surface
before more advanced transport estimation is added.

## `.tlscript` `@net(...)` Hook (Compiler Side)

The `.tlscript` frontend now supports `@net(...)` decorators and a dedicated compiler hook that
extracts network sync metadata without bloating the semantic pass.

Examples:

- `@net(sync="on_change")`
- `@net(unreliable)`

See `docs/tlscript-parallel-runtime.md` for the broader compiler-hook ergonomics pattern used by
both `@net(...)` and parallel execution decorators.

## Current Limits / Next Steps

Current limits:

- no UDP socket loop in `nps` (transport remains runtime-owned)
- no delta compression yet (`on_change` metadata is extracted but not encoded into transport flow)
- no encryption/authentication/session handshake yet

Recommended next steps:

1. `runtime` transport loop (`tokio::UdpSocket`) + NPS queue integration
2. channel bandwidth budgeting and delta compression
3. snapshot interpolation/rollback hooks for `ParadoxPE`
4. telemetry surfaces (`rtt`, `loss`, `jitter`, resend ratios)
