# NPS Runtime Plan (Channels, Ticks, Snapshots)

This document defines the concrete runtime plan for the next NPS milestone.

The purpose is to freeze the transport model before building the real UDP runtime loop in
`runtime/`.

## Scope

This plan covers:

- channel classes
- tick ownership
- packet cadence
- snapshot strategy
- MPS offload points
- ParadoxPE integration boundaries

It does not yet define encryption, matchmaking, or session/authentication flow.

## Canonical Channel Model

NPS should use a small fixed set of traffic lanes.

### 1. `PhysicsState`

Use for:

- quantized transform batches
- authoritative ParadoxPE snapshots
- future state deltas

Transport semantics:

- unreliable
- sequenced
- simulation tick bound

Reason:

- stale physics packets are less useful than fresh ones
- head-of-line blocking is unacceptable for the beta sandbox target

### 2. `PlayerInput`

Use for:

- local player input frames
- client prediction input stream

Transport semantics:

- unreliable
- sequenced
- simulation tick bound

Reason:

- input should arrive quickly
- later input supersedes earlier input
- resends are less useful than keeping cadence tight

### 3. `LifecycleEvent`

Use for:

- spawn/despawn
- authority transfer
- scripted state transitions with persistent meaning

Transport semantics:

- reliable
- ordered
- control tick bound

Reason:

- these events change long-lived game state
- dropping or reordering them is not acceptable

### 4. `UiSync`

Use for:

- low-frequency UI/tooling/chat style traffic

Transport semantics:

- reliable
- ordered

### 5. `ScriptSync`

Use for:

- future `.tlscript @net(...)` state replication not covered by the default physics/input channels

Transport semantics:

- reliable by default
- can later become metadata-driven if `.tlscript` explicitly marks it unsafe for reliability

## Tick Model

Beta should keep the model narrow and explicit.

### Simulation Tick

This is the authoritative fixed-step tick.

Use it for:

- ParadoxPE stepping
- input packets
- physics snapshot packets

Rules:

- every `InputFrame` carries one simulation tick
- every `TransformBatch` / physics snapshot is associated with one simulation tick
- authoritative rewinds and interpolation anchor to this tick

### Control Tick

This is a looser logical timeline for lifecycle changes.

Use it for:

- spawn/despawn
- authority handoff
- scripted lifecycle events

Rules:

- does not need to advance at the same cadence as simulation
- still uses monotonic packet sequencing

### Heartbeat Tick

This exists for:

- ack-only packets
- keepalive cadence

Rules:

- should never become a second gameplay timeline
- transport-only concern

## Snapshot Strategy

Beta should use a two-layer snapshot plan.

### Layer 1: Full Quantized Snapshot Batch

This already exists in basic form through `TransformBatch`.

Use it for:

- initial snapshot path
- correction/resync
- low-complexity beta fallback

Properties:

- quantized positions on Tileline grid
- normalized velocity transport
- handle/object ids stay packed integer values

### Layer 2: Delta-On-Change Snapshot Path

This should come next, but only after the runtime loop is stable.

Use it for:

- bandwidth reduction under larger scenes
- `@net(sync=\"on_change\")` integration

Rules:

- full snapshot/keyframe every `N` ticks
- delta packets between keyframes
- dropped delta bursts must be recoverable by later keyframe

For beta one, full quantized snapshots are enough if performance is acceptable.

## ParadoxPE Integration

ParadoxPE is the first-class state source for NPS physics traffic.

Current bridge:

- `PhysicsWorld::capture_snapshot()`
- `PhysicsInterpolationBuffer`
- `NetworkPacketManager::queue_physics_snapshot(...)`

Required next step:

- make runtime own the snapshot emission cadence
- decide whether snapshots are sent every tick or at a controlled divisor

Recommended beta default:

- step physics every fixed tick
- emit physics snapshot packets every fixed tick in sandbox mode
- allow downsampling later for bandwidth-limited targets

## MPS Offload Points

These should remain explicit.

### Encode Side

Safe MPS work:

- transform batch packing
- delta generation
- packet header emission
- reliable retransmit rebuilds

### Decode Side

Safe MPS work:

- header parse
- payload decode
- delta reconstruction
- snapshot unpack/quantization decode

### Main Runtime Thread Should Keep Ownership Of

- socket I/O loop coordination
- session/peer table mutation
- final application of decoded gameplay state

This keeps packet CPU work parallel without turning transport state into a lock-heavy mess.

## `.tlscript` Integration Plan

`.tlscript` already has `@net(...)` metadata extraction. The next stage should bind it to the
runtime transport model, not directly to wire encoding.

Recommended path:

1. `.tlscript` compiler emits sync metadata
2. runtime maps metadata to one of the canonical NPS lanes
3. NPS packet manager encodes according to lane semantics

This preserves one network policy surface instead of letting scripts invent wire behavior.

## Telemetry Required For Beta

NPS should surface at least:

- packets sent/received by lane
- RTT
- jitter
- packet loss estimate
- retransmit count
- decode/encode task cost
- snapshot bytes per second

Without this, network optimization will be guesswork.

## Recommended Immediate Sequence

1. Add runtime-owned `tokio::UdpSocket` transport loop.
2. Route inbound/outbound datagrams through `NetworkPacketManager`.
3. Emit ParadoxPE snapshots on the canonical simulation tick.
4. Add lane-level metrics and loss/jitter reporting.
5. Only then add delta compression and more advanced sync policies.

## Beta Non-Goals

These should not block the first NPS beta milestone:

- encryption/auth handshake
- NAT traversal
- voice/chat stack
- generalized replication system for every engine subsystem
- rollback netcode polish beyond what ParadoxPE and snapshots immediately need
