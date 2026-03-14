# Runtime MAS (Multi Audio Scheduler)

`MAS` is the audio-side scheduler layer for Tileline Alpha.  
It is implemented in `runtime/src/mas.rs` and uses `MPS` worker lanes for asynchronous audio jobs.

## Why MAS Uses MPS

- Audio mixing is CPU-heavy under high voice counts.
- We already have topology-aware scheduling (`P/E` core routing) in MPS.
- MAS reuses that pipeline instead of creating a second thread scheduler.

## Current API (Alpha A1)

- `MultiAudioScheduler`
- `MasConfig`
- `MasPriority`
- `MasCoreAffinity`
- `AudioBufferBlock`
- `MasMetrics`

## Flow

1. `submit_mix_block(...)` allocates a silent interleaved block.
2. The block is mixed in an MPS worker closure.
3. Successful jobs push the mixed block into a lock-free ready queue.
4. Runtime/audio backend drains ready blocks via `drain_ready_blocks(...)`.

## Safety / Failure Handling

- User mix callback panic is trapped via `catch_unwind`.
- Panic increments MAS failure counters (no hard crash path).
- Ready queue uses a soft capacity cap:
  - if full, block is dropped and `dropped_ready_blocks` increments.

## Telemetry

`MasMetrics` provides:

- submitted/completed/failed job counters
- dropped-ready counter
- current ready queue depth
- snapshot of underlying `mps::SchedulerMetrics`

This allows one HUD to expose both physics/network/render and audio scheduler health.

