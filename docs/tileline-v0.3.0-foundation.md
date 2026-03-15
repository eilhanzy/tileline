# Tileline v0.3.0 Foundation Track

Date: 2026-03-15

This track moves scope from `v0.2.x` polish into `v0.3.0` feature foundations:

- runtime FSR policy surface (mode/quality/sharpness + fail-soft behavior)
- decentralized NPS snapshot topology primitives for multiplayer scaling

## 1) Runtime FSR Foundation

Implemented in `runtime/src/upscaler.rs` and integrated into TLApp runtime:

- `FsrMode`: `Off | Auto | On`
- `FsrQualityPreset`: `Native | UltraQuality | Quality | Balanced | Performance`
- `FsrConfig` and resolved `FsrStatus` (active/fallback reason/render scale/sharpness)
- backend fail-soft resolution for unsupported backends

TLApp CLI additions:

- `--fsr off|auto|on`
- `--fsr-quality native|ultra|quality|balanced|performance`
- `--fsr-sharpness <0..1>`
- `--fsr-scale <0.5..1|auto>`

Runtime telemetry/title now includes FSR mode/active/scale/sharpness.

## 2) NPS Decentralized Topology Foundation

Implemented in `nps/src/model.rs` and wired into `runtime/src/network_transport.rs`:

- `NetworkTopology`: `ClientServer | ListenHost | PeerMesh`
- `MeshFanoutConfig` for bounded direct peer fanout
- deterministic `select_mesh_snapshot_targets(...)` helper
- transport snapshot routing now supports `PeerMesh` fanout instead of forced full broadcast

Additional transport metrics:

- `snapshot_skipped_topology`
- `last_snapshot_ready_peers`
- `last_snapshot_target_peers`

## 3) Next v0.3.0 Steps

1. Connect true FSR render pass path (EASU/RCAS fullscreen pipeline) to renderer targets.
2. Add NPS peer scoring (RTT/loss-aware relay preference) on top of deterministic fanout.
3. Bind topology and FSR profiles into `.tlpfile` scene/project policy.
4. Add Android + Mali/Panthor + Adreno acceptance gates for v0.3.0.

