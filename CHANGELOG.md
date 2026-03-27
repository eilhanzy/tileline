# Changelog

All notable changes to this project are documented in this file.

## [Unreleased]

## [0.4.5.1] - 2026-03-22

2D sprite texture path patch release.

### Added

- `.tlsprite` sprite texture keys:
  - `texture = <path>.png|<path>.svg`
  - `sprite_texture = ...` alias
- Runtime atlas upload path for external `.png/.svg` sprite sources via
  `TlspriteProgram::sprite_texture_bindings()`.
- `scripts/release_v0.4.5.1.sh` release workflow.
- Release notes:
  - `docs/releases/v0.4.5.1.md`
  - `docs/releases/v0.4.5.1-github.md`

### Changed

- `.tlsprite` now infers deterministic external image slot ids (`[64..127]`) when
  `texture_slot` is omitted for image-backed sprites.
- Added parser/runtime test coverage for PNG texture-source inference.

## [0.4.5] - 2026-03-22

2D foundation consolidation and `.tlpfile`-first packaging release.

### Added

- `.tlpfile` scene-dimension model:
  - `[project] default_dimension = 2d|3d`
  - `[scene.*] dimension = 2d|3d`
- Runtime startup/reload integration that applies scene mode from `.tlpfile` bundles.
- Project-root package builder:
  - `scripts/build_tlpfile_pak.sh`
  - numbered shard output with 5GB default cap.
- Release workflow script: `scripts/release_v0.4.5.sh`.
- Release notes:
  - `docs/releases/v0.4.5.md`
  - `docs/releases/v0.4.5-github.md`

### Changed

- `scripts/package_prebeta_pak.sh` now defaults to project mode and delegates to
  `.tlpfile`-root shard packaging; legacy `--src/--out` flow remains supported.
- Demo FBX assets were normalized under `docs/demos/tlapp/` for package-root consistency.

## [0.2.0] - 2026-03-15

Pre-beta packaging and release workflow expansion.

### Added

- Pre-Beta `.pak` packaging support in `runtime`:
  - deterministic pack/list/unpack core API (`runtime/src/pak.rs`)
  - CLI tool example (`runtime/examples/pak_tool.rs`)
  - convenience script (`scripts/package_prebeta_pak.sh`)
  - checksum-verified unpack flow and safe relative path handling
- Release packaging script: `scripts/release_v0.2.0.sh`.
- Release notes document: `docs/releases/v0.2.0.md`.
- Binary distribution artifacts under `dist/v0.2.0/<target-triple>/`:
  - `tlapp`
  - `SHA256SUMS`
  - `build-info.txt`
  - `tileline-v0.2.0-<target-triple>-tlapp.tar.gz`

## [0.1.0] - 2026-03-14

Initial pre-alpha release baseline.

### Added

- Official `runtime` binary target: `tlapp` (`cargo run -p runtime --bin tlapp`).
- Release packaging script: `scripts/release_v0.1.0.sh`.
- Release notes document: `docs/releases/v0.1.0.md`.
- Binary distribution artifacts under `dist/v0.1.0/<target-triple>/`:
  - `tlapp`
  - `SHA256SUMS`
  - `build-info.txt`
  - `tileline-v0.1.0-<target-triple>-tlapp.tar.gz`

### Runtime / Demo Stability

- ParadoxPE contact-guard tuning tightened for heavy ball-count scenarios.
- Post-step scene reconciliation added to reduce one-frame boundary escape visuals.
- TLApp showcase preset tuned for more natural damping/scatter behavior.
