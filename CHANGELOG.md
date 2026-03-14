# Changelog

All notable changes to this project are documented in this file.

## [Unreleased]

### Added

- Pre-Beta `.pak` packaging support in `runtime`:
  - deterministic pack/list/unpack core API (`runtime/src/pak.rs`)
  - CLI tool example (`runtime/examples/pak_tool.rs`)
  - convenience script (`scripts/package_prebeta_pak.sh`)
  - checksum-verified unpack flow and safe relative path handling

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
