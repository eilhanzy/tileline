# Tileline Alpha Plan

This document defines the first Alpha milestone scope after `v0.2.0`.

## Alpha Goals

1. Add a lightweight `.tlsprite` editor path with list-mode inspection/editing.
2. Introduce MAS (Multi Audio Scheduler) as an MPS-integrated audio task layer.
3. Keep runtime architecture `src/`-first (examples remain thin wrappers).
4. Establish a cohesive Alpha visual style (dark lavender base theme).

## Scope (Alpha A1)

### 1) `.tlsprite` List Editor

- Runtime module: `runtime/src/tlsprite_editor.rs`
- Initial frontend: `runtime/examples/tlsprite_list_editor.rs`
- Behavior:
  - parse `.tlsprite` source through `runtime::compile_tlsprite(...)`
  - render deterministic list rows (`section`, `kind`, `slot`, `layer`, transforms, optional `fbx`)
  - expose diagnostics without crashing (`Warning` / `Error`)
  - provide markdown table output for tooling/docs
- Theme preset:
  - `TlspriteEditorTheme::DarkLavender`
  - palette tokens are centralized for future egui/wgpu editor UI

### 2) MAS (Multi Audio Scheduler)

- Runtime module: `runtime/src/mas.rs`
- Core contract:
  - schedule audio mix jobs through `mps::MpsScheduler`
  - keep render thread non-blocking
  - use lock-free ready queue for mixed audio blocks
  - expose soft-fail telemetry (`submitted/completed/failed/dropped`)
- Initial API:
  - `MultiAudioScheduler`
  - `submit_mix_block(...)`
  - `drain_ready_blocks(...)`
  - `MasMetrics`

### 3) Runtime Integration Rules

- MAS runs adjacent to runtime loop (same process), but remains modular.
- `.tlsprite` editor logic lives in `runtime/src` and is reusable by:
  - CLI tooling
  - upcoming Alpha UI shell
  - CI validation scripts

## Deferred to Alpha A2

1. Full interactive editor UI (egui panels, inline property editing, save/apply loop).
2. Audio backend adapter (CPAL/miniaudio) with device hotplug and stream buffering.
3. `.tlsprite` editor icon toolbar and in-app font loading.
4. `.tlsprite` schema-assisted auto-complete.

## Exit Criteria (Alpha A1)

1. `cargo check -p runtime --bin tlapp` passes.
2. `cargo run -p runtime --example tlsprite_list_editor -- --file <path>` works with diagnostics.
3. MAS can submit/drain mix blocks and reports metrics.
4. Documentation updated in:
   - `docs/README.md`
   - `README.md`
   - subsystem notes (`docs/runtime-tlsprite.md`, `docs/runtime-mas.md`)
