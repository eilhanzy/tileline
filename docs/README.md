# Tileline Documentation Index

This directory contains design and implementation notes for subsystems that are intended to live
in `src/` crates (not benchmark-only code).

## Documents

- `tlscript-lexer.md`: zero-copy `.tlscript` token/lexer design and constraints
- `tlscript-parser-plan.md`: parser architecture, AST shape, and V1 grammar plan
- `tlscript-semantic.md`: semantic rules, soft diagnostics, handles, and WASM sandbox policy
- `tlscript-parallel-runtime.md`: parallel contracts, advisor suggestions, and runtime dispatch planning
- `tileline-pre-alpha-transition.md`: immediate Foundation -> Pre-Alpha gates, sprint plan, and validation checks
- `tileline-alpha-plan.md`: Alpha A1 scope for `.tlsprite` list editor, MAS scaffold, and theme baseline
- `tileline-pre-beta-fsr-plan.md`: pre-beta milestones with FSR 1.0, adaptive quality, and full input accessibility
- `tileline-beta-roadmap.md`: phased beta roadmap, integration milestones, and deferred scope
- `tileline-v0.4.5-roadmap.md`: v0.4.5 roadmap for first-class 2D side-view foundation (chunked tile world + dig/place + flat-2D ParadoxPE)
- `tileline-v0.5.0-roadmap.md`: v0.5.0 roadmap for render optimization, effects/textures, ParadoxPE + MPS revision, and Rayon/Bevy/WGPU independence
- `tileline-v0.3.0-foundation.md`: v0.3.0 foundation scope for runtime FSR policy and decentralized NPS topology
- `engine-architecture-api-core.md`: architecture/API-core map for crate boundaries, runtime lifecycle, and integration surfaces
- `tlscript-v0.4.5-guide.md`: strict `v0.4.5` TLScript guide used as canonical source for PDF release docs
- `nps-protocol.md`: NPS UDP bit-packing, reliability, authority handoff, and MPS integration
- `nps-runtime-plan.md`: canonical NPS channel, tick, snapshot, and transport runtime plan
- `paradoxpe-foundation.md`: ParadoxPE handles, SoA storage, broadphase/solver pipeline, snapshot base, and script ABI
- `paradoxpe-tlscript-examples.md`: verified `.tlscript` examples targeting the current ParadoxPE ABI
- `gms-dispatch-planner.md`: GPU scoring, workload assignment, aggressive secondary-lane multi-GPU planning, and Vulkan version gating
- `runtime-bridge-flow.md`: canonical MPS -> `tl-core` -> GMS -> runtime submit/present flow
- `runtime-scene-showcase.md`: runtime-side 3D scene/sprite payload model and bounce-tank showcase controller
- `runtime-scene-workload.md`: runtime scene density to GMS dispatch workload mapping
- `runtime-scheduler-path.md`: runtime automatic scheduler path selection (`GMS` vs `MGS`)
- `runtime-android.md`: Android beta-track runtime policy, scheduler precedence, and fail-soft behavior
- `runtime-draw-hud.md`: `SceneFrameInstances` -> draw-batch compile path and telemetry HUD overlay flow
- `runtime-upscaler.md`: runtime FSR policy surface and TLApp integration points
- `runtime-tlapp-console.md`: in-app `Ctrl+F1` CLI for live graphics and `.tlscript` overrides
- `runtime-pak.md`: `.pak` archive format and pack/list/unpack runtime toolchain
- `runtime-tlpfile-gui.md`: `.tlpfile` project manifest + general-purpose runtime GUI shell
- `runtime-tlsprite.md`: `.tlsprite` parser, runtime sprite program flow, and HUD scaling signals
- `runtime-tljoint.md`: scene-based multi `.tlscript` + multi `.tlsprite` composition manifest
- `runtime-mas.md`: MAS audio scheduler flow and MPS integration contract
- `alpha-foss-ui-assets.md`: candidate FOSS icon/font packs and asset policy
- `examples/tlsprite/runtime_basic_types.tlsprite`: starter `.tlsprite` asset with `hud`, `camera`, and `terrain` kinds
- `runtime-tlscript-showcase.md`: `.tlscript` showcase compile/evaluate bootstrap for scene control
- `mgs-scene-workload.md`: runtime/mobile scene density to MGS bridge hint and tile planning mapping
- `mgs-orangepi5-validation.md`: repeatable Orange Pi 5 validation runner and pass gates for MGS
- `releases/v0.4.5.1.md`: current release notes for 2D `.png/.svg` sprite texture support
- `releases/v0.4.5.1-github.md`: GitHub-ready release summary for `v0.4.5.1`
- `releases/v0.4.5.md`: 2D foundation consolidation release notes
- `releases/v0.4.5-github.md`: GitHub-ready release summary for `v0.4.5`
- `releases/v0.3.0.md`: runtime version command + console file tooling release notes
- `releases/v0.3.0-github.md`: GitHub-ready release summary for `v0.3.0`
- `releases/v0.2.1.md`: stabilization release notes for collision robustness
- `releases/v0.1.0.md`: first pre-alpha release notes and binary packaging flow
- `demos/tlapp/*`: TLApp show-scene assets (`bounce_showcase.tlscript`, `bounce_hud.tlsprite`)
- `demos/`: demo-specific flow notes and show scripts (`docs/demos/*`)

## Documentation Style

- Prefer implementation-adjacent docs that reference real modules and public APIs.
- Keep benchmark docs separate from engine/runtime behavior.
- Favor explicit performance constraints (latency budgets, zero-copy paths, bounded waits).

## PDF Build Utility

- `scripts/build_tlscript_pdf.sh`: builds a TLScript guide PDF from markdown via `pandoc` (default
  engine: `tectonic`) with explicit dependency checks and fail-soft diagnostics.
