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
- `nps-protocol.md`: NPS UDP bit-packing, reliability, authority handoff, and MPS integration
- `nps-runtime-plan.md`: canonical NPS channel, tick, snapshot, and transport runtime plan
- `paradoxpe-foundation.md`: ParadoxPE handles, SoA storage, broadphase/solver pipeline, snapshot base, and script ABI
- `paradoxpe-tlscript-examples.md`: verified `.tlscript` examples targeting the current ParadoxPE ABI
- `gms-dispatch-planner.md`: GPU scoring, workload assignment, aggressive secondary-lane multi-GPU planning, and Vulkan version gating
- `runtime-bridge-flow.md`: canonical MPS -> `tl-core` -> GMS -> runtime submit/present flow
- `runtime-scene-showcase.md`: runtime-side 3D scene/sprite payload model and bounce-tank showcase controller
- `runtime-scene-workload.md`: runtime scene density to GMS dispatch workload mapping
- `runtime-scheduler-path.md`: runtime automatic scheduler path selection (`GMS` vs `MGS`)
- `runtime-draw-hud.md`: `SceneFrameInstances` -> draw-batch compile path and telemetry HUD overlay flow
- `runtime-pak.md`: `.pak` archive format and pack/list/unpack runtime toolchain
- `runtime-tlpfile-gui.md`: `.tlpfile` project manifest + general-purpose runtime GUI shell
- `runtime-tlsprite.md`: `.tlsprite` parser, runtime sprite program flow, and HUD scaling signals
- `runtime-tljoint.md`: scene-based multi `.tlscript` + multi `.tlsprite` composition manifest
- `runtime-mas.md`: MAS audio scheduler flow and MPS integration contract
- `alpha-foss-ui-assets.md`: candidate FOSS icon/font packs and asset policy
- `examples/tlsprite/runtime_basic_types.tlsprite`: starter `.tlsprite` asset with `hud`, `camera`, and `terrain` kinds
- `runtime-tlscript-showcase.md`: `.tlscript` showcase compile/evaluate bootstrap for scene control
- `mgs-scene-workload.md`: runtime/mobile scene density to MGS bridge hint and tile planning mapping
- `releases/v0.2.0.md`: current release notes with pre-beta `.pak` packaging additions
- `releases/v0.1.0.md`: first pre-alpha release notes and binary packaging flow
- `demos/tlapp/*`: TLApp show-scene assets (`bounce_showcase.tlscript`, `bounce_hud.tlsprite`)
- `demos/`: demo-specific flow notes and show scripts (`docs/demos/*`)

## Documentation Style

- Prefer implementation-adjacent docs that reference real modules and public APIs.
- Keep benchmark docs separate from engine/runtime behavior.
- Favor explicit performance constraints (latency budgets, zero-copy paths, bounded waits).
