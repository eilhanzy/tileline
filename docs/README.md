# Tileline Documentation Index

This directory contains design and implementation notes for subsystems that are intended to live
in `src/` crates (not benchmark-only code).

## Documents

- `tlscript-lexer.md`: zero-copy `.tlscript` token/lexer design and constraints
- `tlscript-parser-plan.md`: parser architecture, AST shape, and V1 grammar plan
- `tlscript-semantic.md`: semantic rules, soft diagnostics, handles, and WASM sandbox policy
- `tlscript-parallel-runtime.md`: parallel contracts, advisor suggestions, and runtime dispatch planning
- `nps-protocol.md`: NPS UDP bit-packing, reliability, authority handoff, and MPS integration
- `gms-dispatch-planner.md`: GPU scoring, workload assignment, and multi-GPU planning behavior
- `runtime-bridge-flow.md`: canonical MPS -> `tl-core` -> GMS -> runtime submit/present flow
- `pmta-payload-synthesis.md`: PMTA token decoding, AES-CTR synthesis, and dataset consistency workflow

## Documentation Style

- Prefer implementation-adjacent docs that reference real modules and public APIs.
- Keep benchmark docs separate from engine/runtime behavior.
- Favor explicit performance constraints (latency budgets, zero-copy paths, bounded waits).
