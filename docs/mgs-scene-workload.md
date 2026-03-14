# MGS Scene Workload Mapping

This document describes the runtime-friendly path that maps scene/sprite density to MGS bridge
hints and tile plans for mobile/serial fallback rendering.

## Source Files

- `mgs/src/scene_workload.rs`
- `runtime/src/mobile_scene_workload.rs`
- `mgs/examples/scene_workload_plan.rs`

## Goal

- Keep MGS planning inputs in `src/` crates (not benchmark-only).
- Reuse the same scene density signals used by runtime scene orchestration.
- Preserve serial/mobile-first behavior while still deriving deterministic hint inputs.

## Mapping

1. Runtime scene payload (`SceneFrameInstances`) is converted into `MobileSceneSnapshot`.
2. `mgs::estimate_mps_workload_hint(...)` produces `MpsWorkloadHint`:
   - `transfer_size_kb`
   - `object_count`
   - `target_width/target_height`
   - `latency_budget_ms`
3. `MgsBridge::translate(...)` maps the hint into fallback-aware tile planning.

## Example

```bash
cargo run -p mgs --example scene_workload_plan
```

The example prints resolved fallback level, tile count, and total draw coverage.
