# Runtime Scheduler Path Policy (GMS vs MGS)

This note documents the automatic graphics scheduler selection used by runtime code.

## Source Files

- `runtime/src/scheduler_path.rs`
- `runtime/examples/auto_scene_scheduler.rs`

## Policy

`runtime::choose_scheduler_path(...)` selects:

- `Mgs` for mobile/TBDR-style integrated targets (Mali/Adreno/PowerVR/Apple mobile profile)
- `Gms` for non-mobile throughput-oriented targets (typically discrete desktop GPUs)

`runtime::choose_scheduler_path_for_platform(...)` adds explicit platform overrides:

- Android `auto` policy is deterministic: `auto => Mgs`
- Desktop policy keeps adapter-profile based choice (`Gms` vs `Mgs`)

For `.tlpfile` projects, runtime applies precedence:

1. manifest explicit `scheduler = gms|mgs` is attempted first
2. if explicit value is unsupported on Android (for example `gms` on non-Vulkan), runtime enters
   fail-soft mode with diagnostics
3. `scheduler = auto` delegates to platform policy (Android -> `Mgs`)

The decision includes:

- selected path (`GraphicsSchedulerPath`)
- adapter metadata (`name`, `backend`, `device_type`)
- detected mobile profile
- explainable reason string

## Example

```bash
cargo run -p runtime --example auto_scene_scheduler
```

The example:

1. discovers the preferred adapter
2. chooses `GMS` or `MGS`
3. runs bounce-scene workload synthesis and planning through the selected path
