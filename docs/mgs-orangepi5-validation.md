# MGS Orange Pi 5 Validation

This document defines the repeatable validation path for MGS on Orange Pi 5 (RK3588 / Mali + Panthor-class stack).

## Goal

Before canonical runtime loop freeze work, we keep MGS regression checks reproducible on ARM by running:

1. `cargo check -p mgs`
2. `cargo test -p mgs`
3. `render_benchmark` with fixed CLI parameters

The result is stored under `dist/mgs/orangepi5/<timestamp>/`.

## Runner Script

Use:

```bash
./scripts/test_mgs_orangepi5.sh
```

Default benchmark parameters:

- `--mode stable`
- `--vsync on`
- `--warmup 2`
- `--duration 10`
- `--resolution 1280x720`

Useful overrides:

```bash
./scripts/test_mgs_orangepi5.sh --mode max --vsync off --duration 20 --resolution 1920x1080
./scripts/test_mgs_orangepi5.sh --min-avg-fps 55 --min-stability 80
./scripts/test_mgs_orangepi5.sh --no-bench
```

## Captured Artifacts

Each run directory includes:

- `uname.txt`, `lscpu.txt`, `cpuinfo.txt`
- `gpu_modules.txt` (`panthor/panfrost/mali/...` when available)
- `vulkaninfo_summary.txt` (if `vulkaninfo` exists)
- `cargo_check_mgs.log`
- `cargo_test_mgs.log`
- `mgs_benchmark.log` (when benchmark runs)
- `summary.txt`

## Pass Criteria (Pre-Canonical Loop Gate)

Minimum gate for merge readiness:

- `cargo check -p mgs` passes
- `cargo test -p mgs` passes
- benchmark exits cleanly on device session
- no adapter/present initialization error in `mgs_benchmark.log`

Recommended quality gate for Orange Pi 5 stabilization:

- `--min-stability 80`
- target average FPS threshold set per scene profile (`--min-avg-fps`)

## Notes

- Benchmark requires an active display session (`DISPLAY` or `WAYLAND_DISPLAY`).
- If no display exists, the script skips benchmark and still records check/test outputs.
- Keep benchmark scene and CLI parameters stable when comparing two commits.
