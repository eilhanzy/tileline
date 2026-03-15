# Runtime Upscaler (FSR Policy Surface)

This document describes the v0.3.0 foundation added in `runtime/src/upscaler.rs`.

## What exists now

- `FsrMode`: `Off | Auto | On`
- `FsrQualityPreset`: `Native | UltraQuality | Quality | Balanced | Performance`
- `FsrConfig` and resolved `FsrStatus`
- fail-soft backend gating (`Vulkan/Metal/DX12` accepted by policy, others fallback to native)
- two-stage render path in `WgpuSceneRenderer`:
  - scene pass into internal source target (scaled viewport when FSR active)
  - fullscreen upscale pass with RCAS-like sharpen kernel

Runtime integration points:

- `runtime/src/wgpu_scene_renderer.rs`
  - `set_fsr_config(...)`
  - `fsr_status()`
- `runtime/src/tlapp_app.rs`
  - CLI flags:
    - `--fsr`
    - `--fsr-quality`
    - `--fsr-sharpness`
    - `--fsr-scale`
  - startup and runtime telemetry includes FSR status fields

## Notes

Current pass is an engine-owned spatial upscaler + sharpen implementation for v0.3.0 foundation.
Planned next step is stricter AMD FSR 1.0 parity tuning (`EASU + RCAS` parameterization and
quality validation matrix).
