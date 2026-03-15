# Runtime Upscaler (FSR Policy Surface)

This document describes the v0.3.0 foundation added in `runtime/src/upscaler.rs`.

## What exists now

- `FsrMode`: `Off | Auto | On`
- `FsrQualityPreset`: `Native | UltraQuality | Quality | Balanced | Performance`
- `FsrConfig` and resolved `FsrStatus`
- fail-soft backend gating (`Vulkan/Metal/DX12` accepted by policy, others fallback to native)

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

This is the policy/config foundation pass. Full FSR 1.0 shader pass integration
(`EASU + RCAS`) is the next v0.3.0 renderer milestone.

