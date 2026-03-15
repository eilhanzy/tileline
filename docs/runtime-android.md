# Runtime Android Beta Track

This document defines the Android enablement baseline for Tileline pre-beta (`v0.2.x`) and the
beta gate target (`v0.3.0-beta`).

## Scope (Locked)

- ABI: `arm64-v8a` (`aarch64-linux-android`)
- API level: `29+`
- Graphics policy: `Vulkan-first` with fail-soft fallback screen (no hard crash)
- Runtime scope on Android: TLApp runtime + mini editor V1

## Build Prerequisites

- Android SDK + NDK installed
- `ANDROID_HOME` and `ANDROID_NDK_HOME` configured
- Rust target installed:

```bash
rustup target add aarch64-linux-android
```

- `cargo-apk` installed:

```bash
cargo install cargo-apk
```

## Build and Run

Build TLApp APK:

```bash
cargo apk build -p runtime --bin tlapp --target aarch64-linux-android --release
```

Or use the helper script:

```bash
./scripts/build_android_apk.sh
```

Run on connected device:

```bash
cargo apk run -p runtime --bin tlapp --target aarch64-linux-android --release
```

Build project GUI APK:

```bash
cargo apk build -p runtime --bin tlproject_gui --target aarch64-linux-android --release
```

## Scheduler and Fallback Behavior

Android runtime scheduler precedence for `.tlpfile`:

| Manifest scheduler | Behavior on Android |
| --- | --- |
| `gms` | Attempt explicit GMS. If unsupported, runtime enters fail-soft with diagnostics. |
| `mgs` | Uses MGS directly. |
| `auto` | Deterministic Android rule: `auto => MGS`. |

Fail-soft diagnostics include:

- adapter/backend info
- selected scheduler (or rejection reason)
- Vulkan probe failure reason when Vulkan path is unavailable

These diagnostics are emitted to logs and reflected in runtime status/title text.

## Driver Limit Guard (Panthor / Mali)

Runtime device creation clamps requested `wgpu::Limits::default()` to adapter-supported limits
before `request_device`.

This specifically protects Android/Linux mobile stacks where the driver advertises lower 3D texture
limits (for example `max_texture_dimension_3d = 512` instead of `2048`), so TLApp can still boot
without hard failure.

## Android Mini Editor V1 (Selected Scope)

Included:

- scene preview
- start/pause/stop controls
- compile diagnostics panel
- touch-first preview panning + gamepad kept enabled

Deferred from V1:

- full `.tlpfile` text editor
- full file explorer / asset authoring panels
- advanced layout customization

## Validation Gates

Android beta gates are locked to:

- at least two device classes:
  - one Adreno
  - one Mali/Panthor
- average FPS >= 30
- P95 frame time near 33.3 ms target band
- unsupported Vulkan path must show fail-soft screen (no crash)
