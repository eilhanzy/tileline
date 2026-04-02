# Tileline v0.5.5 Roadmap

Codename: `MPS Standalone SIMD`

## Release Goal

`v0.5.5` splits MPS into a standalone library and adds runtime-dispatched SIMD kernels with a
stable scalar fallback.

## Workstream A: MPS SIMD Foundation

### A1. Runtime SIMD Dispatch

Target:

- select one backend once at scheduler/dispatcher initialization
- support `Scalar`, `X86Avx512`, `Aarch64Neon`, and `PowerPcAltivec`
- keep scalar fallback available on every platform

Acceptance:

- `mps::detect_runtime_simd()` returns a deterministic capability snapshot
- `MpsScheduler::simd_backend()` and dispatcher/thread-pool metrics expose backend + lane count
- worker callbacks receive `ctx.simd` without hot-loop feature probing

### A2. First Kernel Set

Target:

- SoA transform copy/publish helpers
- batched `position += velocity * dt` integration helper
- AABB min/max and pairwise overlap mask helpers

Acceptance:

- SIMD and scalar outputs match within epsilon for supported batches
- unsupported hosts fall back to scalar without crashing

## Workstream B: Hard Extract To Standalone MPS Repo

Target:

- MPS becomes a separate git repository tagged `v0.5.5`
- Tileline workspace depends on that repo through `git + tag`
- the old in-tree `mps/` crate is removed from the workspace

Acceptance:

- standalone `/home/eilhanzy/Projects/mps` builds and tests on its own
- `paradoxpe`, `runtime`, `tl-core`, and `nps` build against the tagged git dependency
- no dual-home source-of-truth remains inside the Tileline workspace

## Test Gates

- `cargo test` in standalone MPS repo
- `cargo check -p paradoxpe -p tl-core -p runtime -p nps`
- `cargo test -p paradoxpe`
- scalar-vs-SIMD golden tests in `mps::simd`
- optional cross-target checks for `aarch64` and `powerpc64le` when targets/toolchains are available

## Defaults

- AVX-512/NEON/AltiVec mode is `auto`
- `MPS_SIMD_FORCE_SCALAR=1` forces scalar fallback for debugging
- the standalone repo name and crate name remain `mps`
