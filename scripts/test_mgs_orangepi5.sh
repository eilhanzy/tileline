#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_ROOT="${ROOT_DIR}/dist/mgs/orangepi5"

MODE="stable"
VSYNC="on"
WARMUP="2"
DURATION="10"
RESOLUTION="1280x720"
RUN_BENCH=1
MIN_AVG_FPS="0"
MIN_STABILITY="0"

print_usage() {
  cat <<'USAGE'
Tileline MGS Orange Pi 5 Validation Runner

Usage:
  ./scripts/test_mgs_orangepi5.sh [options]

Options:
  --mode <auto|stable|max>      Benchmark mode (default: stable)
  --vsync <auto|on|off>         VSync preference (default: on)
  --warmup <sec>                Warmup seconds (default: 2)
  --duration <sec>              Sample seconds (default: 10)
  --resolution <WxH>            Benchmark resolution (default: 1280x720)
  --min-avg-fps <value>         Optional pass gate for average FPS (default: 0)
  --min-stability <value>       Optional pass gate for stability percent (default: 0)
  --no-bench                    Run only cargo check + cargo test
  --help                        Show this help

Artifacts:
  dist/mgs/orangepi5/<timestamp>/
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="$2"
      shift 2
      ;;
    --vsync)
      VSYNC="$2"
      shift 2
      ;;
    --warmup)
      WARMUP="$2"
      shift 2
      ;;
    --duration)
      DURATION="$2"
      shift 2
      ;;
    --resolution)
      RESOLUTION="$2"
      shift 2
      ;;
    --min-avg-fps)
      MIN_AVG_FPS="$2"
      shift 2
      ;;
    --min-stability)
      MIN_STABILITY="$2"
      shift 2
      ;;
    --no-bench)
      RUN_BENCH=0
      shift
      ;;
    -h|--help)
      print_usage
      exit 0
      ;;
    *)
      echo "error: unknown argument '$1'" >&2
      print_usage
      exit 1
      ;;
  esac
done

mkdir -p "${LOG_ROOT}"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="${LOG_ROOT}/${RUN_ID}"
mkdir -p "${RUN_DIR}"

log() {
  echo "[mgs-opi5] $*"
}

capture_cmd() {
  local output_file="$1"
  shift
  {
    echo "$ $*"
    "$@"
  } >"${output_file}" 2>&1 || true
}

log "capturing system info to ${RUN_DIR}"
capture_cmd "${RUN_DIR}/uname.txt" uname -a
capture_cmd "${RUN_DIR}/rustc.txt" rustc -Vv
capture_cmd "${RUN_DIR}/cargo.txt" cargo -V

if command -v lscpu >/dev/null 2>&1; then
  capture_cmd "${RUN_DIR}/lscpu.txt" lscpu
fi
if [[ -r /proc/cpuinfo ]]; then
  cp /proc/cpuinfo "${RUN_DIR}/cpuinfo.txt"
fi
if command -v lsmod >/dev/null 2>&1; then
  capture_cmd "${RUN_DIR}/gpu_modules.txt" bash -lc "lsmod | grep -Ei 'panthor|panfrost|mali|nvidia|amdgpu' || true"
fi
if command -v vulkaninfo >/dev/null 2>&1; then
  capture_cmd "${RUN_DIR}/vulkaninfo_summary.txt" vulkaninfo --summary
else
  echo "vulkaninfo not found" >"${RUN_DIR}/vulkaninfo_summary.txt"
fi
if command -v glxinfo >/dev/null 2>&1; then
  capture_cmd "${RUN_DIR}/glxinfo_B.txt" glxinfo -B
fi

pushd "${ROOT_DIR}" >/dev/null

log "running cargo check -p mgs"
cargo check -p mgs 2>&1 | tee "${RUN_DIR}/cargo_check_mgs.log"

log "running cargo test -p mgs"
cargo test -p mgs -- --nocapture 2>&1 | tee "${RUN_DIR}/cargo_test_mgs.log"

BENCH_EXIT="skipped"
if [[ "${RUN_BENCH}" -eq 1 ]]; then
  if [[ -z "${DISPLAY:-}" && -z "${WAYLAND_DISPLAY:-}" ]]; then
    log "no active display detected (DISPLAY/WAYLAND_DISPLAY empty), skipping benchmark"
    echo "benchmark skipped: no active display" >"${RUN_DIR}/mgs_benchmark.log"
  else
    log "running benchmark: mode=${MODE} vsync=${VSYNC} warmup=${WARMUP}s duration=${DURATION}s resolution=${RESOLUTION}"
    set +e
    cargo run -p mgs --example render_benchmark -- \
      --mode "${MODE}" \
      --vsync "${VSYNC}" \
      --warmup "${WARMUP}" \
      --duration "${DURATION}" \
      --resolution "${RESOLUTION}" \
      2>&1 | tee "${RUN_DIR}/mgs_benchmark.log"
    BENCH_EXIT="$?"
    set -e
  fi
fi

popd >/dev/null

score_line=""
render_fps_line=""
if [[ -f "${RUN_DIR}/mgs_benchmark.log" ]]; then
  score_line="$(grep -E '^Score: ' "${RUN_DIR}/mgs_benchmark.log" | tail -n 1 || true)"
  # Support both historical `FPS:` and current `Render FPS:` summary formats.
  render_fps_line="$(grep -E '^(Render FPS:|FPS: )' "${RUN_DIR}/mgs_benchmark.log" | tail -n 1 || true)"
fi
avg_fps="$(printf '%s' "${render_fps_line}" | sed -E 's/.*avg=([0-9]+\.?[0-9]*).*/\1/' || true)"
stability="$(printf '%s' "${score_line}" | sed -E 's/.*stability=([0-9]+\.?[0-9]*)%.*/\1/' || true)"

if [[ -z "${avg_fps}" || "${avg_fps}" == "${render_fps_line}" ]]; then
  avg_fps="n/a"
fi
if [[ -z "${stability}" || "${stability}" == "${score_line}" ]]; then
  stability="n/a"
fi

{
  echo "MGS Orange Pi 5 Validation Summary"
  echo "run_id=${RUN_ID}"
  echo "run_dir=${RUN_DIR}"
  echo "mode=${MODE}"
  echo "vsync=${VSYNC}"
  echo "warmup_sec=${WARMUP}"
  echo "duration_sec=${DURATION}"
  echo "resolution=${RESOLUTION}"
  echo "benchmark_exit=${BENCH_EXIT}"
  echo "avg_fps=${avg_fps}"
  echo "stability_percent=${stability}"
  if [[ -n "${render_fps_line}" ]]; then
    echo "render_fps_line=${render_fps_line}"
  fi
  if [[ -n "${score_line}" ]]; then
    echo "score_line=${score_line}"
  fi
} | tee "${RUN_DIR}/summary.txt"

fail=0
if [[ "${avg_fps}" != "n/a" ]] && [[ "${MIN_AVG_FPS}" != "0" ]]; then
  if awk -v v="${avg_fps}" -v min="${MIN_AVG_FPS}" 'BEGIN { exit !(v + 0 < min + 0) }'; then
    log "gate failed: avg_fps ${avg_fps} < min_avg_fps ${MIN_AVG_FPS}"
    fail=1
  fi
fi
if [[ "${stability}" != "n/a" ]] && [[ "${MIN_STABILITY}" != "0" ]]; then
  if awk -v v="${stability}" -v min="${MIN_STABILITY}" 'BEGIN { exit !(v + 0 < min + 0) }'; then
    log "gate failed: stability ${stability}% < min_stability ${MIN_STABILITY}%"
    fail=1
  fi
fi
if [[ "${BENCH_EXIT}" != "skipped" ]] && [[ "${BENCH_EXIT}" != "0" ]]; then
  log "benchmark process returned non-zero exit code: ${BENCH_EXIT}"
  fail=1
fi

if [[ "${fail}" -ne 0 ]]; then
  log "validation FAILED"
  exit 1
fi

log "validation OK"
log "artifacts: ${RUN_DIR}"
