#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_PATH="$ROOT_DIR/docs/demos/tlapp/tlapp_project.tlpfile"
OUTPUT_DIR="$ROOT_DIR/dist/prebeta"
BASE_NAME="tileline-assets-prebeta"
MAX_SHARD_GB=5
LIST_AFTER_PACK=1
MODE="project"
LEGACY_SOURCE_DIR=""
LEGACY_OUTPUT_PAK=""

print_usage() {
  cat <<'USAGE'
Tileline Pre-Beta .pak packer

Usage:
  Project mode (default):
    ./scripts/package_prebeta_pak.sh \
      [--project <file.tlpfile>] \
      [--out-dir <dir>] \
      [--base-name <name>] \
      [--max-shard-gb <int>] \
      [--no-list]

  Legacy mode (single archive):
    ./scripts/package_prebeta_pak.sh --src <dir> [--out <file.pak>] [--no-list]

Project defaults:
  --project docs/demos/tlapp/tlapp_project.tlpfile
  --out-dir dist/prebeta
  --base-name tileline-assets-prebeta
  --max-shard-gb 5

Legacy defaults:
  --src <required in legacy mode>
  --out dist/prebeta/tileline-assets-prebeta.pak
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --project)
      PROJECT_PATH="$2"
      MODE="project"
      shift 2
      ;;
    --out-dir)
      OUTPUT_DIR="$2"
      MODE="project"
      shift 2
      ;;
    --base-name)
      BASE_NAME="$2"
      MODE="project"
      shift 2
      ;;
    --max-shard-gb)
      MAX_SHARD_GB="$2"
      MODE="project"
      shift 2
      ;;
    --src)
      LEGACY_SOURCE_DIR="$2"
      MODE="legacy"
      shift 2
      ;;
    --out)
      LEGACY_OUTPUT_PAK="$2"
      MODE="legacy"
      shift 2
      ;;
    --no-list)
      LIST_AFTER_PACK=0
      shift
      ;;
    -h|--help)
      print_usage
      exit 0
      ;;
    *)
      echo "error: unknown argument: $1" >&2
      print_usage
      exit 1
      ;;
  esac
done

if [[ "$MODE" == "legacy" ]]; then
  if [[ -z "$LEGACY_SOURCE_DIR" ]]; then
    echo "error: legacy mode requires --src <dir>" >&2
    exit 1
  fi
  if [[ -z "$LEGACY_OUTPUT_PAK" ]]; then
    LEGACY_OUTPUT_PAK="$ROOT_DIR/dist/prebeta/tileline-assets-prebeta.pak"
  fi
  mkdir -p "$(dirname "$LEGACY_OUTPUT_PAK")"
  pushd "$ROOT_DIR" >/dev/null
  cargo run -p runtime --example pak_tool -- \
    pack --src "$LEGACY_SOURCE_DIR" --out "$LEGACY_OUTPUT_PAK"
  if [[ "$LIST_AFTER_PACK" -eq 1 ]]; then
    cargo run -p runtime --example pak_tool -- \
      list --pak "$LEGACY_OUTPUT_PAK"
  fi
  popd >/dev/null
  exit 0
fi

builder_args=(
  --project "$PROJECT_PATH"
  --out-dir "$OUTPUT_DIR"
  --base-name "$BASE_NAME"
  --max-shard-gb "$MAX_SHARD_GB"
)
if [[ "$LIST_AFTER_PACK" -eq 0 ]]; then
  builder_args+=(--no-list)
fi
"$ROOT_DIR/scripts/build_tlpfile_pak.sh" "${builder_args[@]}"

# Backward-compatibility alias:
# If only one shard is produced, copy it to <base-name>.pak for older scripts/docs.
shopt -s nullglob
parts=( "$OUTPUT_DIR/$BASE_NAME".part*.pak )
if [[ "${#parts[@]}" -eq 1 ]]; then
  cp -f "${parts[0]}" "$OUTPUT_DIR/$BASE_NAME.pak"
  echo "[package_prebeta] single-shard alias created: $OUTPUT_DIR/$BASE_NAME.pak"
fi
shopt -u nullglob
