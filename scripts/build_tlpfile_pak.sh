#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_PATH="$ROOT_DIR/docs/demos/tlapp/tlapp_project.tlpfile"
OUTPUT_DIR="$ROOT_DIR/dist/pak"
BASE_NAME="tlapp-assets"
MAX_SHARD_GB=5
LIST_AFTER_PACK=1

print_usage() {
  cat <<'USAGE'
Tileline .tlpfile -> numbered .pak builder

Usage:
  ./scripts/build_tlpfile_pak.sh \
    [--project <path/to/project.tlpfile>] \
    [--out-dir <dir>] \
    [--base-name <name>] \
    [--max-shard-gb <int>] \
    [--no-list]

Defaults:
  --project docs/demos/tlapp/tlapp_project.tlpfile
  --out-dir dist/pak
  --base-name tlapp-assets
  --max-shard-gb 5

Notes:
  - The directory that contains the .tlpfile is treated as the package root.
  - Path references in .tlpfile/.tljoint/.tlsprite must stay inside this root
    (no absolute paths, no ../ traversal).
  - Output files are numbered: <base-name>.part0001.pak, part0002, ...
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --project)
      PROJECT_PATH="$2"
      shift 2
      ;;
    --out-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --base-name)
      BASE_NAME="$2"
      shift 2
      ;;
    --max-shard-gb)
      MAX_SHARD_GB="$2"
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

if ! [[ "$MAX_SHARD_GB" =~ ^[0-9]+$ ]] || [[ "$MAX_SHARD_GB" -le 0 ]]; then
  echo "error: --max-shard-gb must be a positive integer" >&2
  exit 1
fi

PROJECT_ABS="$(realpath -m "$PROJECT_PATH")"
if [[ ! -f "$PROJECT_ABS" ]]; then
  echo "error: .tlpfile not found: $PROJECT_ABS" >&2
  exit 1
fi
if [[ "${PROJECT_ABS##*.}" != "tlpfile" ]]; then
  echo "error: --project must point to a .tlpfile: $PROJECT_ABS" >&2
  exit 1
fi

PROJECT_ROOT="$(dirname "$PROJECT_ABS")"
OUTPUT_ABS="$(realpath -m "$OUTPUT_DIR")"
if [[ "$OUTPUT_ABS" == "$PROJECT_ROOT"* ]]; then
  echo "error: --out-dir must be outside project root ($PROJECT_ROOT) to avoid recursive packaging" >&2
  exit 1
fi
mkdir -p "$OUTPUT_ABS"

validate_no_root_escape_paths() {
  local root="$1"
  local bad_lines=0
  local escaped
  escaped="$(rg -n \
    '^\s*(tljoint|joint|tlscript|script|tlscripts|scripts|tlsprite|sprite|tlsprites|sprites|fbx)\s*=\s*/|^\s*(tljoint|joint|tlscript|script|tlscripts|scripts|tlsprite|sprite|tlsprites|sprites|fbx)\s*=.*\.\./' \
    "$root" \
    -g '*.tlpfile' -g '*.tljoint' -g '*.tlsprite' 2>/dev/null || true)"
  if [[ -n "$escaped" ]]; then
    echo "error: manifest paths must stay inside tlpfile root (no absolute paths and no ../):" >&2
    echo "$escaped" >&2
    bad_lines=1
  fi
  return "$bad_lines"
}

validate_no_root_escape_paths "$PROJECT_ROOT"

MAX_BYTES=$((MAX_SHARD_GB * 1024 * 1024 * 1024))
STAGING_DIR="$(mktemp -d "${TMPDIR:-/tmp}/tileline-pak-builder.XXXXXX")"
cleanup() {
  rm -rf "$STAGING_DIR"
}
trap cleanup EXIT

declare -a source_files=()
while IFS= read -r -d '' rel_path; do
  rel_path="${rel_path#./}"
  source_files+=("$rel_path")
done < <(cd "$PROJECT_ROOT" && find . -type f ! -name '*.pak' -print0 | LC_ALL=C sort -z)

if [[ "${#source_files[@]}" -eq 0 ]]; then
  echo "error: no files found in project root '$PROJECT_ROOT'" >&2
  exit 1
fi

declare -a shard_sources=()
declare -a shard_bytes=()
declare -a shard_files=()

copy_into_shard() {
  local src="$1"
  local dst="$2"
  mkdir -p "$(dirname "$dst")"
  ln "$src" "$dst" 2>/dev/null || cp --reflink=auto -- "$src" "$dst"
}

current_shard=1
current_bytes=0
current_files=0
current_src_dir="$STAGING_DIR/shard$(printf '%04d' "$current_shard")"
mkdir -p "$current_src_dir"

for rel_path in "${source_files[@]}"; do
  abs_path="$PROJECT_ROOT/$rel_path"
  size_bytes="$(stat -c '%s' "$abs_path")"
  if [[ "$current_bytes" -gt 0 ]] && (( current_bytes + size_bytes > MAX_BYTES )); then
    shard_sources+=("$current_src_dir")
    shard_bytes+=("$current_bytes")
    shard_files+=("$current_files")
    current_shard=$((current_shard + 1))
    current_bytes=0
    current_files=0
    current_src_dir="$STAGING_DIR/shard$(printf '%04d' "$current_shard")"
    mkdir -p "$current_src_dir"
  fi
  copy_into_shard "$abs_path" "$current_src_dir/$rel_path"
  current_bytes=$((current_bytes + size_bytes))
  current_files=$((current_files + 1))
done

if [[ "$current_files" -gt 0 ]]; then
  shard_sources+=("$current_src_dir")
  shard_bytes+=("$current_bytes")
  shard_files+=("$current_files")
fi

manifest_file="$OUTPUT_ABS/${BASE_NAME}.manifest.txt"
{
  echo "tlpfile=$PROJECT_ABS"
  echo "root=$PROJECT_ROOT"
  echo "max_shard_bytes=$MAX_BYTES"
  echo "shards=${#shard_sources[@]}"
} > "$manifest_file"

pushd "$ROOT_DIR" >/dev/null
for i in "${!shard_sources[@]}"; do
  part_number="$(printf '%04d' "$((i + 1))")"
  out_file="$OUTPUT_ABS/${BASE_NAME}.part${part_number}.pak"
  src_dir="${shard_sources[$i]}"
  bytes="${shard_bytes[$i]}"
  files="${shard_files[$i]}"
  echo "[builder] pack shard $part_number files=$files bytes=$bytes -> $out_file"
  cargo run -p runtime --example pak_tool -- \
    pack --src "$src_dir" --out "$out_file"
  if [[ "$LIST_AFTER_PACK" -eq 1 ]]; then
    cargo run -p runtime --example pak_tool -- \
      list --pak "$out_file"
  fi
  echo "part${part_number} file=$out_file bytes=$bytes files=$files" >> "$manifest_file"
done
popd >/dev/null

echo "[builder] done"
echo "[builder] manifest: $manifest_file"
