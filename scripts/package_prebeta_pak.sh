#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_DIR="$ROOT_DIR/docs/demos"
OUTPUT_PAK="$ROOT_DIR/dist/prebeta/tileline-assets-prebeta.pak"
LIST_AFTER_PACK=1

print_usage() {
  cat <<'USAGE'
Tileline Pre-Beta .pak packer

Usage:
  ./scripts/package_prebeta_pak.sh [--src <dir>] [--out <file.pak>] [--no-list]

Defaults:
  --src docs/demos
  --out dist/prebeta/tileline-assets-prebeta.pak
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --src)
      SOURCE_DIR="$2"
      shift 2
      ;;
    --out)
      OUTPUT_PAK="$2"
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

mkdir -p "$(dirname "$OUTPUT_PAK")"

pushd "$ROOT_DIR" >/dev/null
cargo run -p runtime --example pak_tool -- \
  pack --src "$SOURCE_DIR" --out "$OUTPUT_PAK"
if [[ "$LIST_AFTER_PACK" -eq 1 ]]; then
  cargo run -p runtime --example pak_tool -- \
    list --pak "$OUTPUT_PAK"
fi
popd >/dev/null

