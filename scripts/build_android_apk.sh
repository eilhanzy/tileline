#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="${TARGET:-aarch64-linux-android}"
BIN="${BIN:-tlapp}"
PROFILE="${PROFILE:-release}"

require_cmd() {
  local cmd="$1"
  if command -v "$cmd" >/dev/null 2>&1; then
    return 0
  fi
  echo "error: required command not found: $cmd" >&2
  return 1
}

ensure_rust_target() {
  local target="$1"
  if rustup target list --installed | grep -qx "$target"; then
    return 0
  fi
  echo "info: installing rust target $target"
  rustup target add "$target"
}

MODE_FLAG="--release"
if [[ "$PROFILE" != "release" ]]; then
  MODE_FLAG=""
fi

require_cmd rustup
require_cmd cargo

if ! cargo apk --version >/dev/null 2>&1; then
  echo "error: cargo-apk is not installed" >&2
  echo "hint: cargo install cargo-apk" >&2
  exit 1
fi

ensure_rust_target "$TARGET"

pushd "$ROOT_DIR" >/dev/null
if [[ -n "$MODE_FLAG" ]]; then
  cargo apk build -p runtime --bin "$BIN" --target "$TARGET" "$MODE_FLAG"
else
  cargo apk build -p runtime --bin "$BIN" --target "$TARGET"
fi
popd >/dev/null

echo "android apk build complete (bin=$BIN target=$TARGET profile=$PROFILE)"
