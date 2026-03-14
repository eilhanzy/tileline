#!/usr/bin/env bash
set -euo pipefail

VERSION="v0.1.0"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOST_TRIPLE="$(rustc -vV | awk '/host:/ { print $2 }')"
TARGET_TRIPLE="${TARGET_TRIPLE:-$HOST_TRIPLE}"

EXT=""
if [[ "$TARGET_TRIPLE" == *windows* ]]; then
  EXT=".exe"
fi

DIST_DIR="$ROOT_DIR/dist/$VERSION/$TARGET_TRIPLE"
BIN_NAME="tlapp$EXT"
ARCHIVE_NAME="tileline-${VERSION}-${TARGET_TRIPLE}-tlapp.tar.gz"

hash_file() {
  local file="$1"
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$file"
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$file"
  else
    echo "error: neither sha256sum nor shasum is available" >&2
    return 1
  fi
}

mkdir -p "$DIST_DIR"

pushd "$ROOT_DIR" >/dev/null
cargo build -p runtime --release --bin tlapp --target "$TARGET_TRIPLE"
popd >/dev/null

BIN_SRC="$ROOT_DIR/target/$TARGET_TRIPLE/release/$BIN_NAME"
if [[ ! -f "$BIN_SRC" ]]; then
  echo "error: built binary not found at $BIN_SRC" >&2
  exit 1
fi

cp "$BIN_SRC" "$DIST_DIR/$BIN_NAME"

cat > "$DIST_DIR/build-info.txt" <<INFO
version=$VERSION
built_at_utc=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
host_triple=$HOST_TRIPLE
target_triple=$TARGET_TRIPLE
rustc=$(rustc --version)
INFO

pushd "$DIST_DIR" >/dev/null
hash_file "$BIN_NAME" > SHA256SUMS
rm -f "$ARCHIVE_NAME"
tar -czf "$ARCHIVE_NAME" "$BIN_NAME" SHA256SUMS build-info.txt
popd >/dev/null

echo "release artifacts ready: $DIST_DIR"
ls -lh "$DIST_DIR"
