#!/usr/bin/env bash
set -euo pipefail

TARGET_SCOPE="all"
DO_INSTALL=0

print_usage() {
  cat <<USAGE
Usage: $0 [--target all|windows|macos] [--install]

Options:
  --target   Which cross-toolchain family to handle (default: all)
  --install  Attempt package-manager install for supported Linux Windows-GNU deps

Notes:
  - macOS cross-linking from Linux usually requires osxcross (o64-clang + SDK).
  - Without --install, this script runs in doctor mode and prints actionable hints.
USAGE
}

have_cmd() {
  command -v "$1" >/dev/null 2>&1
}

detect_pkg_manager() {
  if have_cmd apt-get; then
    echo "apt"
  elif have_cmd dnf; then
    echo "dnf"
  elif have_cmd pacman; then
    echo "pacman"
  elif have_cmd zypper; then
    echo "zypper"
  else
    echo "unknown"
  fi
}

install_windows_tools_linux() {
  local pm
  pm="$(detect_pkg_manager)"
  case "$pm" in
    apt)
      echo "info: installing mingw-w64 via apt"
      sudo apt-get update
      sudo apt-get install -y mingw-w64
      ;;
    dnf)
      echo "info: installing MinGW toolchain via dnf"
      sudo dnf install -y mingw64-gcc mingw64-binutils
      ;;
    pacman)
      echo "info: installing MinGW toolchain via pacman"
      sudo pacman -Sy --needed mingw-w64-gcc
      ;;
    zypper)
      echo "info: installing MinGW toolchain via zypper"
      sudo zypper install -y mingw64-cross-gcc mingw64-cross-binutils
      ;;
    *)
      echo "warn: unsupported package manager for auto-install"
      return 1
      ;;
  esac
}

check_windows_toolchain() {
  local missing=0
  local host
  host="$(rustc -vV | awk '/host:/ { print $2 }')"
  if ! rustup target list --installed | grep -qx "x86_64-pc-windows-gnu"; then
    echo "warn: rust target missing: x86_64-pc-windows-gnu"
    if [[ "$DO_INSTALL" -eq 1 ]]; then
      rustup target add x86_64-pc-windows-gnu
    else
      echo "hint: run: rustup target add x86_64-pc-windows-gnu"
      missing=1
    fi
  fi

  if [[ "$host" == *linux* ]]; then
    if ! have_cmd x86_64-w64-mingw32-gcc; then
      echo "warn: missing x86_64-w64-mingw32-gcc"
      missing=1
    fi
    if ! have_cmd x86_64-w64-mingw32-dlltool; then
      echo "warn: missing x86_64-w64-mingw32-dlltool"
      missing=1
    fi

    if [[ "$missing" -eq 1 ]]; then
      if [[ "$DO_INSTALL" -eq 1 ]]; then
        install_windows_tools_linux || true
      else
        echo "hint: run: $0 --target windows --install"
      fi
    fi
  fi

  if have_cmd x86_64-w64-mingw32-gcc && have_cmd x86_64-w64-mingw32-dlltool; then
    echo "ok: windows-gnu cross-toolchain looks ready"
    return 0
  fi
  return 1
}

check_macos_toolchain() {
  local missing=0
  local host
  host="$(rustc -vV | awk '/host:/ { print $2 }')"

  if ! rustup target list --installed | grep -qx "aarch64-apple-darwin"; then
    echo "warn: rust target missing: aarch64-apple-darwin"
    if [[ "$DO_INSTALL" -eq 1 ]]; then
      rustup target add aarch64-apple-darwin
    else
      echo "hint: run: rustup target add aarch64-apple-darwin"
      missing=1
    fi
  fi

  if [[ "$host" == *apple-darwin* ]]; then
    if ! have_cmd xcrun; then
      echo "warn: xcrun not found"
      echo "hint: xcode-select --install"
      missing=1
    fi
  else
    if ! have_cmd o64-clang; then
      echo "warn: o64-clang not found (osxcross required for Linux->macOS cross-link)"
      echo "hint: install osxcross and export its bin dir in PATH"
      missing=1
    fi
  fi

  if [[ "$missing" -eq 0 ]]; then
    echo "ok: macOS cross-toolchain preflight looks ready"
    return 0
  fi
  return 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target)
      if [[ $# -lt 2 ]]; then
        echo "error: --target requires a value" >&2
        exit 2
      fi
      TARGET_SCOPE="$2"
      shift 2
      ;;
    --install)
      DO_INSTALL=1
      shift
      ;;
    -h|--help)
      print_usage
      exit 0
      ;;
    *)
      echo "error: unknown option: $1" >&2
      print_usage
      exit 2
      ;;
  esac
done

status=0
case "$TARGET_SCOPE" in
  windows)
    check_windows_toolchain || status=1
    ;;
  macos)
    check_macos_toolchain || status=1
    ;;
  all)
    check_windows_toolchain || status=1
    check_macos_toolchain || status=1
    ;;
  *)
    echo "error: invalid --target value: $TARGET_SCOPE" >&2
    print_usage
    exit 2
    ;;
esac

if [[ "$status" -eq 0 ]]; then
  echo "cross-toolchain doctor: healthy"
else
  echo "cross-toolchain doctor: incomplete"
fi

exit "$status"
