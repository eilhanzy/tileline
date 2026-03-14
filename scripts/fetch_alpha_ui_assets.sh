#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${ROOT_DIR}/docs/assets/alpha-ui"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT

mkdir -p "${OUT_DIR}/fonts" "${OUT_DIR}/icons"

download_and_extract_zip() {
  local url="$1"
  local dst="$2"
  local zip_path="${TMP_DIR}/archive.zip"
  rm -f "${zip_path}"
  curl -L --fail --silent --show-error "${url}" -o "${zip_path}"
  rm -rf "${dst}"
  mkdir -p "${dst}"
  unzip -q "${zip_path}" -d "${dst}"
}

echo "[alpha-assets] downloading JetBrains Mono (OFL-1.1)"
download_and_extract_zip \
  "https://github.com/JetBrains/JetBrainsMono/releases/latest/download/JetBrainsMono.zip" \
  "${OUT_DIR}/fonts/jetbrains-mono"

echo "[alpha-assets] downloading Noto Sans (OFL-1.1)"
download_and_extract_zip \
  "https://github.com/notofonts/noto-fonts/releases/latest/download/07_NotoSans.zip" \
  "${OUT_DIR}/fonts/noto-sans"

echo "[alpha-assets] downloading Tabler Icons (MIT)"
download_and_extract_zip \
  "https://github.com/tabler/tabler-icons/archive/refs/heads/main.zip" \
  "${OUT_DIR}/icons/tabler-icons"

echo "[alpha-assets] downloading Heroicons (MIT)"
download_and_extract_zip \
  "https://github.com/tailwindlabs/heroicons/archive/refs/heads/master.zip" \
  "${OUT_DIR}/icons/heroicons"

cat > "${OUT_DIR}/SOURCES.txt" <<'EOF'
Tileline Alpha UI Assets (FOSS)

Fonts:
- JetBrains Mono (OFL-1.1): https://github.com/JetBrains/JetBrainsMono
- Noto Sans (OFL-1.1): https://github.com/notofonts/noto-fonts

Icons:
- Tabler Icons (MIT): https://github.com/tabler/tabler-icons
- Heroicons (MIT): https://github.com/tailwindlabs/heroicons
EOF

echo "[alpha-assets] done -> ${OUT_DIR}"

