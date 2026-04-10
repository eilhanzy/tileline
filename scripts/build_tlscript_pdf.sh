#!/usr/bin/env bash
set -euo pipefail

SCRIPT_NAME="build_tlscript_pdf"
DEFAULT_ENGINE="tectonic"

usage() {
  cat <<'USAGE'
Usage:
  scripts/build_tlscript_pdf.sh --source <guide.md> --output <guide.pdf> [--engine <pdf_engine>]
  scripts/build_tlscript_pdf.sh <guide.md> <guide.pdf> [--engine <pdf_engine>]

Description:
  Build a PDF from a TLScript markdown guide using pandoc.
  Default PDF engine is 'tectonic'.

Options:
  --source <path>   Source markdown file.
  --output <path>   Output PDF path.
  --engine <name>   Pandoc PDF engine command (default: tectonic).
  -h, --help        Show this help message.

Exit codes:
  0  Success
  2  Invalid arguments
  3  Missing or invalid source file
  4  Missing dependency (pandoc or PDF engine)
  5  PDF conversion failed
  6  Output file missing/empty
USAGE
}

log() {
  echo "[$SCRIPT_NAME] $*"
}

err() {
  echo "[$SCRIPT_NAME] error: $*" >&2
}

dependency_hint() {
  local dep="$1"
  case "$dep" in
    pandoc)
      cat <<'HINT' >&2
Install guidance:
  macOS (Homebrew): brew install pandoc
  Ubuntu/Debian:    sudo apt-get install -y pandoc
  Arch Linux:       sudo pacman -S pandoc
HINT
      ;;
    tectonic)
      cat <<'HINT' >&2
Install guidance:
  macOS (Homebrew): brew install tectonic
  Any platform:     cargo install --locked tectonic
HINT
      ;;
    *)
      cat <<HINT >&2
Install guidance:
  Ensure '$dep' is installed and available in PATH.
HINT
      ;;
  esac
}

SOURCE=""
OUTPUT=""
ENGINE="$DEFAULT_ENGINE"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --source)
      [[ $# -ge 2 ]] || { err "--source requires a value"; usage; exit 2; }
      SOURCE="$2"
      shift 2
      ;;
    --output)
      [[ $# -ge 2 ]] || { err "--output requires a value"; usage; exit 2; }
      OUTPUT="$2"
      shift 2
      ;;
    --engine)
      [[ $# -ge 2 ]] || { err "--engine requires a value"; usage; exit 2; }
      ENGINE="$2"
      shift 2
      ;;
    --*)
      err "unknown option: $1"
      usage
      exit 2
      ;;
    *)
      if [[ -z "$SOURCE" ]]; then
        SOURCE="$1"
      elif [[ -z "$OUTPUT" ]]; then
        OUTPUT="$1"
      else
        err "unexpected positional argument: $1"
        usage
        exit 2
      fi
      shift
      ;;
  esac
done

if [[ -z "$SOURCE" || -z "$OUTPUT" ]]; then
  err "both source and output are required"
  usage
  exit 2
fi

if [[ ! -f "$SOURCE" ]]; then
  err "source markdown not found: $SOURCE"
  exit 3
fi

if ! command -v pandoc >/dev/null 2>&1; then
  err "missing dependency: pandoc"
  dependency_hint "pandoc"
  exit 4
fi

if ! command -v "$ENGINE" >/dev/null 2>&1; then
  err "missing PDF engine: $ENGINE"
  dependency_hint "$ENGINE"
  exit 4
fi

mkdir -p "$(dirname "$OUTPUT")"

TITLE="$(awk '/^# / {sub(/^# /, ""); print; exit}' "$SOURCE")"
if [[ -z "$TITLE" ]]; then
  TITLE="TLScript Guide v0.4.5"
fi

BUILD_DATE="$(date +%Y-%m-%d)"

log "source=$SOURCE"
log "output=$OUTPUT"
log "engine=$ENGINE"

if ! pandoc "$SOURCE" \
  --from="gfm+yaml_metadata_block" \
  --pdf-engine="$ENGINE" \
  --toc \
  --number-sections \
  --variable "papersize:a4" \
  --variable "geometry:margin=2.2cm" \
  --variable "colorlinks:true" \
  --metadata "title:$TITLE" \
  --metadata "date:$BUILD_DATE" \
  --output "$OUTPUT"; then
  err "pandoc conversion failed"
  exit 5
fi

if [[ ! -s "$OUTPUT" ]]; then
  err "output file was not created or is empty: $OUTPUT"
  exit 6
fi

SIZE_BYTES="$(wc -c < "$OUTPUT" | tr -d ' ')"
log "pdf generated successfully (${SIZE_BYTES} bytes)"
