#!/usr/bin/env bash
# Fresh local environment: venv, pip deps, LFS assets, demo checkpoints, smoke test.
#
# Usage:
#   ./scripts/clean_install_local.sh
#   PRIMA_VENV=.venv ./scripts/clean_install_local.sh --skip-data
#   ./scripts/clean_install_local.sh --wipe-data --force-data
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

VENV="${PRIMA_VENV:-.venv}"
SKIP_DATA=0
FORCE_DATA=0
WIPE_DATA=0
EDITABLE=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --venv)
      VENV="$2"
      shift 2
      ;;
    --skip-data)
      SKIP_DATA=1
      shift
      ;;
    --force-data)
      FORCE_DATA=1
      shift
      ;;
    --wipe-data)
      WIPE_DATA=1
      shift
      ;;
    --no-editable)
      EDITABLE=0
      shift
      ;;
    -h|--help)
      echo "Usage: $0 [--venv DIR] [--skip-data] [--force-data] [--wipe-data] [--no-editable]"
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
done

echo "[clean-install] Repository: ${ROOT}"

if command -v git-lfs >/dev/null 2>&1; then
  echo "[clean-install] git lfs pull (demo images / teaser) ..."
  git lfs install
  git lfs pull
else
  echo "[clean-install] WARN: git-lfs not found; demo images may be LFS pointer stubs. Install: brew install git-lfs && git lfs install" >&2
fi

if [[ -d "$VENV" ]]; then
  echo "[clean-install] Removing existing venv: ${VENV}"
  rm -rf "$VENV"
fi

echo "[clean-install] Creating venv: ${VENV}"
python3 -m venv "$VENV"
# shellcheck disable=SC1090
source "${VENV}/bin/activate"

python -m pip install -U pip wheel setuptools

echo "[clean-install] pip install -r requirements.txt (this can take a long time) ..."
python -m pip install -r "${ROOT}/requirements.txt"

if [[ "$EDITABLE" -eq 1 ]]; then
  echo "[clean-install] pip install -e . (editable package) ..."
  if python -m pip install -e "${ROOT}"; then
    :
  else
    echo "[clean-install] WARN: editable install failed (often Detectron2 / mmcv build). Try the conda instructions in README.md, or use --no-editable." >&2
  fi
fi

if [[ "$WIPE_DATA" -eq 1 ]]; then
  echo "[clean-install] Wiping downloaded demo data under data/ ..."
  rm -rf "${ROOT}/data/PRIMAS1" "${ROOT}/data/PRIMAS3" "${ROOT}/data/smal" "${ROOT}/data/amr_vitbb.pth" 2>/dev/null || true
fi

if [[ "$SKIP_DATA" -eq 0 ]]; then
  FORCE_ARGS=()
  if [[ "$FORCE_DATA" -eq 1 ]]; then
    FORCE_ARGS=(--force)
  fi
  echo "[clean-install] Downloading demo assets (large) ..."
  python "${ROOT}/scripts/setup_demo_data.py" "${FORCE_ARGS[@]}"
else
  echo "[clean-install] Skipping setup_demo_data (--skip-data)."
fi

echo "[clean-install] Smoke test: import app + build_demo ..."
python -c "import app; app.build_demo(); print('[clean-install] Gradio demo build: OK')"

echo "[clean-install] Done. Activate with: source ${VENV}/bin/activate"
