#!/usr/bin/env bash
# Fresh local environment: venv, pip deps, LFS assets, demo checkpoints, smoke test.
#
# Requires Python 3.10+ (matches README, Space, and type hints in app.py).
#
# Usage:
#   ./scripts/clean_install_local.sh
#   PRIMA_PYTHON=/opt/homebrew/bin/python3.10 ./scripts/clean_install_local.sh
#   PRIMA_VENV=.venv ./scripts/clean_install_local.sh --skip-data
#   ./scripts/clean_install_local.sh --wipe-data --force-data
set -euo pipefail

# Non-interactive: no pip/git credential prompts on stdin.
export GIT_TERMINAL_PROMPT=0
export PIP_DISABLE_PIP_VERSION_CHECK=1
export HF_HUB_DISABLE_SYMLINKS_WARNING=1

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
      echo "Env: PRIMA_PYTHON=python3.10  PRIMA_VENV=.venv"
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
done

resolve_python() {
  if [[ -n "${PRIMA_PYTHON:-}" ]]; then
    if [[ -x "${PRIMA_PYTHON}" ]] || command -v "${PRIMA_PYTHON}" >/dev/null 2>&1; then
      echo "${PRIMA_PYTHON}"
      return 0
    fi
    echo "[clean-install] ERROR: PRIMA_PYTHON=${PRIMA_PYTHON} is not executable." >&2
    return 1
  fi
  local c p
  for c in python3.12 python3.11 python3.10; do
    if command -v "$c" >/dev/null 2>&1; then
      if "$c" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)'; then
        command -v "$c"
        return 0
      fi
    fi
  done
  for p in /opt/homebrew/bin/python3.10 /usr/local/bin/python3.10; do
    if [[ -x "$p" ]]; then
      echo "$p"
      return 0
    fi
  done
  return 1
}

resolve_torch_index_url() {
  if [[ -n "${PRIMA_TORCH_INDEX_URL:-}" ]]; then
    echo "${PRIMA_TORCH_INDEX_URL}"
    return 0
  fi

  if [[ "$(uname -s)" == "Darwin" ]]; then
    return 1
  fi

  if command -v nvcc >/dev/null 2>&1; then
    local cuda_version
    cuda_version="$(nvcc --version | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -n 1)"
    case "${cuda_version}" in
      11.8)
        echo "https://download.pytorch.org/whl/cu118"
        return 0
        ;;
      12.1)
        echo "https://download.pytorch.org/whl/cu121"
        return 0
        ;;
      "")
        echo "[clean-install] WARN: Could not parse nvcc CUDA version; using pip default PyTorch wheel." >&2
        return 1
        ;;
      *)
        echo "[clean-install] WARN: CUDA ${cuda_version} detected; set PRIMA_TORCH_INDEX_URL if Detectron2 needs a specific PyTorch wheel." >&2
        return 1
        ;;
    esac
  fi

  return 1
}

echo "[clean-install] Repository: ${ROOT}"

if ! PY="$(resolve_python)"; then
  echo "[clean-install] ERROR: Need Python 3.10 or newer (Gradio 5 + app type hints)." >&2
  echo "  macOS: brew install python@3.10" >&2
  echo "  Then: PRIMA_PYTHON=/opt/homebrew/bin/python3.10 $0 ..." >&2
  exit 1
fi
echo "[clean-install] Using Python: $("$PY" -c 'import sys; print(sys.executable, sys.version.split()[0])')"

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
"$PY" -m venv "$VENV"
# shellcheck disable=SC1090
source "${VENV}/bin/activate"

python -m pip install --no-input -U pip wheel
# Match requirements.txt / pyproject pins before pulling the rest
python -m pip install --no-input "setuptools<81" "packaging<25" "Cython<3"

if TORCH_INDEX_URL="$(resolve_torch_index_url)"; then
  echo "[clean-install] Installing PyTorch from ${TORCH_INDEX_URL} ..."
  python -m pip install --no-input --index-url "${TORCH_INDEX_URL}" \
    "torch==2.2.1" "torchvision==0.17.1"
fi

echo "[clean-install] pip install -r requirements.txt (this can take a long time) ..."
REQ_TMP="$(mktemp)"
grep -vE '^[[:space:]]*(deeplabcut|detectron2)' "${ROOT}/requirements.txt" > "${REQ_TMP}"
python -m pip install --no-input -r "${REQ_TMP}"
rm -f "${REQ_TMP}"

if [[ "$(uname -s)" == "Darwin" ]]; then
  echo "[clean-install] macOS: PyTables wheel then DeepLabCut 3.x (SuperAnimal pytorch API) ..."
  python -m pip install --no-input "tables>=3.9.2,<3.11"
  python -m pip install --no-input "deeplabcut==3.0.0rc14" || {
    echo "[clean-install] ERROR: deeplabcut install failed. Try: brew install hdf5 && retry." >&2
    exit 1
  }
else
  python -m pip install --no-input "deeplabcut==3.0.0rc14"
fi

echo "[clean-install] Detectron2 (needs torch in venv; --no-build-isolation) ..."
python -m pip install --no-input --no-build-isolation \
  "detectron2 @ git+https://github.com/facebookresearch/detectron2.git"

# Spaces install Gradio separately; local venv needs it for app.py.
echo "[clean-install] Installing Gradio for local demo (HF Space provides its own) ..."
python -m pip install --no-input "gradio>=5.1,<7"

if [[ "$EDITABLE" -eq 1 ]]; then
  echo "[clean-install] pip install --no-deps -e . (register package; runtime deps from requirements.txt) ..."
  python -m pip install --no-input --no-deps -e "${ROOT}"
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

export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

echo "[clean-install] Smoke test: import app + build_demo + DeepLabCut API ..."
python -c "
import app
app.get_demo_profile.cache_clear()
p = app.get_demo_profile()
print('[clean-install] demo profile:', p.mode)
app.build_demo()
print('[clean-install] DeepLabCut SuperAnimal (may take ~30s on first import) ...')
from deeplabcut.pose_estimation_pytorch.apis import superanimal_analyze_images  # noqa: F401
print('[clean-install] Gradio demo build + DeepLabCut 3.x: OK')
"

echo "[clean-install] Done. Activate with: source ${VENV}/bin/activate"
