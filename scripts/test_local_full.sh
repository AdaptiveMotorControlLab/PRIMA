#!/usr/bin/env bash
# Full local smoke: local CLI inference (no Gradio).
set -euo pipefail

export GIT_TERMINAL_PROMPT=0
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
VENV="${PRIMA_VENV:-.venv}"

if [[ ! -x "${VENV}/bin/python" ]]; then
  echo "ERROR: missing ${VENV}. Run: ./scripts/clean_install_local.sh --skip-data" >&2
  exit 1
fi

# shellcheck disable=SC1090
source "${VENV}/bin/activate"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PRIMA_PRELOAD_ASSETS=0
export PRIMA_DEMO_MODE=local

echo "=== [1/3] Demo profile (local) ==="
python -c "import app; app.get_demo_profile.cache_clear(); p=app.get_demo_profile(); print('profile:', p.mode)"

echo "=== [2/3] DeepLabCut SuperAnimal API ==="
python -c "
from deeplabcut.pose_estimation_pytorch.apis import superanimal_analyze_images  # noqa: F401
print('DeepLabCut SuperAnimal: OK')
"

echo "=== [3/3] PRIMA local CLI inference (beagle, TTA off) ==="
python "${ROOT}/scripts/local_infer.py" --tta_iters 0 2>&1

echo "=== All local checks passed ==="
