#!/usr/bin/env bash
# Deploy working tree to Hugging Face Space MLAdaptiveIntelligence/PRIMA-demo.
#
# Demo PNG/JPG are tracked with Git LFS (Hugging Face Hub Xet bridge); see .gitattributes.
# We rsync the working tree (not ``git archive``) so LFS-tracked files are real bytes here,
# then ``git add`` stores them as LFS objects on push.
#
# Prerequisites: brew install git-lfs git-xet && git xet install && git lfs install
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
SPACE_URL="${HF_SPACE_GIT_URL:-https://huggingface.co/spaces/MLAdaptiveIntelligence/PRIMA-demo.git}"

if ! command -v git-lfs >/dev/null 2>&1; then
  echo "[deploy] ERROR: git-lfs is required. Install: brew install git-lfs && git lfs install" >&2
  exit 1
fi

TMP="$(mktemp -d)"
cleanup() { rm -rf "$TMP"; }
trap cleanup EXIT

echo "[deploy] Rsync working tree from ${ROOT} ..."
rsync -a \
  --exclude=".git/" \
  --exclude="__pycache__/" \
  --exclude="*.pyc" \
  --exclude=".DS_Store" \
  --exclude="data/" \
  --exclude=".venv/" \
  --exclude="venv/" \
  --exclude=".pytest_cache/" \
  --exclude=".tmp_gradio_info.json" \
  --exclude="demo_out_tta_gradio/" \
  --exclude=".gradio/" \
  "${ROOT}/" "${TMP}/"

cd "$TMP"

echo "[deploy] Git init + LFS commit ..."
git init -q
git lfs install
git add -A
git -c user.email="space-deploy@users.noreply.github.com" -c user.name="HF Space deploy" commit -q -m "Deploy snapshot (LFS for demo images per .gitattributes)"

git remote add hf "$SPACE_URL"
echo "[deploy] Force-pushing to Hugging Face Space ..."
GIT_TERMINAL_PROMPT=0 git -c credential.helper=osxkeychain push hf HEAD:main --force
echo "[deploy] Done."
