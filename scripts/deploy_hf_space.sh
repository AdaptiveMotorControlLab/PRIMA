#!/usr/bin/env bash
# Deploy working tree to Hugging Face Space MLAdaptiveIntelligence/PRIMA-demo.
#
# Demo PNG/JPG are tracked with Git LFS (Hugging Face Hub Xet bridge); see .gitattributes.
# We rsync only the Git-tracked files needed by app.py from the working tree
# (not ``git archive``), so tracked LFS files are materialized bytes while
# untracked local files and non-Space project files stay out. Then ``git add``
# stores matching files as LFS objects on push.
#
# Prerequisites: brew install git-lfs git-xet && git xet install && git lfs install
set -euo pipefail

export GIT_TERMINAL_PROMPT=0

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

SPACE_SYNC_PATHS=(
  ".gitattributes"
  "README.md"
  "requirements.txt"
  "pyproject.toml"
  "app.py"
  "demo_tta.py"
  "chumpy"
  "configs/sa_finetune_hrnet_w32.yaml"
  "demo_data"
  "images/teaser.png"
  "prima"
)
SPACE_EXTRA_FILES=(
  "packages.txt"
)

echo "[deploy] Rsync Git-tracked Space files from ${ROOT} ..."
printf '[deploy]   %s\n' "${SPACE_SYNC_PATHS[@]}"
missing_tracked=()
for path in "${SPACE_SYNC_PATHS[@]}"; do
  if [[ -z "$(git ls-files -- "$path")" ]]; then
    missing_tracked+=("$path")
  fi
done
if [[ "${#missing_tracked[@]}" -gt 0 ]]; then
  printf '[deploy] ERROR: Space sync path is not tracked by Git: %s\n' "${missing_tracked[@]}" >&2
  echo "[deploy] Add required new files with git add, or remove them from SPACE_SYNC_PATHS." >&2
  exit 1
fi
git ls-files -z -- "${SPACE_SYNC_PATHS[@]}" | rsync -a --from0 --files-from=- "${ROOT}/" "${TMP}/"

echo "[deploy] Rsync explicit Space config files from ${ROOT} ..."
for path in "${SPACE_EXTRA_FILES[@]}"; do
  if [[ ! -f "$path" ]]; then
    echo "[deploy] ERROR: Missing required Space config file: $path" >&2
    exit 1
  fi
  printf '[deploy]   %s\n' "$path"
  rsync -a --relative "$path" "$TMP/"
done

README_FILE="${TMP}/README.md"
if ! sed -n '1,20p' "$README_FILE" | grep -q '^sdk: gradio$'; then
  echo "[deploy] Adding Hugging Face Space YAML front matter to README.md ..."
  README_TMP="${README_FILE}.tmp"
  {
    cat <<'YAML'
---
title: PRIMA Demo
emoji: 🦮
colorFrom: blue
colorTo: green
sdk: gradio
python_version: "3.10"
app_file: app.py
startup_duration_timeout: 60m
---

YAML
    cat "$README_FILE"
  } > "$README_TMP"
  mv "$README_TMP" "$README_FILE"
fi

cd "$TMP"

echo "[deploy] Git init + LFS commit ..."
git init -q
git lfs install
git add -A
git -c user.email="space-deploy@users.noreply.github.com" -c user.name="HF Space deploy" commit -q -m "Deploy snapshot (LFS for demo images per .gitattributes)"

PUSH_URL="$SPACE_URL"
if [[ "$PUSH_URL" == https://huggingface.co/* && -z "${HF_TOKEN:-}" && -f "${HF_HOME:-$HOME/.cache/huggingface}/token" ]]; then
  HF_TOKEN="$(<"${HF_HOME:-$HOME/.cache/huggingface}/token")"
fi
if [[ "$PUSH_URL" == https://huggingface.co/* && -n "${HF_TOKEN:-}" ]]; then
  PUSH_URL="${PUSH_URL/https:\/\/huggingface.co/https:\/\/hf_user:${HF_TOKEN}@huggingface.co}"
fi

git remote add hf "$PUSH_URL"
echo "[deploy] Force-pushing to Hugging Face Space ..."
GIT_TERMINAL_PROMPT=0 git -c credential.helper= push hf HEAD:main --force
echo "[deploy] Done."
