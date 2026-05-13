#!/usr/bin/env bash
# Deploy current branch to Hugging Face Space MLAdaptiveIntelligence/PRIMA-demo.
# Hugging Face rejects git pushes that contain demo PNG/JPG in-tree; this script
# archives the repo, strips those binaries, makes a single commit, and force-pushes.
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
BRANCH="$(git branch --show-current)"
SPACE_URL="${HF_SPACE_GIT_URL:-https://huggingface.co/spaces/MLAdaptiveIntelligence/PRIMA-demo.git}"

TMP="$(mktemp -d)"
cleanup() { rm -rf "$TMP"; }
trap cleanup EXIT

echo "[deploy] Archiving ${BRANCH} from ${ROOT} ..."
git archive "$BRANCH" | tar -x -C "$TMP"
cd "$TMP"

echo "[deploy] Stripping demo images / teaser from snapshot (still on GitHub raw URLs) ..."
rm -f demo_data/*.png demo_data/*.jpg demo_data/*.jpeg 2>/dev/null || true
rm -f images/teaser.png 2>/dev/null || true

git init -q
git add -A
git -c user.email="space-deploy@users.noreply.github.com" -c user.name="HF Space deploy" commit -q -m "Deploy snapshot (no PNG/JPG in git per HF policy)"

git remote add hf "$SPACE_URL"
echo "[deploy] Force-pushing to Hugging Face Space ..."
GIT_TERMINAL_PROMPT=0 git -c credential.helper=osxkeychain push hf HEAD:main --force
echo "[deploy] Done."
