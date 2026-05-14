#!/usr/bin/env bash
# Clean redeploy of the Hugging Face Space from the current working tree.
# Same as scripts/deploy_hf_space.sh; use after a local clean install or any code change.
set -euo pipefail
ROOT="$(git rev-parse --show-toplevel)"
exec "${ROOT}/scripts/deploy_hf_space.sh"
