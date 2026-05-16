#!/usr/bin/env python3
"""
PRIMA: Boosting Animal Mesh Recovery with Biological Priors and Test-Time Adaptation

Official implementation of the paper:
"PRIMA: Boosting Animal Mesh Recovery with Biological Priors and Test-Time Adaptation"
by Xiaohang Yu, Ti Wang, and Mackenzie Weygandt Mathis
Licensed under a modified MIT license
"""
# Download and arrange PRIMA demo assets into the expected data/ layout.
# Usage:
#   python scripts/setup_demo_data.py
#   python scripts/setup_demo_data.py --force

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from prima.utils.weights import (
    DEFAULT_HF_REPO_ID,
    ensure_demo_assets,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download PRIMA demo checkpoints and data")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Target data directory")
    parser.add_argument("--force", action="store_true", help="Redownload and overwrite existing files")
    parser.add_argument(
        "--hf-repo-id",
        type=str,
        default=DEFAULT_HF_REPO_ID,
        help="Hugging Face repo ID containing demo assets (e.g., org/repo)",
    )
    args = parser.parse_args()
    ensure_demo_assets(
        args.data_dir,
        stages=("PRIMAS1", "PRIMAS3"),
        force=args.force,
        hf_repo_id=args.hf_repo_id,
    )

    print("\n[done] Demo assets ready.")
    print("Run demo:")
    print("  python demo.py --img_folder demo_data/ --out_folder demo_out/")
    print("Run demo with TTA:")
    print("  python demo_tta.py --img_folder demo_data/ --out_folder demo_out_tta/ --tta_lr 1e-6 --tta_num_iters 30")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
