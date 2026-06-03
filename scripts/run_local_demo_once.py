#!/usr/bin/env python3
"""
PRIMA: Boosting Animal Mesh Recovery with Biological Priors and Test-Time Adaptation

Official implementation of the paper:
"PRIMA: Boosting Animal Mesh Recovery with Biological Priors and Test-Time Adaptation"
by Xiaohang Yu, Ti Wang, and Mackenzie Weygandt Mathis
Licensed under a modified MIT license

One-shot local smoke: load PRIMA, run beagle demo (TTA off), print paths to outputs.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("PRIMA_PRELOAD_ASSETS", "0")

import cv2  # noqa: E402

import app  # noqa: E402


def main() -> int:
    out_dir = ROOT / "demo_out_tta_gradio_local_proof"
    out_dir.mkdir(parents=True, exist_ok=True)
    img_path = ROOT / "demo_data" / "beagle.jpg"
    if not img_path.is_file():
        print(f"ERROR: missing {img_path}")
        return 1

    print("[1/4] Loading PRIMA checkpoint …")
    model, cfg, renderer, device = app._load_prima_model()
    print(f"      device={device}")

    print("[2/4] Building detector (Detectron2 if installed, else SuperAnimal detector) …")
    det = app._build_detector()
    print(f"      detector={'detectron2' if det is not None else 'superanimal fallback'}")

    img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
    print(f"[3/4] Running inference on {img_path.name} (TTA iterations=0) …")
    before, after, kpts, _, _ = app._collect_animal_results(
        model,
        cfg,
        renderer,
        device,
        det,
        str(out_dir),
        img,
        1e-6,
        0,
        0.7,
        0.1,
        False,
        False,
    )
    print(f"      renders: before={len(before)} after={len(after)} kpts={len(kpts)}")

    pngs = sorted(out_dir.glob("*.png"))
    print("[4/4] Output files:")
    for p in pngs:
        print(f"      {p}")

    if not pngs:
        print("FAIL: no PNG outputs (often pyrender/display on headless macOS).")
        return 1

    print("OK: local demo produced outputs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
