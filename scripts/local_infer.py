#!/usr/bin/env python3
"""
PRIMA: Boosting Animal Mesh Recovery with Biological Priors and Test-Time Adaptation

Official implementation of the paper:
"PRIMA: Boosting Animal Mesh Recovery with Biological Priors and Test-Time Adaptation"
by Xiaohang Yu, Ti Wang, and Mackenzie Weygandt Mathis
Licensed under a modified MIT license
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Local PRIMA inference (no Gradio).")
    p.add_argument(
        "--image",
        type=str,
        default=str(ROOT / "demo_data" / "beagle.jpg"),
        help="Path to an input image.",
    )
    p.add_argument(
        "--out",
        type=str,
        default=str(ROOT / "demo_out_local_cli"),
        help="Output folder for PNG renders / artifacts.",
    )
    p.add_argument("--tta_lr", type=float, default=1e-6)
    p.add_argument("--tta_iters", type=int, default=0, help="0 disables TTA.")
    p.add_argument("--det_thresh", type=float, default=0.7)
    p.add_argument("--kp_conf_thresh", type=float, default=0.1)
    p.add_argument("--side_view", action="store_true")
    p.add_argument("--save_mesh", action="store_true")
    return p.parse_args()


def main() -> int:
    # Ensure local defaults (GPU if available) but no Space-only preload behavior.
    os.environ.setdefault("PRIMA_DEMO_MODE", "local")
    os.environ.setdefault("PRIMA_PRELOAD_ASSETS", "0")

    import numpy as np  # noqa: E402

    import app  # noqa: E402

    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    img_path = Path(args.image)
    if not img_path.is_file():
        raise FileNotFoundError(f"Missing image: {img_path}")

    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        raise RuntimeError(f"Failed to read image: {img_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.uint8)

    print("[local_infer] Loading PRIMA model ...")
    model, model_cfg, renderer, cam_crop_to_full_fn, device = app._load_prima_model()
    print(f"[local_infer] device={device}")

    print("[local_infer] Building detector (Detectron2 if installed, else fallback) ...")
    detector = app._build_detector()
    print(f"[local_infer] detector={'detectron2' if detector is not None else 'fallback'}")

    print("[local_infer] Running inference ...")
    before, after, kpts, mesh_before, mesh_after = app._collect_animal_results(
        model,
        model_cfg,
        renderer,
        cam_crop_to_full_fn,
        device,
        detector,
        str(out_dir),
        img_rgb,
        tta_lr=float(args.tta_lr),
        tta_num_iters=int(args.tta_iters),
        det_thresh=float(args.det_thresh),
        kp_conf_thresh=float(args.kp_conf_thresh),
        side_view=bool(args.side_view),
        save_mesh=bool(args.save_mesh),
    )

    print(f"[local_infer] renders: before={len(before)} after={len(after)} kpts={len(kpts)}")
    if mesh_before or mesh_after:
        print(f"[local_infer] meshes: before={mesh_before} after={mesh_after}")

    pngs = sorted(out_dir.glob("*.png"))
    for p in pngs:
        print(f"[local_infer] output: {p}")

    if not pngs:
        raise RuntimeError("No PNG outputs produced.")

    print("[local_infer] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

