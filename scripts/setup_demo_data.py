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
import shutil
import sys
from pathlib import Path

import torch

DEFAULT_HF_REPO_ID = "MLAdaptiveIntelligence/PRIMA"


SMAL_ASSET_PATHS = [
    "my_smpl_00781_4_all.pkl",
    "my_smpl_data_00781_4_all.pkl",
    "walking_toy_symmetric_pose_prior_with_cov_35parts.pkl",
]
BACKBONE_ASSET_PATH = "amr_vitbb.pth"
STAGE1_CONFIG_ASSET_PATH = "config_s1_HYDRA.yaml"
STAGE1_CHECKPOINT_ASSET_PATH = "s1ckpt.ckpt"
STAGE3_CONFIG_ASSET_PATH = "config_s3_HYDRA.yaml"
STAGE3_CHECKPOINT_ASSET_PATH = "s3ckpt.ckpt"


def download_from_hub(hf_repo_id: str, remote_filename: str, dest: Path) -> None:
    """Download ``remote_filename`` from the Hub repo to exact path ``dest`` (resumable, uses HF cache)."""
    from huggingface_hub import hf_hub_download

    dest.parent.mkdir(parents=True, exist_ok=True)
    got = hf_hub_download(
        repo_id=hf_repo_id,
        filename=remote_filename,
        local_dir=str(dest.parent),
        local_dir_use_symlinks=False,
    )
    got_path = Path(got).resolve()
    target = dest.resolve()
    if got_path != target:
        if target.exists():
            target.unlink()
        shutil.move(str(got_path), str(target))


def validate_torch_checkpoint(path: Path) -> None:
    try:
        torch.load(path, map_location="cpu")
    except Exception as exc:
        raise RuntimeError(
            f"Checkpoint file is invalid or incomplete: {path}\n"
            "Downloaded checkpoint is not loadable. "
            "Please verify the uploaded Hugging Face file and try again."
        ) from exc


def maybe_download_backbone(data_dir: Path, force: bool, hf_repo_id: str) -> None:
    target = data_dir / "amr_vitbb.pth"
    if target.exists() and not force:
        print(f"[skip] {target} already exists")
        return

    print("[download] pretrained backbone")
    download_from_hub(hf_repo_id, BACKBONE_ASSET_PATH, target)
    print(f"[ok] {target}")


def maybe_download_smal(data_dir: Path, force: bool, hf_repo_id: str) -> None:
    required = [Path(p).name for p in SMAL_ASSET_PATHS]
    smal_dir = data_dir / "smal"
    if smal_dir.exists() and all((smal_dir / n).exists() for n in required) and not force:
        print("[skip] SMAL files already exist")
        return

    print("[download] SMAL assets")
    for asset_path in SMAL_ASSET_PATHS:
        filename = Path(asset_path).name
        target = smal_dir / filename
        download_from_hub(hf_repo_id, asset_path, target)
    print(f"[ok] {smal_dir}")


def maybe_download_stage(
    stage_name: str,
    config_asset_path: str,
    checkpoint_asset_path: str,
    ckpt_name: str,
    data_dir: Path,
    force: bool,
    hf_repo_id: str,
) -> None:
    stage_dir = data_dir / stage_name
    cfg_target = stage_dir / ".hydra" / "config.yaml"
    ckpt_target = stage_dir / "checkpoints" / ckpt_name
    existing_ckpt_valid = False
    if cfg_target.exists() and ckpt_target.exists() and not force:
        try:
            validate_torch_checkpoint(ckpt_target)
            existing_ckpt_valid = True
        except RuntimeError:
            print(f"[warn] {stage_name} checkpoint is incomplete, redownloading checkpoint only.")
    if cfg_target.exists() and existing_ckpt_valid and not force:
        print(f"[skip] {stage_name} assets already exist")
        return

    print(f"[download] {stage_name} assets")
    cfg_target.parent.mkdir(parents=True, exist_ok=True)
    ckpt_target.parent.mkdir(parents=True, exist_ok=True)
    if force or not cfg_target.exists():
        download_from_hub(hf_repo_id, config_asset_path, cfg_target)
    if force or not ckpt_target.exists() or not existing_ckpt_valid:
        download_from_hub(hf_repo_id, checkpoint_asset_path, ckpt_target)
    validate_torch_checkpoint(ckpt_target)
    print(f"[ok] {stage_dir}")


def verify_layout(data_dir: Path) -> None:
    required_paths = [
        data_dir / "smal" / "my_smpl_00781_4_all.pkl",
        data_dir / "smal" / "my_smpl_data_00781_4_all.pkl",
        data_dir / "smal" / "walking_toy_symmetric_pose_prior_with_cov_35parts.pkl",
        data_dir / "amr_vitbb.pth",
        data_dir / "PRIMAS1" / ".hydra" / "config.yaml",
        data_dir / "PRIMAS1" / "checkpoints" / "s1ckpt.ckpt",
        data_dir / "PRIMAS3" / ".hydra" / "config.yaml",
        data_dir / "PRIMAS3" / "checkpoints" / "s3ckpt.ckpt",
    ]
    missing = [p for p in required_paths if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(str(p) for p in missing))
    validate_torch_checkpoint(data_dir / "PRIMAS1" / "checkpoints" / "s1ckpt.ckpt")
    validate_torch_checkpoint(data_dir / "PRIMAS3" / "checkpoints" / "s3ckpt.ckpt")


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
    data_dir = args.data_dir.resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    maybe_download_smal(data_dir, force=args.force, hf_repo_id=args.hf_repo_id)
    maybe_download_backbone(data_dir, force=args.force, hf_repo_id=args.hf_repo_id)
    maybe_download_stage(
        "PRIMAS1",
        STAGE1_CONFIG_ASSET_PATH,
        STAGE1_CHECKPOINT_ASSET_PATH,
        "s1ckpt.ckpt",
        data_dir,
        force=args.force,
        hf_repo_id=args.hf_repo_id,
    )
    maybe_download_stage(
        "PRIMAS3",
        STAGE3_CONFIG_ASSET_PATH,
        STAGE3_CHECKPOINT_ASSET_PATH,
        "s3ckpt.ckpt",
        data_dir,
        force=args.force,
        hf_repo_id=args.hf_repo_id,
    )
    verify_layout(data_dir)

    print("\n[done] Demo assets ready.")
    print("Run demo:")
    print("  python demo.py --checkpoint data/PRIMAS1/checkpoints/s1ckpt.ckpt --img_folder demo_data/ --out_folder demo_out/")
    print("Run demo with TTA:")
    print("  python demo_tta.py --checkpoint data/PRIMAS1/checkpoints/s1ckpt.ckpt --img_folder demo_data/ --out_folder demo_out_tta/ --tta_lr 1e-6 --tta_num_iters 30")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
