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
import subprocess
import sys
import tempfile
from pathlib import Path

import torch


SMAL_FOLDER_URL = "https://drive.google.com/drive/folders/1O1tWYimVMA7hEbnwuPyiDWh90tUGoTPB"
BACKBONE_FILE_URL = "https://drive.google.com/file/d/1jOJXJVPXnWX7W7vqYVt0joJZr4C8x-Yo/view"

# Stage assets are fetched as explicit files (not whole folder download)
# to avoid pulling extra checkpoints and to keep setup deterministic.
STAGE1_CONFIG_URL = "https://drive.google.com/file/d/1Q1uNfkBDUPWjCF64xEOWxw1wygftlJBa/view"
STAGE1_CHECKPOINT_URL = "https://drive.google.com/file/d/12WYwwYE-ru8NT_9VgM7Ock--et-VB43v/view"
STAGE3_CONFIG_URL = "https://drive.google.com/file/d/1gtBhuLShgLv72ZqUo4FiKben_x0toOrB/view"
STAGE3_CHECKPOINT_URL = "https://drive.google.com/file/d/1gLXfqyhRaEUiENgv9shjIdJg_73tonk4/view"


def run_gdown(args: list[str]) -> None:
    cmd = [sys.executable, "-m", "gdown", *args]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        sys.stderr.write(result.stdout)
        sys.stderr.write(result.stderr)
        raise RuntimeError(
            "gdown failed. Install it with: pip install gdown\n"
            f"Failed command: {' '.join(cmd)}"
        )


def try_run_gdown(args: list[str]) -> tuple[bool, str]:
    cmd = [sys.executable, "-m", "gdown", *args]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        return True, ""
    error_output = f"{result.stdout}\n{result.stderr}".strip()
    return False, error_output


def validate_torch_checkpoint(path: Path) -> None:
    try:
        torch.load(path, map_location="cpu")
    except Exception as exc:
        raise RuntimeError(
            f"Checkpoint file is invalid or incomplete: {path}\n"
            "Google Drive may have returned a partial/quota-limited file. "
            "Please retry later or download manually from the README links and place it in data/."
        ) from exc


def copy_required_file(search_root: Path, filename: str, dst: Path) -> None:
    matches = list(search_root.rglob(filename))
    if not matches:
        raise FileNotFoundError(f"Could not find {filename} under {search_root}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(matches[0], dst)


def maybe_download_backbone(data_dir: Path, force: bool) -> None:
    target = data_dir / "amr_vitbb.pth"
    if target.exists() and not force:
        print(f"[skip] {target} already exists")
        return

    print("[download] pretrained backbone")
    with tempfile.TemporaryDirectory(prefix="prima_backbone_") as tmp:
        tmpdir = Path(tmp)
        run_gdown(["--fuzzy", BACKBONE_FILE_URL, "-O", str(tmpdir)])
        files = [p for p in tmpdir.rglob("*") if p.is_file()]
        if not files:
            raise FileNotFoundError("Backbone download produced no file")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(files[0], target)
    print(f"[ok] {target}")


def maybe_download_smal(data_dir: Path, force: bool) -> None:
    required = [
        "my_smpl_00781_4_all.pkl",
        "my_smpl_data_00781_4_all.pkl",
        "walking_toy_symmetric_pose_prior_with_cov_35parts.pkl",
    ]
    smal_dir = data_dir / "smal"
    if smal_dir.exists() and all((smal_dir / n).exists() for n in required) and not force:
        print("[skip] SMAL files already exist")
        return

    print("[download] SMAL assets")
    with tempfile.TemporaryDirectory(prefix="prima_smal_") as tmp:
        tmpdir = Path(tmp)
        run_gdown(["--folder", SMAL_FOLDER_URL, "-O", str(tmpdir)])
        for filename in required:
            copy_required_file(tmpdir, filename, smal_dir / filename)
    print(f"[ok] {smal_dir}")


def maybe_download_stage(
    stage_name: str,
    config_url: str,
    checkpoint_url: str,
    ckpt_name: str,
    data_dir: Path,
    force: bool,
    fallback_checkpoint_url: str | None = None,
) -> None:
    stage_dir = data_dir / stage_name
    cfg_target = stage_dir / ".hydra" / "config.yaml"
    ckpt_target = stage_dir / "checkpoints" / ckpt_name
    if cfg_target.exists() and ckpt_target.exists() and not force:
        print(f"[skip] {stage_name} assets already exist")
        return

    print(f"[download] {stage_name} assets")
    cfg_target.parent.mkdir(parents=True, exist_ok=True)
    ckpt_target.parent.mkdir(parents=True, exist_ok=True)
    run_gdown(["--fuzzy", config_url, "-O", str(cfg_target)])
    ok, err = try_run_gdown(["--fuzzy", checkpoint_url, "-O", str(ckpt_target)])
    if not ok:
        quota_error = "Too many users have viewed or downloaded this file recently" in err
        if fallback_checkpoint_url and quota_error:
            print(
                f"[warn] {stage_name} primary checkpoint is quota-limited. "
                "Trying fallback checkpoint from the same Drive folder."
            )
            run_gdown(["--fuzzy", fallback_checkpoint_url, "-O", str(ckpt_target)])
        else:
            sys.stderr.write(err + "\n")
            raise RuntimeError(
                "gdown failed. Install it with: pip install gdown\n"
                f"Failed command: {sys.executable} -m gdown --fuzzy {checkpoint_url} -O {ckpt_target}"
            )
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
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    maybe_download_smal(data_dir, force=args.force)
    maybe_download_backbone(data_dir, force=args.force)
    maybe_download_stage(
        "PRIMAS1",
        STAGE1_CONFIG_URL,
        STAGE1_CHECKPOINT_URL,
        "s1ckpt.ckpt",
        data_dir,
        force=args.force,
    )
    maybe_download_stage(
        "PRIMAS3",
        STAGE3_CONFIG_URL,
        STAGE3_CHECKPOINT_URL,
        "s3ckpt.ckpt",
        data_dir,
        force=args.force,
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
