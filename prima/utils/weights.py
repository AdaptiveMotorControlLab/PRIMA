"""
PRIMA: Boosting Animal Mesh Recovery with Biological Priors and Test-Time Adaptation

Official implementation of the paper:
"PRIMA: Boosting Animal Mesh Recovery with Biological Priors and Test-Time Adaptation"
by Xiaohang Yu, Ti Wang, and Mackenzie Weygandt Mathis
Licensed under a modified MIT license
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Optional, Union

HF_REPO_ID = "MLAdaptiveIntelligence/PRIMA"
DEFAULT_HF_REPO_ID = HF_REPO_ID

MODEL_DIR_NAME = "PRIMA"
CHECKPOINT_FILENAME = "best_inference.ckpt"
DEFAULT_CHECKPOINT = Path("data") / MODEL_DIR_NAME / "checkpoints" / CHECKPOINT_FILENAME

SMAL_ASSET_PATHS = [
    "my_smpl_00781_4_all.pkl",
    "my_smpl_data_00781_4_all.pkl",
    "walking_toy_symmetric_pose_prior_with_cov_35parts.pkl",
]
BACKBONE_ASSET_PATH = "amr_vitbb.pth"
CONFIG_ASSET_PATH = "config.yaml"
CHECKPOINT_ASSET_PATH = "best_inference.ckpt"

RELATIVE_CHECKPOINT = Path(MODEL_DIR_NAME) / "checkpoints" / CHECKPOINT_FILENAME

PathLike = Union[str, Path]


def _resolve_hf_repo_id(hf_repo_id: Optional[str]) -> str:
    return hf_repo_id or os.environ.get("PRIMA_HF_REPO_ID", HF_REPO_ID)


def _default_checkpoint_path(data_dir: PathLike = "data") -> Path:
    return Path(data_dir) / RELATIVE_CHECKPOINT


def _config_path_for_checkpoint(checkpoint_path: PathLike) -> Path:
    checkpoint_path = Path(checkpoint_path)
    return checkpoint_path.parent.parent / ".hydra" / "config.yaml"


def _is_default_checkpoint(checkpoint_path: PathLike) -> bool:
    """True if the path follows the standard ``<data>/PRIMA/checkpoints/`` layout."""
    checkpoint_path = Path(checkpoint_path)
    if len(checkpoint_path.parents) < 2:
        return False
    if checkpoint_path.parent.parent.name != MODEL_DIR_NAME:
        return False
    return checkpoint_path.name == CHECKPOINT_FILENAME


def _download_file(
    hf_repo_id: str,
    remote_filename: str,
    destination: Path,
    force_download: bool = False,
) -> None:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError(
            "huggingface_hub is required to download PRIMA demo assets. "
            "Install it with:  pip install huggingface_hub\n"
            "Or download the assets manually and pass a local checkpoint path."
        ) from None

    destination.parent.mkdir(parents=True, exist_ok=True)
    downloaded = hf_hub_download(
        repo_id=hf_repo_id,
        filename=remote_filename,
        local_dir=str(destination.parent),
        local_dir_use_symlinks=False,
        force_download=force_download,
    )
    downloaded_path = Path(downloaded).resolve()
    target = destination.resolve()
    if downloaded_path != target:
        if target.exists():
            target.unlink()
        shutil.move(str(downloaded_path), str(target))


def _validate_torch_checkpoint(path: Path) -> None:
    import inspect
    import pickle
    import zipfile

    import torch

    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as checkpoint_zip:
            corrupt_member = checkpoint_zip.testzip()
        if corrupt_member is not None:
            raise RuntimeError(
                f"Checkpoint file is invalid or incomplete: {path}\n"
                f"Corrupt archive member: {corrupt_member}\n"
                "Please redownload the checkpoint and try again."
            )

    supports_weights_only = "weights_only" in inspect.signature(torch.load).parameters
    load_kwargs = {"map_location": "cpu"}
    if supports_weights_only:
        load_kwargs["weights_only"] = True

    try:
        torch.load(path, **load_kwargs)
    except pickle.UnpicklingError as exc:
        message = str(exc)
        if (
            supports_weights_only
            and "Weights only load failed" in message
            and ("Unsupported global" in message or "Unsupported class" in message)
        ):
            return
        raise RuntimeError(
            f"Checkpoint file is invalid or incomplete: {path}\n"
            "Downloaded checkpoint is not loadable. "
            "Please verify the uploaded Hugging Face file and try again."
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            f"Checkpoint file is invalid or incomplete: {path}\n"
            "Downloaded checkpoint is not loadable. "
            "Please verify the uploaded Hugging Face file and try again."
        ) from exc


def _ensure_backbone(data_dir: Path, force: bool, hf_repo_id: str) -> None:
    target = data_dir / "amr_vitbb.pth"
    if target.exists() and not force:
        print(f"[skip] {target} already exists")
        return

    print("[download] pretrained backbone")
    _download_file(hf_repo_id, BACKBONE_ASSET_PATH, target, force_download=force)
    print(f"[ok] {target}")


def _ensure_smal_assets(data_dir: Path, force: bool, hf_repo_id: str) -> None:
    required = [Path(p).name for p in SMAL_ASSET_PATHS]
    smal_dir = data_dir / "smal"
    if smal_dir.exists() and all((smal_dir / n).exists() for n in required) and not force:
        print("[skip] SMAL files already exist")
        return

    print("[download] SMAL assets")
    for asset_path in SMAL_ASSET_PATHS:
        target = smal_dir / Path(asset_path).name
        _download_file(hf_repo_id, asset_path, target, force_download=force)
    print(f"[ok] {smal_dir}")


def _ensure_model_assets(
    data_dir: Path,
    force: bool,
    hf_repo_id: str,
    validate_existing: bool = True,
) -> None:
    model_dir = data_dir / MODEL_DIR_NAME
    config_target = model_dir / ".hydra" / "config.yaml"
    checkpoint_target = model_dir / "checkpoints" / CHECKPOINT_FILENAME
    redownload_checkpoint = False

    if config_target.exists() and checkpoint_target.exists() and not force:
        if validate_existing:
            try:
                _validate_torch_checkpoint(checkpoint_target)
            except RuntimeError:
                print("[warn] PRIMA checkpoint is incomplete, redownloading checkpoint only.")
                redownload_checkpoint = True
            else:
                print("[skip] PRIMA model assets already exist")
                return
        else:
            print("[skip] PRIMA model assets already exist")
            return

    print("[download] PRIMA model assets")
    config_target.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_target.parent.mkdir(parents=True, exist_ok=True)
    if force or not config_target.exists():
        _download_file(hf_repo_id, CONFIG_ASSET_PATH, config_target, force_download=force)
    if redownload_checkpoint and checkpoint_target.exists():
        checkpoint_target.unlink()
    if force or redownload_checkpoint or not checkpoint_target.exists():
        _download_file(
            hf_repo_id,
            CHECKPOINT_ASSET_PATH,
            checkpoint_target,
            force_download=force or redownload_checkpoint,
        )
    _validate_torch_checkpoint(checkpoint_target)
    print(f"[ok] {model_dir}")


def _verify_assets(data_dir: Path) -> None:
    model_dir = data_dir / MODEL_DIR_NAME
    required_paths = [
        data_dir / "smal" / "my_smpl_00781_4_all.pkl",
        data_dir / "smal" / "my_smpl_data_00781_4_all.pkl",
        data_dir / "smal" / "walking_toy_symmetric_pose_prior_with_cov_35parts.pkl",
        data_dir / "amr_vitbb.pth",
        model_dir / ".hydra" / "config.yaml",
        model_dir / "checkpoints" / CHECKPOINT_FILENAME,
    ]

    missing = [p for p in required_paths if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(str(p) for p in missing))

    _validate_torch_checkpoint(model_dir / "checkpoints" / CHECKPOINT_FILENAME)


def _ensure_assets_for_checkpoint(
    checkpoint_path: PathLike,
    force: bool = False,
    hf_repo_id: Optional[str] = None,
) -> None:
    checkpoint_path = Path(checkpoint_path)
    config_path = _config_path_for_checkpoint(checkpoint_path)
    if not _is_default_checkpoint(checkpoint_path):
        if checkpoint_path.exists() and config_path.exists() and not force:
            print(f"[skip] Using local PRIMA checkpoint {checkpoint_path}")
            return
        raise FileNotFoundError(
            "Missing checkpoint or config for a custom path:\n"
            f"  checkpoint: {checkpoint_path}\n"
            f"  config: {config_path}\n"
            "Auto-download supports the standard PRIMA demo layout only:\n"
            f"  data/{MODEL_DIR_NAME}/checkpoints/{CHECKPOINT_FILENAME}\n"
            "Pass that path, or download/copy your custom checkpoint manually."
        )

    data_dir = checkpoint_path.parent.parent.parent
    repo_id = _resolve_hf_repo_id(hf_repo_id)
    print(f"[download] Ensuring PRIMA demo assets under {data_dir}")
    _ensure_smal_assets(data_dir, force=force, hf_repo_id=repo_id)
    _ensure_backbone(data_dir, force=force, hf_repo_id=repo_id)
    _ensure_model_assets(
        data_dir,
        force=force,
        hf_repo_id=repo_id,
        validate_existing=False,
    )


def ensure_demo_assets(
    data_dir: PathLike = "data",
    *,
    force: bool = False,
    hf_repo_id: Optional[str] = None,
) -> None:
    """Ensure PRIMA demo assets exist in the expected ``data/`` layout."""
    data_dir = Path(data_dir).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    repo_id = _resolve_hf_repo_id(hf_repo_id)

    _ensure_smal_assets(data_dir, force=force, hf_repo_id=repo_id)
    _ensure_backbone(data_dir, force=force, hf_repo_id=repo_id)
    _ensure_model_assets(data_dir, force=force, hf_repo_id=repo_id)
    _verify_assets(data_dir)


def resolve_prima_checkpoint_path(
    checkpoint_path: PathLike = "",
    *,
    data_dir: PathLike = "data",
    auto_download: bool = True,
    hf_repo_id: Optional[str] = None,
    force: bool = False,
) -> str:
    """Return a PRIMA checkpoint path, downloading default demo assets if needed."""
    resolved = Path(checkpoint_path) if checkpoint_path else _default_checkpoint_path(data_dir)
    if auto_download:
        _ensure_assets_for_checkpoint(resolved, force=force, hf_repo_id=hf_repo_id)
    return str(resolved)


__all__ = [
    "DEFAULT_CHECKPOINT",
    "DEFAULT_HF_REPO_ID",
    "HF_REPO_ID",
    "ensure_demo_assets",
    "resolve_prima_checkpoint_path",
]
