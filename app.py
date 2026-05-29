"""
PRIMA: Boosting Animal Mesh Recovery with Biological Priors and Test-Time Adaptation

Official implementation of the paper:
"PRIMA: Boosting Animal Mesh Recovery with Biological Priors and Test-Time Adaptation"
by Xiaohang Yu, Ti Wang, and Mackenzie Weygandt Mathis
Licensed under a modified MIT license
"""

"""Gradio demo for PRIMA + SuperAnimal + TTA.

This script wraps the ``demo_tta.py`` pipeline into an interactive
Gradio interface. The overall logic follows:

1. Given an input image, run Detectron2 to detect animals.
2. For each detected animal, run PRIMA for 3D pose/shape estimation.
3. Run the fine-tuned DeepLabCut SuperAnimal model to obtain PRIMA 26-keypoint
   2D predictions.
4. Run test-time adaptation (TTA) with user-specified lr and iters.
5. Render and save before/after TTA results and keypoint visualizations.

"""

import argparse
import concurrent.futures
import os
import queue
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass
from functools import lru_cache
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Tuple
from pathlib import Path

# macOS: PyRender (OpenGL) and DeepLabCut/pyglet must run on the main thread.
if sys.platform == "darwin":
    os.environ.setdefault("PYGLET_HEADLESS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

import cv2
import gradio as gr
import numpy as np
import torch
import torch.utils.data

if sys.platform == "darwin":
    torch.set_num_threads(1)

# Repo-local minimal ``chumpy`` shim (see ``chumpy/__init__.py``) so SMAL pickles load
# without installing the full chumpy package in Space builds.
_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from prima.utils.weights import (
    DEFAULT_HF_REPO_ID,
    resolve_prima_checkpoint_path,
)
from prima.utils.detection import select_animal_boxes


# Default checkpoint path following README instructions
DEFAULT_CHECKPOINT = str(_REPO_ROOT / "data" / "PRIMAS1" / "checkpoints" / "s1ckpt_inference.ckpt")
DEFAULT_HF_ASSET_REPO = DEFAULT_HF_REPO_ID

# Output folder for rendered images/meshes and keypoints
DEFAULT_OUT_FOLDER = "demo_out_tta_gradio"
DEFAULT_SERVER_NAME = os.environ.get("PRIMA_GRADIO_HOST", "0.0.0.0")
DEFAULT_SERVER_PORT = int(os.environ.get("PRIMA_GRADIO_PORT", "7860"))

_D2_R50_CFG = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
_D2_R50_URL = (
    "https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/"
    "faster_rcnn_R_50_FPN_3x/137849458/model_final_280758.pkl"
)
_D2_X101_CFG = "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"
_D2_X101_URL = (
    "https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/"
    "faster_rcnn_X_101_32x8d_FPN_3x/139173657/model_final_68b088.pkl"
)

# Gradio example row: (image_rel, tta_lr, tta_iters, det_thresh, kp_thresh, side_view, save_mesh)
ExampleRow = Tuple[str, float, int, float, float, bool, bool]


@dataclass(frozen=True)
class DemoProfile:
    """Runtime settings for either the full local app or the lightweight HF Space demo."""

    mode: str
    prima_device: str  # "auto" (CUDA if available) or "cpu"
    detectron_config_yaml: str
    detectron_weights_url: str
    detectron_device: str  # "auto" or "cpu"
    default_tta_iters: int
    max_tta_iters: int
    default_save_mesh: bool
    default_side_view: bool
    preload_assets: bool
    example_rows: Tuple[ExampleRow, ...]
    description: str
    interface_title: str

    def resolve_prima_device(self) -> torch.device:
        if self.prima_device == "cpu":
            return torch.device("cpu")
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def resolve_detectron_device(self) -> str:
        if self.detectron_device == "cpu":
            return "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"


LOCAL_DEMO_PROFILE = DemoProfile(
    mode="local",
    prima_device="auto",
    detectron_config_yaml=_D2_X101_CFG,
    detectron_weights_url=_D2_X101_URL,
    detectron_device="auto",
    default_tta_iters=30,
    max_tta_iters=100,
    default_save_mesh=True,
    default_side_view=False,
    preload_assets=False,
    example_rows=(
        ("demo_data/000000015956_horse.png", 1e-6, 30, 0.7, 0.1, False, True),
        ("demo_data/n02412080_12159.png", 1e-6, 30, 0.7, 0.1, False, True),
        ("demo_data/000000315905_zebra.jpg", 1e-6, 30, 0.7, 0.1, False, True),
        ("demo_data/beagle.jpg", 1e-6, 30, 0.7, 0.1, False, True),
        ("demo_data/shepherd_hati.jpg", 1e-6, 30, 0.7, 0.1, False, True),
    ),
    description=(
        "**Local demo** — full pipeline on your machine (GPU when available).\n\n"
        "Detectron2 **X-101-FPN**, PRIMA mesh recovery, optional **DeepLabCut SuperAnimal + TTA**. "
        "Set TTA iterations to **0** to skip adaptation. Outputs are saved under "
        f"`{DEFAULT_OUT_FOLDER}`."
    ),
    interface_title=(
        "PRIMA local demo (GPU/CPU) — detection, mesh recovery, optional TTA"
    ),
)

SPACE_DEMO_PROFILE = DemoProfile(
    mode="space",
    prima_device="cpu",
    detectron_config_yaml=_D2_R50_CFG,
    detectron_weights_url=_D2_R50_URL,
    detectron_device="cpu",
    default_tta_iters=30,
    max_tta_iters=30,
    default_save_mesh=False,
    default_side_view=False,
    preload_assets=True,
    example_rows=(
        ("demo_data/000000015956_horse.png", 1e-6, 30, 0.7, 0.1, False, False),
        ("demo_data/n02412080_12159.png", 1e-6, 30, 0.7, 0.1, False, False),
        ("demo_data/000000315905_zebra.jpg", 1e-6, 30, 0.7, 0.1, False, False),
        ("demo_data/beagle.jpg", 1e-6, 30, 0.7, 0.1, False, False),
        ("demo_data/shepherd_hati.jpg", 1e-6, 30, 0.7, 0.1, False, False),
    ),
    description=(
        "**Hugging Face Space (cpu-basic)** — lightweight demo: **CPU-only** PRIMA inference. "
        "The Space build skips Detectron2 and uses a full-image crop fallback. TTA is optional "
        "(30 iterations by default, matching the local demo; set to 0 to skip). Mesh `.obj` export "
        "is off by default to save time and disk."
    ),
    interface_title="PRIMA on Hugging Face — lightweight CPU demo",
)


def _is_truthy_env(var_name: str) -> bool:
    return os.environ.get(var_name, "").strip().lower() in {"1", "true", "yes", "on"}


def _running_on_space() -> bool:
    return bool(os.environ.get("SPACE_ID") or os.environ.get("HF_SPACE_ID"))


@lru_cache(maxsize=1)
def get_demo_profile() -> DemoProfile:
    """Select local vs Space profile. Override with ``PRIMA_DEMO_MODE=local|space``."""
    override = os.environ.get("PRIMA_DEMO_MODE", "").strip().lower()
    if override == "local":
        return LOCAL_DEMO_PROFILE
    if override == "space":
        return SPACE_DEMO_PROFILE
    return SPACE_DEMO_PROFILE if _running_on_space() else LOCAL_DEMO_PROFILE


def _gradio_examples_for_interface(profile: DemoProfile) -> List[List]:
    """Gradio prefetches example media at startup (paths must exist beside ``app.py``)."""
    if _is_truthy_env("PRIMA_DISABLE_GRADIO_EXAMPLES"):
        return []
    rows: List[List] = []
    for rel, *rest in profile.example_rows:
        p = _REPO_ROOT / rel
        if p.is_file():
            rows.append([str(p), *rest])
    return rows


def _should_use_gradio_queue(profile: DemoProfile) -> bool:
    """Whether to enable Gradio's background queue.

    On macOS local, the queue runs handlers in worker threads; PyRender and
    pyglet/AppKit then crash with "setting the main menu on a non-main thread".
    """
    if _is_truthy_env("PRIMA_GRADIO_NO_QUEUE"):
        return False
    if _is_truthy_env("PRIMA_GRADIO_QUEUE"):
        return True
    return not (sys.platform == "darwin" and profile.mode == "local")


def _warmup_runtime_cache(checkpoint_path: str, profile: DemoProfile) -> Dict[str, Any]:
    """Load model + DLC on the main thread (recommended for macOS local)."""
    print("[startup] Preloading DeepLabCut on main thread…")
    _deeplabcut_available()
    print("[startup] Loading PRIMA + Detectron2 on main thread (first run can take several minutes)…")
    model, model_cfg, renderer, cam_crop_to_full_fn, device, detector = _load_model_and_detector_for_demo(
        checkpoint_path, profile
    )
    return {
        "model": model,
        "model_cfg": model_cfg,
        "renderer": renderer,
        "cam_crop_to_full_fn": cam_crop_to_full_fn,
        "device": device,
        "detector": detector,
    }


def _should_preload_assets(profile: DemoProfile) -> bool:
    preload_env = os.environ.get("PRIMA_PRELOAD_ASSETS")
    if preload_env is not None:
        return _is_truthy_env("PRIMA_PRELOAD_ASSETS")
    return profile.preload_assets

def _deeplabcut_available() -> bool:
    try:
        from deeplabcut.pose_estimation_pytorch.apis import superanimal_analyze_images  # noqa: F401

        return True
    except Exception:
        return False


def _preload_assets_once(checkpoint_path: str) -> None:
    print("[startup] Ensuring demo assets from Hugging Face Hub...")
    resolve_prima_checkpoint_path(
        checkpoint_path,
        data_dir=_REPO_ROOT / "data",
        auto_download=True,
        hf_repo_id=os.environ.get("PRIMA_HF_REPO_ID", DEFAULT_HF_ASSET_REPO),
    )
    print("[startup] Asset preload complete.")


def _load_prima_model(checkpoint_path: str = DEFAULT_CHECKPOINT):
    """Load PRIMA model and renderer once for the Gradio app."""
    from prima.models import load_prima
    from prima.utils.renderer import Renderer, cam_crop_to_full

    checkpoint_path = resolve_prima_checkpoint_path(
        checkpoint_path,
        data_dir=_REPO_ROOT / "data",
        auto_download=True,
        hf_repo_id=os.environ.get("PRIMA_HF_REPO_ID", DEFAULT_HF_ASSET_REPO),
    )
    checkpoint = Path(checkpoint_path)
    cfg_path = checkpoint.parent.parent / ".hydra" / "config.yaml"
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Missing checkpoint: {checkpoint}. Download demo checkpoints/data as described in README."
        )
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"Missing model config: {cfg_path}. Ensure the full checkpoint folder layout from README is present."
        )

    profile = get_demo_profile()
    model, model_cfg = load_prima(checkpoint_path)
    device = profile.resolve_prima_device()
    model = model.to(device)
    model.eval()

    renderer = Renderer(model_cfg, faces=model.smal.faces)
    return model, model_cfg, renderer, cam_crop_to_full, device


def _build_detector(profile: Optional[DemoProfile] = None):
    """Build Detectron2 animal detector (profile selects X-101+GPU locally vs R50+CPU on Space)."""
    try:
        import detectron2.config
        import detectron2.engine
        from detectron2 import model_zoo
    except Exception as e:
        print(f"[warn] Detectron2 unavailable ({type(e).__name__}: {e}); using full-image fallback bbox.")
        return None

    if profile is None:
        profile = get_demo_profile()
    config_yaml = profile.detectron_config_yaml
    weights = profile.detectron_weights_url
    device_str = profile.resolve_detectron_device()
    print(f"[detectron2] mode={profile.mode} config={config_yaml} device={device_str}")

    cfg = detectron2.config.get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(config_yaml))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.WEIGHTS = weights
    cfg.MODEL.DEVICE = device_str
    detector = detectron2.engine.DefaultPredictor(cfg)
    return detector


def _load_model_and_detector_for_demo(checkpoint_path: str, profile: DemoProfile):
    """Load PRIMA and Detectron2 once for the Gradio session (main thread only)."""
    model, model_cfg, renderer, cam_crop_to_full_fn, device = _load_prima_model(checkpoint_path)
    detector = _build_detector(profile)
    return model, model_cfg, renderer, cam_crop_to_full_fn, device, detector


def _detect_animal_boxes(
    detector,
    img_bgr: np.ndarray,
    det_thresh: float,
) -> Optional[np.ndarray]:
    """Return Nx4 XYXY boxes or None if no animal detections."""
    if detector is None:
        h, w = img_bgr.shape[:2]
        return np.array([[0.0, 0.0, float(max(1, w - 1)), float(max(1, h - 1))]], dtype=np.float32)

    det_out = detector(img_bgr)
    det_instances = det_out["instances"]
    boxes, suppressed = select_animal_boxes(det_instances, score_threshold=float(det_thresh))
    if suppressed > 0:
        print(f"[INFO] Suppressed {suppressed} duplicate animal detection(s)")
    if len(boxes) == 0:
        return None
    return boxes


# SuperAnimal defaults (same as in demo_tta parser)
SUPER_ANIMAL_ARGS = SimpleNamespace(
    superanimal_name="superanimal_quadruped",
    superanimal_model_name="hrnet_w32",
    superanimal_detector_name="fasterrcnn_resnet50_fpn_v2",
    superanimal_max_individuals=1,
    saved_2d_model_path="",
    pytorch_config_2d_path=str(_REPO_ROOT / "configs" / "sa_finetune_hrnet_w32.yaml"),
)


def _collect_animal_results(
    model,
    model_cfg,
    renderer,
    cam_crop_to_full_fn,
    device,
    detector,
    out_folder: str,
    img_rgb: np.ndarray,
    tta_lr: float,
    tta_num_iters: int,
    det_thresh: float,
    kp_conf_thresh: float,
    side_view: bool,
    save_mesh: bool,
    boxes: Optional[np.ndarray] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], str | None, str | None]:
    """Run detection + PRIMA + SuperAnimal + TTA on a single RGB image.

    Returns:
        before_imgs: list of HxWx3 RGB images (before TTA) for all animals
        after_imgs: list of HxWx3 RGB images (after TTA) for all animals
        kpt_imgs: list of HxWx3 RGB keypoint visualizations
        first_before_mesh: path to first animal's before-TTA mesh (.obj) or None
        first_after_mesh: path to first animal's after-TTA mesh (.obj) or None
    """
    from prima.utils import recursive_to
    from prima.datasets.vitdet_dataset import ViTDetDataset
    from demo_tta import (
        denorm_patch_to_rgb,
        resolve_sa_weights_path,
        run_superanimal_on_patch,
        save_keypoint_vis,
        tta_optimize,
    )

    def report(message: str) -> None:
        if progress_callback is not None:
            progress_callback(message)

    if int(tta_num_iters) > 0 and not SUPER_ANIMAL_ARGS.saved_2d_model_path:
        report("Resolving SuperAnimal weights...")
        SUPER_ANIMAL_ARGS.saved_2d_model_path = resolve_sa_weights_path("")

    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    if boxes is None:
        if detector is None:
            report("Detectron2 unavailable; using full-image crop...")
        else:
            report("Detecting animals with Detectron2...")
        boxes = _detect_animal_boxes(detector, img_bgr, det_thresh)
    if boxes is None:
        return [], [], [], None, None

    report(f"Detected {len(boxes)} animal(s). Preparing crops...")
    dataset = ViTDetDataset(model_cfg, img_bgr, boxes)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    before_imgs: List[np.ndarray] = []
    after_imgs: List[np.ndarray] = []
    kpt_imgs: List[np.ndarray] = []
    before_mesh_paths: List[str] = []
    after_mesh_paths: List[str] = []

    img_token = next(tempfile._get_candidate_names())

    total_batches = len(dataloader)
    for batch_idx, batch in enumerate(dataloader, start=1):
        batch = recursive_to(batch, device)

        report(f"Animal {batch_idx}/{total_batches}: running PRIMA...")
        with torch.no_grad():
            out_before = model(batch)

        animal_id = int(batch["animalid"][0])

        # Save/render before TTA
        img_fn = f"{img_token}"
        from demo_tta import render_and_save  # imported lazily to avoid circular issues

        report(f"Animal {batch_idx}/{total_batches}: rendering before TTA...")
        render_and_save(
            renderer,
            cam_crop_to_full_fn,
            out_before,
            batch,
            img_fn,
            animal_id,
            out_folder,
            suffix="before_tta",
            side_view=side_view,
            save_mesh=save_mesh,
        )

        before_png_path = os.path.join(out_folder, f"{img_fn}_{animal_id}_before_tta.png")
        if os.path.exists(before_png_path):
            before_bgr = cv2.imread(before_png_path)
            if before_bgr is not None:
                before_imgs.append(cv2.cvtColor(before_bgr, cv2.COLOR_BGR2RGB))

        if save_mesh:
            before_obj_path = os.path.join(out_folder, f"{img_fn}_{animal_id}_before_tta.obj")
            if os.path.exists(before_obj_path):
                before_mesh_paths.append(before_obj_path)

        if int(tta_num_iters) <= 0:
            report(f"Animal {batch_idx}/{total_batches}: rendering final output...")
            render_and_save(
                renderer,
                cam_crop_to_full_fn,
                out_before,
                batch,
                img_fn,
                animal_id,
                out_folder,
                suffix="after_tta",
                side_view=side_view,
                save_mesh=save_mesh,
            )

            after_png_path = os.path.join(out_folder, f"{img_fn}_{animal_id}_after_tta.png")
            if os.path.exists(after_png_path):
                after_bgr = cv2.imread(after_png_path)
                if after_bgr is not None:
                    after_imgs.append(cv2.cvtColor(after_bgr, cv2.COLOR_BGR2RGB))

            if save_mesh:
                after_obj_path = os.path.join(out_folder, f"{img_fn}_{animal_id}_after_tta.obj")
                if os.path.exists(after_obj_path):
                    after_mesh_paths.append(after_obj_path)
            continue

        # Prepare patch for SuperAnimal
        report(f"Animal {batch_idx}/{total_batches}: running SuperAnimal keypoints...")
        patch_rgb = denorm_patch_to_rgb(batch["img"][0])
        with tempfile.TemporaryDirectory(prefix=f"dlc_{img_fn}_{animal_id}_") as tmp_dir:
            bodyparts_xyc = run_superanimal_on_patch(patch_rgb, SUPER_ANIMAL_ARGS, tmp_dir)

        if bodyparts_xyc is None:
            # No keypoints => skip TTA for this animal
            continue

        kpts_xyc = bodyparts_xyc
        kpts_xyc[kpts_xyc[:, 2] < float(kp_conf_thresh), 2] = 0.0

        # Save keypoint visualization and npy
        kpt_png_path = os.path.join(out_folder, f"{img_fn}_{animal_id}_prima26_kpts.png")
        save_keypoint_vis(patch_rgb, kpts_xyc, kpt_png_path)
        npy_path = os.path.join(out_folder, f"{img_fn}_{animal_id}_prima26_kpts.npy")
        np.save(npy_path, kpts_xyc)

        if os.path.exists(kpt_png_path):
            kpt_bgr = cv2.imread(kpt_png_path)
            if kpt_bgr is not None:
                kpt_imgs.append(cv2.cvtColor(kpt_bgr, cv2.COLOR_BGR2RGB))

        # Normalize keypoints to [-0.5, 0.5] as in demo_tta
        patch_h, patch_w = patch_rgb.shape[:2]
        kpts_norm = kpts_xyc.copy()
        kpts_norm[:, 0] = kpts_norm[:, 0] / float(patch_w) - 0.5
        kpts_norm[:, 1] = kpts_norm[:, 1] / float(patch_h) - 0.5
        gt_kpts_norm = torch.from_numpy(kpts_norm[None]).to(device=device, dtype=batch["img"].dtype)

        # Run TTA
        report(f"Animal {batch_idx}/{total_batches}: running TTA ({int(tta_num_iters)} iterations)...")
        out_after = tta_optimize(
            model,
            batch,
            gt_kpts_norm,
            num_iters=int(tta_num_iters),
            lr=float(tta_lr),
        )

        report(f"Animal {batch_idx}/{total_batches}: rendering after TTA...")
        render_and_save(
            renderer,
            cam_crop_to_full_fn,
            out_after,
            batch,
            img_fn,
            animal_id,
            out_folder,
            suffix="after_tta",
            side_view=side_view,
            save_mesh=save_mesh,
        )

        after_png_path = os.path.join(out_folder, f"{img_fn}_{animal_id}_after_tta.png")
        if os.path.exists(after_png_path):
            after_bgr = cv2.imread(after_png_path)
            if after_bgr is not None:
                after_imgs.append(cv2.cvtColor(after_bgr, cv2.COLOR_BGR2RGB))

        if save_mesh:
            after_obj_path = os.path.join(out_folder, f"{img_fn}_{animal_id}_after_tta.obj")
            if os.path.exists(after_obj_path):
                after_mesh_paths.append(after_obj_path)

    first_before_mesh = before_mesh_paths[0] if before_mesh_paths else None
    first_after_mesh = after_mesh_paths[0] if after_mesh_paths else None

    report("Collecting outputs...")
    return before_imgs, after_imgs, kpt_imgs, first_before_mesh, first_after_mesh


def build_demo(
    checkpoint_path: str = DEFAULT_CHECKPOINT,
    out_folder: str = DEFAULT_OUT_FOLDER,
    runtime_cache: Optional[Dict[str, Any]] = None,
) -> gr.Interface:
    profile = get_demo_profile()
    print(
        f"[demo] profile={profile.mode} prima={profile.resolve_prima_device()} "
        f"detectron={profile.detectron_config_yaml} d2_device={profile.resolve_detectron_device()}"
    )
    if _should_use_gradio_queue(profile):
        print("[demo] Gradio queue enabled (background worker threads).")
    else:
        print("[demo] Gradio queue disabled (inference runs on main thread; required on macOS local).")
    os.makedirs(out_folder, exist_ok=True)
    runtime_cache = runtime_cache or {
        "model": None,
        "model_cfg": None,
        "renderer": None,
        "cam_crop_to_full_fn": None,
        "device": None,
        "detector": None,
    }

    def gradio_inference(
        image: np.ndarray,
        tta_lr: float,
        tta_num_iters: int,
        det_thresh: float,
        kp_conf_thresh: float,
        side_view: bool,
        save_mesh: bool,
    ):
        """Wrapper for Gradio. ``image`` is an RGB numpy array.

        Yields intermediate status so long first-run (Hub downloads + model load)
        and long inference do not hit silent client/proxy WebSocket timeouts.
        """

        if image is None:
            yield None, None, None, "No image provided."
            return

        if int(tta_num_iters) > 0 and not _deeplabcut_available():
            yield (
                None,
                None,
                None,
                "DeepLabCut is not installed. Set **TTA iterations** to **0** for PRIMA-only inference, "
                "or install `deeplabcut` (see README / requirements.txt).",
            )
            return

        if image.dtype != np.uint8:
            img_rgb = np.clip(image, 0, 255).astype(np.uint8)
        else:
            img_rgb = image

        yield None, None, None, "Queued; preparing run…"

        if runtime_cache["model"] is None:
            yield (
                None,
                None,
                None,
                "First run: downloading demo assets from Hugging Face (large checkpoint) "
                "and loading the model. This can take many minutes.",
            )
            try:
                model, model_cfg, renderer, cam_crop_to_full_fn, device, detector = _load_model_and_detector_for_demo(
                    checkpoint_path, profile
                )
            except Exception:
                yield None, None, None, f"Model initialization failed:\n{traceback.format_exc()}"
                return
            runtime_cache["model"] = model
            runtime_cache["model_cfg"] = model_cfg
            runtime_cache["renderer"] = renderer
            runtime_cache["cam_crop_to_full_fn"] = cam_crop_to_full_fn
            runtime_cache["device"] = device
            runtime_cache["detector"] = detector
            yield None, None, None, "Model loaded."

        try:
            yield None, None, None, "Running animal detection…"
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            boxes = _detect_animal_boxes(runtime_cache["detector"], img_bgr, det_thresh)
            if boxes is None:
                yield (
                    None,
                    None,
                    None,
                    "No animal detected. Try lowering the detection threshold or another image.",
                )
                return
            yield (
                None,
                None,
                None,
                f"Detected {len(boxes)} animal region(s). Running PRIMA (+ SuperAnimal/TTA if enabled)...",
            )

            def run_collect(progress_callback: Optional[Callable[[str], None]] = None):
                return _collect_animal_results(
                    runtime_cache["model"],
                    runtime_cache["model_cfg"],
                    runtime_cache["renderer"],
                    runtime_cache["cam_crop_to_full_fn"],
                    runtime_cache["device"],
                    runtime_cache["detector"],
                    out_folder,
                    img_rgb,
                    tta_lr=tta_lr,
                    tta_num_iters=tta_num_iters,
                    det_thresh=det_thresh,
                    kp_conf_thresh=kp_conf_thresh,
                    side_view=side_view,
                    save_mesh=save_mesh,
                    boxes=boxes,
                    progress_callback=progress_callback,
                )

            if _should_use_gradio_queue(profile):
                stage_updates: queue.Queue[str] = queue.Queue()

                def report_stage(message: str) -> None:
                    stage_updates.put(message)

                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    fut = pool.submit(
                        run_collect,
                        report_stage,
                    )
                    t0 = time.monotonic()
                    latest_stage = "Starting inference..."
                    while True:
                        while True:
                            try:
                                latest_stage = stage_updates.get_nowait()
                            except queue.Empty:
                                break
                            else:
                                elapsed = int(time.monotonic() - t0)
                                yield None, None, None, f"{latest_stage}\nElapsed: {elapsed}s"
                        try:
                            before_imgs, after_imgs, kpt_imgs, mesh_before, mesh_after = fut.result(
                                timeout=1.0
                            )
                            break
                        except concurrent.futures.TimeoutError:
                            elapsed = int(time.monotonic() - t0)
                            yield None, None, None, (
                                f"{latest_stage}\n"
                                f"Elapsed: {elapsed}s\n"
                                "CPU inference can take several minutes."
                            )
            else:
                before_imgs, after_imgs, kpt_imgs, mesh_before, mesh_after = run_collect()
        except Exception:
            yield None, None, None, f"Inference failed:\n{traceback.format_exc()}"
            return

        first_before = before_imgs[0] if before_imgs else None
        first_after = after_imgs[0] if after_imgs else None
        first_kpts = kpt_imgs[0] if kpt_imgs else None
        if first_before is None and first_after is None:
            yield (
                None,
                None,
                None,
                "No output generated. Try an image with a clearly visible quadruped.",
            )
            return
        yield first_before, first_after, first_kpts, "OK"

    _gradio_examples = _gradio_examples_for_interface(profile)
    _iface_kw = dict(
        fn=gradio_inference,
        analytics_enabled=False,
        cache_examples=False,
        inputs=[
            gr.Image(
                label="Input image",
                type="numpy",
                sources=["upload", "clipboard"],
            ),
            gr.Slider(
                label="TTA learning rate",
                minimum=1e-7,
                maximum=1e-4,
                value=1e-6,
                step=1e-7,
            ),
            gr.Slider(
                label="TTA iterations",
                minimum=0,
                maximum=profile.max_tta_iters,
                value=profile.default_tta_iters,
                step=1,
                info="Set to 0 to disable TTA and reuse the initial PRIMA prediction.",
            ),
            gr.Slider(
                label="Detection threshold",
                minimum=0.3,
                maximum=0.9,
                value=0.7,
                step=0.05,
            ),
            gr.Slider(
                label="Keypoint confidence threshold",
                minimum=0.0,
                maximum=1.0,
                value=0.1,
                step=0.05,
            ),
            gr.Checkbox(label="Render side view", value=profile.default_side_view),
            gr.Checkbox(label="Save meshes (.obj)", value=profile.default_save_mesh),
        ],
        outputs=[
            gr.Image(label="Before TTA"),
            gr.Image(label="After TTA"),
            gr.Image(label="PRIMA 26 keypoints"),
            gr.Textbox(label="Status / Traceback", lines=12),
        ],
        title=profile.interface_title,
        description=profile.description,
    )
    if _gradio_examples:
        _iface_kw["examples"] = _gradio_examples
    demo = gr.Interface(**_iface_kw)
    if _should_use_gradio_queue(profile):
        demo.queue(max_size=8, default_concurrency_limit=1)
    return demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gradio demo for PRIMA + SuperAnimal + TTA")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=DEFAULT_CHECKPOINT,
        help="Path to the pretrained PRIMA checkpoint",
    )
    parser.add_argument(
        "--out_folder",
        type=str,
        default=DEFAULT_OUT_FOLDER,
        help="Folder used to save rendered outputs and meshes",
    )
    parser.add_argument(
        "--server_name",
        type=str,
        default=DEFAULT_SERVER_NAME,
        help="Host/interface used by Gradio. Use 0.0.0.0 for Run:AI port-forward.",
    )
    parser.add_argument(
        "--server_port",
        type=int,
        default=DEFAULT_SERVER_PORT,
        help="Port used by Gradio.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    profile = get_demo_profile()
    if _should_preload_assets(profile):
        _preload_assets_once(args.checkpoint)
    runtime_cache: Optional[Dict[str, Any]] = None
    if (
        sys.platform == "darwin"
        and profile.mode == "local"
        and _is_truthy_env("PRIMA_WARMUP")
    ):
        runtime_cache = _warmup_runtime_cache(args.checkpoint, profile)
    demo = build_demo(
        checkpoint_path=args.checkpoint,
        out_folder=args.out_folder,
        runtime_cache=runtime_cache,
    )
    demo.launch(
        inbrowser=False,
        ssr_mode=False,
        server_name=args.server_name,
        server_port=args.server_port,
    )
