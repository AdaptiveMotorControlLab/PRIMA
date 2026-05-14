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
3. Run DeepLabCut SuperAnimal to obtain 2D keypoints.
4. Map SuperAnimal 39 keypoints to the 26 PRIMA keypoints.
5. Run test-time adaptation (TTA) with user-specified lr and iters.
6. Render and save before/after TTA results and keypoint visualizations.

"""

import argparse
import concurrent.futures
import os
import sys
import tempfile
import time
import traceback
from types import SimpleNamespace
from typing import List, Tuple
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import torch
import torch.utils.data

# Repo-local minimal ``chumpy`` shim (see ``chumpy/__init__.py``) so SMAL pickles load
# without installing the full chumpy package in Space builds.
_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# Default checkpoint path following README instructions
DEFAULT_CHECKPOINT = "data/PRIMAS1/checkpoints/s1ckpt.ckpt"
DEFAULT_HF_ASSET_REPO = "MLAdaptiveIntelligence/PRIMA"

# Output folder for rendered images/meshes and keypoints
DEFAULT_OUT_FOLDER = "demo_out_tta_gradio"


def _is_truthy_env(var_name: str) -> bool:
    return os.environ.get(var_name, "").strip().lower() in {"1", "true", "yes", "on"}


def _running_on_space() -> bool:
    return bool(os.environ.get("SPACE_ID") or os.environ.get("HF_SPACE_ID"))


def _gradio_examples_for_interface() -> List[List]:
    """Gradio prefetches example media at startup.

    Demo images are tracked with Git LFS / Xet (see ``.gitattributes``) so they can live
    in the Hugging Face Space repo. Use absolute paths only when files exist beside ``app.py``.
    """
    if _is_truthy_env("PRIMA_DISABLE_GRADIO_EXAMPLES"):
        return []
    rows: List[List] = []
    template: List[Tuple[str, float, int, float, float, bool, bool]] = [
        ("demo_data/000000015956_horse.png", 1e-6, 30, 0.7, 0.1, False, True),
        ("demo_data/n02412080_12159.png", 1e-6, 30, 0.7, 0.1, False, True),
        ("demo_data/000000315905_zebra.jpg", 1e-6, 30, 0.7, 0.1, False, True),
        ("demo_data/beagle.jpg", 1e-6, 0, 0.7, 0.1, False, True),
        ("demo_data/shepherd_hati.jpg", 1e-6, 0, 0.7, 0.1, False, True),
    ]
    for rel, *rest in template:
        p = _REPO_ROOT / rel
        if p.is_file():
            rows.append([str(p), *rest])
    return rows


def _should_preload_assets() -> bool:
    """Default to preload on Spaces; configurable via PRIMA_PRELOAD_ASSETS."""
    preload_env = os.environ.get("PRIMA_PRELOAD_ASSETS")
    if preload_env is not None:
        return _is_truthy_env("PRIMA_PRELOAD_ASSETS")
    return _running_on_space()


def _gradio_heartbeat_interval_sec() -> float:
    """How often to yield status while waiting on long CPU/GPU work (keeps WebSockets alive).

    Set ``PRIMA_GRADIO_HEARTBEAT_SEC`` to ``0`` to run long work on the Gradio thread (old behavior).
    """
    raw = os.environ.get("PRIMA_GRADIO_HEARTBEAT_SEC", "25").strip()
    try:
        v = float(raw)
    except ValueError:
        return 25.0
    return max(0.0, v)


def _ensure_demo_assets(checkpoint_path: str) -> None:
    """Download required demo assets when running in a clean environment."""
    from scripts.setup_demo_data import (
        maybe_download_smal,
        maybe_download_backbone,
        maybe_download_stage,
    )

    checkpoint = Path(checkpoint_path)
    data_dir = checkpoint.parents[2]
    hf_repo_id = os.environ.get("PRIMA_HF_REPO_ID", DEFAULT_HF_ASSET_REPO)

    maybe_download_smal(data_dir, force=False, hf_repo_id=hf_repo_id)
    maybe_download_backbone(data_dir, force=False, hf_repo_id=hf_repo_id)
    maybe_download_stage(
        "PRIMAS1",
        "config_s1_HYDRA.yaml",
        "s1ckpt.ckpt",
        "s1ckpt.ckpt",
        data_dir,
        force=False,
        hf_repo_id=hf_repo_id,
    )


def _preload_assets_once(checkpoint_path: str) -> None:
    checkpoint = Path(checkpoint_path)
    cfg_path = checkpoint.parent.parent / ".hydra" / "config.yaml"
    if checkpoint.exists() and cfg_path.exists():
        print("[startup] Assets already present; skipping preload.")
        return
    print("[startup] Preloading demo assets from Hugging Face Hub...")
    _ensure_demo_assets(checkpoint_path)
    print("[startup] Asset preload complete.")


def _load_prima_model(checkpoint_path: str = DEFAULT_CHECKPOINT):
    """Load PRIMA model and renderer once for the Gradio app."""
    from prima.models import load_prima
    from prima.utils.renderer import Renderer

    checkpoint = Path(checkpoint_path)
    cfg_path = checkpoint.parent.parent / ".hydra" / "config.yaml"
    if not checkpoint.exists() or not cfg_path.exists():
        _ensure_demo_assets(checkpoint_path)
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Missing checkpoint: {checkpoint}. Download demo checkpoints/data as described in README."
        )
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"Missing model config: {cfg_path}. Ensure the full checkpoint folder layout from README is present."
        )

    model, model_cfg = load_prima(checkpoint_path)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    model.eval()

    renderer = Renderer(model_cfg, faces=model.smal.faces)
    return model, model_cfg, renderer, device


def _build_detector():
    """Build Detectron2 animal detector (same config as demo_tta/demo.py)."""
    try:
        import detectron2.config
        import detectron2.engine
        from detectron2 import model_zoo
    except Exception as e:
        print(f"[warn] Detectron2 unavailable ({type(e).__name__}: {e}); using full-image fallback bbox.")
        return None

    cfg = detectron2.config.get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")
    )
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.WEIGHTS = (
        "https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/"
        "faster_rcnn_X_101_32x8d_FPN_3x/139173657/model_final_68b088.pkl"
    )
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    detector = detectron2.engine.DefaultPredictor(cfg)
    return detector


def _load_model_and_detector_for_demo(checkpoint_path: str):
    """Run on a worker thread when using heartbeat polling (single entry point for executor)."""
    model, model_cfg, renderer, device = _load_prima_model(checkpoint_path)
    detector = _build_detector()
    return model, model_cfg, renderer, device, detector


# SuperAnimal defaults (same as in demo_tta parser)
SUPER_ANIMAL_ARGS = SimpleNamespace(
    superanimal_name="superanimal_quadruped",
    superanimal_model_name="hrnet_w32",
    superanimal_detector_name="fasterrcnn_resnet50_fpn_v2",
    superanimal_max_individuals=1,
)


def _collect_animal_results(
    model,
    model_cfg,
    renderer,
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
        ANIMAL_COCO_IDS,
        denorm_patch_to_rgb,
        map_superanimal_to_prima,
        run_superanimal_on_patch,
        save_keypoint_vis,
        tta_optimize,
    )

    # Detect animals
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    if detector is None:
        # Fallback for environments where Detectron2 is unavailable: process full image as one crop.
        h, w = img_bgr.shape[:2]
        boxes = np.array([[0.0, 0.0, float(max(1, w - 1)), float(max(1, h - 1))]], dtype=np.float32)
    else:
        det_out = detector(img_bgr)
        det_instances = det_out["instances"]

        valid_idx = [
            i
            for i, (c, s) in enumerate(zip(det_instances.pred_classes, det_instances.scores))
            if (int(c) in ANIMAL_COCO_IDS) and (float(s) > float(det_thresh))
        ]
        if len(valid_idx) == 0:
            return [], [], [], None, None

        boxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()

    dataset = ViTDetDataset(model_cfg, img_bgr, boxes)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    before_imgs: List[np.ndarray] = []
    after_imgs: List[np.ndarray] = []
    kpt_imgs: List[np.ndarray] = []
    before_mesh_paths: List[str] = []
    after_mesh_paths: List[str] = []

    img_token = next(tempfile._get_candidate_names())

    for batch in dataloader:
        batch = recursive_to(batch, device)

        with torch.no_grad():
            out_before = model(batch)

        animal_id = int(batch["animalid"][0])

        # Save/render before TTA
        img_fn = f"{img_token}"
        from demo_tta import render_and_save  # imported lazily to avoid circular issues

        render_and_save(
            renderer,
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
            render_and_save(
                renderer,
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
        patch_rgb = denorm_patch_to_rgb(batch["img"][0])
        with tempfile.TemporaryDirectory(prefix=f"dlc_{img_fn}_{animal_id}_") as tmp_dir:
            bodyparts_xyc = run_superanimal_on_patch(patch_rgb, SUPER_ANIMAL_ARGS, tmp_dir)

        if bodyparts_xyc is None:
            # No keypoints => skip TTA for this animal
            continue

        mapped_xyc = map_superanimal_to_prima(bodyparts_xyc)
        mapped_xyc[mapped_xyc[:, 2] < float(kp_conf_thresh), 2] = 0.0

        # Save keypoint visualization and npy
        kpt_png_path = os.path.join(out_folder, f"{img_fn}_{animal_id}_prima26_kpts.png")
        save_keypoint_vis(patch_rgb, mapped_xyc, kpt_png_path)
        npy_path = os.path.join(out_folder, f"{img_fn}_{animal_id}_prima26_kpts.npy")
        np.save(npy_path, mapped_xyc)

        if os.path.exists(kpt_png_path):
            kpt_bgr = cv2.imread(kpt_png_path)
            if kpt_bgr is not None:
                kpt_imgs.append(cv2.cvtColor(kpt_bgr, cv2.COLOR_BGR2RGB))

        # Normalize keypoints to [-0.5, 0.5] as in demo_tta
        patch_h, patch_w = patch_rgb.shape[:2]
        mapped_norm = mapped_xyc.copy()
        mapped_norm[:, 0] = mapped_norm[:, 0] / float(patch_w) - 0.5
        mapped_norm[:, 1] = mapped_norm[:, 1] / float(patch_h) - 0.5
        gt_kpts_norm = torch.from_numpy(mapped_norm[None]).to(device=device, dtype=batch["img"].dtype)

        # Run TTA
        out_after = tta_optimize(
            model,
            batch,
            gt_kpts_norm,
            num_iters=int(tta_num_iters),
            lr=float(tta_lr),
        )

        render_and_save(
            renderer,
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

    return before_imgs, after_imgs, kpt_imgs, first_before_mesh, first_after_mesh


def build_demo(checkpoint_path: str = DEFAULT_CHECKPOINT, out_folder: str = DEFAULT_OUT_FOLDER) -> gr.Interface:
    os.makedirs(out_folder, exist_ok=True)
    runtime_cache = {
        "model": None,
        "model_cfg": None,
        "renderer": None,
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

        if image.dtype != np.uint8:
            img_rgb = np.clip(image, 0, 255).astype(np.uint8)
        else:
            img_rgb = image

        yield None, None, None, "Queued; preparing run…"

        hb = _gradio_heartbeat_interval_sec()

        if runtime_cache["model"] is None:
            yield (
                None,
                None,
                None,
                "First run: downloading demo assets from Hugging Face (large checkpoint) "
                "and loading the model. This can take many minutes; status updates here "
                "mean the session is still alive.",
            )
            try:
                if hb <= 0:
                    model, model_cfg, renderer, device, detector = _load_model_and_detector_for_demo(
                        checkpoint_path
                    )
                else:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                        fut = pool.submit(_load_model_and_detector_for_demo, checkpoint_path)
                        t0 = time.monotonic()
                        while True:
                            try:
                                model, model_cfg, renderer, device, detector = fut.result(timeout=hb)
                                break
                            except concurrent.futures.TimeoutError:
                                elapsed = int(time.monotonic() - t0)
                                yield None, None, None, (
                                    f"First run: still loading model and assets ({elapsed}s). "
                                    f"Updates every ~{int(hb)}s keep the browser connection open on Spaces."
                                )
            except Exception:
                yield None, None, None, f"Model initialization failed:\n{traceback.format_exc()}"
                return
            runtime_cache["model"] = model
            runtime_cache["model_cfg"] = model_cfg
            runtime_cache["renderer"] = renderer
            runtime_cache["device"] = device
            runtime_cache["detector"] = detector
            yield None, None, None, "Model loaded. Running detection and inference…"

        try:
            if hb <= 0:
                before_imgs, after_imgs, kpt_imgs, mesh_before, mesh_after = _collect_animal_results(
                    runtime_cache["model"],
                    runtime_cache["model_cfg"],
                    runtime_cache["renderer"],
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
                )
            else:
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    fut = pool.submit(
                        _collect_animal_results,
                        runtime_cache["model"],
                        runtime_cache["model_cfg"],
                        runtime_cache["renderer"],
                        runtime_cache["device"],
                        runtime_cache["detector"],
                        out_folder,
                        img_rgb,
                        tta_lr,
                        tta_num_iters,
                        det_thresh,
                        kp_conf_thresh,
                        side_view,
                        save_mesh,
                    )
                    t0 = time.monotonic()
                    while True:
                        try:
                            before_imgs, after_imgs, kpt_imgs, mesh_before, mesh_after = fut.result(
                                timeout=hb
                            )
                            break
                        except concurrent.futures.TimeoutError:
                            elapsed = int(time.monotonic() - t0)
                            yield None, None, None, (
                                f"Inference still running ({elapsed}s). "
                                f"Detection, SuperAnimal, and TTA can take several minutes; "
                                f"updates every ~{int(hb)}s keep the connection alive."
                            )
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

    _gradio_examples = _gradio_examples_for_interface()
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
                maximum=100,
                value=30,
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
            gr.Checkbox(label="Render side view", value=False),
            gr.Checkbox(label="Save meshes (.obj)", value=True),
        ],
        outputs=[
            gr.Image(label="Before TTA"),
            gr.Image(label="After TTA"),
            gr.Image(label="PRIMA 26 keypoints"),
            gr.Textbox(label="Status / Traceback", lines=12),
        ],
        title="PRIMA: Boosting Animal Mesh Recovery with Biological Priors and Test-Time Adaptation",
        description=(
            "Upload an animal image. The demo runs Detectron2 for animal detection, "
            "PRIMA for 3D pose/shape, DeepLabCut SuperAnimal for 2D keypoints, and "
            "test-time adaptation (TTA) with configurable learning rate and iterations. "
            "Set TTA iterations to 0 to disable adaptation.\n\n"
            "Results (PNG/OBJ and 26-keypoint visualizations) are saved under "
            f"'{out_folder}'."
        ),
    )
    if _gradio_examples:
        _iface_kw["examples"] = _gradio_examples
    demo = gr.Interface(**_iface_kw)
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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if _should_preload_assets():
        _preload_assets_once(args.checkpoint)
    demo = build_demo(checkpoint_path=args.checkpoint, out_folder=args.out_folder)
    demo.launch(inbrowser=False)
