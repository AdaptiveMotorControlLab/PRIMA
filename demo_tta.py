"""
PRIMA: Boosting Animal Mesh Recovery with Biological Priors and Test-Time Adaptation

Official implementation of the paper:
"PRIMA: Boosting Animal Mesh Recovery with Biological Priors and Test-Time Adaptation"
by Xiaohang Yu, Ti Wang, and Mackenzie Weygandt Mathis
Licensed under a modified MIT license

demo_tta.py: PRIMA inference with fine-tuned DeepLabCut SuperAnimal TTA

Pipeline:
1. Run Detectron2 to detect animals in the input image.
2. Run PRIMA on each detected animal to obtain 3D pose/shape estimation.
3. Run a fine-tuned DeepLabCut SuperAnimal pose model (Animal3D 26-joint
   layout) to obtain 2D keypoints already in PRIMA topology -- no 39 -> 26
   remapping needed. The fine-tuned snapshot is wired into DLC's
   ``superanimal_analyze_images`` via the ``customized_pose_checkpoint``
   and ``customized_model_config`` kwargs.
4. Run test-time adaptation (TTA) with user-specified lr and num_iters
   to further optimize the 3D pose and shape estimation.
5. Render and save before/after TTA results (PNG + OBJ) and the
   26-keypoint visualization (PNG).
"""


from pathlib import Path
import argparse
import copy
import os
import tempfile
import warnings

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
from tqdm import tqdm

from prima.models import load_prima
from prima.utils import recursive_to
from prima.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
from prima.utils.detection import ANIMAL_COCO_IDS, select_animal_boxes
from prima.utils.weights import DEFAULT_HF_REPO_ID, resolve_prima_checkpoint_path

warnings.filterwarnings("ignore")

LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)
GREEN = (0.65, 0.86, 0.74)

REPO_ROOT = Path(__file__).resolve().parent


def load_renderer_components():
    try:
        from prima.utils.renderer import Renderer, cam_crop_to_full
    except Exception as exc:
        raise RuntimeError(
            "Cannot initialize the PRIMA renderer. Rendering requires a working "
            "pyrender/OpenGL backend such as EGL or OSMesa. Install the missing "
            "OpenGL runtime for this environment, or run in an environment where "
            "PYOPENGL_PLATFORM=egl/osmesa works."
        ) from exc
    return Renderer, cam_crop_to_full


def denorm_patch_to_rgb(img_tensor: torch.Tensor) -> np.ndarray:
    patch = (img_tensor.detach().cpu() * (DEFAULT_STD[:, None, None]) + DEFAULT_MEAN[:, None, None]) / 255.0
    patch = patch.permute(1, 2, 0).numpy()
    return np.clip(patch, 0.0, 1.0)


def save_keypoint_vis(patch_rgb: np.ndarray, kpts_xyc: np.ndarray, save_path: str) -> None:
    vis = cv2.cvtColor((patch_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR).copy()
    num_kpts = len(kpts_xyc)

    for i, (x, y, c) in enumerate(kpts_xyc):
        if c <= 0:
            continue

        # Use distinct color for each keypoint (OpenCV uses BGR)
        hue = int(179 * i / max(1, num_kpts - 1))
        color_bgr = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]
        color_bgr = (int(color_bgr[0]), int(color_bgr[1]), int(color_bgr[2]))

        cx, cy = int(round(float(x))), int(round(float(y)))
        cv2.circle(vis, (cx, cy), 3, color_bgr, -1)
        cv2.putText(vis, str(i), (cx + 3, cy - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imwrite(save_path, vis)


def resolve_sa_weights_path(local_path: str) -> str:
    """Return a local path to the fine-tuned SuperAnimal .pt snapshot.

    If ``local_path`` is empty, downloads ``sa_finetune_hrnet_w32.pt`` from the
    ``MLAdaptiveIntelligence/FMPose3D`` Hugging Face repo (cached under
    ``~/.cache/huggingface``).
    """
    if local_path:
        return local_path
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError(
            "huggingface_hub is required to auto-download the fine-tuned "
            "SuperAnimal weights. Install with `pip install huggingface_hub`, "
            "or pass --saved_2d_model_path with a local .pt file."
        ) from None
    repo_id = "MLAdaptiveIntelligence/FMPose3D"
    filename = "sa_finetune_hrnet_w32.pt"
    try:
        cached_path = hf_hub_download(repo_id=repo_id, filename=filename, local_files_only=True)
    except Exception:
        print(f"No --saved_2d_model_path provided; downloading '{filename}' from {repo_id}...")
        return hf_hub_download(repo_id=repo_id, filename=filename)

    print(f"Using cached SuperAnimal weights: {cached_path}")
    return cached_path


def run_superanimal_on_patch(patch_rgb: np.ndarray, args, tmp_dir: str):
    """Predict 26-joint 2D keypoints on a single PRIMA patch using a
    fine-tuned DeepLabCut SuperAnimal snapshot.

    Returns an ``(26, 3)`` array of ``(x, y, confidence)`` in patch
    pixel coordinates, or ``None`` if no individual was detected.
    """
    try:
        from deeplabcut.pose_estimation_pytorch.apis import superanimal_analyze_images
    except Exception as e:
        raise RuntimeError(
            "Cannot import DeepLabCut SuperAnimal API. Please install deeplabcut with pose_estimation_pytorch support."
        ) from e

    patch_path = os.path.join(tmp_dir, "patch.png")
    cv2.imwrite(patch_path, cv2.cvtColor((patch_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

    preds = superanimal_analyze_images(
        superanimal_name=args.superanimal_name,
        model_name=args.superanimal_model_name,
        detector_name=args.superanimal_detector_name,
        images=patch_path,
        max_individuals=args.superanimal_max_individuals,
        out_folder=tmp_dir,
        customized_model_config=args.pytorch_config_2d_path,
        customized_pose_checkpoint=args.saved_2d_model_path,
    )

    payload = preds.get(patch_path, None)
    if payload is None:
        return None
    bodyparts = payload.get("bodyparts", None)
    if bodyparts is None or len(bodyparts) == 0:
        return None

    best_idx = int(np.argmax(bodyparts[..., 2].mean(axis=1)))
    return bodyparts[best_idx].astype(np.float32)


def render_and_save(renderer, cam_crop_to_full_fn, out, batch, img_fn, animal_id, out_folder, suffix, side_view, save_mesh):
    pred_cam = out['pred_cam']
    box_center = batch['box_center'].float()
    box_size = batch['box_size'].float()
    img_size = batch['img_size'].float()
    scaled_focal_length = batch['focal_length'][0, 0] / batch['img'].shape[-1] * img_size.max()
    pred_cam_t_full = cam_crop_to_full_fn(pred_cam, box_center, box_size, img_size, scaled_focal_length)

    white_img = (torch.ones_like(batch['img'][0]).cpu() - DEFAULT_MEAN[:, None, None] / 255) / (
        DEFAULT_STD[:, None, None] / 255
    )
    input_patch = denorm_patch_to_rgb(batch['img'][0])

    regression_img = renderer(
        out['pred_vertices'][0].detach().cpu().numpy(),
        out['pred_cam_t'][0].detach().cpu().numpy(),
        batch['img'][0],
        mesh_base_color=GREEN,
        scene_bg_color=(1, 1, 1),
    )

    final_img = np.concatenate([input_patch, regression_img], axis=1)
    if side_view:
        side_img = renderer(
            out['pred_vertices'][0].detach().cpu().numpy(),
            out['pred_cam_t'][0].detach().cpu().numpy(),
            white_img,
            mesh_base_color=GREEN,
            scene_bg_color=(1, 1, 1),
            side_view=True,
        )
        final_img = np.concatenate([final_img, side_img], axis=1)

    cv2.imwrite(
        os.path.join(out_folder, f'{img_fn}_{animal_id}_{suffix}.png'),
        cv2.cvtColor((255 * final_img).astype(np.uint8), cv2.COLOR_RGB2BGR),
    )

    if save_mesh:
        verts = out['pred_vertices'][0].detach().cpu().numpy()
        cam_t = pred_cam_t_full[0].detach().cpu().numpy()
        tmesh = renderer.vertices_to_trimesh(verts, cam_t.copy(), LIGHT_BLUE)
        tmesh.export(os.path.join(out_folder, f'{img_fn}_{animal_id}_{suffix}.obj'))


def tta_optimize(model, batch, gt_kpts_norm, num_iters, lr):
    model.eval()

    if hasattr(model, 'backbone'):
        for p in model.backbone.parameters():
            p.requires_grad = False

    orig_smal_head_state = copy.deepcopy(model.smal_head.state_dict())
    model.smal_head.freeze_except_regression_heads()
    tta_params = model.smal_head.get_tta_parameters(mode='all')
    optimizer = torch.optim.Adam(tta_params, lr=lr)

    valid_mask = (gt_kpts_norm[..., 2] > 0).float().unsqueeze(-1)
    gt_xy = gt_kpts_norm[..., :2]

    for _ in range(num_iters):
        optimizer.zero_grad()
        out = model(batch)
        pred_xy = out['pred_keypoints_2d']
        loss = F.mse_loss(pred_xy * valid_mask, gt_xy * valid_mask, reduction='sum') / (valid_mask.sum() + 1e-6)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        out_after = model(batch)

    model.smal_head.load_state_dict(orig_smal_head_state)
    model.smal_head.unfreeze_all()

    return out_after


def main():
    parser = argparse.ArgumentParser(description='PRIMA + SuperAnimal + TTA demo')
    parser.add_argument('--checkpoint', type=str, default='',
                        help='Path to pretrained PRIMA checkpoint. Empty -> auto-download the default Stage 1 checkpoint.')
    parser.add_argument('--hf-repo-id', '--hf_repo_id', dest='hf_repo_id',
                        type=str, default=os.environ.get("PRIMA_HF_REPO_ID", DEFAULT_HF_REPO_ID),
                        help='Hugging Face repo ID containing PRIMA demo assets')
    parser.add_argument('--no-auto-download', '--no_auto_download', dest='no_auto_download', action='store_true',
                        help='Disable automatic download of missing PRIMA demo assets')
    parser.add_argument('--img_path', type=str, default=None, help='Single image path')
    parser.add_argument('--img_folder', type=str, default='demo_data/', help='Folder with input images')
    parser.add_argument('--out_folder', type=str, default='demo_out_tta', help='Output folder')
    parser.add_argument('--side_view', dest='side_view', action='store_true', default=False, help='Render side view')
    parser.add_argument('--save_mesh', dest='save_mesh', action='store_true', default=False, help='Save meshes')
    parser.add_argument('--file_type', nargs='+', default=['*.jpg', '*.png', '*.jpeg', '*.JPEG'], help='Image globs')
    parser.add_argument('--det_thresh', type=float, default=0.7, help='Detectron2 score threshold for animals')

    parser.add_argument('--tta_lr', type=float, default=1e-6, help='TTA learning rate')
    parser.add_argument('--tta_num_iters', type=int, default=30, help='TTA iterations')
    parser.add_argument('--kp_conf_thresh', type=float, default=0.1, help='Keypoint confidence threshold')

    parser.add_argument('--superanimal_name', type=str, default='superanimal_quadruped')
    parser.add_argument('--superanimal_model_name', type=str, default='hrnet_w32')
    parser.add_argument('--superanimal_detector_name', type=str, default='fasterrcnn_resnet50_fpn_v2')
    parser.add_argument('--superanimal_max_individuals', type=int, default=1)
    parser.add_argument('--saved_2d_model_path', type=str, default='',
                        help='Path to the fine-tuned SuperAnimal 26-joint .pt snapshot. '
                             'Empty -> auto-download sa_finetune_hrnet_w32.pt from '
                             'MLAdaptiveIntelligence/FMPose3D on Hugging Face Hub.')
    parser.add_argument('--pytorch_config_2d_path', type=str,
                        default=str(Path(__file__).resolve().parent / 'configs' / 'sa_finetune_hrnet_w32.yaml'),
                        help='Path to the DLC pytorch config yaml for the fine-tuned snapshot. '
                             'Defaults to the bundled configs/sa_finetune_hrnet_w32.yaml.')

    args = parser.parse_args()
    checkpoint_path = resolve_prima_checkpoint_path(
        args.checkpoint,
        data_dir=REPO_ROOT / "data",
        auto_download=not args.no_auto_download,
        hf_repo_id=args.hf_repo_id,
    )
    args.saved_2d_model_path = resolve_sa_weights_path(args.saved_2d_model_path)

    model, model_cfg = load_prima(checkpoint_path)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    model.eval()

    Renderer, cam_crop_to_full_fn = load_renderer_components()
    renderer = Renderer(model_cfg, faces=model.smal.faces)
    os.makedirs(args.out_folder, exist_ok=True)

    import detectron2.config
    import detectron2.engine
    from detectron2 import model_zoo

    cfg = detectron2.config.get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.WEIGHTS = "https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x/139173657/model_final_68b088.pkl"
    cfg.MODEL.DEVICE = device.type
    detector = detectron2.engine.DefaultPredictor(cfg)

    if args.img_path is not None:
        img_paths = [Path(args.img_path)]
    else:
        img_paths = sorted([img for end in args.file_type for img in Path(args.img_folder).glob(end)])

    for img_path in img_paths:
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"[WARN] Cannot read image: {img_path}")
            continue
        det_out = detector(img_bgr)
        det_instances = det_out['instances']
        boxes, suppressed = select_animal_boxes(
            det_instances,
            animal_class_ids=ANIMAL_COCO_IDS,
            score_threshold=args.det_thresh,
        )
        if suppressed > 0:
            print(f"[INFO] Suppressed {suppressed} duplicate animal detection(s) in {img_path}")

        if len(boxes) == 0:
            print(f"[INFO] No animal detected in {img_path}")
            continue

        dataset = ViTDetDataset(model_cfg, img_bgr, boxes)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

        for batch in tqdm(dataloader, desc=f"{img_path.name}"):
            batch = recursive_to(batch, device)
            with torch.no_grad():
                out_before = model(batch)

            img_fn = img_path.stem
            animal_id = int(batch['animalid'][0])

            render_and_save(
                renderer,
                cam_crop_to_full_fn,
                out_before,
                batch,
                img_fn,
                animal_id,
                args.out_folder,
                suffix='before_tta',
                side_view=args.side_view,
                save_mesh=args.save_mesh,
            )

            patch_rgb = denorm_patch_to_rgb(batch['img'][0])
            with tempfile.TemporaryDirectory(prefix=f"dlc_{img_fn}_{animal_id}_") as tmp_dir:
                kpts_xyc = run_superanimal_on_patch(patch_rgb, args, tmp_dir)

            if kpts_xyc is None:
                print(f"[WARN] No SuperAnimal keypoints for {img_fn}_{animal_id}, skip TTA")
                continue

            kpts_xyc[kpts_xyc[:, 2] < args.kp_conf_thresh, 2] = 0.0

            save_keypoint_vis(
                patch_rgb,
                kpts_xyc,
                os.path.join(args.out_folder, f"{img_fn}_{animal_id}_prima26_kpts.png"),
            )
            np.save(os.path.join(args.out_folder, f"{img_fn}_{animal_id}_prima26_kpts.npy"), kpts_xyc)

            patch_h, patch_w = patch_rgb.shape[:2]
            kpts_norm = kpts_xyc.copy()
            kpts_norm[:, 0] = kpts_norm[:, 0] / float(patch_w) - 0.5
            kpts_norm[:, 1] = kpts_norm[:, 1] / float(patch_h) - 0.5
            gt_kpts_norm = torch.from_numpy(kpts_norm[None]).to(device=device, dtype=batch['img'].dtype)

            out_after = tta_optimize(
                model,
                batch,
                gt_kpts_norm,
                num_iters=args.tta_num_iters,
                lr=args.tta_lr,
            )

            render_and_save(
                renderer,
                cam_crop_to_full_fn,
                out_after,
                batch,
                img_fn,
                animal_id,
                args.out_folder,
                suffix='after_tta',
                side_view=args.side_view,
                save_mesh=args.save_mesh,
            )


if __name__ == '__main__':
    main()
