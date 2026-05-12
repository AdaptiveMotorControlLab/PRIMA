"""
PRIMA: Boosting Animal Mesh Recovery with Biological Priors and Test-Time Adaptation

Official implementation of the paper:
"PRIMA: Boosting Animal Mesh Recovery with Biological Priors and Test-Time Adaptation"
by Xiaohang Yu, Ti Wang, and Mackenzie Weygandt Mathis
Licensed under a modified MIT license
"""

"""
demo_tta.py: PRIMA inference with DeepLabCut SuperAnimal TTA

Pipeline:
1. Run Detectron2 to detect animals in the input image.
2. Run PRIMA on each detected animal to obtain 3D pose/shape estimation.
3. Run DeepLabCut SuperAnimal to obtain 2D keypoint estimation.
4. Map the 39 SuperAnimal keypoints to the 26 PRIMA keypoints.
5. Run test-time adaptation (TTA) with user-specified lr and num_iters
   to further optimize the 3D pose and shape estimation.
6. Render and save before/after TTA results (PNG + OBJ) and the
   26-keypoint visualization (PNG).

Reference code:
- Test-time adaptation: prima/../eval_with_tta.py
- DeepLabCut: https://github.com/AdaptiveMotorControlLab/FMPose3D/blob/main/animals/demo/vis_animals.py
- Keypoint mapping (SuperAnimal 39 → PRIMA 26):
    keypoint_mapping = {"quadruped80k":[10, 5, -1, 26, 29, 30, 35, 22, 24, 27, 31, 32, -1, -1,
                                     25, 28, 33, 34, 15, 23, 11, 6, 4, 3, 0, -1]}
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
from prima.utils.renderer import Renderer, cam_crop_to_full

warnings.filterwarnings("ignore")

LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)
GREEN = (0.65, 0.86, 0.74)

ANIMAL_COCO_IDS = [15, 16, 17, 18, 19, 21, 22]
keypoint_mapping = {
    "quadruped80k": [10, 5, -1, 26, 29, 30, 35, 22, 24, 27, 31, 32, -1, -1, 25, 28, 33, 34, 15, 23, 11, 6, 4, 3, 0, -1]
}


def denorm_patch_to_rgb(img_tensor: torch.Tensor) -> np.ndarray:
    patch = (img_tensor.detach().cpu() * (DEFAULT_STD[:, None, None]) + DEFAULT_MEAN[:, None, None]) / 255.0
    patch = patch.permute(1, 2, 0).numpy()
    return np.clip(patch, 0.0, 1.0)


def map_superanimal_to_prima(bodyparts_xyc: np.ndarray) -> np.ndarray:
    mapping = keypoint_mapping["quadruped80k"]
    num_src = bodyparts_xyc.shape[0]
    mapped = np.zeros((len(mapping), 3), dtype=np.float32)

    for tgt_i, src_i in enumerate(mapping):
        if src_i >= 0 and src_i < num_src:
            mapped[tgt_i] = bodyparts_xyc[src_i]

    return mapped


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


def run_superanimal_on_patch(patch_rgb: np.ndarray, args, tmp_dir: str):
    try:
        from deeplabcut.pose_estimation_pytorch.apis import superanimal_analyze_images
    except Exception as e:
        raise RuntimeError(
            "Cannot import DeepLabCut SuperAnimal API. Please install deeplabcut with pose_estimation_pytorch support."
        ) from e

    patch_path = os.path.join(tmp_dir, "patch.png")
    cv2.imwrite(patch_path, cv2.cvtColor((patch_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

    preds = superanimal_analyze_images(
        args.superanimal_name,
        args.superanimal_model_name,
        args.superanimal_detector_name,
        patch_path,
        args.superanimal_max_individuals,
        out_folder=tmp_dir,
    )

    payload = preds.get(patch_path, None)
    if payload is None:
        return None
    bodyparts = payload.get("bodyparts", None)
    if bodyparts is None or len(bodyparts) == 0:
        return None

    best_idx = int(np.argmax(bodyparts[..., 2].mean(axis=1)))
    return bodyparts[best_idx]


def render_and_save(renderer, out, batch, img_fn, animal_id, out_folder, suffix, side_view, save_mesh):
    pred_cam = out['pred_cam']
    box_center = batch['box_center'].float()
    box_size = batch['box_size'].float()
    img_size = batch['img_size'].float()
    scaled_focal_length = batch['focal_length'][0, 0] / batch['img'].shape[-1] * img_size.max()
    pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length)

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
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to pretrained PRIMA checkpoint')
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

    args = parser.parse_args()

    model, model_cfg = load_prima(args.checkpoint)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    model.eval()

    renderer = Renderer(model_cfg, faces=model.smal.faces)
    os.makedirs(args.out_folder, exist_ok=True)

    import detectron2.config
    import detectron2.engine
    from detectron2 import model_zoo

    cfg = detectron2.config.get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.WEIGHTS = "https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x/139173657/model_final_68b088.pkl"
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
        valid_idx = [
            i for i, (c, s) in enumerate(zip(det_instances.pred_classes, det_instances.scores))
            if (int(c) in ANIMAL_COCO_IDS) and (float(s) > args.det_thresh)
        ]
        boxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()

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
                bodyparts_xyc = run_superanimal_on_patch(patch_rgb, args, tmp_dir)

            if bodyparts_xyc is None:
                print(f"[WARN] No SuperAnimal keypoints for {img_fn}_{animal_id}, skip TTA")
                continue

            mapped_xyc = map_superanimal_to_prima(bodyparts_xyc)
            mapped_xyc[mapped_xyc[:, 2] < args.kp_conf_thresh, 2] = 0.0

            save_keypoint_vis(
                patch_rgb,
                mapped_xyc,
                os.path.join(args.out_folder, f"{img_fn}_{animal_id}_prima26_kpts.png"),
            )
            np.save(os.path.join(args.out_folder, f"{img_fn}_{animal_id}_prima26_kpts.npy"), mapped_xyc)

            patch_h, patch_w = patch_rgb.shape[:2]
            mapped_norm = mapped_xyc.copy()
            mapped_norm[:, 0] = mapped_norm[:, 0] / float(patch_w) - 0.5
            mapped_norm[:, 1] = mapped_norm[:, 1] / float(patch_h) - 0.5
            gt_kpts_norm = torch.from_numpy(mapped_norm[None]).to(device=device, dtype=batch['img'].dtype)

            out_after = tta_optimize(
                model,
                batch,
                gt_kpts_norm,
                num_iters=args.tta_num_iters,
                lr=args.tta_lr,
            )

            render_and_save(
                renderer,
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

