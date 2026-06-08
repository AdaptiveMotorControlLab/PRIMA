"""
PRIMA video demo: run Stage 1 inference frame-by-frame on a video.

For each frame, Detectron2 animal detections are filtered by class and score,
then only the highest-confidence bounding box is passed to PRIMA.
"""

from pathlib import Path
import argparse
import os
import warnings

import cv2
import detectron2
import detectron2.config
import detectron2.engine
import numpy as np
import torch
import torch.utils
import torch.utils.data
from detectron2 import model_zoo
from tqdm import tqdm

from prima.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
from prima.models import load_prima
from prima.utils import recursive_to
from prima.utils.detection import ANIMAL_COCO_IDS
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


def select_top_confidence_animal_box(det_instances, score_threshold=0.7):
    classes = det_instances.pred_classes.detach().cpu().numpy()
    scores = det_instances.scores.detach().cpu().numpy()
    class_ids = set(int(class_id) for class_id in ANIMAL_COCO_IDS)
    valid_idx = np.array(
        [
            i
            for i, (class_id, score) in enumerate(zip(classes, scores))
            if int(class_id) in class_ids and float(score) > float(score_threshold)
        ],
        dtype=np.int64,
    )
    if len(valid_idx) == 0:
        return np.zeros((0, 4), dtype=np.float32), None

    top_idx = valid_idx[int(np.argmax(scores[valid_idx]))]
    box = det_instances.pred_boxes.tensor[top_idx].detach().cpu().numpy().astype(np.float32)
    return box[None], float(scores[top_idx])


def depth_to_viridis_rgb(depth_img):
    valid_mask = depth_img > 0
    if np.sum(valid_mask) == 0:
        depth_norm = np.zeros_like(depth_img)
    else:
        min_val = np.min(depth_img[valid_mask])
        max_val = np.max(depth_img[valid_mask])
        if min_val == max_val:
            depth_norm = np.zeros_like(depth_img)
        else:
            depth_norm = (depth_img - min_val) / (max_val - min_val + 1e-8)
    depth_norm[~valid_mask] = 0

    depth_vis = (depth_norm * 255).astype(np.uint8)
    depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_VIRIDIS)
    depth_vis = cv2.cvtColor(depth_vis, cv2.COLOR_BGR2RGB)
    depth_vis = depth_vis.astype(np.float32) / 255.0
    depth_vis[~valid_mask] = 0
    return depth_vis


def make_empty_output_frame(frame_bgr, img_res, num_panels):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    first_panel = cv2.resize(frame_rgb, (img_res, img_res)).astype(np.float32) / 255.0
    blank_panel = np.ones((img_res, img_res, 3), dtype=np.float32)
    panels = [first_panel] + [blank_panel.copy() for _ in range(num_panels - 1)]
    return np.concatenate(panels, axis=1)


def make_full_frame_output(frame_bgr):
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def get_video_rotation(cap, rotate_arg):
    if rotate_arg != "auto":
        return rotate_arg
    orientation_prop = getattr(cv2, "CAP_PROP_ORIENTATION_META", None)
    if orientation_prop is None:
        return "none"
    orientation = int(cap.get(orientation_prop) or 0) % 360
    if orientation == 90:
        return "90cw"
    if orientation == 180:
        return "180"
    if orientation == 270:
        return "90ccw"
    return "none"


def rotate_frame(frame_bgr, rotation):
    if rotation == "90cw":
        return cv2.rotate(frame_bgr, cv2.ROTATE_90_CLOCKWISE)
    if rotation == "90ccw":
        return cv2.rotate(frame_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if rotation == "180":
        return cv2.rotate(frame_bgr, cv2.ROTATE_180)
    return frame_bgr


def main():
    parser = argparse.ArgumentParser(description="PRIMA video demo")
    parser.add_argument("--checkpoint", type=str, default="",
                        help="Path to pretrained model checkpoint. Empty -> auto-download the default Stage 1 checkpoint.")
    parser.add_argument("--hf-repo-id", "--hf_repo_id", dest="hf_repo_id",
                        type=str, default=os.environ.get("PRIMA_HF_REPO_ID", DEFAULT_HF_REPO_ID),
                        help="Hugging Face repo ID containing PRIMA demo assets")
    parser.add_argument("--no-auto-download", "--no_auto_download", dest="no_auto_download", action="store_true",
                        help="Disable automatic download of missing PRIMA demo assets")
    parser.add_argument("--video_path", type=str, required=True, help="Input video path")
    parser.add_argument("--out_video", type=str, default="demo_video_out.mp4", help="Output rendered video path")
    parser.add_argument("--out_folder", type=str, default="demo_video_out", help="Output folder for optional meshes")
    parser.add_argument("--det_thresh", type=float, default=0.7, help="Animal detection confidence threshold")
    parser.add_argument("--side_view", dest="side_view", action="store_true", default=False,
                        help="If set, render side view also")
    parser.add_argument("--render_depth", dest="render_depth", action="store_true", default=False,
                        help="If set, render depth map also")
    parser.add_argument("--full_frame", dest="full_frame", action="store_true", default=False,
                        help="Render the mesh overlay on the full video frame instead of crop-panel output")
    parser.add_argument("--save_mesh", dest="save_mesh", action="store_true", default=False,
                        help="If set, save one mesh per processed frame")
    parser.add_argument("--max_frames", type=int, default=-1,
                        help="Maximum number of frames to process. Use -1 for the full video.")
    parser.add_argument("--frame_stride", type=int, default=1,
                        help="Process every Nth frame. Output video contains processed frames only.")
    parser.add_argument("--rotate", type=str, default="auto",
                        choices=["auto", "none", "90cw", "90ccw", "180"],
                        help="Rotate input frames before detection/rendering. "
                             "auto uses video orientation metadata when OpenCV exposes it.")

    args = parser.parse_args()
    os.makedirs(args.out_folder, exist_ok=True)
    out_video_parent = os.path.dirname(args.out_video)
    if out_video_parent:
        os.makedirs(out_video_parent, exist_ok=True)

    checkpoint_path = resolve_prima_checkpoint_path(
        args.checkpoint,
        data_dir=REPO_ROOT / "data",
        auto_download=not args.no_auto_download,
        hf_repo_id=args.hf_repo_id,
    )

    model, model_cfg = load_prima(checkpoint_path)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    model.eval()

    Renderer, cam_crop_to_full = load_renderer_components()
    renderer = Renderer(model_cfg, faces=model.smal.faces)

    cfg = detectron2.config.get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.WEIGHTS = "https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x/139173657/model_final_68b088.pkl"
    cfg.MODEL.DEVICE = device.type
    detector = detectron2.engine.DefaultPredictor(cfg)

    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video_path}")
    orientation_auto_prop = getattr(cv2, "CAP_PROP_ORIENTATION_AUTO", None)
    if orientation_auto_prop is not None:
        cap.set(orientation_auto_prop, 0)
    frame_rotation = get_video_rotation(cap, args.rotate)
    print(f"[video] frame rotation: {frame_rotation}")

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    fps = src_fps / max(1, args.frame_stride) if src_fps and src_fps > 0 else 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if args.max_frames > 0:
        total_steps = min(total_frames, args.max_frames * max(1, args.frame_stride))
    else:
        total_steps = total_frames

    img_res = int(model_cfg.MODEL.IMAGE_SIZE)
    num_panels = 2 + int(args.side_view) + int(args.render_depth)
    writer = None
    out_size = None

    video_stem = Path(args.video_path).stem
    frame_idx = 0
    processed_frames = 0
    rendered_frames = 0
    skipped_frames = 0

    pbar = tqdm(total=total_steps if total_steps > 0 else None, desc="Processing video")
    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            frame_bgr = rotate_frame(frame_bgr, frame_rotation)

            if args.max_frames > 0 and processed_frames >= args.max_frames:
                break

            should_process = frame_idx % max(1, args.frame_stride) == 0
            if not should_process:
                frame_idx += 1
                pbar.update(1)
                continue

            det_out = detector(frame_bgr)
            boxes, top_score = select_top_confidence_animal_box(
                det_out["instances"],
                score_threshold=args.det_thresh,
            )

            if len(boxes) == 0:
                if args.full_frame:
                    final_img = make_full_frame_output(frame_bgr)
                else:
                    final_img = make_empty_output_frame(frame_bgr, img_res, num_panels)
                skipped_frames += 1
            else:
                dataset = ViTDetDataset(model_cfg, frame_bgr, boxes)
                batch = next(iter(torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)))
                batch = recursive_to(batch, device)
                with torch.no_grad():
                    out = model(batch)

                pred_cam = out["pred_cam"]
                box_center = batch["box_center"].float()
                box_size = batch["box_size"].float()
                img_size = batch["img_size"].float()
                scaled_focal_length = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
                pred_cam_t_full = cam_crop_to_full(
                    pred_cam,
                    box_center,
                    box_size,
                    img_size,
                    scaled_focal_length,
                ).detach().cpu().numpy()

                if args.full_frame:
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    final_img = renderer(
                        out["pred_vertices"][0].detach().cpu().numpy(),
                        pred_cam_t_full[0],
                        frame_rgb,
                        full_frame=True,
                        mesh_base_color=GREEN,
                        scene_bg_color=(1, 1, 1),
                        focal_length=float(scaled_focal_length.detach().cpu().numpy()),
                    )
                else:
                    white_img = (torch.ones_like(batch["img"][0]).cpu() - DEFAULT_MEAN[:, None, None] / 255) / (
                        DEFAULT_STD[:, None, None] / 255
                    )
                    input_patch = (
                        batch["img"][0].cpu() * DEFAULT_STD[:, None, None] + DEFAULT_MEAN[:, None, None]
                    ) / 255.0
                    input_patch = input_patch.permute(1, 2, 0).numpy()

                    regression_img = renderer(
                        out["pred_vertices"][0].detach().cpu().numpy(),
                        out["pred_cam_t"][0].detach().cpu().numpy(),
                        batch["img"][0],
                        mesh_base_color=GREEN,
                        scene_bg_color=(1, 1, 1),
                    )
                    final_img = np.concatenate([input_patch, regression_img], axis=1)

                    if args.side_view:
                        side_img = renderer(
                            out["pred_vertices"][0].detach().cpu().numpy(),
                            out["pred_cam_t"][0].detach().cpu().numpy(),
                            white_img,
                            mesh_base_color=GREEN,
                            scene_bg_color=(1, 1, 1),
                            side_view=True,
                        )
                        final_img = np.concatenate([final_img, side_img], axis=1)

                    if args.render_depth:
                        depth_img = renderer(
                            out["pred_vertices"][0].detach().cpu().numpy(),
                            out["pred_cam_t"][0].detach().cpu().numpy(),
                            white_img,
                            mesh_base_color=GREEN,
                            scene_bg_color=(1, 1, 1),
                            depth=True,
                        )
                        final_img = np.concatenate([final_img, depth_to_viridis_rgb(depth_img)], axis=1)

                if args.save_mesh:
                    verts = out["pred_vertices"][0].detach().cpu().numpy()
                    cam_t = pred_cam_t_full[0]
                    tmesh = renderer.vertices_to_trimesh(verts, cam_t.copy(), LIGHT_BLUE)
                    mesh_name = f"{video_stem}_frame{frame_idx:06d}_score{top_score:.3f}.obj"
                    tmesh.export(os.path.join(args.out_folder, mesh_name))

                rendered_frames += 1

            frame_out = cv2.cvtColor((255 * final_img).astype(np.uint8), cv2.COLOR_RGB2BGR)
            if writer is None:
                out_size = (frame_out.shape[1], frame_out.shape[0])
                writer = cv2.VideoWriter(args.out_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, out_size)
                if not writer.isOpened():
                    raise RuntimeError(f"Cannot open output video writer: {args.out_video}")
            elif (frame_out.shape[1], frame_out.shape[0]) != out_size:
                frame_out = cv2.resize(frame_out, out_size)
            writer.write(frame_out)

            processed_frames += 1
            frame_idx += 1
            pbar.update(1)
    finally:
        pbar.close()
        cap.release()
        if writer is not None:
            writer.release()

    print(
        f"[done] Processed {processed_frames} frame(s) from {args.video_path}; "
        f"rendered {rendered_frames}, no-detection placeholders {skipped_frames}. "
        f"Saved video to {args.out_video}."
    )


if __name__ == "__main__":
    main()
