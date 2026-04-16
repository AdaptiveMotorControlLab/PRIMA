"""
PRIMA: Boosting Animal Mesh Recovery with Biological Priors and Test-Time Adaptation

Official implementation of the paper:
"PRIMA: Boosting Animal Mesh Recovery with Biological Priors and Test-Time Adaptation"
by Xiaohang Yu, Ti Wang, and Mackenzie Weygandt Mathis
Licensed under a modified MIT license
"""

'''
this scripts if to convert DeepLabCut labeled data format (20 keypoints) to COCO format (26 keypoints ), also image should be extracted from the raw video to save as frames.

Usage:
    python dlc2coco.py --dataset_dir /path/to/dataset --output_dir /path/to/output

for camera x
dlc keypoint data: <dataset_dir>/<behavior>/fte_pw/camx_fte.csv,
    where video frame index from the video and keypoint coordinates are stored
raw video: <dataset_dir>/<behavior>/camx.mp4

for coco format, please refer to:
    ./datasets/quadruped2d/test.json

also, the relationship of multiview should be saved.


keypoint mapping from acinoset to animal3d :
keypoint_mapping = {"acinoset":[2, 1, -1, 13, 10, 19, 16, 5, -1, -1, -1, -1, 11, 8, 12, 9, 18, 15, 3, 7, -1,-1,-1,-1, 0, 6]}


'''

import argparse
import os
import json
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# DLC keypoints (20 keypoints from acinoset):
# 0: nose, 1: r_eye, 2: l_eye, 3: neck_base, 4: spine, 5: tail_base, 6: tail1, 7: tail2,
# 8: r_shoulder, 9: r_front_knee, 10: r_front_ankle, 11: l_shoulder, 12: l_front_knee, 13: l_front_ankle,
# 14: r_hip, 15: r_back_knee, 16: r_back_ankle, 17: l_hip, 18: l_back_knee, 19: l_back_ankle

# Animal3D keypoints (26 keypoints):
# Based on the mapping: [2, 1, -1, 13, 10, 19, 16, 5, -1, -1, -1, -1, 11, 8, 12, 9, 18, 15, 3, 7, -1,-1,-1,-1, 0, 6]
# This means: animal3d_idx 0 maps to acinoset_idx 2 (l_eye), animal3d_idx 1 maps to acinoset_idx 1 (r_eye), etc.

# Keypoint mapping from acinoset (DLC) to animal3d (COCO format)
KEYPOINT_MAPPING = [2, 1, -1, 13, 10, 19, 16, 5, -1, -1, -1, -1, 11, 8, 12, 9, 18, 15, 3, 7, -1, -1, -1, -1, 0, 6]

def read_dlc_csv(csv_path):
    """
    Read DeepLabCut CSV file and extract keypoint data
    Returns: DataFrame with frame index and keypoint coordinates
    """
    # Read the CSV file, skip the first 2 rows (header rows)
    df = pd.read_csv(csv_path, skiprows=2)
    
    # Replace NaN with 0
    df = df.fillna(0)
    
    # The first column is frame index
    frame_indices = df.iloc[:, 0].values
    
    # Extract keypoint coordinates (x, y, likelihood)
    # DLC format: each keypoint has 3 columns (x, y, likelihood)
    num_keypoints = 20
    keypoints_data = []
    
    for idx, frame_idx in enumerate(frame_indices):
        keypoints = []
        for kp_idx in range(num_keypoints):
            col_start = 1 + kp_idx * 3
            x = float(df.iloc[idx, col_start])
            y = float(df.iloc[idx, col_start + 1])
            likelihood = float(df.iloc[idx, col_start + 2])
            
            # If likelihood is 0 (from NaN), but x and y are not 0, assume it's a valid point
            if likelihood == 0 and (x != 0 or y != 0):
                likelihood = 1.0  # Default to high confidence
            
            keypoints.append([x, y, likelihood])
        
        keypoints_data.append({
            'frame_idx': int(frame_idx),
            'keypoints': keypoints
        })
    
    return keypoints_data

def map_keypoints_to_animal3d(acinoset_keypoints):
    """
    Map 20 DLC keypoints to 26 Animal3D keypoints using the provided mapping
    acinoset_keypoints: list of [x, y, likelihood] for 20 keypoints
    Returns: list of [x, y, visibility] for 26 keypoints
    """
    animal3d_keypoints = []
    
    for animal3d_idx, acinoset_idx in enumerate(KEYPOINT_MAPPING):
        if acinoset_idx == -1:
            # Missing keypoint, set to [0, 0, 0]
            animal3d_keypoints.append([0.0, 0.0, 0.0])
        else:
            x, y, likelihood = acinoset_keypoints[acinoset_idx]
            # Replace NaN with 0
            if np.isnan(x):
                x = 0.0
            if np.isnan(y):
                y = 0.0
            if np.isnan(likelihood):
                likelihood = 0.0
            
            # Convert likelihood to visibility flag (2 = visible, 1 = occluded, 0 = not labeled)
            # If the keypoint has valid coordinates, mark as visible
            if x != 0.0 or y != 0.0:
                visibility = 2.0
            else:
                visibility = 0.0
            
            animal3d_keypoints.append([float(x), float(y), visibility])
    
    return animal3d_keypoints

def extract_frames_from_video(video_path, output_dir, frame_indices, behavior, camera_id):
    """
    Extract specific frames from video and save as images
    Returns: dict mapping frame_idx to image path
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return {}
    
    frame_paths = {}
    
    # Sort frame indices for efficient extraction
    sorted_frames = sorted(set(frame_indices))  # Remove duplicates
    
    pbar = tqdm(total=len(sorted_frames), desc=f"Extracting frames from {video_path.name}")
    
    for target_frame in sorted_frames:
        # Use CAP_PROP_POS_FRAMES to seek to the exact frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ret, frame = cap.read()
        
        if ret and frame is not None:
            # Save frame as image with behavior name in filename
            img_filename = f"{behavior}_cam{camera_id}_frame_{target_frame:06d}.jpg"
            img_path = output_dir / img_filename
            cv2.imwrite(str(img_path), frame)
            frame_paths[target_frame] = str(img_path.relative_to(output_dir.parent.parent))
        else:
            print(f"Warning: Failed to read frame {target_frame} from {video_path.name}")
        
        pbar.update(1)
    
    pbar.close()
    cap.release()
    
    return frame_paths

def compute_bbox_from_keypoints(keypoints):
    """
    Compute bounding box from keypoints
    keypoints: list of [x, y, visibility]
    Returns: [x, y, width, height]
    """
    valid_points = [(kp[0], kp[1]) for kp in keypoints if kp[2] > 0]
    
    if not valid_points:
        return [0, 0, 0, 0]
    
    xs, ys = zip(*valid_points)
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    
    # Add some padding
    padding = 20
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    width = (x_max - x_min) + 2 * padding
    height = (y_max - y_min) + 2 * padding
    
    return [float(x_min), float(y_min), float(width), float(height)]

def process_camera(camera_id, base_dir, output_dir, behavior):
    """
    Process one camera: read CSV, extract frames, convert to COCO format
    behavior: name of the behavior (e.g., 'run', 'flick')
    """
    base_dir = Path(base_dir)
    output_dir = Path(output_dir)
    
    # Paths
    csv_path = base_dir / "fte_pw" / f"cam{camera_id}_fte.csv"
    video_path = base_dir / f"cam{camera_id}.mp4"
    
    print(f"\nProcessing Camera {camera_id} - Behavior: {behavior}...")
    print(f"CSV: {csv_path}")
    print(f"Video: {video_path}")
    
    # Read keypoint data from CSV
    keypoints_data = read_dlc_csv(csv_path)
    print(f"Found {len(keypoints_data)} frames with keypoints")
    
    # Extract frames from video
    frame_indices = [kp_data['frame_idx'] for kp_data in keypoints_data]
    images_dir = output_dir / "images" / behavior / f"cam{camera_id}"
    frame_paths = extract_frames_from_video(video_path, images_dir, frame_indices, behavior, camera_id)
    
    # Convert to COCO format
    coco_data = []
    for kp_data in tqdm(keypoints_data, desc=f"Converting cam{camera_id} to COCO format"):
        frame_idx = kp_data['frame_idx']
        
        if frame_idx not in frame_paths:
            continue
        
        # Map keypoints from acinoset (20) to animal3d (26)
        acinoset_kps = kp_data['keypoints']
        animal3d_kps = map_keypoints_to_animal3d(acinoset_kps)
        
        # Compute bounding box
        bbox = compute_bbox_from_keypoints(animal3d_kps)
        
        # Create COCO entry
        img_path = frame_paths[frame_idx]
        coco_entry = {
            "img_path": img_path,
            "mask_path": img_path,  # Same as img_path
            "bbox": bbox,
            "keypoint_2d": animal3d_kps,
            "camera_id": camera_id,
            "frame_idx": frame_idx,
            "behavior": behavior
        }
        
        coco_data.append(coco_entry)
    
    return coco_data

def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert DeepLabCut labeled data to COCO format"
    )
    parser.add_argument(
        "--dataset_dir", type=str, default=".",
        help="Root directory containing behavior subdirectories (run, flick, etc.)"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Output directory for COCO format data (default: {dataset_dir}/coco_format)"
    )
    parser.add_argument(
        "--behaviors", type=str, nargs="+", default=["run", "flick"],
        help="Behavior names to process (default: run flick)"
    )
    parser.add_argument(
        "--cameras", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6],
        help="Camera IDs to process (default: 1 2 3 4 5 6)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir) if args.output_dir else dataset_dir / "coco_format"
    output_dir.mkdir(parents=True, exist_ok=True)

    behaviors = args.behaviors
    camera_ids = args.cameras
    
    all_data = []
    behavior_data = {}
    camera_data = {}
    
    for behavior in behaviors:
        behavior_dir = dataset_dir / behavior
        behavior_data[behavior] = []
        
        print(f"\n{'='*60}")
        print(f"Processing Behavior: {behavior.upper()}")
        print(f"{'='*60}")
        
        for cam_id in camera_ids:
            coco_data = process_camera(cam_id, behavior_dir, output_dir, behavior)
            all_data.extend(coco_data)
            behavior_data[behavior].extend(coco_data)
            
            # Store per-camera-behavior data
            key = f"{behavior}_cam{cam_id}"
            camera_data[key] = coco_data
    
    # Save combined data (all behaviors and cameras)
    output_json = output_dir / "all_data.json"
    with open(output_json, 'w') as f:
        json.dump({"data": all_data}, f, indent=4)
    
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Saved combined data to {output_json}")
    print(f"Total entries: {len(all_data)}")
    
    # Save per-behavior data
    for behavior in behaviors:
        behavior_json = output_dir / f"{behavior}.json"
        with open(behavior_json, 'w') as f:
            json.dump({"data": behavior_data[behavior]}, f, indent=4)
        print(f"\nSaved {behavior} data to {behavior_json} ({len(behavior_data[behavior])} entries)")
    
    # Save per-camera-behavior data
    for behavior in behaviors:
        for cam_id in camera_ids:
            key = f"{behavior}_cam{cam_id}"
            cam_json = output_dir / f"{behavior}_cam{cam_id}.json"
            with open(cam_json, 'w') as f:
                json.dump({"data": camera_data[key]}, f, indent=4)
            print(f"  - {behavior}_cam{cam_id}: {len(camera_data[key])} entries")
    
    # Save multiview relationships
    # Group by behavior and frame index to establish multiview correspondence
    multiview_data = {}
    for entry in all_data:
        behavior = entry['behavior']
        frame_idx = entry['frame_idx']
        cam_id = entry['camera_id']
        
        if behavior not in multiview_data:
            multiview_data[behavior] = {}
        
        if frame_idx not in multiview_data[behavior]:
            multiview_data[behavior][frame_idx] = {}
        
        multiview_data[behavior][frame_idx][f"cam{cam_id}"] = {
            "img_path": entry['img_path'],
            "keypoint_2d": entry['keypoint_2d'],
            "bbox": entry['bbox']
        }
    
    multiview_json = output_dir / "multiview_mapping.json"
    with open(multiview_json, 'w') as f:
        json.dump(multiview_data, f, indent=4)
    
    print(f"\nSaved multiview mapping to {multiview_json}")
    for behavior in behaviors:
        print(f"  - {behavior}: {len(multiview_data.get(behavior, {}))} synchronized frames")
    
    print(f"\n{'='*60}")
    print("Conversion complete!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()