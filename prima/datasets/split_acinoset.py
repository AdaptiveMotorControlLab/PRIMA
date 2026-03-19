"""
Split acinoset multiview_mapping.json into train and test sets (7:3 ratio)
"""

import json
import random
from pathlib import Path
from collections import defaultdict

def split_multiview_data(input_json, output_dir, train_ratio=0.7, seed=42):
    """
    Split multiview mapping data into train and test sets.
    
    Args:
        input_json: Path to multiview_mapping.json
        output_dir: Directory to save train.json and test.json
        train_ratio: Ratio of training data (default 0.7 for 70%)
        seed: Random seed for reproducibility
    """
    # Set random seed
    random.seed(seed)
    
    # Load data
    print(f"Loading data from {input_json}...")
    with open(input_json, 'r') as f:
        data = json.load(f)
    
    # Initialize train and test splits
    train_data = defaultdict(dict)
    test_data = defaultdict(dict)
    
    # Process each behavior
    for behavior, frames in data.items():
        print(f"\nProcessing behavior: {behavior}")
        
        # Get all frame indices
        frame_indices = list(frames.keys())
        total_frames = len(frame_indices)
        
        # Shuffle frame indices
        random.shuffle(frame_indices)
        
        # Calculate split point
        train_size = int(total_frames * train_ratio)
        
        # Split frames
        train_frames = frame_indices[:train_size]
        test_frames = frame_indices[train_size:]
        
        print(f"  Total frames: {total_frames}")
        print(f"  Train frames: {len(train_frames)}")
        print(f"  Test frames: {len(test_frames)}")
        
        # Assign to train and test
        for frame_idx in train_frames:
            train_data[behavior][frame_idx] = frames[frame_idx]
        
        for frame_idx in test_frames:
            test_data[behavior][frame_idx] = frames[frame_idx]
    
    # Save train and test splits
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_json = output_dir / "train.json"
    test_json = output_dir / "test.json"
    
    print(f"\nSaving train data to {train_json}...")
    with open(train_json, 'w') as f:
        json.dump(dict(train_data), f, indent=4)
    
    print(f"Saving test data to {test_json}...")
    with open(test_json, 'w') as f:
        json.dump(dict(test_data), f, indent=4)
    
    # Print summary
    print("\n" + "="*50)
    print("Summary:")
    print("="*50)
    
    total_train_frames = sum(len(frames) for frames in train_data.values())
    total_test_frames = sum(len(frames) for frames in test_data.values())
    total_frames = total_train_frames + total_test_frames
    
    print(f"Total frames: {total_frames}")
    print(f"Train frames: {total_train_frames} ({total_train_frames/total_frames*100:.1f}%)")
    print(f"Test frames: {total_test_frames} ({total_test_frames/total_frames*100:.1f}%)")
    print("\nPer behavior:")
    for behavior in train_data.keys():
        train_count = len(train_data[behavior])
        test_count = len(test_data[behavior])
        total_count = train_count + test_count
        print(f"  {behavior}: train={train_count}, test={test_count}, total={total_count}")
    
    print("\nDone!")

if __name__ == "__main__":
    # Configuration
    input_json = "/home/xiaohang/Xiaohang_workspace/vggt_animal/AniMer/datasets/acinoset/multiview_mapping.json"
    output_dir = "/home/xiaohang/Xiaohang_workspace/vggt_animal/AniMer/datasets/acinoset"
    
    # Split data
    split_multiview_data(
        input_json=input_json,
        output_dir=output_dir,
        train_ratio=0.7,
        seed=42
    )
