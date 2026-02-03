import os 
#from prepare_data import create_video_frames_df
from src.utils.data_utils import create_video_frames_df
import random
import numpy as np
import torch
import torchvision.transforms as T
import pandas as pd
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from src.data.dataset import VideoYoloDataset, TemporalWaggleCollator
from src.data.augmentation import WaggleAugmentations
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn  as nn
from src.utils.data_utils import fix_dataframe_with_video_lengths
import datetime
from torch.utils.tensorboard import SummaryWriter
from src.utils.eval_utils import get_preds_gt, yolo_to_img_space, yolo_to_img_space_gt, get_eval_metrics
from src.utils.nms import batch_postprocess_predictions
from src.utils.vis_utils import reverse_transform, save_frames
import argparse
import matplotlib.pyplot as plt

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def denormalize_tensor(tensor, mean, std):
    """
    Reverse torchvision.transforms.Normalize on a tensor.
    Args:
        tensor: torch.Tensor of shape (C, T, H, W)
        mean: list of 3 floats
        std: list of 3 floats
    Returns:
        Denormalized tensor (values roughly in [0,1])
    """
    mean = torch.tensor(mean).view(-1, 1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1, 1)
    return tensor * std + mean

def save_sample_with_waggle_visualization(sample, sample_idx, output_dir="test_dataloader_vis", denormalize=True):
    """
    Save frames with waggle position and direction visualization.
    Similar to test_window_augmentations function.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract sample data
    frames = sample["video"]
    targets = sample["targets"]
    metadata = sample["metadata"]
    
    video_name = metadata['video_name']
    start_frame = metadata['start_frame']
    end_frame = metadata['end_frame']
    
    print(f"\nVisualizing sample {sample_idx}:")
    print(f"  Video: {video_name}")
    print(f"  Frame range: {start_frame} to {end_frame}")
    print(f"  Frames shape: {frames.shape}")
    print(f"  Targets shape: {targets.shape}")
    
    if denormalize == True:
        # Denormalize frames for visualization
        frames_denorm = denormalize_tensor(
            frames,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    else:
        frames_denorm = frames
    
    # Extract bee position and direction from target tensor
    grid_size = 28
    bee_pos = None
    bee_dir = None
    waggle_start_frame = -1
    waggle_end_frame = -1
    
    # Find where object exists in target tensor
    obj_mask = targets[:, :, 0, 0] == 1.0
    if obj_mask.any():
        grid_y, grid_x = torch.where(obj_mask)
        grid_y, grid_x = grid_y[0].item(), grid_x[0].item()
        
        cell_x = targets[grid_y, grid_x, 0, 1].item()
        cell_y = targets[grid_y, grid_x, 0, 2].item()
        dir_x = targets[grid_y, grid_x, 0, 3].item()
        dir_y = targets[grid_y, grid_x, 0, 4].item()
        
        # Convert to pixel coordinates
        bee_x = (grid_x + cell_x) * (224 / grid_size)
        bee_y = (grid_y + cell_y) * (224 / grid_size)
        bee_pos = (bee_x, bee_y)
        bee_dir = (dir_x, dir_y)  # Direction vector normalized
        
        # Get waggle timing info
        start_norm = targets[grid_y, grid_x, 0, 5].item()
        end_norm = targets[grid_y, grid_x, 0, 6].item()
        
        # Convert normalized times to frame indices
        num_frames = frames_denorm.shape[1]
        if start_norm != -1 and end_norm != -1:
            waggle_start_frame = int(start_norm * (num_frames - 1))
            waggle_end_frame = round(end_norm * (num_frames - 1))
            print(f"  Waggle timing: frames {waggle_start_frame} to {waggle_end_frame} (out of {num_frames} frames)")
    
    # Save each frame with waggle visualization
    num_frames = frames_denorm.shape[1]
    
    sample_output_dir = os.path.join(output_dir, f"sample_{sample_idx}")
    os.makedirs(sample_output_dir, exist_ok=True)
    
    for frame_idx in range(num_frames):
        frame = frames_denorm[:, frame_idx, :, :]  # (C, H, W)
        frame_np = frame.permute(1, 2, 0).numpy()
        
        # Ensure values are in [0, 1]
        if frame_np.min() < 0 or frame_np.max() > 1.0:
            frame_np = np.clip(frame_np, 0, 1)
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.imshow(frame_np)
        
        # Draw waggle position and direction only on waggle frames
        if bee_pos is not None:
            # Check if this frame is within waggle duration
            is_waggle_frame = (waggle_start_frame <= frame_idx <= waggle_end_frame)
            
            if is_waggle_frame:
                # Waggle frame draw annotation
                circle = plt.Circle(bee_pos, radius=5, color='red', fill=True, alpha=0.7)
                ax.add_patch(circle)
                
                # Draw direction arrow
                if bee_dir is not None:
                    arrow_length = 30
                    ax.arrow(bee_pos[0], bee_pos[1], 
                            bee_dir[0] * arrow_length, bee_dir[1] * arrow_length,
                            head_width=5, head_length=5, fc='yellow', ec='yellow', alpha=0.8)
                
                # Add text annotation with position, direction, and norm
                dir_norm = np.sqrt(bee_dir[0]**2 + bee_dir[1]**2)
                text_str = f"Pos: ({bee_pos[0]:.1f}, {bee_pos[1]:.1f})\n"
                text_str += f"Dir: ({bee_dir[0]:.3f}, {bee_dir[1]:.3f})\n"
                text_str += f"||Dir||: {dir_norm:.4f}"
                ax.text(10, 20, text_str, 
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                       fontsize=9, color='black', verticalalignment='top')
                
                frame_status = "waggle"
            else:
                # Non-waggle frame draw NOTHING
                frame_status = "no waggle"
        else:
            frame_status = "no detection"
        
        # Set title with frame status
        title = f"Sample {sample_idx}, Frame {frame_idx} ({frame_status})\n{video_name}"
        ax.set_title(title, fontsize=10)
        ax.axis('off')
        
        # Save frame
        filename = f"frame_{frame_idx:03d}.png"
        filepath = os.path.join(sample_output_dir, filename)
        plt.savefig(filepath, dpi=100, bbox_inches='tight')
        plt.close()
    
    # Also save a grid of all frames
    create_frame_grid(frames_denorm, bee_pos, bee_dir, video_name, sample_output_dir, 
                      waggle_start_frame, waggle_end_frame)
    
    print(f"  Saved visualization to: {sample_output_dir}/")
    
    return sample_output_dir

def create_frame_grid(video_tensor, bee_pos, bee_dir, video_name, output_dir, 
                      waggle_start_frame=-1, waggle_end_frame=-1):
    """Create a grid of all frames with waggle visualization."""
    num_frames = video_tensor.shape[1]
    cols = 4
    rows = (num_frames + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 3.5))
    axes = axes.flatten()
    
    for i in range(num_frames):
        ax = axes[i]
        frame = video_tensor[:, i, :, :]
        frame_np = frame.permute(1, 2, 0).numpy()
        
        if frame_np.min() < 0 or frame_np.max() > 1.0:
            frame_np = np.clip(frame_np, 0, 1)
        
        ax.imshow(frame_np)
        
        # Draw waggle only on frames where it occurs
        if bee_pos is not None:
            is_waggle_frame = (waggle_start_frame <= i <= waggle_end_frame)
            
            if is_waggle_frame:
                # Waggle frame red circle + arrow
                circle = plt.Circle(bee_pos, radius=5, color='red', fill=True, alpha=0.7)
                ax.add_patch(circle)
                
                if bee_dir is not None:
                    arrow_length = 30
                    ax.arrow(bee_pos[0], bee_pos[1], 
                            bee_dir[0] * arrow_length, bee_dir[1] * arrow_length,
                            head_width=5, head_length=5, fc='yellow', ec='yellow', alpha=0.8)
                
                # Add text annotation with position, direction, and norm
                dir_norm = np.sqrt(bee_dir[0]**2 + bee_dir[1]**2)
                text_str = f"Pos: ({bee_pos[0]:.1f}, {bee_pos[1]:.1f})\n"
                text_str += f"Dir: ({bee_dir[0]:.2f}, {bee_dir[1]:.2f})\n"
                text_str += f"||Dir||: {dir_norm:.3f}"
                ax.text(5, 15, text_str, 
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                       fontsize=7, color='black', verticalalignment='top')
                
                frame_label = f"Frame {i} (Waggle)"
            else:
                # Non-waggle frame - draw NOTHING
                frame_label = f"Frame {i}"
        else:
            frame_label = f"Frame {i}"
        
        ax.set_title(frame_label, fontsize=9)
        ax.axis('off')
    
    # Hide unused subplots
    for i in range(num_frames, len(axes)):
        axes[i].axis('off')
    
    # Add video info to title
    title = f"Sample Visualization: {video_name}\n"
    if waggle_start_frame != -1 and waggle_end_frame != -1:
        title += f"Waggle: frames {waggle_start_frame} to {waggle_end_frame}"
    else:
        title += "No waggle"
    
    plt.suptitle(title, fontsize=14)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save grid
    grid_path = os.path.join(output_dir, "frames_grid.png")
    plt.savefig(grid_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved grid: frames_grid.png")

def main(args):
    batch_size = args.batch_size

    csv_path = args.csv_path
    data = pd.read_csv(csv_path)
    target_video = '001_2304_1760.mp4'
    data = data[data['video_name'] == target_video].reset_index(drop=True)
    print(f"After filtering to video '{target_video}': {len(data)} samples")
    

    video_frames_dict = create_video_frames_df(data["video_name"].unique())
    data = fix_dataframe_with_video_lengths(data, video_frames_dict)
    data = data.sample(frac=1).reset_index(drop=True)

    transforms = T.Compose([
        T.ToPILImage(),
        T.Resize((224, 224)),
        T.ToTensor(),
        #T.Normalize(mean=[0.485, 0.456, 0.406],
        #        std=[0.229, 0.224, 0.225])
    ])

    # Create augmentation
    train_augmentation = WaggleAugmentations(
        width=224, height=224, 
        prob_flip_h=0.5, prob_flip_v=0.0, 
        prob_rotate=1.0, rotate_range=(-45, 45), 
        prob_scale=1.0, scale_range=(0.9, 1.1),
        prob_translate=0.3, translate_range=0.1,
        prob_hsv=0.0, hsv_hue=0.1, hsv_saturation=0.9, hsv_value=0.9,
        prob_brightness=1.0, brightness_range=0.4, 
        prob_contrast=1.0, contrast_range=0.4,
        prob_gamma=0.0, gamma_range=(0.8, 1.2),
        prob_blur=0.1, blur_range=(0.5, 2.0),
        prob_clahe=0.1, clahe_clip_limit=2.0, clahe_tile_grid_size=(8, 8),
        prob_color_shuffle=0.0,
        prob_posterize=0.0, posterize_bits=(4, 7),
        prob_greyscale=0.0,
        normalize=True,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        debug=False)

    total_len = len(data)
    train_len = int(0.8 * total_len)

    train_indices = list(range(train_len))
    test_indices = list(range(train_len, total_len))

    train_df = data.iloc[train_indices].reset_index(drop=True)
    test_df = data.iloc[test_indices].reset_index(drop=True)

    # Create datasets
    train_dataset = VideoYoloDataset(
        train_df,
        args.data_dir,
        transforms,
        width=224,
        height=224,
        clip_len=16,
        grid_size=28,
        max_detections_per_cell=1,
        num_classes=1,
        augment=train_augmentation,
        is_training=True
    )
    
    test_dataset = VideoYoloDataset(
        train_df,
        args.data_dir,
        transforms,
        width=224,
        height=224,
        clip_len=16,
        grid_size=28,
        max_detections_per_cell=1,
        num_classes=1,
        augment=None,
        is_training=False 
    )


    print(f'Train data len: {len(train_dataset)}')
    
    # Change if you want to visalise more than 1 sample from the dataset
    num_samples_to_visualize = 10
    for i in range(min(num_samples_to_visualize, len(train_dataset))):
        sample = train_dataset[i]
        save_sample_with_waggle_visualization(sample, 
                                              i, 
                                              output_dir="tests/test_dataloader_vis/train", denormalize=True)
    
    # Visualize first few samples from test dataset
    for i in range(min(num_samples_to_visualize, len(test_dataset))):
        sample = test_dataset[i]
        save_sample_with_waggle_visualization(sample, i, 
                                              output_dir="tests/test_dataloader_vis/test", 
                                              denormalize=False)
    
    sample = train_dataset[0]
    frames = sample["video"]
    print("\nFrames shape:", frames.shape)  # (C, T, H, W)

    # YOLO-style target tensor
    targets = sample["targets"]
    print("Targets shape:", targets.shape)  # (grid_size, grid_size, max_det, 7)

    # Label
    label = sample["label"]
    print("Label:", label)  # 0 or 1

    # Metadata
    meta = sample["metadata"]
    print("Metadata:", meta)

def get_args():
    parser = argparse.ArgumentParser(description="Waggle detection training")

    parser.add_argument("--data_dir", type=str, default='/home/prajna/complete_data/videos/', help="Root directory containing video files")

    parser.add_argument("--csv_path", type=str, default="/home/prajna/complete_data/annotations/fps_multires_full_data.csv", help="Path to annotations csv")

    parser.add_argument("--batch_size", type=int, default=4)

    parser.add_argument("--num_workers", type=int, default=4)

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    main(args)