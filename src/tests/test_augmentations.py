import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torchvision import transforms as T
from src.data.augmentation import WaggleAugmentations
from src.data.dataset import VideoYoloDataset

def perpendicular(v):
    """Return a perpendicular vector (rotated +90°)."""
    return (-v[1], v[0])

def extend_to_border(origin, direction, width, height):
    """
    Extend a ray from origin in direction until it hits the image border.
    Returns the endpoint (x, y).
    """
    x0, y0 = origin
    dx, dy = direction

    eps = 1e-6
    dx = dx if abs(dx) > eps else eps
    dy = dy if abs(dy) > eps else eps

    t_vals = []

    # Left / right borders
    t_vals.append((0 - x0) / dx)
    t_vals.append((width - x0) / dx)

    # Top / bottom borders
    t_vals.append((0 - y0) / dy)
    t_vals.append((height - y0) / dy)

    # Keep only forward intersections
    t_vals = [t for t in t_vals if t > 0]

    t_min = min(t_vals)
    return x0 + dx * t_min, y0 + dy * t_min


def test_window_augmentations(augmentation_config=None, test_name="test"):
    """
    Test augmentations on a single window.
    Shows waggle position and direction on every frame.
    
    Args:
        augmentation_config: WaggleAugmentations instance or None for no augmentation
        test_name: Name for the output folder
    """
    # Create output folder
    output_dir = f"tests/aug_test_{test_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Current directory: {os.getcwd()}")
    print(f"Saving to: {output_dir}")
    
    # Use your actual data paths
    data_dir = '/home/prajna/complete_data/videos/'
    annotations_path = '/home/prajna/complete_data/annotations/fps_multires_full_data.csv'
    
    # Load annotations
    df = pd.read_csv(annotations_path)
    
    # Take first sample 
    test_sample = df.head(1).copy()
    
    # Create transforms without normalization for clearer visualization
    transforms_simple = T.Compose([
        T.ToPILImage(),
        T.ToTensor(),
    ])
    
    # Create dataset
    dataset = VideoYoloDataset(
        test_sample,
        data_dir,
        transforms_simple,
        width=224,
        height=224,
        clip_len=16,
        grid_size=28,
        max_detections_per_cell=1,
        num_classes=1,
        augment=augmentation_config,  # Pass augmentation config
        is_training=True if augmentation_config else False
    )
    
    # Get the sample
    sample = dataset[0]
    video_name = sample['metadata']['video_name']
    start_frame = sample['metadata']['start_frame']
    end_frame = sample['metadata']['end_frame']
    aug_info = sample['metadata'].get('augmentation', ['No augmentation'])
    
    print(f"Video: {video_name}")
    print(f"Frame range: {start_frame} to {end_frame}")
    print(f"Augmentations: {aug_info}")
    
    # Extract bee position and direction FROM TARGET TENSOR
    # This should be the same for all frames in the window
    targets = sample['targets']
    bee_pos = None
    bee_dir = None
    
    grid_size = 28
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
        bee_dir = (dir_x, dir_y)  # Direction vector (normalized)
        
        # Also get waggle timing info
        start_norm = targets[grid_y, grid_x, 0, 5].item()
        end_norm = targets[grid_y, grid_x, 0, 6].item()
        if start_norm != -1 and end_norm != -1:
            print(f"Waggle timing: frames {int(start_norm * 16)} to {int(end_norm * 16)}")
        
    # Save each frame with waggle visualization
    video_tensor = sample['video']  # (C, T, H, W)
    num_frames = video_tensor.shape[1]
    
    print(f"Saving {num_frames} frames to: {output_dir}/")
    
    for frame_idx in range(num_frames):
        frame = video_tensor[:, frame_idx, :, :]  # (C, H, W)
        frame_np = frame.permute(1, 2, 0).numpy()
        
        # Ensure values are in [0, 1]
        if frame_np.min() < 0 or frame_np.max() > 1.0:
            frame_np = np.clip(frame_np, 0, 1)
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.imshow(frame_np)
        
        # Draw waggle position and direction on every frame
        if bee_pos is not None:
            # Draw position circle
            circle = plt.Circle(bee_pos, radius=5, color='red', fill=True, alpha=0.7)
            ax.add_patch(circle)
            
            # Draw direction arrow (scale for visibility)
            if bee_dir is not None:
                img_w, img_h = 224, 224
                arrow_length = 30

                #arrow_length = 30
                #ax.arrow(bee_pos[0], bee_pos[1], 
                #        bee_dir[0] * arrow_length, bee_dir[1] * arrow_length,
                #        head_width=5, head_length=5, fc='yellow', ec='yellow', alpha=0.8)
                # Main direction (yellow)
                #ax.arrow(
                #    bee_pos[0], bee_pos[1],
                #    bee_dir[0] * arrow_length, bee_dir[1] * arrow_length,
                #    head_width=5, head_length=5,
                #    fc='yellow', ec='yellow', alpha=0.9
                #)

                # Perpendicular direction (cyan)
                #perp = perpendicular(bee_dir)
                #ax.arrow(
                #    bee_pos[0], bee_pos[1],
                #    perp[0] * arrow_length, perp[1] * arrow_length,
                #    head_width=5, head_length=5,
                #    fc='cyan', ec='cyan', alpha=0.7
                #)
                # === Direction vectors ===
                dx, dy = bee_dir
                perp_dx, perp_dy = -dy, dx

                # Normalize perpendicular just to be safe
                norm = np.sqrt(perp_dx**2 + perp_dy**2)
                perp_dx /= norm
                perp_dy /= norm

                # ---- Short direction arrow (local) ----
                ax.arrow(
                    bee_pos[0], bee_pos[1],
                    dx * arrow_length, dy * arrow_length,
                    head_width=5, head_length=5,
                    fc='yellow', ec='yellow', alpha=0.9
                )

                # ---- Dotted direction ray to image border ----
                end_dir = extend_to_border(
                    bee_pos, (dx, dy), img_w, img_h
                )

                ax.plot(
                    [bee_pos[0], end_dir[0]],
                    [bee_pos[1], end_dir[1]],
                    linestyle='--',
                    linewidth=2,
                    color='yellow',
                    alpha=0.6
                )

                # ---- Perpendicular ray to image border ----
                end_perp = extend_to_border(
                    bee_pos, (perp_dx, perp_dy), img_w, img_h
                )

                ax.plot(
                    [bee_pos[0], end_perp[0]],
                    [bee_pos[1], end_perp[1]],
                    linestyle='-',
                    linewidth=2,
                    color='cyan',
                    alpha=0.8
                )


        # Set title
        if augmentation_config:
            aug_title = ", ".join(aug_info[:2]) if aug_info else "No augmentation"
            title = f"Frame {frame_idx}\n{aug_title}"
        else:
            title = f"Original Frame {frame_idx}"
        
        ax.set_title(title, fontsize=10)
        ax.axis('off')
        
        # Save frame
        filename = f"frame_{frame_idx:03d}.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=100, bbox_inches='tight')
        plt.close()
    
    # Also save a grid of all frames
    create_frame_grid(video_tensor, bee_pos, bee_dir, aug_info, output_dir)
    
    print(f"\n Saved all frames to: {output_dir}/")
    
    return output_dir, bee_pos, bee_dir

def create_frame_grid(video_tensor, bee_pos, bee_dir, aug_info, output_dir):
    """Create a grid of all frames with waggle visualization."""
    num_frames = video_tensor.shape[1]
    cols = 4
    rows = (num_frames + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 3.5))
    axes = axes.flatten()
    
    img_w, img_h = 224, 224
    arrow_length = 30
    
    for i in range(num_frames):
        ax = axes[i]
        frame = video_tensor[:, i, :, :]
        frame_np = frame.permute(1, 2, 0).numpy()
        
        if frame_np.min() < 0 or frame_np.max() > 1.0:
            frame_np = np.clip(frame_np, 0, 1)
        
        ax.imshow(frame_np)
        
        # Draw waggle on every frame in the grid too
        if bee_pos is not None:
            circle = plt.Circle(bee_pos, radius=5, color='red', fill=True, alpha=0.7)
            ax.add_patch(circle)
            
            if bee_dir is not None:
                dx, dy = bee_dir
                perp_dx, perp_dy = -dy, dx
                
                # Normalize perpendicular
                norm = np.sqrt(perp_dx**2 + perp_dy**2)
                perp_dx /= norm
                perp_dy /= norm
                
                # Short direction arrow (local)
                ax.arrow(
                    bee_pos[0], bee_pos[1],
                    dx * arrow_length, dy * arrow_length,
                    head_width=5, head_length=5,
                    fc='yellow', ec='yellow', alpha=0.9
                )
                
                # Dotted direction ray to image border
                end_dir = extend_to_border(
                    bee_pos, (dx, dy), img_w, img_h
                )
                ax.plot(
                    [bee_pos[0], end_dir[0]],
                    [bee_pos[1], end_dir[1]],
                    linestyle='--',
                    linewidth=2,
                    color='yellow',
                    alpha=0.6
                )
                
                # Perpendicular ray to image border
                end_perp = extend_to_border(
                    bee_pos, (perp_dx, perp_dy), img_w, img_h
                )
                ax.plot(
                    [bee_pos[0], end_perp[0]],
                    [bee_pos[1], end_perp[1]],
                    linestyle='-',
                    linewidth=2,
                    color='cyan',
                    alpha=0.8
                )
        
        ax.set_title(f"Frame {i}", fontsize=9)
        ax.axis('off')
    
    # Hide unused subplots
    for i in range(num_frames, len(axes)):
        axes[i].axis('off')
    
    # Add augmentation info to title
    if aug_info and aug_info[0] != "No augmentation":
        aug_text = ", ".join(aug_info[:3])
        plt.suptitle(f"All Frames with Waggle Visualization\n{aug_text}", fontsize=14)
    else:
        plt.suptitle("All Original Frames with Waggle Visualization", fontsize=14)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save grid
    grid_path = os.path.join(output_dir, "frames_grid.png")
    plt.savefig(grid_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved grid: frames_grid.png")

# Helper function to compare multiple tests
def compare_augmentations(test_configs):
    """Compare multiple augmentation configurations side-by-side."""
    results = []
    
    for config_name, aug_config in test_configs:
        
        output_dir, bee_pos, bee_dir = test_window_augmentations(
            augmentation_config=aug_config,
            test_name=config_name
        )
        
        results.append({
            'name': config_name,
            'dir': output_dir,
            'pos': bee_pos,
            'dir': bee_dir
        })
    
    return results

if __name__ == "__main__":
    print("Augmentation Test Script")

    augs = WaggleAugmentations(
        width=224, height=224, 
        prob_flip_h=0.0, prob_flip_v=0.0, 
        prob_rotate=1.0, rotate_range=(-90, 90), 
        prob_scale=0.0, scale_range=(0.9, 1.1),
        prob_translate=0.0, translate_range=0.1,
        prob_hsv=0.0, hsv_hue=0.1, hsv_saturation=0.9, hsv_value=0.9,
        prob_brightness=0.0, brightness_range=0.4, 
        prob_contrast=0.0, contrast_range=0.4,
        prob_gamma=0.0, gamma_range=(0.8, 1.2),
        prob_blur=0.0, blur_range=(0.5, 2.0),
        prob_clahe=0.0, clahe_clip_limit=2.0, clahe_tile_grid_size=(8, 8),
        prob_color_shuffle=0.0,
        prob_posterize=0.0, posterize_bits=(4, 7),
        prob_greyscale=0.0,
        normalize=False,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        debug=False)

    test_window_augmentations(
        augmentation_config=augs,
        test_name="aug_rotate"
    )