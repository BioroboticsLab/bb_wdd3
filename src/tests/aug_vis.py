import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrow
import torchvision.transforms.functional as F

# This file is abondened and an earlier iteration of code we dont use anymore
def visualize_augmentation(frames, target_dict, augmentation_fn=None, num_frames_to_show=4):
    """
    Visualize original vs augmented frames with target annotations.
    
    Args:
        frames: List of frame tensors [C, H, W]
        target_dict: Dict with 'x', 'y', 'dir_x', 'dir_y'
        augmentation_fn: WaggleAugmentations instance (or None for original only)
        num_frames_to_show: Number of frames to display from the clip
    """
    original_frames = frames.copy()
    original_target = target_dict.copy()
    
    # Apply augmentation if provided
    if augmentation_fn is not None:
        aug_frames, aug_target = augmentation_fn(frames, target_dict.copy())
    else:
        aug_frames = frames
        aug_target = target_dict
    
    # Select frames to display
    frame_indices = np.linspace(0, len(original_frames)-1, num_frames_to_show, dtype=int)
    
    # Create figure
    fig, axes = plt.subplots(2, num_frames_to_show, figsize=(4*num_frames_to_show, 8))
    if num_frames_to_show == 1:
        axes = axes.reshape(2, 1)
    
    fig.suptitle('Augmentation Verification: Original (top) vs Augmented (bottom)', 
                 fontsize=14, fontweight='bold')
    
    for idx, frame_idx in enumerate(frame_indices):
        # Original frame
        plot_frame_with_target(
            axes[0, idx], 
            original_frames[frame_idx], 
            original_target,
            f"Original Frame {frame_idx}",
            original_width=960,
            original_height=540
        )
        
        # Augmented frame
        plot_frame_with_target(
            axes[1, idx], 
            aug_frames[frame_idx], 
            aug_target,
            f"Augmented Frame {frame_idx}",
            original_width=960,
            original_height=540
        )
    
    plt.tight_layout()
    return fig, original_target, aug_target


def plot_frame_with_target(ax, frame_tensor, target_dict, title, original_width=960, original_height=540):
    """
    Plot a single frame with target point and direction vector.
    
    Args:
        ax: Matplotlib axis
        frame_tensor: Tensor [C, H, W]
        target_dict: Dict with 'x', 'y', 'dir_x', 'dir_y' in original coords (960x540)
        title: Title for the subplot
        original_width: Original video width (960)
        original_height: Original video height (540)
    """
    # Convert tensor to numpy for plotting
    frame_np = frame_tensor.permute(1, 2, 0).numpy()
    
    # Clip values to [0, 1] for display
    frame_np = np.clip(frame_np, 0, 1)
    
    H, W = frame_np.shape[:2]
    
    # Convert target coordinates from original space to current frame space
    x_display = target_dict['x'] * (W / original_width)
    y_display = target_dict['y'] * (H / original_height)
    
    # Display the frame
    ax.imshow(frame_np)
    ax.set_title(title, fontsize=10)
    ax.axis('off')
    
    # Plot target point
    circle = Circle((x_display, y_display), radius=5, color='red', fill=True, alpha=0.7)
    ax.add_patch(circle)
    
    # Plot direction vector
    dir_x = target_dict['dir_x']
    dir_y = target_dict['dir_y']
    
    # Scale direction for visibility (arrow length)
    arrow_length = 30
    dx = dir_x * arrow_length
    dy = dir_y * arrow_length
    
    arrow = FancyArrow(
        x_display, y_display, dx, dy,
        width=3, head_width=8, head_length=6,
        color='yellow', edgecolor='black', linewidth=1.5, alpha=0.8
    )
    ax.add_patch(arrow)
    
    # Add text annotation
    text = f"Pos: ({target_dict['x']:.1f}, {target_dict['y']:.1f})\n"
    text += f"Dir: ({dir_x:.2f}, {dir_y:.2f})"
    ax.text(5, H-5, text, color='white', fontsize=8, 
            verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))


def test_augmentations_multiple_times(frames, target_dict, augmentation_fn, n_tests=3):
    """
    Apply augmentation multiple times to see variation.
    
    Args:
        frames: List of frame tensors
        target_dict: Original target dict
        augmentation_fn: WaggleAugmentations instance
        n_tests: Number of different augmentations to show
    """
    fig, axes = plt.subplots(n_tests + 1, 4, figsize=(16, 4*(n_tests+1)))
    
    fig.suptitle('Multiple Augmentation Samples (showing first 4 frames)', 
                 fontsize=14, fontweight='bold')
    
    # Show original in first row
    frame_indices = np.linspace(0, len(frames)-1, 4, dtype=int)
    for idx, frame_idx in enumerate(frame_indices):
        plot_frame_with_target(
            axes[0, idx], 
            frames[frame_idx], 
            target_dict,
            f"Original Frame {frame_idx}",
            original_width=960,
            original_height=540
        )
    
    # Show augmented versions
    for test_idx in range(n_tests):
        aug_frames, aug_target = augmentation_fn(frames.copy(), target_dict.copy())
        
        for idx, frame_idx in enumerate(frame_indices):
            plot_frame_with_target(
                axes[test_idx + 1, idx], 
                aug_frames[frame_idx], 
                aug_target,
                f"Aug {test_idx+1} Frame {frame_idx}",
                original_width=960,
                original_height=540
            )
    
    plt.tight_layout()
    return fig


def compare_targets(original_target, augmented_target):
    """
    Print comparison of target values before and after augmentation.
    """
    print("=" * 60)
    print("TARGET COMPARISON")
    print("=" * 60)
    print(f"{'Attribute':<15} {'Original':<20} {'Augmented':<20} {'Change':<10}")
    print("-" * 60)
    
    for key in ['x', 'y', 'dir_x', 'dir_y']:
        orig_val = original_target[key]
        aug_val = augmented_target[key]
        change = aug_val - orig_val
        print(f"{key:<15} {orig_val:<20.4f} {aug_val:<20.4f} {change:<10.4f}")
    
    print("=" * 60)
    
    # Verify direction vector is normalized
    dir_norm = np.sqrt(augmented_target['dir_x']**2 + augmented_target['dir_y']**2)
    print(f"Direction vector magnitude: {dir_norm:.6f} (should be ~1.0)")
    print("=" * 60)


def demo_visualization(dataset, idx=0, augmentation_fn=None):
    """
    Demonstrate visualization using your dataset.
    
    Args:
        dataset: Your VideoYoloDataset instance
        idx: Index of sample to visualize
        augmentation_fn: WaggleAugmentations instance
    """
    # Get a sample from your dataset
    sample = dataset[idx]
    
    # Get the video name and frames
    video_name = sample['metadata']['video_name']
    start_frame = sample['metadata']['start_frame']
    end_frame = sample['metadata']['end_frame']
    
    print(f"Visualizing: {video_name}")
    print(f"Frames: {start_frame} to {end_frame}")
    print(f"Label: {sample['label']}")
    
    if sample['label'] == 1:
        # Get original frames (before augmentation)
        frames = dataset.video_frames_dict[video_name][start_frame:end_frame]
        
        # Apply transform to get processed frames
        if dataset.transform:
            processed_frames = [dataset.transform(frame) for frame in frames]
        else:
            processed_frames = frames
        
        # Get original target
        original_target = {
            'x': dataset.data.iloc[idx]['origin_x'],
            'y': dataset.data.iloc[idx]['origin_y'],
            'dir_x': dataset.data.iloc[idx]['direction_x'],
            'dir_y': dataset.data.iloc[idx]['direction_y']
        }
        
        # Visualize
        fig1, orig_target, aug_target = visualize_augmentation(
            processed_frames, 
            original_target, 
            augmentation_fn
        )
        
        # Compare targets
        compare_targets(orig_target, aug_target)
        
        # Show multiple augmentations
        if augmentation_fn is not None:
            fig2 = test_augmentations_multiple_times(
                processed_frames, 
                original_target, 
                augmentation_fn, 
                n_tests=3
            )
        
        save_path="debug_vis.png"
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig2)

    else:
        print("Sample has no waggle (label=0), skipping visualization.")

