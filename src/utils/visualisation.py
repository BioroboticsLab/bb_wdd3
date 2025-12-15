import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from typing import Optional


def denormalize_tensor(tensor, mean, std):
    """
    Denormalize a tensor using mean and std.
    
    Args:
        tensor: Tensor of shape (C, H, W) or (C, T, H, W)
        mean: List of mean values per channel
        std: List of std values per channel
        
    Returns:
        Denormalized tensor
    """
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)
    
    if tensor.ndim == 4:  # (C, T, H, W)
        mean = mean.unsqueeze(-2)
        std = std.unsqueeze(-2)
    elif tensor.ndim != 3:  # Not (C, H, W)
        raise ValueError("Tensor must be 3D (C, H, W) or 4D (C, T, H, W)")
    
    return tensor * std + mean


def visualize_gt_video_snippets(
    dataset,
    idx: int = 0,
    grid_size: int = 28,
    mean: list = [0.485, 0.456, 0.406],
    std: list = [0.229, 0.224, 0.225],
    save_dir: str = './visualizations',
    num_augs: int = 3,
    fps: int = 15
):
    """
    Visualize one video snippet with and without augmentations.
    
    Args:
        dataset: VideoYoloDataset instance
        idx: Sample index to visualize
        grid_size: Grid size used in dataset
        mean: Normalization mean
        std: Normalization std
        save_dir: Output directory
        num_augs: Number of augmented versions to generate
        fps: Frames per second for output videos
    """
    import copy
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Get original sample (no augmentation)
    orig_dataset = copy.deepcopy(dataset)
    orig_dataset.augment = False
    orig_sample = orig_dataset[idx]
    
    def save_video_with_gt(sample, tag):
        """Helper to save a single video with GT overlay."""
        video = sample['video']  # (C, T, H, W)
        target = sample['targets']
        meta = sample.get('metadata', {})
        aug_info = meta.get('augmentation', [])
        aug_text = "_".join(aug_info) if aug_info else "None"
        T = video.shape[1]
        
        # Extract GT position from grid
        obj_mask = target[:, :, 0, 0] > 0.5
        px_x = px_y = dir_x = dir_y = None
        
        if obj_mask.any():
            grid_y = torch.where(obj_mask)[0][0].item()
            grid_x = torch.where(obj_mask)[1][0].item()
            cell_x = target[grid_y, grid_x, 0, 1].item()
            cell_y = target[grid_y, grid_x, 0, 2].item()
            dir_x = target[grid_y, grid_x, 0, 3].item()
            dir_y = target[grid_y, grid_x, 0, 4].item()
            
            # Convert to pixel coordinates
            img_w = img_h = 224
            x_norm = (grid_x + cell_x) / grid_size
            y_norm = (grid_y + cell_y) / grid_size
            px_x = x_norm * img_w
            px_y = y_norm * img_h
        
        # Create frames with GT overlay
        frames = []
        for t in range(T):
            frame = denormalize_tensor(video[:, t, :, :], mean, std)
            frame_np = (frame.permute(1, 2, 0).cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
            
            # Create matplotlib figure
            fig, ax = plt.subplots(figsize=(3.2, 3.2), dpi=100)
            ax.imshow(frame_np)
            ax.axis('off')
            
            # Draw GT if available
            if px_x is not None and px_y is not None:
                ax.plot(px_x, px_y, 'ro', markersize=6)
                if abs(dir_x) > 0.01 or abs(dir_y) > 0.01:
                    ax.arrow(px_x, px_y, dir_x * 20, dir_y * 20,
                            head_width=4, head_length=4,
                            fc='yellow', ec='yellow', lw=1.5)
            
            # Convert figure to numpy array
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            frame_with_gt = np.frombuffer(renderer.buffer_rgba(), dtype=np.uint8)
            frame_with_gt = frame_with_gt.reshape(
                fig.canvas.get_width_height()[::-1] + (4,)
            )
            frame_with_gt = frame_with_gt[..., :3]
            plt.close(fig)
            frames.append(frame_with_gt)
        
        # Save as video
        video_path = os.path.join(save_dir, f'sample_{tag}_{aug_text}.mp4')
        imageio.mimsave(video_path, frames, fps=fps)
        print(f"  Saved {video_path} ({T} frames)")
    
    # Save original
    save_video_with_gt(orig_sample, "original")
    
    # Save augmented versions
    for i in range(num_augs):
        aug_sample = dataset[idx]
        save_video_with_gt(aug_sample, f"aug{i+1}")
    
    print(f"✓ Saved {1 + num_augs} versions for sample {idx}")


def save_middle_frame_grid(save_dir: str, out_name: str = 'middle_frames.png'):
    """
    Create a grid of middle frames from all videos in save_dir.
    
    Args:
        save_dir: Directory containing MP4 files
        out_name: Output filename for the grid image
    """
    video_files = [f for f in os.listdir(save_dir) if f.endswith('.mp4')]
    
    if not video_files:
        print("No video files found in", save_dir)
        return
    
    # Sort: original first, then rest
    video_files.sort()
    video_files = sorted(video_files, key=lambda x: 0 if "original" in x.lower() else 1)
    
    frames = []
    captions = []
    
    # Extract middle frames
    for idx, vf in enumerate(video_files):
        vid_path = os.path.join(save_dir, vf)
        reader = imageio.get_reader(vid_path)
        num_frames = reader.count_frames()
        mid_idx = 0
        mid_frame = reader.get_data(mid_idx)
        reader.close()
        
        frames.append(mid_frame)
        
        # Create caption
        if "original" in vf.lower():
            caption = "a. Original"
        else:
            augtext = vf.replace(".mp4", "").replace("sample_", "")
            parts = augtext.split("_")
            
            # Remove augmentation indices
            clean_parts = []
            for p in parts:
                if p.lower().startswith("aug") and p[3:].isdigit():
                    continue
                clean_parts.append(p)
            
            augtext = " ".join(clean_parts)
            prefix = chr(ord('a') + len(captions)) + ". "
            caption = prefix + augtext
        
        captions.append(caption)
    
    # Create grid
    N = len(frames)
    cols = min(N, 4)
    rows = int(np.ceil(N / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    axes = np.array(axes).reshape(rows, cols)
    
    for i, ax in enumerate(axes.flat):
        ax.axis("off")
        if i < N:
            ax.imshow(frames[i])
            ax.text(0.5, -0.12, captions[i],
                   transform=ax.transAxes,
                   ha='center', va='top',
                   fontsize=12)
    
    out_path = os.path.join(save_dir, out_name)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved grid: {out_path}")


def create_dataset_heatmap(
    dataset,
    grid_size: int = 28,
    output_path: str = './dataset_heatmap.png'
):
    """
    Create a heatmap showing distribution of waggle dances across the grid.
    
    Args:
        dataset: VideoYoloDataset instance
        grid_size: Grid size
        output_path: Output file path
    """
    from torch.utils.data import DataLoader
    from src.data import TemporalWaggleCollator
    
    heatmap = np.zeros((grid_size, grid_size), dtype=np.int64)
    
    loader = DataLoader(dataset, batch_size=8, collate_fn=TemporalWaggleCollator())
    
    print("Computing heatmap...")
    for batch in loader:
        targets = batch['targets']
        obj_mask = targets[..., 0] > 0
        heatmap += obj_mask.sum(dim=(0, -1)).numpy()
    
    # Plot heatmap
    plt.figure(figsize=(8, 8))
    plt.imshow(heatmap, cmap='hot', interpolation='nearest')
    plt.colorbar(label="Number of Waggle Dances")
    plt.title(f"Dataset Waggle Dance Distribution ({grid_size}×{grid_size})")
    plt.xlabel("Grid X")
    plt.ylabel("Grid Y")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    print(f"✓ Heatmap saved to {output_path}")


def visualize_single_sample_side_by_side(dataset, idx=0, save_path="./comparison_sample.png",
                                         mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Visualize one video sample from dataset as side-by-side:
    - Left: original frame (original resolution)
    - Right: processed 224x224 frame
    - Both annotated with GT
    """
    sample = dataset[idx]
    video_tensor = sample['video']      # (C, T, H, W)
    targets = sample['targets']
    metadata = sample['metadata']

    print(metadata)
    
    # Load original video frame
    video_path = os.path.join("/home/prajnanab99/yolo_old_method/data/resvideos",metadata['video_name'])
    start_frame = metadata['start_frame']
    
    reader = imageio.get_reader(video_path)
    orig_frame = reader.get_data(start_frame)
    reader.close()
    
    # Processed middle frame
    T = video_tensor.shape[1]
    mid_idx = T // 2
    frame_tensor = video_tensor[:, mid_idx, :, :]
    frame_np = (denormalize_tensor(frame_tensor, mean, std)
                .permute(1, 2, 0).cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
    
    # Extract GT position
    obj_mask = targets[:, :, 0, 0] > 0
    orig_x, orig_y = metadata["original_coords"]
    px_x_orig, px_y_orig = orig_x, orig_y
    px_x_proc = px_y_proc = None
    if obj_mask.any():
        grid_y, grid_x = torch.where(obj_mask)
        grid_y, grid_x = grid_y[0].item(), grid_x[0].item()
        cell_x = targets[grid_y, grid_x, 0, 1].item()
        cell_y = targets[grid_y, grid_x, 0, 2].item()
        dir_x = targets[grid_y, grid_x, 0, 3].item()
        dir_y = targets[grid_y, grid_x, 0, 4].item()
        
        # # Original frame GT
        # px_x_orig = (grid_x + cell_x) / targets.shape[0] * original_w
        # px_y_orig = (grid_y + cell_y) / targets.shape[1] * original_h
        
        # Processed 224x224 GT
        px_x_proc = (grid_x + cell_x) / targets.shape[0] * 224
        px_y_proc = (grid_y + cell_y) / targets.shape[1] * 224
    
    # Plot side by side
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    
    # Original frame
    axes[0].imshow(orig_frame)
    axes[0].set_title("Original Resolution")
    axes[0].axis("off")
    if px_x_orig is not None:
        axes[0].plot(px_x_orig, px_y_orig, 'ro', markersize=6)
        print(px_x_orig, px_y_orig)
        axes[0].arrow(px_x_orig, px_y_orig, dir_x*20, dir_y*20,
                      head_width=5, head_length=5, fc='yellow', ec='yellow')
    
    # Processed 224x224 frame
    axes[1].imshow(frame_np)
    axes[1].set_title("Processed 224x224")
    axes[1].axis("off")
    if px_x_proc is not None:
        axes[1].plot(px_x_proc, px_y_proc, 'ro', markersize=6)
        axes[1].arrow(px_x_proc, px_y_proc, dir_x*20, dir_y*20,
                      head_width=5, head_length=5, fc='yellow', ec='yellow')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    
    print(f"✓ Saved visualization: {save_path}")


import os
import cv2
import pandas as pd


def visualize_preds_and_gt(video_path, preds_csv=None, gt_csv=None, output_path="vis_output.mp4"):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    preds_by_frame = {}
    if preds_csv is not None:
        for _, row in preds_csv.iterrows():
            for f in range(int(row.start), int(row.end) + 1):
                preds_by_frame.setdefault(f, []).append(row)

    gts_by_frame = {}
    if gt_csv is not None:
        for _, row in gt_csv.iterrows():
            for f in range(int(row.start_frame), int(row.end_frame) + 1):
                gts_by_frame.setdefault(f, []).append(row)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ---------- Predictions (RED) ----------
        if preds_csv is not None and frame_idx in preds_by_frame:
            for det in preds_by_frame[frame_idx]:
                x, y = int(float(det.x)), int(float(det.y))
                dx, dy = float(det.dir_x), float(det.dir_y)

                cv2.circle(frame, (x, y), 8, (0, 0, 255), 2)  # outline only
                arrow_length = 30
                end_x = int(x + dx * arrow_length)
                end_y = int(y + dy * arrow_length)
                cv2.arrowedLine(frame, (x, y), (end_x, end_y), (0, 0, 255), 2)
                cv2.putText(frame, f"P:{det.confidence:.2f}", (x + 10, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # ---------- Ground Truth (GREEN) ----------
        if gt_csv is not None and frame_idx in gts_by_frame:
            for gt in gts_by_frame[frame_idx]:
                x, y = int(gt.x1), int(gt.y1)
                dx, dy = float(gt.direction_x), float(gt.direction_y)

                cv2.circle(frame, (x, y), 8, (0, 255, 0), 2)  # outline only
                arrow_length = 40
                end_x = int(x + dx * arrow_length)
                end_y = int(y + dy * arrow_length)
                cv2.arrowedLine(frame, (x, y), (end_x, end_y), (0, 255, 0), 2)
                cv2.putText(frame, "GT", (x + 10, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        info_text = f"Frame {frame_idx}/{total_frames-1}"
        cv2.putText(frame, info_text, (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"Visualization saved to {output_path}")



def visualise_all_videos_preds_and_gt(video_folder="./data/eval_videos/multi_res/", gt_path=None,gt_folder=None, preds_folder=None,output_folder="./eval_results/visualised_videos", confidence=0.0):
    os.makedirs(output_folder, exist_ok=True)
    gt=False

    video_files = [f for f in os.listdir(video_folder) if f.endswith(('.mp4', '.MP4', '.avi', '.mov'))]
    
    if not video_files:
        print(f"No video files found in {video_folder}")
        return {}
    
    print(f"\nFound {len(video_files)} video(s) to visualise")
    
    for video_file in video_files:
        video_path = os.path.join(video_folder, video_file)
        video_name = os.path.splitext(video_file)[0]
        if preds_folder is not None:
            preds_csv_path = os.path.join(preds_folder, f"{video_name}_preds.csv")
            preds_df = pd.read_csv(preds_csv_path)
            preds_df = preds_df[preds_df['confidence']>=confidence]

        if gt_folder is not None:
            gt=True
            gt_csv_path = os.path.join(gt_folder, f"{video_name}_waggle_annotations_expanded.csv")
            # print(gt_csv_path)
            if not os.path.exists(gt_csv_path):
                continue
            gt_df = pd.read_csv(gt_csv_path)
        elif gt_path is not None:
            gt=True
            gt_df = pd.read_csv(gt_path)
            gt_df = gt_df[gt_df['video_name']==(video_name+".mp4")]

        output_video_path = os.path.join(output_folder, f"{video_name}_after_post_process.mp4")
        visualize_preds_and_gt(video_path,preds_df if preds_folder is not None else None, gt_df if gt is not None else None,output_video_path)

        print("Completed processing ", video_name)
    print("Visualisation Completed!")