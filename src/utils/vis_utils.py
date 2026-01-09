import numpy as np
import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
import os
from PIL import Image, ImageDraw
import cv2 

def reverse_transform(tensor, original_size=(960, 540)):
    """
    Helper functions for batchwise reverse transform.
    Reverse transform for entire sequence tensor [3, 16, 224, 224]
    Returns list of 16 PIL images at original_size
    """
    # tensor shape: [3, 16, 224, 224]
    frames = []
    
    # Reverse normalization for entire batch
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)  # [3, 1, 1, 1]
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)   # [3, 1, 1, 1]
    
    tensor = tensor * std + mean
    tensor = torch.clamp(tensor, 0, 1)
    
    # Convert each frame to PIL and resize
    for frame_idx in range(tensor.shape[1]):
        frame = tensor[:, frame_idx, :, :]  # [3, 224, 224]
        pil_image = T.ToPILImage()(frame)
        pil_image = pil_image.resize(original_size)
        frames.append(pil_image)
    
    return frames

def reverse_transform_batch(batch_tensor, original_size=(960, 540)):
    """
    Reverse transform for batch tensor [B, 3, 16, 224, 224]
    Returns list of lists: [[batch_0_frames], [batch_1_frames], ...]
    """
    batch_frames = []
    
    for i in range(batch_tensor.shape[0]):
        # Extract single sequence: [3, 16, 224, 224]
        sequence_tensor = batch_tensor[i]
        frames = reverse_transform(sequence_tensor, original_size)
        batch_frames.append(frames)
    
    return batch_frames


def save_frames(frames, output_dir, start_frame_idx, base_filename="frame"):
    """
    Save list of PIL images to a folder using global frame numbering
    
    Args:
        frames: List of 16 PIL Images (the waggle sequence)
        output_dir: Directory to save frames
        start_frame_idx: Global starting frame index for this sequence
        base_filename: Base name for frames
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Validate we have exactly 16 frames
    if len(frames) != 16:
        print(f"Warning: Expected 16 frames, got {len(frames)}")
    
    # Save each frame with global frame numbering
    for i, frame in enumerate(frames):
        # Calculate global frame number
        global_frame_idx = start_frame_idx + i
        
        # Create filename with global frame number
        filename = f"{base_filename}_{global_frame_idx:06d}.jpg"  # Use 6 digits for larger numbers
        filepath = os.path.join(output_dir, filename)
        
        # Save as JPEG
        frame.save(filepath, "JPEG", quality=95)
        
    print(f"Saved {len(frames)} frames (global indices {start_frame_idx}-{start_frame_idx + len(frames) - 1}) to {output_dir}")


def save_frames(frames, output_dir, start_frame_idx, base_filename="frame"):
    """
    Save list of PIL images to a folder using global frame numbering
    
    Args:
        frames: List of 16 PIL Images (the waggle sequence)
        output_dir: Directory to save frames
        start_frame_idx: Global starting frame index for this sequence
        base_filename: Base name for frames
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Validate we have exactly 16 frames
    if len(frames) != 16:
        print(f"Warning: Expected 16 frames, got {len(frames)}")
    
    # Save each frame with global frame numbering
    for i, frame in enumerate(frames):
        # Calculate global frame number
        global_frame_idx = start_frame_idx + i
        
        # Create filename with global frame number
        filename = f"{base_filename}_{global_frame_idx:06d}.jpg"  # Use 6 digits for larger numbers
        filepath = os.path.join(output_dir, filename)
        
        # Save as JPEG
        frame.save(filepath, "JPEG", quality=95)
        
    print(f"Saved {len(frames)} frames (global indices {start_frame_idx}-{start_frame_idx + len(frames) - 1}) to {output_dir}")

