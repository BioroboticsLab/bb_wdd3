import pandas as pd
from typing import List, Dict, Tuple
import torch 
import torchvision.transforms as T

def df_to_detections(df: pd.DataFrame) -> List[Dict]:
    """Convert a pandas DataFrame back into a list of detection dictionaries.
    
    Expected DataFrame columns:
    start, end, x, y, dir_x, dir_y, confidence
    """
    detections = []

    if df.empty:
        return detections

    for _, row in df.iterrows():
        detections.append({
            "temporal_offsets": (row["start_frame"], row["end_frame"]),
            "position": (row["x1"], row["y1"]),
            "direction": (row["direction_x"], row["direction_y"])
            #, "confidence": row['confidence']
        })

    return detections


def detections_to_df(detections: List[Dict]) -> pd.DataFrame:
    """Convert detection dictionaries to pandas DataFrame.
    
    Args:
        detections: List of detection dictionaries with keys:
            - temporal_offsets: (start, end) tuple
            - position: (x, y) tuple
            - direction: (dir_x, dir_y) tuple
            - confidence: float
    
    Returns:
        DataFrame with columns: start, end, x, y, dir_x, dir_y, confidence
    """
    if not detections:
        return pd.DataFrame(columns=["start", "end", "x", "y", "dir_x", "dir_y", "confidence"])
    
    rows = []
    for det in detections:
        rows.append({
            "start": det["temporal_offsets"][0],
            "end": det["temporal_offsets"][1],
            "x": det["position"][0],
            "y": det["position"][1],
            "dir_x": det["direction"][0],
            "dir_y": det["direction"][1],
            "confidence": det["confidence"]
        })
    return pd.DataFrame(rows)

def transform_yolo_to_image_coords(
    norm_x: float, 
    norm_y: float, 
    grid_cell_j: int, 
    grid_cell_i: int, 
    grid_size: int, 
    original_size: tuple = (960, 540)
) -> tuple:
    """Transform YOLO normalized coordinates to image space coordinates."""
    orig_w, orig_h = original_size
    cell_width = orig_w / grid_size
    cell_height = orig_h / grid_size
    
    pos_x = (grid_cell_j + norm_x) * cell_width
    pos_y = (grid_cell_i + norm_y) * cell_height
    
    # clamp to image boundaries
    pos_x = max(0, min(pos_x, orig_w - 1))
    pos_y = max(0, min(pos_y, orig_h - 1))
    
    return float(pos_x), float(pos_y)

def compute_temporal_window(
    start_offset: float,
    end_offset: float,
    window_start: int,
    window_end: int,
    window_size: int = 16
) -> tuple:
    """Compute start and end frames from offsets."""
    start_frame = int(start_offset * window_size + window_start)
    end_frame = int(end_offset * window_size + window_start)
    
    # Clamp to valid ranges
    start_frame = max(window_start, min(start_frame, window_end))
    end_frame = max(start_frame, min(end_frame, window_end))
    
    return int(start_frame), int(end_frame)

def yolo_to_img_space(
    model_output: torch.Tensor,
    all_starts: List[int],  # window start frame
    all_ends: List[int],    # window end frames
    window_size: int = 16,
    confidence_threshold: float = 0.8,
    all_original_size: List[tuple] = (960,540),
    all_fps: int=60
) -> List[List[Dict]]:
    all_detections = []
    batch_size, grid_size, _, _, _ = model_output.shape
    
    for b in range(batch_size):
        sample_detections = []
        window_start = all_starts[b]  
        window_end = all_ends[b]
        original_size = all_original_size[b]
        fps = all_fps[b]
        
        for i in range(grid_size):
            for j in range(grid_size):
                for k in range(model_output.shape[3]):
                    detection = model_output[b, i, j, k].detach().cpu()
                    confidence = torch.sigmoid(detection[0]).item()
                    
                    if confidence > confidence_threshold:
                        norm_x = detection[1].item()
                        norm_y = detection[2].item()
                        dir_x = detection[3].item()
                        dir_y = detection[4].item()
                        start_offset = detection[5].item()
                        end_offset = detection[6].item()
                        
                        pos_x, pos_y = transform_yolo_to_image_coords(
                            norm_x, norm_y, j, i, grid_size, original_size
                        )
                        start_frame, end_frame = compute_temporal_window(
                            start_offset, end_offset, window_start, window_end, window_size
                        )
                        
                        sample_detections.append({
                            "confidence": float(confidence),
                            "position": [pos_x, pos_y],
                            "direction": [float(dir_x), float(dir_y)],
                            "grid_cell": [i, j, k],
                            "temporal_offsets": [start_frame, end_frame],
                            "fps": fps
                        })
        
        all_detections.append(sample_detections)
    
    return all_detections 

def yolo_to_img_space_gt(
    model_output: torch.Tensor,
    all_starts: List[int],
    all_ends: List[int],
    window_size: int = 16,
    all_original_size: List[tuple] = (960,540),
    all_fps: int=60
) -> List[List[Dict]]:
    """Transform ground truth YOLO format to image space (no confidence thresholding)"""
    all_detections = []
    batch_size, grid_size, _, _, _ = model_output.shape
    for b in range(batch_size):
        sample_detections = []
        window_start = all_starts[b]  
        window_end = all_ends[b]
        original_size = all_original_size[b]
        fps = all_fps[b]
        
        for i in range(grid_size):
            for j in range(grid_size):
                for k in range(model_output.shape[3]):
                    detection = model_output[b, i, j, k].detach().cpu()
                    confidence = detection[0].item()  # gt confidence is 0 or 1
                    
                    # only process if there's actually an object (confidence > 0)
                    if confidence > 0:
                        norm_x = detection[1].item()
                        norm_y = detection[2].item()
                        dir_x = detection[3].item()
                        dir_y = detection[4].item()
                        start_offset = detection[5].item()
                        end_offset = detection[6].item()
                        
                        pos_x, pos_y = transform_yolo_to_image_coords(
                            norm_x, norm_y, j, i, grid_size, original_size
                        )
                        start_frame, end_frame = compute_temporal_window(
                            start_offset, end_offset, window_start, window_end, window_size
                        )
                        
                        sample_detections.append({
                            "confidence": float(confidence),  # is 1.0 for GT
                            "position": [pos_x, pos_y],
                            "direction": [float(dir_x), float(dir_y)],
                            "grid_cell": [i, j, k],
                            "temporal_offsets": [start_frame, end_frame],
                            "is_ground_truth": True,  # Optional: mark as GT for debug
                            "fps": fps
                        })
        
        all_detections.append(sample_detections)
    
    return all_detections


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