import pandas as pd 
import numpy as np 
from typing import List, Dict, Tuple, Optional
import torch 
import os
import cv2
import yaml
from src.utils.video_utils import get_video_category
cv2.setNumThreads(4)

def load_preds_csv(csv_path):
    # Load the CSV file
    df = pd.read_csv(csv_path)
    
    return df

def detections_to_df(detections: List[Dict]) -> "pd.DataFrame":
    # On Single Batch so no batch informations 
    if not detections:
        return pd.DataFrame(columns=["start","end","x","y","dir_x","dir_y","confidence"])
    
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

def fix_dataframe_with_video_lengths(df, video_frame_counts):
    """
    Filter dataframe based on video frame counts.
    video_frame_counts: dict mapping video_name -> frame_count (int)
    """
    df = df.copy()
    valid_rows = []
    
    for idx, row in df.iterrows():
        video_name = row['video_name']
        if video_name not in video_frame_counts:
            continue
        
        total_frames = video_frame_counts[video_name]  # Use int
        end_frame = min(int(row['end_frame']), total_frames)
        start_frame = int(row['start_frame'])
        
        if (end_frame - start_frame) >= 4:
            new_row = row.copy()
            new_row['start_frame'] = start_frame
            new_row['end_frame'] = end_frame
            valid_rows.append(new_row)
    
    return pd.DataFrame(valid_rows).reset_index(drop=True)
    
def prepare_dataset(csv_path, clip_len=32, pos=False):
    df = pd.read_csv(csv_path)

    all_clips_data = []

    for id, row in df.iterrows():
        if pos and row['waggle']==0:
            continue

        video_duration_frames = row["end_frame"] - row["start_frame"]
        
        if video_duration_frames < clip_len:
            num_segments_to_create = 1            
            num_segments_to_create = max(1, video_duration_frames - clip_len + 1)
            
        else:
            num_segments_to_create = video_duration_frames - clip_len + 1

        for i in range(num_segments_to_create):
            clip_info = {
                "video_name": row["video_name"],
                "start_frame": row["start_frame"] + i,
                "end_frame": row["start_frame"] + i + clip_len -1 if video_duration_frames >= clip_len else row["end_frame"],
                "x1": row["x1"],
                "y1": row["y1"],
                "x2": row["x2"],
                "y2": row["y2"],
                "angle": row["angle"],
                "waggle": row["waggle"]
            }
            all_clips_data.append(clip_info)
            
    data = pd.DataFrame(all_clips_data)
    return data


def extract_frames_from_video(video_name, width=224, height=224):
    video_path = os.path.join(os.curdir, "/home/prajna/complete_data/videos/", video_name)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break 

        frame = cv2.resize(frame, (width, height))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frames.append(frame)

    cap.release()
    return frames


def create_video_frames_df(video_names, video_dir='/home/prajna/complete_data/videos/'):
    """Get frame counts for videos without loading frames"""
    frame_counts = {}
    for video_name in video_names:
        video_path = os.path.join(video_dir, video_name)
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_counts[video_name] = frame_count
            cap.release()
        else:
            print(f"Warning: Could not open {video_name}")
            frame_counts[video_name] = 0
    return frame_counts


def find_overlapping_rows(df):
    """Find rows that have temporal overlaps within the same video"""
    overlap_indices = set()
    
    for video_name, group in df.groupby('video_name'):
        group_sorted = group.sort_values('start_frame').reset_index(drop=True)
        
        # Check each segment against subsequent segments
        for i in range(len(group_sorted)):
            current_end = group_sorted.loc[i, 'end_frame']
            
            for j in range(i + 1, len(group_sorted)):
                next_start = group_sorted.loc[j, 'start_frame']
                next_end = group_sorted.loc[j, 'end_frame']
                
                # If next segment starts after current ends, no more overlaps possible
                if next_start > current_end:
                    break
                
                # If we get here, they overlap
                if next_start <= current_end:  # This means they overlap
                    # Get original indices
                    overlap_indices.add(group_sorted.loc[i, 'Unnamed: 0'])
                    overlap_indices.add(group_sorted.loc[j, 'Unnamed: 0'])
    
    # Return rows with original indices
    return df[df['Unnamed: 0'].isin(overlap_indices)].reset_index(drop=True)


def preds_to_df(predictions, prediction_type='raw', sequence_offset=0):
    """
    Convert predictions list to pandas DataFrame for analysis
    
    Args:
        predictions: List of prediction sequences
        prediction_type: 'raw' or 'postprocessed' 
        sequence_offset: Starting sequence index (useful if processing subsets)
    """
    predictions_list = []
    
    for seq_idx, seq_preds in enumerate(predictions):
        global_seq_idx = seq_idx + sequence_offset
        for pred in seq_preds:
            row = {
                'sequence_idx': global_seq_idx,
                'position_x': pred['position'][0],
                'position_y': pred['position'][1],
                'direction_x': pred['direction'][0],
                'direction_y': pred['direction'][1],
                'start_frame': pred['temporal_offsets'][0],
                'end_frame': pred['temporal_offsets'][1],
                'confidence': pred['confidence'],
                'type': prediction_type
            }
            
            # Add post-processing specific fields
            if prediction_type == 'postprocessed':
                row.update({
                    'num_original_detections': pred['num_original_detections'],
                    'cluster_id': pred['cluster_id'],
                    'aggregation_mode': pred['aggregation_mode']
                })
            
            predictions_list.append(row)
    
    return pd.DataFrame(predictions_list)

def save_preds_to_csv(predictions, filename, prediction_type='raw', output_dir='outputs'):
    """
    Convert predictions to DataFrame and save as CSV in specified directory
    
    Args:
        predictions: List of prediction sequences
        filename: Output CSV filename
        prediction_type: 'raw' or 'postprocessed'
        output_dir: Directory to save CSV files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert to DataFrame
    df = preds_to_df(predictions, prediction_type)
    
    # Save to file
    filepath = os.path.join(output_dir, filename)
    df.to_csv(filepath, index=False)
    print(f"Saved {len(df)} {prediction_type} predictions to {filepath}")
    
    return df

def load_config(config_path='config.yaml'):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def balance_sample(data, divisor):
    # used to balance subset sample so each recording is balanced based on groups recording
    # landgraf group, sharaonis group and niehs group
    if divisor == 1:
        return data
        
    data['_category'] = data['video_name'].apply(get_video_category)
        
    grouped = data.groupby('_category')
    n_per_category = len(data) // (divisor * len(grouped))
        
    sampled = grouped.apply(lambda g: g.sample(n=min(n_per_category, len(g)), random_state=42))
    sampled = sampled.reset_index(drop=True).drop(columns='_category')
        
    return sampled