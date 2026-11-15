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
        
    data = data.copy()
    data['_category'] = data['video_name'].apply(get_video_category)
        
    grouped = data.groupby('_category')
    n_per_category = len(data) // (divisor * len(grouped))
        
    sampled = grouped.apply(lambda g: g.sample(n=min(n_per_category, len(g)), random_state=42))
    sampled = sampled.reset_index(drop=True)
    if '_category' in sampled.columns:
        sampled = sampled.drop(columns='_category')

    return sampled


def cap_lab_samples(data, caps, seed=42):
    """Randomly subsample specific lab categories down to a fixed row cap,
    leaving every other category untouched. Unlike balance_sample() (equal
    split across all categories), this lets some labs (e.g. the smaller
    Berlin/Bristol ones) stay fully intact while only capping the
    over-represented ones (e.g. Sharoni/Nieh).

    Caller is responsible for applying this to train_df only, AFTER the
    stem-based train/val split -- capping before the split would shrink the
    video population feeding train_val_split_videos (risking train/val
    membership drifting from an uncapped run) and would also shrink
    validation itself, which should stay on the true, natural distribution.

    Args:
        data: DataFrame with a 'video_name' column.
        caps: dict mapping get_video_category() codes ('0', 'T', 'C') to a
            max row count. Categories not in this dict are kept in full.
        seed: random seed for reproducibility.
    """
    data = data.copy()
    data['_category'] = data['video_name'].apply(get_video_category)

    parts = []
    for category, group in data.groupby('_category'):
        cap = caps.get(category)
        if cap is not None and len(group) > cap:
            group = group.sample(n=cap, random_state=seed)
        parts.append(group)

    out = pd.concat(parts, ignore_index=True)
    return out.drop(columns='_category')


def cap_lab_samples_by_dance(data, caps, n_duration_bins=3, seed=42):
    """Like cap_lab_samples(), but samples whole dances instead of individual
    rows, keeping an equal NUMBER OF DANCES across short/medium/long
    duration buckets (not an equal number of rows).

    cap_lab_samples() samples rows uniformly at random. Two problems with
    that: (1) longer dances have way more rows than short ones (duration and
    row count are correlated ~0.87 on this data), so a plain random row
    sample ends up mostly made of windows from long dances, which can bias
    what the model learns toward long-dance features; (2) it can keep some
    windows of one dance while dropping others, so a single dance's
    start/middle/end don't all stay together.

    An earlier version of this function fixed both of those by capping each
    duration bucket to an equal ROW budget (cap / n_duration_bins). That
    backfires: since long dances contain far more rows each, an equal row
    budget buys noticeably fewer long dances than short ones, so the dance
    *count* per bucket ends up skewed anyway -- just in the opposite
    direction. What we actually want is an equal number of dance *instances*
    per duration bucket. This version:
      1. groups rows into whole dances (video_name, waggle_run_id)
      2. splits each lab's dances into n_duration_bins duration buckets
         (by quantile, so each bucket starts with roughly the same number
         of dances)
      3. shuffles dances within each bucket, then takes them one "round" at
         a time, one dance from every bucket per round, stopping as soon as
         either the smallest bucket runs out of dances, or the next round
         would push the total over the row cap -- whichever comes first.
         This keeps the dance count IDENTICAL across buckets, while still
         respecting the overall row budget.
      4. puts the kept dances' rows back together

    Args:
        data: DataFrame with 'video_name', 'waggle_run_id', 'waggle_start',
            'waggle_end' columns.
        caps: dict mapping get_video_category() codes to a max row count,
            same as cap_lab_samples().
        n_duration_bins: how many duration groups (e.g. short/medium/long)
            to split each lab's dances into before sampling.
        seed: random seed for reproducibility.
    """
    data = data.copy()
    data['_category'] = data['video_name'].apply(get_video_category)
    data['_duration'] = data['waggle_end'] - data['waggle_start']

    parts = []
    for category, group in data.groupby('_category'):
        cap = caps.get(category)
        if cap is None or len(group) <= cap:
            parts.append(group.drop(columns=['_duration']))
            continue

        # one row per dance instance, with its row count and duration
        dances = group.groupby(['video_name', 'waggle_run_id']).agg(
            n_rows=('_duration', 'size'),
            duration=('_duration', 'max'),
        ).reset_index()
        dances['_bin'] = pd.qcut(dances['duration'], n_duration_bins,
                                  duplicates='drop', labels=False)

        # shuffled, per-bucket lists of dances to draw from round-by-round
        bucket_lists = {
            _bin: bin_dances.sample(frac=1, random_state=seed).to_dict('records')
            for _bin, bin_dances in dances.groupby('_bin')
        }

        kept_keys = []
        total_rows = 0
        round_idx = 0
        while all(round_idx < len(lst) for lst in bucket_lists.values()):
            round_rows = sum(lst[round_idx]['n_rows'] for lst in bucket_lists.values())
            if total_rows + round_rows > cap:
                break
            for lst in bucket_lists.values():
                dance_row = lst[round_idx]
                kept_keys.append((dance_row['video_name'], dance_row['waggle_run_id']))
            total_rows += round_rows
            round_idx += 1

        kept_mask = group.set_index(['video_name', 'waggle_run_id']).index.isin(kept_keys)
        parts.append(group[kept_mask].drop(columns=['_duration']))

    out = pd.concat(parts, ignore_index=True)
    return out.drop(columns='_category')