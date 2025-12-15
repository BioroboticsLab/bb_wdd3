import torch
from typing import List, Dict, Tuple
from tqdm import tqdm

def extract_detections(model_output: torch.Tensor,
                          window_frames: Tuple[int, int],
                          window_size: int = 16,
                          frame_idx: int = 0,
                          confidence_threshold: float = 0.5,
                          original_size: tuple = (960, 540)) -> List[Dict]:
        
        """
            Extract high-confidence detections from model output grid.
            
            This function converts the model's grid-based predictions into a list
            of detection dictionaries with absolute coordinates and frame indices.
            
            Args:
                model_output: Model predictions of shape (1, grid_h, grid_w, max_det, 7)
                window_frames: (start_frame, end_frame) of the current window
                window_size: Number of frames in the input clip
                frame_idx: Current frame index for tracking
                confidence_threshold: Minimum confidence to keep a detection
                original_size: (width, height) of the original video
                
            Returns:
                List of detection dictionaries with keys:
                - 'frame_idx': Current processing frame
                - 'confidence': Detection confidence score
                - 'position': [x, y] in original video coordinates
                - 'direction': [dx, dy] normalized direction vector
                - 'grid_cell': [i, j, k] source cell indices
                - 'temporal_offsets': [start_frame, end_frame] in video coordinates
        """
        detections = []
        grid_h, grid_w = model_output.shape[1], model_output.shape[2]
        orig_w, orig_h = original_size
        window_start, window_end = window_frames
        
        cell_width = orig_w / grid_w
        cell_height = orig_h / grid_h
        
        for i in range(grid_h):
            for j in range(grid_w):
                for k in range(model_output.shape[3]):
                    detection = model_output[0, i, j, k].detach().cpu()
                    
                    confidence = torch.sigmoid(detection[0]).item()
                    
                    if confidence > confidence_threshold:
                        norm_x = detection[1].item()
                        norm_y = detection[2].item()
                        dir_x = detection[3].item()
                        dir_y = detection[4].item()
                        start_offset = detection[5].item()
                        end_offset = detection[6].item()
                        
                        pos_x = (j + norm_x) * cell_width
                        pos_y = (i + norm_y) * cell_height
                        
                        pos_x = max(0, min(pos_x, orig_w - 1))
                        pos_y = max(0, min(pos_y, orig_h - 1))
                        
                        start_frame = int(start_offset * window_size + window_start)
                        end_frame = int(end_offset * window_size + window_start)
                        
                        start_frame = max(window_start, min(start_frame, window_end))
                        end_frame = max(start_frame, min(end_frame, window_end))
                        
                        detections.append({
                            "frame_idx": frame_idx,
                            "confidence": float(confidence),
                            "position": [float(pos_x), float(pos_y)],
                            "direction": [float(dir_x), float(dir_y)],
                            "grid_cell": [i, j, k],
                            "temporal_offsets": [int(start_frame), int(end_frame)]
                        })
        
        return detections

import numpy as np
import pandas as pd
from typing import List, Dict
from sklearn.cluster import DBSCAN


def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def temporal_iou(start1: int, end1: int, start2: int, end2: int) -> float:
    """Calculate temporal Intersection over Union between two intervals."""
    inter_start = max(start1, start2)
    inter_end = min(end1, end2)
    intersection = max(0, inter_end - inter_start + 1)

    union = (end1 - start1 + 1) + (end2 - start2 + 1) - intersection

    if union == 0:
        return 0.0

    tIoU = intersection / union
    return tIoU

def temporal_iou_for_merging(a, b):
    inter = max(0, min(a.end, b.end) - max(a.start, b.start))
    union = max(a.end, b.end) - min(a.start, b.start)
    return inter / union if union > 0 else 0

def merge_detections(df, dist_thresh=50, tIoU_thresh=0.3, dir_thresh=0.8):
    df = df.copy()
    clusters = [-1] * len(df)
    cluster_id = 0
    
    # Assign cluster IDs
    for i in range(len(df)):
        if clusters[i] != -1:
            continue

        clusters[i] = cluster_id

        for j in range(i+1, len(df)):
            if clusters[j] != -1:
                continue

            a, b = df.iloc[i], df.iloc[j]

            # Spatial distance
            dist = np.hypot(a.x - b.x, a.y - b.y)

            # Temporal IoU
            tIoU = temporal_iou_for_merging(a, b)

            # Direction similarity
            cs = cosine_similarity(a, b)

            if (tIoU > tIoU_thresh and cs > dir_thresh) or \
               (dist < dist_thresh and cs > dir_thresh):
                clusters[j] = cluster_id

        cluster_id += 1
    
    df["cluster"] = clusters
    
    # ---- MERGING CLUSTERS ----
    merged_rows = []

    for c in df["cluster"].unique():
        group = df[df["cluster"] == c]

        # pick the row with earliest start frame
        first_idx = group["start"].idxmin()
        first_row = df.loc[first_idx]

        merged = {
            "start": group["start"].min(),
            "end": group["end"].max(),

            # <-- IMPORTANT: use (x, y) from earliest detection
            "x": first_row.x,
            "y": first_row.y,

            # average directions (still useful)
            "dir_x": group["dir_x"].mean(),
            "dir_y": group["dir_y"].mean(),

            "confidence": group["confidence"].mean(),
        }

        merged_rows.append(merged)

    return pd.DataFrame(merged_rows)

def nms_spatial(
    detections: List[Dict], 
    spatial_thresh: float = 50
) -> List[Dict]:
    """Apply Non-Maximum Suppression based on spatial proximity only.
    
    Args:
        detections: List of detection dictionaries
        spatial_thresh: Spatial distance threshold for suppression
    
    Returns:
        List of detections after NMS
    """
    if not detections:
        return []
    
    detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
    
    keep = []
    suppressed = [False] * len(detections)
    
    for i, det_i in enumerate(detections):
        if suppressed[i]:
            continue
        
        keep.append(det_i)
        
        for j in range(i + 1, len(detections)):
            if suppressed[j]:
                continue
            
            det_j = detections[j]
            
            dx = det_i['position'][0] - det_j['position'][0]
            dy = det_i['position'][1] - det_j['position'][1]
            spatial_dist = np.sqrt(dx**2 + dy**2)
            
            if spatial_dist < spatial_thresh:
                suppressed[j] = True
    
    return keep

def cluster_and_consolidate_waggles_v2(predictions, spatial_threshold=30.0, temporal_threshold=10, 
                                       min_confidence=0.8, mode='median',
                                       apply_nms_after=True, nms_spatial_thresh=30.0, 
                                       nms_temporal_iou_thresh=0.5):
    """
    Enhanced version: Cluster waggle detections, consolidate, then apply spatiotemporal NMS.
    
    Args:
        predictions: List of waggle detection dictionaries
        spatial_threshold: DBSCAN spatial clustering threshold (pixels)
        temporal_threshold: DBSCAN temporal clustering threshold (frames)  
        min_confidence: Minimum confidence threshold
        mode: 'mean' or 'median' for aggregation method
        apply_nms_after: Whether to apply NMS after clustering (RECOMMENDED: True)
        nms_spatial_thresh: Spatial threshold for post-clustering NMS
        nms_temporal_iou_thresh: Temporal IoU threshold for post-clustering NMS
    """
    # Filter by confidence first
    preds = [p for p in predictions if p['confidence'] >= min_confidence]
    if not preds:
        return []

    # Prepare features for clustering: spatial + temporal
    features = []
    for pred in preds:
        x, y = pred['position']
        # Use frame offset as temporal feature
        frame_mid = (pred['temporal_offsets'][0] + pred['temporal_offsets'][1]) / 2
        features.append([x, y, frame_mid])
    
    features = np.array(features)
    
    # Scale features so spatial and temporal are comparable
    scaled_features = []
    for feature in features:
        x, y, frame_mid = feature
        # Scale temporal dimension: convert frames to "pixel-equivalent" units
        temporal_scale = spatial_threshold / temporal_threshold
        scaled_features.append([x, y, frame_mid * temporal_scale])
    
    scaled_features = np.array(scaled_features)
    
    # Now use spatial_threshold for the scaled features
    clustering = DBSCAN(eps=spatial_threshold, min_samples=1).fit(scaled_features)
    labels = clustering.labels_

    consolidated = []
    for cluster_id in np.unique(labels):
        cluster_points = [p for p, lbl in zip(preds, labels) if lbl == cluster_id]
        
        if not cluster_points:
            continue
            
        # Choose aggregation function based on mode
        if mode == 'mean':
            agg_func = np.mean
        elif mode == 'median':
            agg_func = np.median
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'mean' or 'median'")
        
        # Compute position using selected aggregation
        positions = np.array([p['position'] for p in cluster_points])
        aggregated_position = tuple(agg_func(positions, axis=0))
        
        # Compute direction using selected aggregation
        directions = np.array([p['direction'] for p in cluster_points])
        aggregated_direction = tuple(agg_func(directions, axis=0))
        
        # Compute merged temporal range (always min/max)
        start_times = [p['temporal_offsets'][0] for p in cluster_points]
        end_times = [p['temporal_offsets'][1] for p in cluster_points]
        merged_start = min(start_times)
        merged_end = max(end_times)
        
        # Compute mean confidence
        mean_confidence = np.mean([p['confidence'] for p in cluster_points])
        
        # Create consolidated detection
        consolidated_detection = {
            'position': aggregated_position,
            'direction': aggregated_direction,
            'temporal_offsets': (merged_start, merged_end),
            'confidence': mean_confidence,
            'num_original_detections': len(cluster_points),
            'cluster_id': cluster_id,
            'aggregation_mode': mode
        }
        consolidated.append(consolidated_detection)
    
    if apply_nms_after and len(consolidated) > 1:
        consolidated = nms_spatiotemporal(
            consolidated, 
            spatial_thresh=nms_spatial_thresh,
            temporal_iou_thresh=nms_temporal_iou_thresh
        )
    
    return consolidated

def nms_spatiotemporal(
    detections: List[Dict],
    spatial_thresh: float = 40,
    temporal_iou_thresh: float = 0.5
) -> List[Dict]:
    """Apply NMS using both spatial and temporal criteria.
    
    Args:
        detections: List of detection dictionaries
        spatial_thresh: Spatial distance threshold
        temporal_iou_thresh: Temporal IoU threshold
    
    Returns:
        List of detections after spatiotemporal NMS
    """
    if not detections:
        return []

    detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)

    keep = []
    suppressed = [False] * len(detections)

    for i, det_i in enumerate(detections):
        if suppressed[i]:
            continue

        keep.append(det_i)

        for j in range(i + 1, len(detections)):
            if suppressed[j]:
                continue

            det_j = detections[j]

            dx = det_i['position'][0] - det_j['position'][0]
            dy = det_i['position'][1] - det_j['position'][1]
            spatial_dist = np.sqrt(dx**2 + dy**2)

            if spatial_dist < spatial_thresh:
                t_iou = temporal_iou(
                    det_i['temporal_offsets'][0], det_i['temporal_offsets'][1],
                    det_j['temporal_offsets'][0], det_j['temporal_offsets'][1]
                )
                if t_iou > temporal_iou_thresh:
                    suppressed[j] = True

    return keep


def dbscan_merge_detections(
    detections: list,
    time_eps: int = 5,
    space_eps: int = 20,
    dir_thresh: float = 0.5,
):
    """
    detections = [
        {
            "confidence": float,
            "position": [x, y],
            "direction": [dx, dy],
            "grid_cell": [i, j, k],
            "temporal_offsets": [start, end]
        },
        ...
    ]
    """

    if len(detections) == 0:
        return []

    # Extract arrays
    starts = np.array([d["temporal_offsets"][0] for d in detections], dtype=float)
    ends   = np.array([d["temporal_offsets"][1] for d in detections], dtype=float)

    pos   = np.array([d["position"] for d in detections], dtype=float)   # (N, 2)
    dirs  = np.array([d["direction"] for d in detections], dtype=float)  # (N, 2)
    confs = np.array([d["confidence"] for d in detections], dtype=float)

    # Normalize direction vectors
    norms = np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-8
    unit_dirs = dirs / norms

    # Build DBSCAN feature vector
    # [scaled_time, scaled_x, scaled_y, scaled_dir_x, scaled_dir_y]
    N = len(detections)
    X = np.zeros((N, 5), dtype=float)

    X[:, 0] = starts / time_eps
    X[:, 1] = pos[:, 0] / space_eps
    X[:, 2] = pos[:, 1] / space_eps

    # Direction scaling based on threshold
    dir_scale = 1.0 / max((1 - dir_thresh), 1e-3)
    X[:, 3:5] = unit_dirs * dir_scale

    # Clustering
    clustering = DBSCAN(eps=1.0, min_samples=1).fit(X)
    labels = clustering.labels_

    merged = []

    for c in np.unique(labels):
        idx = np.where(labels == c)[0]

        merged_detection = {
            "temporal_offsets": [
                int(starts[idx].min()),
                int(ends[idx].max())
            ],
            "position": [
                float(pos[idx, 0].mean()),
                float(pos[idx, 1].mean())
            ],
            "direction": [
                float(dirs[idx, 0].mean()),
                float(dirs[idx, 1].mean())
            ],
            "confidence": float(confs[idx].mean()),
        }

        merged.append(merged_detection)

    return merged

def dbscan_merge(pred_df: pd.DataFrame, time_eps: int = 5, space_eps: int = 20, 
                 dir_thresh: float = 0.5) -> pd.DataFrame:
    """Merge nearby detections using DBSCAN clustering.
    
    Args:
        pred_df: DataFrame with columns: start, end, x, y, dir_x, dir_y, confidence
        time_eps: Temporal clustering epsilon (frames)
        space_eps: Spatial clustering epsilon (pixels)
        dir_thresh: Direction similarity threshold (unused in current implementation)
    
    Returns:
        Merged DataFrame with clustered detections averaged
    """
    # Prepare data for clustering: combine start frame and position
    X = pred_df[["start", "x", "y"]].values.copy()

    # Scale time and space to balance distances
    X[:, 0] = X[:, 0] / time_eps
    X[:, 1] = X[:, 1] / space_eps
    X[:, 2] = X[:, 2] / space_eps

    # DBSCAN clustering
    clustering = DBSCAN(eps=1.0, min_samples=1).fit(X)
    pred_df_copy = pred_df.copy()
    pred_df_copy["cluster"] = clustering.labels_

    merged_rows = []
    for c in pred_df_copy["cluster"].unique():
        group = pred_df_copy[pred_df_copy["cluster"] == c]
        merged_row = {
            "start": group["start"].min(),
            "end": group["end"].max(),
            "x": group["x"].mean(),
            "y": group["y"].mean(),
            "dir_x": group["dir_x"].mean(),
            "dir_y": group["dir_y"].mean(),
            "confidence": group["confidence"].mean()
        }
        merged_rows.append(merged_row)

    merged_df = pd.DataFrame(merged_rows)
    return merged_df



def cluster_and_consolidate_waggles(predictions, spatial_threshold=30.0, temporal_threshold=10, min_confidence=0.8, mode='median'):
    """
    Cluster waggle detections and consolidate into single detections with:
    - mean/median position, mean/median direction, merged temporal range
    
    Args:
        predictions: List of waggle detection dictionaries
        spatial_threshold: DBSCAN spatial clustering threshold (pixels)
        temporal_threshold: DBSCAN temporal clustering threshold (frames)  
        min_confidence: Minimum confidence threshold
        mode: 'mean' or 'median' for aggregation method
    """
    # Filter by confidence first
    preds = [p for p in predictions if p['confidence'] >= min_confidence]
    if not preds:
        return []

    # Prepare features for clustering: spatial + temporal
    features = []
    for pred in preds:
        x, y = pred['position']
        # Use frame offset as temporal feature
        frame_mid = (pred['temporal_offsets'][0] + pred['temporal_offsets'][1]) / 2
        features.append([x, y, frame_mid])
    
    features = np.array(features)
    
    # Scale features so spatial and temporal are comparable
    scaled_features = []
    for feature in features:
        x, y, frame_mid = feature
        # Scale temporal dimension: convert frames to "pixel-equivalent" units
        # temporal_threshold frames should be equivalent to spatial_threshold pixels
        temporal_scale = spatial_threshold / temporal_threshold
        scaled_features.append([x, y, frame_mid * temporal_scale])
    
    scaled_features = np.array(scaled_features)
    
    # Now use spatial_threshold for the scaled features
    clustering = DBSCAN(eps=spatial_threshold, min_samples=1).fit(scaled_features)
    labels = clustering.labels_

    consolidated = []
    for cluster_id in np.unique(labels):
        cluster_points = [p for p, lbl in zip(preds, labels) if lbl == cluster_id]
        
        if not cluster_points:
            continue
            
        # Choose aggregation function based on mode
        if mode == 'mean':
            agg_func = np.mean
        elif mode == 'median':
            agg_func = np.median
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'mean' or 'median'")
        
        # Compute position using selected aggregation
        positions = np.array([p['position'] for p in cluster_points])
        aggregated_position = tuple(agg_func(positions, axis=0))
        
        # Compute direction using selected aggregation
        directions = np.array([p['direction'] for p in cluster_points])
        aggregated_direction = tuple(agg_func(directions, axis=0))
        
        # Compute merged temporal range (always min/max)
        start_times = [p['temporal_offsets'][0] for p in cluster_points]
        end_times = [p['temporal_offsets'][1] for p in cluster_points]
        merged_start = min(start_times)
        merged_end = max(end_times)
        
        # Compute mean confidence
        mean_confidence = np.mean([p['confidence'] for p in cluster_points])
        
        # Create consolidated detection
        consolidated_detection = {
            'position': aggregated_position,
            'direction': aggregated_direction,
            'temporal_offsets': (merged_start, merged_end),
            'confidence': mean_confidence,
            'num_original_detections': len(cluster_points),
            'cluster_id': cluster_id,
            'aggregation_mode': mode
        }
        consolidated.append(consolidated_detection)
    
    return consolidated
    

def point_nms(predictions, min_confidence=0.8, spatial_threshold=30.0):
    """
    Combined confidence filtering and spatial NMS
    """
    # First filter by confidence only
    filtered = [pred for pred in predictions if pred['confidence'] >= min_confidence]
    
    # If no predictions after filtering, return empty
    if not filtered:
        return []
    
    # Sort by confidence (descending) for NMS
    filtered = sorted(filtered, key=lambda x: x['confidence'], reverse=True)
    
    # Apply spatial NMS
    kept = []
    
    while filtered:
        best = filtered.pop(0)
        kept.append(best)
        
        # Remove predictions that are spatially close
        filtered = [pred for pred in filtered if 
                   spatial_distance(best, pred) >= spatial_threshold]
    
    return kept

def spatial_distance(pred1, pred2):
    """Euclidean distance between two points using 'position' key"""
    pos1 = pred1['position']
    pos2 = pred2['position']
    return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

def max_confidence_filter(predictions, min_confidence=0.8):
    """Return only the single highest-confidence prediction"""
    if not predictions:
        return []
    best = max(predictions, key=lambda x: x['confidence'])
    return [best] if best['confidence'] >= min_confidence else []

def weighted_mean_point(predictions, min_confidence=0.8, spatial_threshold=30.0):
    """
    Select the most confident prediction, then compute
    a confidence-weighted mean of nearby points to refine its position
    """
    filtered = [p for p in predictions if p['confidence'] >= min_confidence]
    if not filtered:
        return []
    
    best = max(filtered, key=lambda x: x['confidence'])
    nearby = [p for p in filtered if spatial_distance(best, p) < spatial_threshold]
    
    # Weighted average of positions
    xs = np.array([p['position'][0] for p in nearby])
    ys = np.array([p['position'][1] for p in nearby])
    confs = np.array([p['confidence'] for p in nearby])
    
    x_mean = np.average(xs, weights=confs)
    y_mean = np.average(ys, weights=confs)
    
    refined = dict(best)
    refined['position'] = (x_mean, y_mean)
    refined['refined_from'] = len(nearby)
    return [refined]

def cos_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

def alignment_score(d1, d2, pos1, pos2):
    """Alignment of vector (pos2 - pos1) with direction d1."""
    v = pos2 - pos1
    return cos_sim(v, d1)

def line_nms_group(
    detections,
    dir_thresh=0.5,
    align_thresh=0.4,
    spatial_thresh=150,
    temporal_gap_thresh=10,
    temporal_overlap_bonus=0.1,
):
    """
    detections: list of dicts
        {
            "confidence": float,
            "position": [x, y],
            "direction": [dx, dy],
            "temporal_offsets": [start, end],
            ...
        }
    """

    N = len(detections)
    groups = [-1] * N
    group_id = 0

    # Convert to numpy-friendly structures
    starts = np.array([d["temporal_offsets"][0] for d in detections], dtype=float)
    ends   = np.array([d["temporal_offsets"][1] for d in detections], dtype=float)
    pos    = np.array([d["position"] for d in detections], dtype=float)     # Nx2
    dirs   = np.array([d["direction"] for d in detections], dtype=float)    # Nx2
    confs  = np.array([d["confidence"] for d in detections], dtype=float)

    for i in range(N):

        if groups[i] != -1:
            continue

        groups[i] = group_id

        for j in range(i+1, N):

            if groups[j] != -1:
                continue

            d1 = dirs[i]
            d2 = dirs[j]

            # 1. Direction similarity
            if cos_sim(d1, d2) < dir_thresh:
                continue

            # 2. Straight-line alignment
            align_ij = max(
                alignment_score(d1, d2, pos[i], pos[j]),
                alignment_score(d2, d1, pos[j], pos[i])
            )
            if align_ij < align_thresh:
                continue

            # 3. Spatial closeness
            dist = np.linalg.norm(pos[i] - pos[j])
            if dist >= spatial_thresh:
                continue

            # 4. Temporal logic
            start_i, end_i = starts[i], ends[i]
            start_j, end_j = starts[j], ends[j]

            temporal_gap = max(start_i, start_j) - min(end_i, end_j)

            # Temporal IoU
            inter = max(0, min(end_i, end_j) - max(start_i, start_j))
            union = (end_i - start_i) + (end_j - start_j) - inter
            tIoU = inter / (union + 1e-8)

            has_overlap = tIoU > temporal_overlap_bonus
            has_small_gap = temporal_gap <= temporal_gap_thresh
            is_very_close_spatially = dist < (spatial_thresh * 0.5)

            temporal_ok = has_overlap or (has_small_gap and is_very_close_spatially)

            if temporal_ok:
                groups[j] = group_id

        group_id += 1

    return groups


def line_nms_collapse(detections, groups):
    detections = list(detections)
    groups = np.array(groups)
    
    merged = []
    
    for gid in np.unique(groups):
        idx = np.where(groups == gid)[0]

        # Confidence-weighted representative
        best_idx = idx[np.argmax([detections[i]["confidence"] for i in idx])]
        rep = detections[best_idx]

        merged_detection = {
            "temporal_offsets": [
                int(min(detections[i]["temporal_offsets"][0] for i in idx)),
                int(max(detections[i]["temporal_offsets"][1] for i in idx)),
            ],
            "position": list(rep["position"]),
            "direction": list(rep["direction"]),
            "confidence": float(np.mean([detections[i]["confidence"] for i in idx])),
            "members": list(idx)
        }

        merged.append(merged_detection)

    return merged


def cluster_and_select_max(predictions, min_confidence=0.8, spatial_threshold=30.0):
    """
    Cluster spatially close predictions and select the highest-confidence point per cluster.
    """
    preds = [p for p in predictions if p['confidence'] >= min_confidence]
    if not preds:
        return []

    positions = np.array([p['position'] for p in preds])
    clustering = DBSCAN(eps=spatial_threshold, min_samples=1).fit(positions)
    labels = clustering.labels_

    kept = []
    for cluster_id in np.unique(labels):
        cluster_points = [p for p, lbl in zip(preds, labels) if lbl == cluster_id]
        best = max(cluster_points, key=lambda x: x['confidence'])
        kept.append(best)
    return kept

def postprocess_predictions(predictions, strategy='cluster_consolidate',
                          spatial_threshold=30.0, temporal_threshold=10,
                          confidence_threshold=0.8, mode='median',
                          nms_spatial_thresh=30.0, nms_temporal_iou_thresh=0.5):
    """
    Run post-processing using the specified strategy.
    
    Strategies:
        'cluster_consolidate_v2': Enhanced clustering with spatiotemporal NMS (RECOMMENDED)
        'cluster_consolidate': Original clustering without NMS
        'nms_spatiotemporal': Direct spatiotemporal NMS without clustering
        'nms': Spatial NMS only
        'max': Keep only highest confidence
        'weighted': Weighted mean refinement
        'cluster_max': Cluster then select max per cluster
    """ 
    if strategy == 'cluster_consolidate_v2':
        preds = cluster_and_consolidate_waggles_v2(
            predictions, 
            spatial_threshold=spatial_threshold,
            temporal_threshold=temporal_threshold,
            min_confidence=confidence_threshold, 
            mode=mode,
            apply_nms_after=True,
            nms_spatial_thresh=nms_spatial_thresh,
            nms_temporal_iou_thresh=nms_temporal_iou_thresh
        )
    elif strategy == 'cluster_consolidate':
        preds = cluster_and_consolidate_waggles(
            predictions, 
            spatial_threshold=spatial_threshold,
            temporal_threshold=temporal_threshold,
            min_confidence=confidence_threshold, 
            mode=mode
        )
    elif strategy == 'nms_spatiotemporal':
        filtered = [p for p in predictions if p['confidence'] >= confidence_threshold]
        preds = nms_spatiotemporal(
            filtered,
            spatial_thresh=nms_spatial_thresh,
            temporal_iou_thresh=nms_temporal_iou_thresh
        )
    elif strategy == 'nms':
        preds = point_nms(predictions, confidence_threshold, spatial_threshold)
    elif strategy == 'max':
        preds = max_confidence_filter(predictions, confidence_threshold)
    elif strategy == 'weighted':
        preds = weighted_mean_point(predictions, confidence_threshold, spatial_threshold)
    elif strategy == 'cluster_max': 
        preds = cluster_and_select_max(predictions, confidence_threshold, spatial_threshold)
    elif strategy == 'dbscan':
        preds = dbscan_merge_detections(predictions, confidence_threshold)
    elif strategy == 'line_nms':
        # predictions = dbscan_merge_detections(predictions, confidence_threshold)
        predictions = cluster_and_consolidate_waggles(
            predictions, 
            spatial_threshold=spatial_threshold,
            temporal_threshold=temporal_threshold,
            min_confidence=confidence_threshold, 
            mode=mode
        )
        groups = line_nms_group(predictions, temporal_gap_thresh=temporal_threshold)
        preds = line_nms_collapse(predictions, groups)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    return {'filtered_predictions': preds}


def batch_postprocess_predictions(batch_predictions,
                                  spatial_threshold=30.0,
                                  temporal_threshold=10,
                                  confidence_threshold=0.9,
                                  strategy='cluster_consolidate_v2',
                                  mode='median',
                                  nms_spatial_thresh=30.0,
                                  nms_temporal_iou_thresh=0.5):
    """Apply post-processing to a batch of prediction lists"""
    processed_batch = []
    for sample_predictions in batch_predictions:
        if not sample_predictions:
            processed_batch.append([])
            continue
        processed = postprocess_predictions(
            sample_predictions,
            strategy=strategy,
            spatial_threshold=spatial_threshold,
            temporal_threshold=temporal_threshold,
            confidence_threshold=confidence_threshold,
            mode=mode,
            nms_spatial_thresh=nms_spatial_thresh,
            nms_temporal_iou_thresh=nms_temporal_iou_thresh
        )
        processed_batch.append(processed['filtered_predictions'])
    return processed_batch