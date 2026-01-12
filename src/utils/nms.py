import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import DBSCAN
from typing import List, Dict, Tuple, Optional
import pandas as pd 

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

def postprocess_predictions(predictions, strategy='cluster_consolidate',
                          spatial_threshold=30.0, temporal_threshold=10,
                          confidence_threshold=0.8, mode='mean'):
    """
    Run post-processing using the specified strategy:
    'nms' | 'max' | 'weighted' | 'cluster' | 'cluster_consolidate' | cluster_consolidate_v2
    """ 
    if strategy == 'cluster_consolidate':
        preds = cluster_and_consolidate_waggles(
            predictions, 
            spatial_threshold=spatial_threshold,
            temporal_threshold=temporal_threshold,
            min_confidence=confidence_threshold, 
            mode=mode
        )
    elif strategy == 'nms':
        preds = point_nms(predictions, confidence_threshold, spatial_threshold)
    elif strategy == 'max':
        preds = max_confidence_filter(predictions, confidence_threshold)
    elif strategy == 'weighted':
        preds = weighted_mean_point(predictions, confidence_threshold, spatial_threshold)
    elif strategy == 'cluster_max': 
        preds = cluster_and_select_max(predictions, confidence_threshold, spatial_threshold)
    elif strategy == 'line_nms':
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
                                  strategy='nms', mode='mean'):
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
            mode=mode
        )
        processed_batch.append(processed['filtered_predictions'])
    return processed_batch

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