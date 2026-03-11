import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import DBSCAN, HDBSCAN
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


def remove_outliers_density(predictions, min_neighbors=2, spatial_radius=50, temporal_radius=15):
    """
    Remove predictions that don't have enough neighbors in space-time.
    Simpler and more interpretable than Isolation Forest.
    
    Args:
        predictions: List of prediction dicts
        min_neighbors: Minimum number of neighbors to be considered inlier
        spatial_radius: Spatial neighborhood radius (pixels)
        temporal_radius: Temporal neighborhood radius (frames)
    """
    if len(predictions) < min_neighbors + 1:
        return predictions
    
    features = []
    for pred in predictions:
        x, y = pred['position']
        frame_mid = (pred['temporal_offsets'][0] + pred['temporal_offsets'][1]) / 2
        features.append([x, y, frame_mid])
    
    features = np.array(features)
    
    # Scale temporal to match spatial units
    temporal_scale = spatial_radius / temporal_radius
    scaled_features = features.copy()
    scaled_features[:, 2] *= temporal_scale
    
    # Count neighbors for each prediction
    inliers = []
    outlier_count = 0
    
    for i, pred in enumerate(predictions):
        # Calculate distances to all other points
        distances = np.sqrt(np.sum((scaled_features - scaled_features[i])**2, axis=1))
        
        # Count neighbors within radius (excluding self)
        num_neighbors = np.sum((distances > 0) & (distances <= spatial_radius))
        
        if num_neighbors >= min_neighbors:
            inliers.append(pred)
        else:
            outlier_count += 1
    
    if outlier_count > 0:
        print(f"Removed {outlier_count} sparse outliers ({outlier_count/len(predictions)*100:.1f}%)")
    
    return inliers


def remove_outliers_isolation_forest(predictions, contamination=0.1):
    """
    Remove spatial-temporal outliers using Isolation Forest.
    
    Args:
        predictions: List of prediction dicts
        contamination: Expected proportion of outliers (0.05-0.2 typical)
    """
    from sklearn.ensemble import IsolationForest
    
    if len(predictions) < 5:  # Too few to detect outliers meaningfully
        return predictions
    
    # Extract spatial-temporal features
    features = []
    for pred in predictions:
        x, y = pred['position']
        frame_mid = (pred['temporal_offsets'][0] + pred['temporal_offsets'][1]) / 2
        features.append([x, y, frame_mid])
    
    features = np.array(features)
    
    # Fit Isolation Forest
    clf = IsolationForest(contamination=contamination, random_state=42)
    outlier_labels = clf.fit_predict(features)  # 1 = inlier, -1 = outlier
    
    # Keep only inliers
    inliers = [pred for pred, label in zip(predictions, outlier_labels) if label == 1]
    
    removed = len(predictions) - len(inliers)
    if removed > 0:
        print(f"Removed {removed} outliers ({removed/len(predictions)*100:.1f}%)")
    
    return inliers


def cluster_and_consolidate_waggles(predictions, spatial_threshold=30.0, temporal_threshold=10, 
                                   min_confidence=0.8, mode='median',
                                   remove_outliers=True, outlier_method='density',
                                   outlier_min_neighbors=2, hdbscan_min_cluster_size=2,
                                   clustering_method='dbscan'):
    """
    Cluster waggle detections and consolidate into single detections with:
    - mean/median position, mean/median direction, merged temporal range
    
    Args:
        predictions: List of waggle detection dictionaries
        spatial_threshold: DBSCAN spatial clustering threshold (pixels)
        temporal_threshold: DBSCAN temporal clustering threshold (frames)  
        min_confidence: Minimum confidence threshold
        mode: 'mean' or 'median' for aggregation method
        remove_outliers: Whether to remove outliers before clustering
        outlier_method: 'density' or 'isolation_forest'
        outlier_min_neighbors: Minimum neighbors for density-based outlier removal
    """
    # Filter by confidence first
    preds = [p for p in predictions if p['confidence'] >= min_confidence]
    if not preds:
        return []
    
    # Remove outliers before clustering
    if remove_outliers and len(preds) >= 5:
        if outlier_method == 'density':
            preds = remove_outliers_density(
                preds, 
                min_neighbors=outlier_min_neighbors,
                spatial_radius=spatial_threshold * 1.5,  # Slightly larger than cluster radius
                temporal_radius=temporal_threshold * 1.5
            )
        elif outlier_method == 'isolation_forest':
            preds = remove_outliers_isolation_forest(preds, contamination=0.4)
        else:
            raise ValueError(f"Unknown outlier_method: {outlier_method}. Use 'density' or 'isolation_forest'")
        
        if not preds:
            return []

    # Check direction vector norms before clustering
    direction_norms_before = []
    for p in preds:
        dx, dy = p['direction']
        norm = np.sqrt(dx**2 + dy**2)
        direction_norms_before.append(norm)
    
    direction_norms_before = np.array(direction_norms_before)
    not_normed_before = np.abs(direction_norms_before - 1.0) > 0.1
    
    if not_normed_before.any():
        print(f"WARNING: {not_normed_before.sum()} direction vectors deviate >0.1 from norm=1.0")

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
        # Scale temporal dimension: convert frames to pixel-equivalent units
        # temporal_threshold frames should be equivalent to spatial_threshold pixels
        temporal_scale = spatial_threshold / temporal_threshold
        scaled_features.append([x, y, frame_mid * temporal_scale])
    
    scaled_features = np.array(scaled_features)
    # print('Scalef featues shape:', scaled_features.shape)
    # Use spatial_threshold for the scaled features in dbscan, hdbscan does this byitself
    if clustering_method == 'dbscan':
        clustering = DBSCAN(eps=spatial_threshold, min_samples=1).fit(scaled_features)
    elif clustering_method == 'hdbscan':
        if len(scaled_features) < 2:
            clustering = DBSCAN(eps=spatial_threshold, min_samples=1).fit(scaled_features)
        else:
            clustering = HDBSCAN(min_cluster_size=hdbscan_min_cluster_size).fit(scaled_features)

    else:
        raise ValueError(f"Unknown clustering_method: {clustering_method}. Use 'dbscan' or 'hdbscan'")
    
    labels = clustering.labels_
    
    # dbscan silently skips outliers, hdbscan gives them -1 so we can count them
    noise_count = np.sum(labels == -1)
    if noise_count > 0:
        print(f"[{clustering_method}] {noise_count} points assigned as noise and skipped")

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
        
        # Compute direction using mean and normalize
        directions = np.array([p['direction'] for p in cluster_points])
        mean_dir = np.mean(directions, axis=0)
        aggregated_direction = tuple(mean_dir / np.linalg.norm(mean_dir))  # norm = 1.0
        
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
    
    # Check direction vector norms after clustering
    # print("\n=== Direction Vector Normalization Check (After Clustering) ===")
    direction_norms_after = []
    for p in consolidated:
        dx, dy = p['direction']
        norm = np.sqrt(dx**2 + dy**2)
        direction_norms_after.append(norm)
    
    direction_norms_after = np.array(direction_norms_after)
    not_normed_after = np.abs(direction_norms_after - 1.0) > 0.1
    
    #print(f"Total consolidated predictions: {len(consolidated)}")
    if not_normed_after.any():
        print(f"WARNING: {not_normed_after.sum()} direction vectors deviate > 0.1 from norm=1.0")
    #else:
    #    print("✓ All direction vectors are properly normalized")
    #print("=" * 70 + "\n")
    
    return consolidated

def postprocess_predictions(predictions, strategy='cluster_consolidate',
                          spatial_threshold=30.0, temporal_threshold=10,
                          confidence_threshold=0.8, mode='mean',
                          remove_outliers=True, outlier_method='density',
                          outlier_min_neighbors=2,
                          clustering_method='dbscan', hdbscan_min_cluster_size=2):
    """
    Run post-processing using the specified strategy:
    'nms' | 'max' | 'weighted' | 'cluster' | 'cluster_consolidate' | 'line_nms'
    
    Args:
        predictions: List of prediction dicts
        strategy: Post-processing strategy to use
        spatial_threshold: Spatial distance threshold (pixels)
        temporal_threshold: Temporal distance threshold (frames)
        confidence_threshold: Minimum confidence to keep
        mode: 'mean' or 'median' for cluster consolidation
        remove_outliers: Whether to remove outliers (only for cluster_consolidate)
        outlier_method: 'density' or 'isolation_forest'
        outlier_min_neighbors: Min neighbors for density-based outlier removal
    """ 
    if strategy == 'cluster_consolidate':
        preds = cluster_and_consolidate_waggles(
            predictions, 
            spatial_threshold=spatial_threshold,
            temporal_threshold=temporal_threshold,
            min_confidence=confidence_threshold, 
            mode=mode,
            remove_outliers=remove_outliers,
            outlier_method=outlier_method,
            outlier_min_neighbors=outlier_min_neighbors,
            clustering_method=clustering_method,
            hdbscan_min_cluster_size=hdbscan_min_cluster_size
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
            mode=mode,
            remove_outliers=remove_outliers,
            outlier_method=outlier_method,
            outlier_min_neighbors=outlier_min_neighbors,
            clustering_method=clustering_method,
            hdbscan_min_cluster_size=hdbscan_min_cluster_size
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
                                  strategy='nms', mode='mean',
                                  remove_outliers=True,
                                  outlier_method='density',
                                  outlier_min_neighbors=2,
                                  clustering_method='dbscan',
                                  hdbscan_min_cluster_size=2):
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
            remove_outliers=remove_outliers,
            outlier_method=outlier_method,
            outlier_min_neighbors=outlier_min_neighbors,
            clustering_method=clustering_method, 
            hdbscan_min_cluster_size=hdbscan_min_cluster_size 
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


# Helper functions for line_nms (you may already have these)
def cos_sim(v1, v2):
    """Cosine similarity between two vectors"""
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)


def alignment_score(d1, d2, p1, p2):
    """
    How well does the vector from p1 to p2 align with direction d1?
    Returns cosine similarity.
    """
    vec = p2 - p1
    vec_norm = np.linalg.norm(vec)
    if vec_norm < 1e-6:
        return 0.0
    vec = vec / vec_norm
    return np.dot(d1, vec)