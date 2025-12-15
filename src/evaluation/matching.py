import numpy as np
from typing import Dict, List, Tuple
from scipy.optimize import linear_sum_assignment


def match_detections_to_gt(
    preds: List[Dict],
    gts: List[Dict],
    max_spatial_dist: float = 50.0,
    max_angular_error: float = 45.0,
    min_temporal_iou: float = 0.3,
    w_spatial: float = 1.0,
    w_angular: float = 0.3
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """Match predicted detections to ground truth using Hungarian algorithm.
    
    Args:
        preds: List of prediction dictionaries with keys: start, end, x, y, dir_x, dir_y
        gts: List of ground truth dictionaries with same structure
        max_spatial_dist: Maximum spatial distance (pixels) for valid match
        max_angular_error: Maximum angular error (degrees) for valid match
        min_temporal_iou: Minimum temporal IoU for valid match
        w_spatial: Weight for spatial distance in cost calculation
        w_angular: Weight for angular error in cost calculation
    
    Returns:
        Tuple of (matches, unmatched_gts, unmatched_preds)
        - matches: List of (gt_idx, pred_idx) tuples
        - unmatched_gts: List of unmatched GT indices
        - unmatched_preds: List of unmatched prediction indices
    """
    if not preds or not gts:
        return [], list(range(len(gts))), list(range(len(preds)))

    n_gts = len(gts)
    n_preds = len(preds)
    
    cost_matrix = np.full((n_gts, n_preds), 1e9)
    
    for i, gt in enumerate(gts):
        gt_start, gt_end = gt["start"], gt["end"]
        gt_pos = np.array([gt["x"], gt["y"]])
        gt_dir = np.array([gt["dir_x"], gt["dir_y"]])
        
        for j, pred in enumerate(preds):
            pred_start, pred_end = pred["start"], pred["end"]
            
            # Temporal IoU check
            intersection = max(0, min(gt_end, pred_end) - max(gt_start, pred_start))
            union = (pred_end - pred_start) + (gt_end - gt_start) - intersection
            temporal_iou = intersection / (union + 1e-8)
            
            if temporal_iou < min_temporal_iou:
                continue
            
            # Spatial distance check
            pred_pos = np.array([pred["x"], pred["y"]])
            spatial_dist = np.linalg.norm(gt_pos - pred_pos)
            
            if spatial_dist > max_spatial_dist:
                continue
            
            # Angular error check
            pred_dir = np.array([pred["dir_x"], pred["dir_y"]])
            cos_sim = np.clip(np.dot(gt_dir, pred_dir), -1.0, 1.0)
            angular_error = np.degrees(np.arccos(cos_sim))
            
            # Combined cost
            cost = w_spatial * spatial_dist + w_angular * angular_error
            cost_matrix[i, j] = cost
    
    # Hungarian algorithm for optimal matching
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    matches = []
    for i, j in zip(row_ind, col_ind):
        if cost_matrix[i, j] < 1e8:
            matches.append((i, j))
    
    matched_gts = {m[0] for m in matches}
    matched_preds = {m[1] for m in matches}
    unmatched_gts = list(set(range(n_gts)) - matched_gts)
    unmatched_preds = list(set(range(n_preds)) - matched_preds)
    
    return matches, unmatched_gts, unmatched_preds