
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
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
            
            intersection = max(0, min(gt_end, pred_end) - max(gt_start, pred_start))
            union = (pred_end - pred_start) + (gt_end - gt_start) - intersection
            temporal_iou = intersection / (union + 1e-8)

            
            if temporal_iou < min_temporal_iou:
                continue
            
            pred_pos = np.array([pred["x"], pred["y"]])
            spatial_dist = np.linalg.norm(gt_pos - pred_pos)
            
            if spatial_dist > max_spatial_dist:
                continue
            
            pred_dir = np.array([pred["dir_x"], pred["dir_y"]])
            cos_sim = np.clip(np.dot(gt_dir, pred_dir), -1.0, 1.0)
            angular_error = np.degrees(np.arccos(cos_sim))
            
            if angular_error > max_angular_error:
                continue
            
            cost = w_spatial * spatial_dist + w_angular * angular_error
            cost_matrix[i, j] = cost
    
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

def get_metrics(pred_df: pd.DataFrame, gt_df: pd.DataFrame, 
                max_spatial_dist: float = 50.0, 
                max_angular_error: float = 45.0,
                min_temporal_iou: float = 0.3) -> Dict:
    gts = []
    for _, row in gt_df.iterrows():
        gts.append({
            "start": row["start_frame"],
            "end": row["end_frame"],
            "x": row["x1"],
            "y": row["y1"],
            "dir_x": row["direction_x"],
            "dir_y": row["direction_y"]
        })

    preds = pred_df.to_dict("records")
    matches, missed, false = match_detections_to_gt(
        preds, gts, 
        max_spatial_dist=max_spatial_dist,
        max_angular_error=max_angular_error,
        min_temporal_iou=min_temporal_iou
    )

    match_mapping = []
    for gt_idx, pred_idx in matches:
        gt = gts[gt_idx]
        pred = preds[pred_idx]
        
        spatial_dist = np.sqrt((pred["x"] - gt["x"])**2 + (pred["y"] - gt["y"])**2)
        
        dot = np.clip(pred["dir_x"]*gt["dir_x"] + pred["dir_y"]*gt["dir_y"], -1.0, 1.0)
        angular_error = np.degrees(np.arccos(dot))
        
        match_mapping.append({
            "gt_index": int(gt_idx),
            "pred_index": int(pred_idx),
            "gt_start": int(gt["start"]),
            "gt_end": int(gt["end"]),
            "pred_start": int(pred["start"]),
            "pred_end": int(pred["end"]),
            "pred_confidence": float(pred['confidence']),
            "gt_x": float(gt['x']),
            "gt_y": float(gt['y']),
            "pred_x": float(pred['x']),
            "pred_y": float(pred['y']),
            "spatial_distance": float(spatial_dist),
            "gt_dir_x": float(gt['dir_x']),
            "gt_dir_y": float(gt['dir_y']),
            "pred_dir_x": float(pred['dir_x']),
            "pred_dir_y": float(pred['dir_y']),
            "angular_error": float(angular_error)
        })

    tp, fn, fp = len(matches), len(missed), len(false)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    spatial_errs, temporal_start_err, temporal_end_err, duration_errs, dir_errs, temporal_iou_list = [], [], [], [], [], []

    for gt_idx, pred_idx in matches:
        gt, pred = gts[gt_idx], preds[pred_idx]

        dx, dy = pred["x"] - gt["x"], pred["y"] - gt["y"]
        spatial_errs.append(np.sqrt(dx**2 + dy**2))

        temporal_start_err.append(abs(pred["start"] - gt["start"]))
        temporal_end_err.append(abs(pred["end"] - gt["end"]))

        pred_start, pred_end = pred["start"], pred["end"]
        gt_start, gt_end = gt["start"], gt["end"]

        intersection = max(0, min(pred_end, gt_end) - max(pred_start, gt_start))
        union = max(pred_end, gt_end) - min(pred_start, gt_start)
        temporal_iou_list.append(intersection / (union + 1e-8))

        gt_dur = gt["end"] - gt["start"]
        pred_dur = pred["end"] - pred["start"]
        duration_errs.append(abs(pred_dur - gt_dur))

        dot = np.clip(pred["dir_x"]*gt["dir_x"] + pred["dir_y"]*gt["dir_y"], -1.0, 1.0)
        dir_errs.append(np.degrees(np.arccos(dot)))

    results = {
        "True Positives": int(tp),
        "False Positives": int(fp),
        "False Negatives": int(fn),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "spatial_error_mean": float(np.mean(spatial_errs)) if spatial_errs else None,
        "spatial_error_std": float(np.std(spatial_errs)) if spatial_errs else None,
        "temporal_start_error_mean": float(np.mean(temporal_start_err)) if temporal_start_err else None,
        "temporal_start_error_std": float(np.std(temporal_start_err)) if temporal_start_err else None,
        "temporal_end_error_mean": float(np.mean(temporal_end_err)) if temporal_end_err else None,
        "temporal_end_error_std": float(np.std(temporal_end_err)) if temporal_end_err else None,
        "temporal_iou_mean": float(np.mean(temporal_iou_list)) if temporal_iou_list else None,
        "duration_error_mean": float(np.mean(duration_errs)) if duration_errs else None,
        "angular_error_mean": float(np.mean(dir_errs)) if dir_errs else None,
        "angular_error_std": float(np.std(dir_errs)) if dir_errs else None,
        "match_mapping": match_mapping
    }
    return results