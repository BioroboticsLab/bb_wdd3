import pandas as pd
from typing import Dict
from .matching import match_detections_to_gt 
from .postprocessing import dbscan_merge, merge_detections
import numpy as np
import os
import json

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

def get_metrix_for_all_videos(video_folder="./data/eval_videos/multi_res/", gt_path="./data/eval_gt/eval_gt.csv", preds_folder="./eval_results/epoch_300", confidence=0.0, comp=True, output_video_folder="./eval_results/epoch_300"):
    video_files = [f for f in os.listdir(video_folder) if f.endswith(('.mp4', '.MP4', '.avi', '.mov'))]
    os.makedirs(output_video_folder, exist_ok=True)
    all_results = {}
    if not video_files:
        print(f"No video files found in {video_folder}")
        return {}
    
    print(f"\nFound {len(video_files)} video(s) to visualise")

    rows=[]
    
    for video_file in video_files:
        video_path = os.path.join(video_folder, video_file)
        video_name = os.path.splitext(video_file)[0]

        preds_csv_path = os.path.join(preds_folder, f"{video_name}_preds.csv")
        preds_df = pd.read_csv(preds_csv_path)
        preds_df=preds_df[preds_df['confidence']>=confidence]

        merged_df = dbscan_merge(preds_df)
        merged_df = merge_detections(merged_df)

        gt_df = pd.read_csv(gt_path)
        gt_df = gt_df[gt_df['video_name']==(video_name+".mp4")]

        before = get_metrics(preds_df, gt_df)

        if comp:

            after = get_metrics(merged_df, gt_df)

            row = {
                "Video": video_name,
                "TP_before": before["True Positives"],
                "TP_after": after["True Positives"],
                "FP_before": before["False Positives"],
                "FP_after": after["False Positives"],
                "FN_before": before["False Negatives"],
                "FN_after": after["False Negatives"]
            }
            rows.append(row)

            num_detections = len(merged_df)
            avg_confidence = float(merged_df['confidence'].mean()) if num_detections > 0 else 0.0

            # print(os.path.join(output_video_folder, video_name + "_detailed_metrics.json"))
        
            with open(os.path.join(output_video_folder,video_name + "_detailed_metrics.json"), "w") as f:
                json.dump(after, f, indent=4)
            
            after['num_detections'] = num_detections
            after['avg_confidence'] = avg_confidence
            after['confidence_threshold'] = confidence

            print("\n" + "="*60)
            print(f"EVALUATION RESULTS - {video_name}")
            print("="*60)
            print(f"True Positives: {after['True Positives']}")
            print(f"False Positives: {after['False Positives']}")
            print(f"False Negatives: {after['False Negatives']}")
            print(f"Precision: {after['precision']:.4f}")
            print(f"Recall: {after['recall']:.4f}")
            print(f"F1 Score: {after['f1']:.4f}")
            print(f"Total Detections: {num_detections}")
            print(f"Average Confidence: {avg_confidence:.4f}")
            
            if after['spatial_error_mean'] is not None:
                print(f"\nSpatial Error: {after['spatial_error_mean']:.2f} ± {after['spatial_error_std']:.2f} pixels")
                print(f"Angular Error: {after['angular_error_mean']:.2f} ± {after['angular_error_std']:.2f} degrees")
                print(f"Temporal IoU: {after['temporal_iou_mean']:.4f}")
            print("="*60)
            all_results[video_name] = after
        else:
            num_detections = len(preds_df)
            avg_confidence = float(preds_df['confidence'].mean()) if num_detections > 0 else 0.0

            # print(os.path.join(output_video_folder, video_name + "_detailed_metrics.json"))
        
            with open(os.path.join(output_video_folder,video_name + "_detailed_metrics.json"), "w") as f:
                json.dump(before, f, indent=4)
            
            before['num_detections'] = num_detections
            before['avg_confidence'] = avg_confidence
            before['confidence_threshold'] = confidence

            print("\n" + "="*60)
            print(f"EVALUATION RESULTS - {video_name}")
            print("="*60)
            print(f"True Positives: {before['True Positives']}")
            print(f"False Positives: {before['False Positives']}")
            print(f"False Negatives: {before['False Negatives']}")
            print(f"Precision: {before['precision']:.4f}")
            print(f"Recall: {before['recall']:.4f}")
            print(f"F1 Score: {before['f1']:.4f}")
            print(f"Total Detections: {num_detections}")
            print(f"Average Confidence: {avg_confidence:.4f}")
            
            if before['spatial_error_mean'] is not None:
                print(f"\nSpatial Error: {before['spatial_error_mean']:.2f} ± {before['spatial_error_std']:.2f} pixels")
                print(f"Angular Error: {before['angular_error_mean']:.2f} ± {before['angular_error_std']:.2f} degrees")
                print(f"Temporal IoU: {before['temporal_iou_mean']:.4f}")
            print("="*60)
            all_results[video_name] = before
    if comp:
        df_metrics = pd.DataFrame(rows)
        df_metrics["TP_change"] = df_metrics["TP_after"] - df_metrics["TP_before"]
        df_metrics["FP_change"] = df_metrics["FP_after"] - df_metrics["FP_before"]
        df_metrics["FN_change"] = df_metrics["FN_after"] - df_metrics["FN_before"]
        df_metrics.to_csv(os.path.join(output_video_folder,"compare_eval_results_merge.csv"))
    
    evaluated_videos = {k: v for k, v in all_results.items() if v is not None}

    if evaluated_videos:
        avg_precision = np.mean([v['precision'] for v in evaluated_videos.values()])
        avg_recall = np.mean([v['recall'] for v in evaluated_videos.values()])
        avg_f1 = np.mean([v['f1'] for v in evaluated_videos.values()])
        avg_num_detections = np.mean([v['num_detections'] for v in evaluated_videos.values()])
        avg_confidence = np.mean([v['avg_confidence'] for v in evaluated_videos.values()])
        
        print(f"Videos evaluated with GT: {len(evaluated_videos)}")
        print(f"Average Precision: {avg_precision:.4f}")
        print(f"Average Recall: {avg_recall:.4f}")
        print(f"Average F1: {avg_f1:.4f}")
        print(f"Average Num Detections: {avg_num_detections:.2f}")
        print(f"Average Confidence: {avg_confidence:.4f}")

    
# def calculate_position_metrics_comprehensive(preds, gts, pos_thresholds=[5, 10, 15, 20, 25, 30]):
#     """
#     Calculate position metrics with comprehensive thresholds.
#     """
#     # handle single threshold input
#     if isinstance(pos_thresholds, (int, float)):
#         pos_thresholds = [pos_thresholds]
    
#     all_metrics = {}
#     all_matched_pairs = {}
    
#     # calculate for each threshold
#     for threshold in pos_thresholds:
#         all_distances = []
#         matched_pairs = []
        
#         for batch_preds, batch_gts in zip(preds, gts):
#             # for each GT, find closest prediction within threshold
#             for gt in batch_gts:
#                 min_dist = float('inf')
#                 best_pred = None
                
#                 for pred in batch_preds:
#                     gt_pos = np.array(gt['position'])
#                     pred_pos = np.array(pred['position'])
#                     distance = np.linalg.norm(gt_pos - pred_pos)
                    
#                     if distance < min_dist and distance <= threshold:
#                         min_dist = distance
#                         best_pred = pred
                
#                 if best_pred is not None:
#                     all_distances.append(min_dist)
#                     matched_pairs.append((gt, best_pred))
        
#         # total_predictions = sum(len(batch_preds) for batch_preds in preds)
#         all_preds_flat = [pred for batch_preds in preds for pred in batch_preds] #fix for count
#         total_predictions = len(all_preds_flat)
#         total_gts = sum(len(batch_gts) for batch_gts in gts)
        
#         # calculate precision, recall, and F1
#         precision = len(matched_pairs) / total_predictions if total_predictions > 0 else 0.0
#         recall = len(matched_pairs) / total_gts if total_gts > 0 else 0.0
#         f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
#         # store metrics for this threshold
#         metrics_at_threshold = {
#             f'mean_position_error_{threshold}': np.mean(all_distances) if all_distances else float('inf'),
#             f'position_precision_{threshold}': precision,
#             f'position_recall_{threshold}': recall,
#             f'position_f1_{threshold}': f1,
#             f'matched_detections_{threshold}': len(matched_pairs),
#         }
        
#         if all_distances:
#             metrics_at_threshold[f'position_error_std_{threshold}'] = np.std(all_distances)
        
#         all_metrics.update(metrics_at_threshold)
#         all_matched_pairs[threshold] = matched_pairs
    
#     # calculate mean metrics across all thresholds
#     precisions = [all_metrics[f'position_precision_{t}'] for t in pos_thresholds]
#     recalls = [all_metrics[f'position_recall_{t}'] for t in pos_thresholds]
#     f1_scores = [all_metrics[f'position_f1_{t}'] for t in pos_thresholds]
#     position_errors = [all_metrics[f'mean_position_error_{t}'] for t in pos_thresholds]
    
#     summary_metrics = {
#         'precision': np.mean(precisions),
#         'recall': np.mean(recalls),
#         'f1': np.mean(f1_scores),
#         'mean_error': np.mean(position_errors),
#         'total_predictions': total_predictions,
#         'total_ground_truths': total_gts
#     }
    
#     return summary_metrics, all_matched_pairs

import numpy as np

def calculate_position_metrics_comprehensive(
    preds,
    gts,
    pos_thresholds=[5, 10, 15, 20, 25, 30],
    confidence_key="confidence",
    confidence_threshold=None
):
    """
    Compute position-based metrics across multiple thresholds with proper 1-to-1 matching.
    preds, gts: lists of batches, each batch = list of dicts with fields:
        - "position": (x, y)
        - optionally "confidence"
    """

    # Normalize thresholds
    if isinstance(pos_thresholds, (int, float)):
        pos_thresholds = [pos_thresholds]

    all_metrics = {}
    all_matched_pairs = {}

    # Count predictions *once*, correctly
    all_preds_flat = [
        pred
        for batch_preds in preds
        for pred in batch_preds
        if confidence_threshold is None
        or pred.get(confidence_key, 1.0) >= confidence_threshold
    ]
    total_predictions = len(all_preds_flat)

    # Count ground truths once
    total_gts = sum(len(batch_gts) for batch_gts in gts)

    # Pre-flatten preds/gts and filter predictions by confidence once
    flat_preds = []
    flat_gts = []
    for batch_preds, batch_gts in zip(preds, gts):
        batch_preds_filtered = [
            p for p in batch_preds
            if confidence_threshold is None
            or p.get(confidence_key, 1.0) >= confidence_threshold
        ]
        flat_preds.append(batch_preds_filtered)
        flat_gts.append(batch_gts)

    # Process each threshold
    for threshold in pos_thresholds:
        matched_pairs = []
        all_distances = []

        for batch_preds, batch_gts in zip(flat_preds, flat_gts):
            used_pred_indices = set()

            for gt in batch_gts:
                gt_pos = np.array(gt["position"])

                min_dist = float("inf")
                best_pred_idx = None

                for i, pred in enumerate(batch_preds):
                    if i in used_pred_indices:
                        continue

                    pred_pos = np.array(pred["position"])
                    distance = np.linalg.norm(gt_pos - pred_pos)

                    if distance <= threshold and distance < min_dist:
                        min_dist = distance
                        best_pred_idx = i

                # Match -> lock pred to avoid reuse
                if best_pred_idx is not None:
                    used_pred_indices.add(best_pred_idx)
                    all_distances.append(min_dist)
                    matched_pairs.append((gt, batch_preds[best_pred_idx]))

        # Compute metrics
        TP = len(matched_pairs)
        precision = TP / total_predictions if total_predictions > 0 else 0.0
        recall = TP / total_gts if total_gts > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        metrics_at_threshold = {
            f"mean_position_error_{threshold}": np.mean(all_distances) if all_distances else float("inf"),
            f"position_precision_{threshold}": precision,
            f"position_recall_{threshold}": recall,
            f"position_f1_{threshold}": f1,
            f"matched_detections_{threshold}": TP,
        }

        if all_distances:
            metrics_at_threshold[f"position_error_std_{threshold}"] = np.std(all_distances)

        all_metrics.update(metrics_at_threshold)
        all_matched_pairs[threshold] = matched_pairs

    # Summary averages over thresholds
    summary_metrics = {
        "precision": np.mean([all_metrics[f"position_precision_{t}"] for t in pos_thresholds]),
        "recall": np.mean([all_metrics[f"position_recall_{t}"] for t in pos_thresholds]),
        "f1": np.mean([all_metrics[f"position_f1_{t}"] for t in pos_thresholds]),
        "mean_error": np.mean([all_metrics[f"mean_position_error_{t}"] for t in pos_thresholds]),
        "total_predictions": total_predictions,
        "total_ground_truths": total_gts,
    }

    return summary_metrics, all_matched_pairs


def calculate_direction_metrics_comprehensive(matched_pairs, angular_thresholds=[10, 15, 20]):
    """
    Calculate direction metrics with comprehensive angular error thresholds.
    """
    # handle single threshold input
    if isinstance(angular_thresholds, (int, float)):
        angular_thresholds = [angular_thresholds]
    
    if not matched_pairs:
        return {
            'accuracy': 0.0,
            'mean_error': float('inf'),
            'mean_cosine_similarity': 0.0
        }
    
    angular_errors = []
    cosine_similarities = []
    
    for gt, pred in matched_pairs:
        gt_dir = np.array(gt['direction'])
        pred_dir = np.array(pred['direction'])
        
        # norm vectors
        gt_dir_norm = gt_dir / (np.linalg.norm(gt_dir) + 1e-8)
        pred_dir_norm = pred_dir / (np.linalg.norm(pred_dir) + 1e-8)
        
        # cosine sim
        cos_sim = np.dot(gt_dir_norm, pred_dir_norm)
        cosine_similarities.append(cos_sim)
        
        # angular error in degrees
        angular_error = np.degrees(np.arccos(np.clip(cos_sim, -1.0, 1.0)))
        angular_errors.append(angular_error)
    
    angular_errors = np.array(angular_errors)
    
    # compute accuracy across all thresholds
    accuracy_values = []
    for threshold in angular_thresholds:
        accuracy_at_threshold = np.mean(angular_errors <= threshold)
        accuracy_values.append(accuracy_at_threshold)
    
    return {
        'accuracy': np.mean(accuracy_values),
        'mean_error': np.mean(angular_errors),
        'mean_cosine_similarity': np.mean(cosine_similarities)
    }

def calculate_temporal_metrics(matched_pairs, iou_threshold_range=(0.25, 0.75)):
    """
    Calculate temporal metrics for matched predictions.
    """
    # hangle single threshold input
    if isinstance(iou_threshold_range, (int, float)):
        iou_threshold_range = (iou_threshold_range, iou_threshold_range)
    elif len(iou_threshold_range) == 1:
        iou_threshold_range = (iou_threshold_range[0], iou_threshold_range[0])
    
    if not matched_pairs:
        return {
            'mean_iou': 0.0,
            'mean_start_error': float('inf'),
            'mean_end_error': float('inf')
        }
    
    start_errors = []
    end_errors = []
    temporal_iou_scores = []
    
    for gt, pred in matched_pairs:
        gt_start, gt_end = gt['temporal_offsets']
        pred_start, pred_end = pred['temporal_offsets']
        
        # start/end frame errors
        start_errors.append(abs(gt_start - pred_start))
        end_errors.append(abs(gt_end - pred_end))
        
        # temporal IoU
        intersection_start = max(gt_start, pred_start)
        intersection_end = min(gt_end, pred_end)
        intersection = max(0, intersection_end - intersection_start)
        
        union_start = min(gt_start, pred_start)
        union_end = max(gt_end, pred_end)
        union = union_end - union_start
        
        temporal_iou = intersection / union if union > 0 else 0
        temporal_iou_scores.append(temporal_iou)
    
    temporal_iou_scores = np.array(temporal_iou_scores)
    
    # calculate IoU thresholds in the specified range with step of 0.05
    iou_thresholds = np.arange(iou_threshold_range[0], iou_threshold_range[1] + 0.05, 0.05)
    iou_values = []
    
    for threshold in iou_thresholds:
        threshold = round(threshold, 2)
        iou_at_threshold = np.mean(temporal_iou_scores >= threshold)
        iou_values.append(iou_at_threshold)
    
    return {
        'mean_iou': np.mean(temporal_iou_scores),
        'mean_start_error': np.mean(start_errors),
        'mean_end_error': np.mean(end_errors)
    }

def calculate_detection_metrics(preds, gts, pos_thresholds=[5, 10, 15, 20, 25, 30], 
                              iou_threshold_range=(0.25, 0.75), angular_thresholds=[10, 15, 20]):
    """
    Comprehensive detection metrics across ALL THREE dimensions.
    """

    
    # handle flexible input formats
    if isinstance(pos_thresholds, (int, float)):
        pos_thresholds = [pos_thresholds]
    if isinstance(iou_threshold_range, (int, float)):
        iou_threshold_range = (iou_threshold_range, iou_threshold_range)
    elif len(iou_threshold_range) == 1:
        iou_threshold_range = (iou_threshold_range[0], iou_threshold_range[0])
    if isinstance(angular_thresholds, (int, float)):
        angular_thresholds = [angular_thresholds]
    
    # handle IoU thresholds
    iou_thresholds = np.arange(iou_threshold_range[0], iou_threshold_range[1] + 0.05, 0.05)
    iou_thresholds = [round(t, 2) for t in iou_thresholds]
    
    comprehensive_precisions = []
    comprehensive_recalls = []
    comprehensive_f1_scores = []
    
    for pos_threshold in pos_thresholds:
        for iou_threshold in iou_thresholds:
            for angular_threshold in angular_thresholds:
                true_positives = 0
                false_positives = 0
                false_negatives = 0
                
                for batch_preds, batch_gts in zip(preds, gts):
                    matched_gts = set()
                    
                    for pred in batch_preds:
                        best_gt_idx = None
                        best_iou = 0
                        
                        for gt_idx, gt in enumerate(batch_gts):
                            if gt_idx in matched_gts:
                                continue
                            
                            # position distance
                            pos_dist = np.linalg.norm(np.array(gt['position']) - np.array(pred['position']))
                            
                            # temporal IoU
                            gt_start, gt_end = gt['temporal_offsets']
                            pred_start, pred_end = pred['temporal_offsets']
                            intersection = max(0, min(gt_end, pred_end) - max(gt_start, pred_start))
                            union = max(gt_end, pred_end) - min(gt_start, pred_start)
                            temp_iou = intersection / union if union > 0 else 0
                            
                            # angular error
                            gt_dir = np.array(gt['direction'])
                            pred_dir = np.array(pred['direction'])
                            gt_dir_norm = gt_dir / (np.linalg.norm(gt_dir) + 1e-8)
                            pred_dir_norm = pred_dir / (np.linalg.norm(pred_dir) + 1e-8)
                            cos_sim = np.dot(gt_dir_norm, pred_dir_norm)
                            angular_error = np.degrees(np.arccos(np.clip(cos_sim, -1.0, 1.0)))
                            
                            # check if all 3 conditions apply
                            if (pos_dist <= pos_threshold and 
                                temp_iou >= iou_threshold and 
                                angular_error <= angular_threshold and 
                                temp_iou > best_iou):
                                best_iou = temp_iou
                                best_gt_idx = gt_idx
                        
                        if best_gt_idx is not None:
                            true_positives += 1
                            matched_gts.add(best_gt_idx)
                        else:
                            false_positives += 1
                    
                    false_negatives += len(batch_gts) - len(matched_gts)
                
                precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
                recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                comprehensive_precisions.append(precision)
                comprehensive_recalls.append(recall)
                comprehensive_f1_scores.append(f1)
    
    return {
        'precision': np.mean(comprehensive_precisions),
        'recall': np.mean(comprehensive_recalls),
        'f1': np.mean(comprehensive_f1_scores)
    }

def get_eval_metrics(preds, gts, config=None):
    """
    Run complete evaluation with hierarchical metrics structure.
    
    Args:
        preds: List of prediction batches
        gts: List of ground truth batches
        config: Dictionary with evaluation configuration
            - pos_thresholds: List of spatial thresholds or single value [10] or [5, 10, 15, 20, 25, 30]
            - iou_threshold_range: Tuple for temporal IoU range (0.5,) or (0.25, 0.75) or single value
            - angular_thresholds: List of angular thresholds or single value [10] or [10, 15, 20]
    
    Returns:
        Hierarchical dictionary with metrics
    """
    # default configuration
    if config is None:
        config = {
            'pos_thresholds': [30],
            'iou_threshold_range': (0.5),
            'angular_thresholds': [20]
        }
    
    # handle flexible input formats
    # convert single pos_threshold to list
    if isinstance(config['pos_thresholds'], (int, float)):
        config['pos_thresholds'] = [config['pos_thresholds']]
    
    # handle single iou_threshold_range value
    if isinstance(config['iou_threshold_range'], (int, float)):
        config['iou_threshold_range'] = (config['iou_threshold_range'], config['iou_threshold_range'])
    elif len(config['iou_threshold_range']) == 1:
        config['iou_threshold_range'] = (config['iou_threshold_range'][0], config['iou_threshold_range'][0])
    
    # convert single angular_threshold to list
    if isinstance(config['angular_thresholds'], (int, float)):
        config['angular_thresholds'] = [config['angular_thresholds']]
    
    # get matched pairs for detailed metrics
    position_metrics, all_matched_pairs = calculate_position_metrics_comprehensive(
        preds, gts, config['pos_thresholds']
    )
    
    # use matched pairs from the first threshold for direction/temporal metrics
    primary_matched_pairs = all_matched_pairs.get(config['pos_thresholds'][0], [])
    
    # calculate all metrics
    comprehensive_metrics = calculate_detection_metrics(
        preds, gts, 
        config['pos_thresholds'],
        config['iou_threshold_range'],
        config['angular_thresholds']
    )
    
    directional_metrics = calculate_direction_metrics_comprehensive(
        primary_matched_pairs, 
        config['angular_thresholds']
    )
    
    temporal_metrics = calculate_temporal_metrics(
        primary_matched_pairs,
        config['iou_threshold_range']
    )
    
    # combine into hierarchical structure for clean prints and outputs
    metrics = {
        'comprehensive': comprehensive_metrics,
        'spatial': position_metrics,
        'temporal': temporal_metrics,
        'directional': directional_metrics,
        'counts': {
            'predictions': position_metrics['total_predictions'],
            'ground_truths': position_metrics['total_ground_truths']
        },
        'config': config
    }
    
    return metrics

def print_evaluation_results(test_metrics, post_test_metrics):
    """Print comprehensive evaluation results before and after post-processing"""
    
    print(f"Before Post-Processing:")
    print(f"  Predictions: {test_metrics['counts']['predictions']}, Ground Truths: {test_metrics['counts']['ground_truths']}")
    print(f"  Comprehensive Detection:")
    print(f"    Precision: {test_metrics['comprehensive']['precision']:.3f}")
    print(f"    Recall: {test_metrics['comprehensive']['recall']:.3f}") 
    print(f"    F1: {test_metrics['comprehensive']['f1']:.3f}")
    print(f"  Spatial Detection:")
    print(f"    Precision: {test_metrics['spatial']['precision']:.3f}")
    print(f"    Recall: {test_metrics['spatial']['recall']:.3f}")
    print(f"    F1: {test_metrics['spatial']['f1']:.3f}")
    print(f"    Mean Error: {test_metrics['spatial']['mean_error']:.1f}px")
    print(f"  Temporal Detection:")
    print(f"    Mean IoU: {test_metrics['temporal']['mean_iou']:.3f}")
    print(f"  Directional Detection:")
    print(f"    Accuracy: {test_metrics['directional']['accuracy']:.3f}")
    print(f"    Mean Angular Error: {test_metrics['directional']['mean_error']:.1f}°")
    print(f"#"*60)
    print(f"After Post-Processing:")
    print(f"  Predictions: {post_test_metrics['counts']['predictions']}, Ground Truths: {post_test_metrics['counts']['ground_truths']}")
    print(f"  Comprehensive Detection:")
    print(f"    Precision: {post_test_metrics['comprehensive']['precision']:.3f}")
    print(f"    Recall: {post_test_metrics['comprehensive']['recall']:.3f}") 
    print(f"    F1: {post_test_metrics['comprehensive']['f1']:.3f}")
    print(f"  Spatial Detection:")
    print(f"    Precision: {post_test_metrics['spatial']['precision']:.3f}")
    print(f"    Recall: {post_test_metrics['spatial']['recall']:.3f}")
    print(f"    F1: {post_test_metrics['spatial']['f1']:.3f}")
    print(f"    Mean Error: {post_test_metrics['spatial']['mean_error']:.1f}px")
    print(f"  Temporal Detection:")
    print(f"    Mean IoU: {post_test_metrics['temporal']['mean_iou']:.3f}")
    print(f"  Directional Detection:")
    print(f"    Accuracy: {post_test_metrics['directional']['accuracy']:.3f}")
    print(f"    Mean Angular Error: {post_test_metrics['directional']['mean_error']:.1f}°")


def get_detailed_metrics( preds, gts, eval_config, stage_name):
        """
        Calculate and print detailed metrics for each threshold combination.
        """
        config = eval_config
        
        # Get all matched pairs for each position threshold
        _, all_matched_pairs = calculate_position_metrics_comprehensive(
            preds, gts, config['pos_thresholds']
        )
        
        # Overall counts
        total_preds = sum(len(batch_preds) for batch_preds in preds)
        total_gts = sum(len(batch_gts) for batch_gts in gts)
        
        print(f"\nOverall Statistics {stage_name}:")
        print(f"  Total Predictions: {total_preds}")
        print(f"  Total Ground Truths: {total_gts}")
        print(f"  Confidence Threshold: {config.get('confidence_threshold', 'N/A')}")

        
        # Results storage
        all_results = {}
        
        # Process each position threshold
        for pos_thresh in config['pos_thresholds']:
            print(f"\n{'─'*80}")
            print(f"POSITION THRESHOLD: {pos_thresh} pixels")
            print(f"{'─'*80}")
            
            matched_pairs = all_matched_pairs.get(pos_thresh, [])
            
            if not matched_pairs:
                print(f"  ⚠ No matches found at this threshold")
                continue
            
            # Calculate spatial statistics
            spatial_errors = []
            for gt, pred in matched_pairs:
                gt_pos = np.array(gt['position'])
                pred_pos = np.array(pred['position'])
                dist = np.linalg.norm(gt_pos - pred_pos)
                spatial_errors.append(dist)
            
            tp = len(matched_pairs)
            fp = total_preds - tp
            fn = total_gts - tp
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            print(f"\n  Spatial Metrics:")
            print(f"    True Positives: {tp}")
            print(f"    False Positives: {fp}")
            print(f"    False Negatives: {fn}")
            print(f"    Precision: {precision:.4f}")
            print(f"    Recall: {recall:.4f}")
            print(f"    F1-Score: {f1:.4f}")
            print(f"    Position Error: {np.mean(spatial_errors):.2f} ± {np.std(spatial_errors):.2f} pixels")
            print(f"    Min Error: {np.min(spatial_errors):.2f} px")
            print(f"    Max Error: {np.max(spatial_errors):.2f} px")
            print(f"    Median Error: {np.median(spatial_errors):.2f} px")
            
            # Calculate temporal statistics
            temporal_start_errors = []
            temporal_end_errors = []
            temporal_ious = []
            # temporal_start_errors_sec = []
            # temporal_end_errors_sec = []
            
            for gt, pred in matched_pairs:
                gt_start, gt_end = gt['temporal_offsets']
                pred_start, pred_end = pred['temporal_offsets']
                
                start_err = abs(pred_start - gt_start)
                end_err = abs(pred_end - gt_end)
                temporal_start_errors.append(start_err)
                temporal_end_errors.append(end_err)
                
                # # Convert to seconds
                # temporal_start_errors_sec.append(start_err / fps)
                # temporal_end_errors_sec.append(end_err / fps)
                
                # Temporal IoU
                intersection = max(0, min(pred_end, gt_end) - max(pred_start, gt_start))
                union = max(pred_end, gt_end) - min(pred_start, gt_start)
                iou = intersection / union if union > 0 else 0
                temporal_ious.append(iou)
            
            print(f"\n  Temporal Metrics:")
            print(f"    Start Frame Error: {np.mean(temporal_start_errors):.2f} ± {np.std(temporal_start_errors):.2f} frames")
            print(f"    End Frame Error: {np.mean(temporal_end_errors):.2f} ± {np.std(temporal_end_errors):.2f} frames")
            # print(f"    Start Time Error: {np.mean(temporal_start_errors_sec):.3f} ± {np.std(temporal_start_errors_sec):.3f} seconds")
            # print(f"    End Time Error: {np.mean(temporal_end_errors_sec):.3f} ± {np.std(temporal_end_errors_sec):.3f} seconds")
            print(f"    Temporal IoU: {np.mean(temporal_ious):.4f} ± {np.std(temporal_ious):.4f}")
            print(f"    Min IoU: {np.min(temporal_ious):.4f}")
            print(f"    Max IoU: {np.max(temporal_ious):.4f}")
            
            # Calculate directional statistics for each angular threshold
            for ang_thresh in config['angular_thresholds']:
                angular_errors = []
                cosine_sims = []
                
                for gt, pred in matched_pairs:
                    gt_dir = np.array(gt['direction'])
                    pred_dir = np.array(pred['direction'])
                    
                    # Normalize
                    gt_dir_norm = gt_dir / (np.linalg.norm(gt_dir) + 1e-8)
                    pred_dir_norm = pred_dir / (np.linalg.norm(pred_dir) + 1e-8)
                    
                    # Cosine similarity
                    cos_sim = np.dot(gt_dir_norm, pred_dir_norm)
                    cosine_sims.append(cos_sim)
                    
                    # Angular error
                    angular_error = np.degrees(np.arccos(np.clip(cos_sim, -1.0, 1.0)))
                    angular_errors.append(angular_error)
                
                correct_direction = np.sum(np.array(angular_errors) <= ang_thresh)
                accuracy = correct_direction / len(angular_errors) if angular_errors else 0
                
                print(f"\n  Directional Metrics (Angular Threshold: {ang_thresh}°):")
                print(f"    Accuracy: {accuracy:.4f} ({correct_direction}/{len(angular_errors)})")
                print(f"    Angular Error: {np.mean(angular_errors):.2f} ± {np.std(angular_errors):.2f} degrees")
                print(f"    Min Error: {np.min(angular_errors):.2f}°")
                print(f"    Max Error: {np.max(angular_errors):.2f}°")
                print(f"    Median Error: {np.median(angular_errors):.2f}°")
                print(f"    Cosine Similarity: {np.mean(cosine_sims):.4f} ± {np.std(cosine_sims):.4f}")
            
            # Store results for this threshold
            all_results[pos_thresh] = {
                'spatial': {
                    'tp': tp, 'fp': fp, 'fn': fn,
                    'precision': precision, 'recall': recall, 'f1': f1,
                    'mean_error': np.mean(spatial_errors),
                    'std_error': np.std(spatial_errors)
                },
                'temporal': {
                    'mean_start_error_frames': np.mean(temporal_start_errors),
                    'std_start_error_frames': np.std(temporal_start_errors),
                    'mean_end_error_frames': np.mean(temporal_end_errors),
                    'std_end_error_frames': np.std(temporal_end_errors),
                    # 'mean_start_error_sec': np.mean(temporal_start_errors_sec),
                    # 'mean_end_error_sec': np.mean(temporal_end_errors_sec),
                    'mean_iou': np.mean(temporal_ious),
                    'std_iou': np.std(temporal_ious)
                },
                'directional': {
                    ang_thresh: {
                        'accuracy': accuracy,
                        'mean_error': np.mean(angular_errors),
                        'std_error': np.std(angular_errors),
                        'mean_cosine_sim': np.mean(cosine_sims)
                    } for ang_thresh in config['angular_thresholds']
                }
            }
        
        # Overall comprehensive metrics
        print(f"\n{'─'*80}")
        print(f"COMPREHENSIVE METRICS (All Thresholds Combined)")
        print(f"{'─'*80}")
        
        comprehensive_metrics = calculate_detection_metrics(
            preds, gts,
            config['pos_thresholds'],
            config['iou_threshold_range'],
            config['angular_thresholds']
        )
        
        print(f"\n  Average across all threshold combinations:")
        print(f"    Precision: {comprehensive_metrics['precision']:.4f}")
        print(f"    Recall: {comprehensive_metrics['recall']:.4f}")
        print(f"    F1-Score: {comprehensive_metrics['f1']:.4f}")
        
        return {
            'per_threshold': all_results,
            'comprehensive': comprehensive_metrics,
            'counts': {'predictions': total_preds, 'ground_truths': total_gts}
        }


def print_comparison_summary( before_metrics, after_metrics):
        """
        Print side-by-side comparison of before/after post-processing.
        """
        print("\n" + "="*80)
        print("SUMMARY COMPARISON: Before vs After Post-Processing")
        print("="*80)
        
        print(f"\n{'Metric':<40} {'Before':<20} {'After':<20} {'Change':<15}")
        print("─"*80)
        
        # Count comparison
        before_preds = before_metrics['counts']['predictions']
        after_preds = after_metrics['counts']['predictions']
        print(f"{'Number of Predictions':<40} {before_preds:<20} {after_preds:<20} {after_preds - before_preds:<15}")
        
        # Comprehensive metrics
        before_comp = before_metrics['comprehensive']
        after_comp = after_metrics['comprehensive']
        
        print(f"{'Precision':<40} {before_comp['precision']:<20.4f} {after_comp['precision']:<20.4f} {after_comp['precision'] - before_comp['precision']:+.4f}")
        print(f"{'Recall':<40} {before_comp['recall']:<20.4f} {after_comp['recall']:<20.4f} {after_comp['recall'] - before_comp['recall']:+.4f}")
        print(f"{'F1-Score':<40} {before_comp['f1']:<20.4f} {after_comp['f1']:<20.4f} {after_comp['f1'] - before_comp['f1']:+.4f}")
        
        print("\n" + "="*80)


# if __name__=="__main__":
#     get_metrix_for_all_videos(video_folder="../data/eval_new_berlin_videos/downsampled", gt_path="../data/eval_new_berlin_gt/berlin_new_gt_downsampled.csv", preds_folder="../eval_new_berlin_results/downsampled/epoch_300", confidence=0.9, comp=True, output_video_folder="../eval_new_berlin_results/downsampled/epoch_300")


def print_clean_summary(metrics, stage_name="Evaluation"):
    """
    Print a clean, minimal summary of metrics.
    
    Args:
        metrics: Dictionary with 'per_threshold' and 'counts' keys from get_detailed_metrics
        stage_name: Name to display (e.g., "Before (Category 0)", "After (Overall)")
    """
    print(f"\n{'='*60}")
    print(f"{stage_name}")
    print(f"{'='*60}")
    
    # Get counts
    total_preds = metrics['counts']['predictions']
    total_gts = metrics['counts']['ground_truths']
    
    print(f"Predictions: {total_preds} | Ground Truths: {total_gts}")
    
    # Get the first threshold's results (usually the main one)
    first_thresh = list(metrics['per_threshold'].keys())[0]
    results = metrics['per_threshold'][first_thresh]
    
    # Detection counts
    spatial = results['spatial']
    print(f"\nDetection Metrics:")
    print(f"  TP: {spatial['tp']:3d} | FP: {spatial['fp']:3d} | FN: {spatial['fn']:3d}")
    print(f"  Precision: {spatial['precision']:.4f}")
    print(f"  Recall:    {spatial['recall']:.4f}")
    print(f"  F1-Score:  {spatial['f1']:.4f}")
    
    # Spatial error
    temporal = results['temporal']
    print(f"\nError Metrics:")
    print(f"  Spatial:   {spatial['mean_error']:.2f} ± {spatial['std_error']:.2f} px")
    print(f"  Temporal Start: {temporal['mean_start_error_frames']:.2f} ± {temporal['std_start_error_frames']:.2f} frames")
    print(f"  Temporal End:   {temporal['mean_end_error_frames']:.2f} ± {temporal['std_end_error_frames']:.2f} frames")
    print(f"  Temporal IoU:   {temporal['mean_iou']:.4f} ± {temporal['std_iou']:.4f}")
    
    # Directional error (get first angular threshold)
    first_ang_thresh = list(results['directional'].keys())[0]
    directional = results['directional'][first_ang_thresh]
    print(f"  Angular:   {directional['mean_error']:.2f} ± {directional['std_error']:.2f}°")
    print(f"  Accuracy:  {directional['accuracy']:.4f}")


def print_clean_comparison(before_metrics, after_metrics, category="Overall"):
    """
    Print clean before/after comparison.
    
    Args:
        before_metrics: Metrics before post-processing
        after_metrics: Metrics after post-processing
        category: Category name for display
    """
    print(f"\n{'#'*60}")
    print(f"COMPARISON: {category}")
    print(f"{'#'*60}")
    
    # Before
    print_clean_summary(before_metrics, f"Before Post-Processing")
    
    # After
    print_clean_summary(after_metrics, f"After Post-Processing")
    
    # Change summary
    print(f"\n{'-'*60}")
    print(f"Changes")
    print(f"{'-'*60}")
    
    # Get first threshold results
    first_thresh = list(before_metrics['per_threshold'].keys())[0]
    before_spatial = before_metrics['per_threshold'][first_thresh]['spatial']
    after_spatial = after_metrics['per_threshold'][first_thresh]['spatial']
    
    before_preds = before_metrics['counts']['predictions']
    after_preds = after_metrics['counts']['predictions']
    
    pred_change = after_preds - before_preds
    prec_change = after_spatial['precision'] - before_spatial['precision']
    rec_change = after_spatial['recall'] - before_spatial['recall']
    f1_change = after_spatial['f1'] - before_spatial['f1']
    
    print(f"  Predictions: {before_preds} → {after_preds} ({pred_change:+d})")
    print(f"  Precision:   {before_spatial['precision']:.4f} → {after_spatial['precision']:.4f} ({prec_change:+.4f})")
    print(f"  Recall:      {before_spatial['recall']:.4f} → {after_spatial['recall']:.4f} ({rec_change:+.4f})")
    print(f"  F1-Score:    {before_spatial['f1']:.4f} → {after_spatial['f1']:.4f} ({f1_change:+.4f})")
    print(f"{'#'*60}\n")

def get_detailed_metrics_silent(preds, gts, eval_config):
    """
    Calculate detailed metrics WITHOUT printing (silent version).
    Returns the same structure as get_detailed_metrics but without console output.
    """
    config = eval_config
    
    # Get all matched pairs for each position threshold
    _, all_matched_pairs = calculate_position_metrics_comprehensive(
        preds, gts, config['pos_thresholds']
    )
    
    # Overall counts
    total_preds = sum(len(batch_preds) for batch_preds in preds)
    total_gts = sum(len(batch_gts) for batch_gts in gts)
    
    # Results storage
    all_results = {}
    
    # Process each position threshold
    for pos_thresh in config['pos_thresholds']:
        matched_pairs = all_matched_pairs.get(pos_thresh, [])
        
        if not matched_pairs:
            # Store empty results
            all_results[pos_thresh] = {
                'spatial': {
                    'tp': 0, 'fp': total_preds, 'fn': total_gts,
                    'precision': 0, 'recall': 0, 'f1': 0,
                    'mean_error': float('inf'),
                    'std_error': 0
                },
                'temporal': {
                    'mean_start_error_frames': float('inf'),
                    'std_start_error_frames': 0,
                    'mean_end_error_frames': float('inf'),
                    'std_end_error_frames': 0,
                    'mean_iou': 0,
                    'std_iou': 0
                },
                'directional': {
                    ang_thresh: {
                        'accuracy': 0,
                        'mean_error': float('inf'),
                        'std_error': 0,
                        'mean_cosine_sim': 0
                    } for ang_thresh in config['angular_thresholds']
                }
            }
            continue
        
        # Calculate spatial statistics
        spatial_errors = []
        for gt, pred in matched_pairs:
            gt_pos = np.array(gt['position'])
            pred_pos = np.array(pred['position'])
            dist = np.linalg.norm(gt_pos - pred_pos)
            spatial_errors.append(dist)
        
        tp = len(matched_pairs)
        fp = total_preds - tp
        fn = total_gts - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Calculate temporal statistics
        temporal_start_errors = []
        temporal_end_errors = []
        temporal_ious = []
        
        for gt, pred in matched_pairs:
            gt_start, gt_end = gt['temporal_offsets']
            pred_start, pred_end = pred['temporal_offsets']
            
            start_err = abs(pred_start - gt_start)
            end_err = abs(pred_end - gt_end)
            temporal_start_errors.append(start_err)
            temporal_end_errors.append(end_err)
            
            # Temporal IoU
            intersection = max(0, min(pred_end, gt_end) - max(pred_start, gt_start))
            union = max(pred_end, gt_end) - min(pred_start, gt_start)
            iou = intersection / union if union > 0 else 0
            temporal_ious.append(iou)
        
        # Calculate directional statistics for each angular threshold
        directional_results = {}
        for ang_thresh in config['angular_thresholds']:
            angular_errors = []
            cosine_sims = []
            
            for gt, pred in matched_pairs:
                gt_dir = np.array(gt['direction'])
                pred_dir = np.array(pred['direction'])
                
                # Normalize
                gt_dir_norm = gt_dir / (np.linalg.norm(gt_dir) + 1e-8)
                pred_dir_norm = pred_dir / (np.linalg.norm(pred_dir) + 1e-8)
                
                # Cosine similarity
                cos_sim = np.dot(gt_dir_norm, pred_dir_norm)
                cosine_sims.append(cos_sim)
                
                # Angular error
                angular_error = np.degrees(np.arccos(np.clip(cos_sim, -1.0, 1.0)))
                angular_errors.append(angular_error)
            
            correct_direction = np.sum(np.array(angular_errors) <= ang_thresh)
            accuracy = correct_direction / len(angular_errors) if angular_errors else 0
            
            directional_results[ang_thresh] = {
                'accuracy': accuracy,
                'mean_error': np.mean(angular_errors),
                'std_error': np.std(angular_errors),
                'mean_cosine_sim': np.mean(cosine_sims)
            }
        
        # Store results for this threshold
        all_results[pos_thresh] = {
            'spatial': {
                'tp': tp, 'fp': fp, 'fn': fn,
                'precision': precision, 'recall': recall, 'f1': f1,
                'mean_error': np.mean(spatial_errors),
                'std_error': np.std(spatial_errors)
            },
            'temporal': {
                'mean_start_error_frames': np.mean(temporal_start_errors),
                'std_start_error_frames': np.std(temporal_start_errors),
                'mean_end_error_frames': np.mean(temporal_end_errors),
                'std_end_error_frames': np.std(temporal_end_errors),
                'mean_iou': np.mean(temporal_ious),
                'std_iou': np.std(temporal_ious)
            },
            'directional': directional_results
        }
    
    # Calculate comprehensive metrics (silent)
    comprehensive_metrics = calculate_detection_metrics(
        preds, gts,
        config['pos_thresholds'],
        config['iou_threshold_range'],
        config['angular_thresholds']
    )
    
    return {
        'per_threshold': all_results,
        'comprehensive': comprehensive_metrics,
        'counts': {'predictions': total_preds, 'ground_truths': total_gts}
    }