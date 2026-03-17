import torch 
from tqdm import tqdm
import numpy as np
from typing import Dict, List, Tuple, Any

def get_preds_gt(model, dataloader, device, return_frames=False, batch_idx_for_frames=0):
    """
    Get all predictions, targets, metadata, and optionally frames from the model on a dataloader
    """
    model.eval()
    all_outputs = []
    all_targets = []
    all_starts = []
    all_ends = []
    all_video_names = []
    all_original_res = [] 
    all_frames = None
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc='Fetching ground truth values.', leave=False)
        
        for batch_idx, batch in enumerate(progress_bar):
            # move batch to device
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            inputs = batch['video'].to(device)
            targets = batch['targets'].to(device)
            metadata = batch['metadata']
            
            # get frame info from metadata
            for meta in metadata:
                all_starts.append(meta['start_frame'])
                all_ends.append(meta['end_frame'])
                all_video_names.append(meta['video_name'])
                all_original_res.append(meta['res'])  # original H W resolutions 
            
            # used to store frames only from specified batch if wanted (optional)
            if return_frames and batch_idx == batch_idx_for_frames:
                all_frames = inputs.cpu()
            
            # get model preds
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(inputs)
            
            # push preds and targets to cpu
            all_outputs.append(outputs.cpu())
            all_targets.append(targets.cpu())
            
            progress_bar.set_postfix({
                'Batches Processed': batch_idx + 1,
                'Frames Stored': f'Batch {batch_idx}' if (return_frames and batch_idx == batch_idx_for_frames) else 'No'
            })
    
    # lists to tensor arrays
    all_outputs = torch.cat(all_outputs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    all_starts = np.array(all_starts)
    all_ends = np.array(all_ends)
    all_video_names = np.array(all_video_names)
    all_original_res = np.array(all_original_res)  # shape (N, 2)
    
    if return_frames and all_frames is not None:
        return all_outputs, all_targets, all_starts, all_ends, all_video_names, all_frames, all_original_res
    else:
        return all_outputs, all_targets, all_starts, all_ends, all_video_names, None, all_original_res

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
    original_size: tuple = (960, 540)
) -> List[List[Dict]]:
    all_detections = []
    batch_size, grid_size, _, _, _ = model_output.shape
    
    for b in range(batch_size):
        sample_detections = []
        window_start = all_starts[b]  
        window_end = all_ends[b]
        
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
                            "temporal_offsets": [start_frame, end_frame]
                        })
        
        all_detections.append(sample_detections)
    
    return all_detections 

def yolo_to_img_space_gt(
    model_output: torch.Tensor,
    all_starts: List[int],
    all_ends: List[int],
    window_size: int = 16,
    original_size: tuple = (960, 540)
) -> List[List[Dict]]:
    """Transform ground truth YOLO format to image space (no confidence thresholding)"""
    all_detections = []
    batch_size, grid_size, _, _, _ = model_output.shape
    for b in range(batch_size):
        sample_detections = []
        window_start = all_starts[b]  
        window_end = all_ends[b]
        
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
                            "is_ground_truth": True  # Optional: mark as GT for debug
                        })
        
        all_detections.append(sample_detections)
    
    return all_detections

def calculate_position_metrics_comprehensive(matched_pairs_per_combo, total_predictions, total_gts):
    """
    Calculate spatial position metrics conditioned on joint matched pairs.
    Same matched pairs as comprehensive/temporal/directional for full consistency.
    Mean position error tells you: on correct joint detections, how many pixels
    off is the predicted position on average.
    """
    spatial_errors = []
    spatial_precisions, spatial_recalls, spatial_f1s = [], [], []

    for pairs in matched_pairs_per_combo:
        if not pairs:
            continue

        distances = [
            np.linalg.norm(np.array(gt['position']) - np.array(pred['position']))
            for gt, pred in pairs
        ]
        spatial_errors.extend(distances)

        tp = len(pairs)
        precision = tp / total_predictions if total_predictions > 0 else 0.0
        recall = tp / total_gts if total_gts > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        spatial_precisions.append(precision)
        spatial_recalls.append(recall)
        spatial_f1s.append(f1)

    return {
        'precision': np.mean(spatial_precisions) if spatial_precisions else 0.0,
        'recall': np.mean(spatial_recalls) if spatial_recalls else 0.0,
        'f1': np.mean(spatial_f1s) if spatial_f1s else 0.0,
        'mean_error': np.mean(spatial_errors) if spatial_errors else float('inf'),
        'total_predictions': total_predictions,
        'total_ground_truths': total_gts
    }

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
            'mean_end_error': float('inf'),
            'me_start': float('inf'), # signed: negative means too early, positive means too late
            'me_end': float('inf'),
            'mse_start': float('inf'),
            'mse_end': float('inf'),
            'duration_accuracy': 0.0,
        }
    
    start_errors = []
    end_errors = []
    start_errors_signed = []
    end_errors_signed = []
    duration_accuracies = []
    temporal_iou_scores = []
    
    for gt, pred in matched_pairs:
        gt_start, gt_end = gt['temporal_offsets']
        pred_start, pred_end = pred['temporal_offsets']
        
        # start/end frame errors (abs start and end error)
        start_errors.append(abs(gt_start - pred_start))
        end_errors.append(abs(gt_end - pred_end))

        # start/end frame signed errors: negative means predicted too early
        s_err = pred_start - gt_start
        e_err = pred_end - gt_end
        start_errors_signed.append(s_err)
        end_errors_signed.append(e_err)

        # duration accuracy
        pred_dur = pred_end - pred_start
        gt_dur = gt_end - gt_start
        dur_acc = 1 - abs(pred_dur - gt_dur) / max(pred_dur, gt_dur) if max(pred_dur, gt_dur) > 0 else 1.0
        duration_accuracies.append(dur_acc)

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
    
    start_errors_signed = np.array(start_errors_signed)
    end_errors_signed = np.array(end_errors_signed)
    
    return {
        'mean_iou': np.mean(temporal_iou_scores),
        'mean_start_error': np.mean(start_errors),
        'mean_end_error': np.mean(end_errors),
        'me_start': float(np.mean(start_errors_signed)),   # bias: <0 too early, >0 too late
        'me_end': float(np.mean(end_errors_signed)),
        'mse_start': float(np.mean(start_errors_signed**2)), 
        'mse_end': float(np.mean(end_errors_signed**2)),
        'duration_accuracy': float(np.mean(duration_accuracies)),  # 1.0 perfect, 0.0 worst
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
    all_matched_pairs_per_combo = [] # matched pair one per threshold combo or values 
    
    for pos_threshold in pos_thresholds:
        for iou_threshold in iou_thresholds:
            for angular_threshold in angular_thresholds:
                true_positives = 0
                false_positives = 0
                false_negatives = 0
                pairs_this_combo = []
                
                for batch_preds, batch_gts in zip(preds, gts):
                    matched_gts = set()
                    
                    for pred in batch_preds:
                        best_score = -1
                        best_gt_idx = None
                        best_gt = None
                        
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
                                angular_error <= angular_threshold):
                            
                                combined_score = temp_iou * (1 - pos_dist / pos_threshold) * (1 - angular_error / angular_threshold)
                                if combined_score > best_score:
                                    best_score = combined_score
                                    best_gt_idx = gt_idx
                                    best_gt = gt

                        if best_gt_idx is not None:
                            true_positives += 1
                            matched_gts.add(best_gt_idx)
                            pairs_this_combo.append((best_gt, pred))
                        else:
                            false_positives += 1
                    
                    false_negatives += len(batch_gts) - len(matched_gts)
                
                all_matched_pairs_per_combo.append(pairs_this_combo)
                precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
                recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                comprehensive_precisions.append(precision)
                comprehensive_recalls.append(recall)
                comprehensive_f1_scores.append(f1)
    
    return {
        'precision': np.mean(comprehensive_precisions),
        'recall': np.mean(comprehensive_recalls),
        'f1': np.mean(comprehensive_f1_scores),
        'matched_pairs_per_combo': all_matched_pairs_per_combo
    }

def get_eval_metrics(
    preds, 
    gts, 
    pos_thresholds=None,
    iou_threshold_range=None,
    angular_thresholds=None
):
    """
    Run complete evaluation with hierarchical metrics structure.
    All auxiliary metrics (directional, temporal) are computed on the same
    matched pairs as the comprehensive F1, averaged across all threshold combos.
    
    Args:
        preds: List of prediction batches
        gts: List of ground truth batches
        pos_thresholds: List of spatial thresholds or single value. 
                       Default: [5, 10, 15, 20, 25, 30]
        iou_threshold_range: Tuple for temporal IoU range or single value.
                            Default: (0.25, 0.75)
        angular_thresholds: List of angular thresholds or single value.
                           Default: [10, 15, 20]
    
    Returns:
        Hierarchical dictionary with metrics containing:
        - comprehensive: Detection metrics across all thresholds
        - spatial: Position-based metrics (kept for backwards compat / diagnostics)
        - temporal: Temporal IoU metrics averaged across all threshold combos
        - directional: Angular direction metrics averaged across all threshold combos
        - counts: Prediction and ground truth counts
        - config: Configuration used for evaluation
    """
    if pos_thresholds is None:
        pos_thresholds = [5, 10, 15, 20, 25, 30]
    if iou_threshold_range is None:
        iou_threshold_range = (0.25, 0.75)
    if angular_thresholds is None:
        angular_thresholds = [10, 15, 20]
    
    if isinstance(pos_thresholds, (int, float)):
        pos_thresholds = [pos_thresholds]
    if isinstance(iou_threshold_range, (int, float)):
        iou_threshold_range = (iou_threshold_range, iou_threshold_range)
    elif len(iou_threshold_range) == 1:
        iou_threshold_range = (iou_threshold_range[0], iou_threshold_range[0])
    if isinstance(angular_thresholds, (int, float)):
        angular_thresholds = [angular_thresholds]

    # comprehensive metrics + matched pairs per combo — single source of truth for matching
    comprehensive_metrics = calculate_detection_metrics(
        preds, gts, 
        pos_thresholds,
        iou_threshold_range,
        angular_thresholds
    )
    matched_pairs_per_combo = comprehensive_metrics.pop('matched_pairs_per_combo')

    # spatial metrics kept separately for diagnostics (position error reporting)
    total_predictions = sum(len(b) for b in preds)
    total_gts = sum(len(b) for b in gts)

    position_metrics = calculate_position_metrics_comprehensive(
        matched_pairs_per_combo, total_predictions, total_gts
    )

    # directional + temporal: average across all threshold combos,
    # using the same matched pairs as comprehensive F1
    dir_accuracies, dir_errors, dir_cosines = [], [], []
    temp_ious, temp_start_errors, temp_end_errors = [], [], []
    temp_me_starts, temp_me_ends = [], []
    temp_mse_starts, temp_mse_ends, temp_dur_accs = [], [], []

    for pairs in matched_pairs_per_combo:
        if not pairs:
            continue

        dir_m = calculate_direction_metrics_comprehensive(pairs, angular_thresholds)
        dir_accuracies.append(dir_m['accuracy'])
        dir_errors.append(dir_m['mean_error'])
        dir_cosines.append(dir_m['mean_cosine_similarity'])

        temp_m = calculate_temporal_metrics(pairs, iou_threshold_range)
        temp_ious.append(temp_m['mean_iou'])
        temp_start_errors.append(temp_m['mean_start_error'])
        temp_end_errors.append(temp_m['mean_end_error'])
        temp_me_starts.append(temp_m['me_start'])
        temp_me_ends.append(temp_m['me_end'])
        temp_mse_starts.append(temp_m['mse_start'])
        temp_mse_ends.append(temp_m['mse_end'])
        temp_dur_accs.append(temp_m['duration_accuracy'])

    directional_metrics = {
        'accuracy': np.mean(dir_accuracies) if dir_accuracies else 0.0,
        'mean_error': np.mean(dir_errors) if dir_errors else float('inf'),
        'mean_cosine_similarity': np.mean(dir_cosines) if dir_cosines else 0.0
    }

    temporal_metrics = {
        'mean_iou': np.mean(temp_ious) if temp_ious else 0.0,
        'mean_start_error': np.mean(temp_start_errors) if temp_start_errors else float('inf'),
        'mean_end_error': np.mean(temp_end_errors) if temp_end_errors else float('inf'),
        'me_start': np.mean(temp_me_starts) if temp_me_starts else float('inf'),
        'me_end': np.mean(temp_me_ends) if temp_me_ends else float('inf'),
        'mse_start': np.mean(temp_mse_starts) if temp_mse_starts else float('inf'),
        'mse_end': np.mean(temp_mse_ends) if temp_mse_ends else float('inf'),
        'duration_accuracy': np.mean(temp_dur_accs) if temp_dur_accs else 0.0,
    }

    metrics = {
        'comprehensive': comprehensive_metrics,
        'spatial': position_metrics,
        'temporal': temporal_metrics,
        'directional': directional_metrics,
        'counts': {
            'predictions': position_metrics['total_predictions'],
            'ground_truths': position_metrics['total_ground_truths']
        },
        'config': {
            'pos_thresholds': pos_thresholds,
            'iou_threshold_range': iou_threshold_range,
            'angular_thresholds': angular_thresholds
        }
    }

    return metrics

def print_evaluation_results(test_metrics, post_test_metrics):
    """Print comprehensive evaluation results before and after post-processing"""
    
    print("\n" + "="*80)
    print("Evaluation Results".center(80))
    print("="*80)
    
    # Header
    print(f"{'Metric':<35} {'Before Post-Proc':<20} {'After Post-Proc':<20}")
    print("-"*80)
    
    # Counts
    print(f"{'Predictions':<35} {test_metrics['counts']['predictions']:<20} {post_test_metrics['counts']['predictions']:<20}")
    print(f"{'Ground Truths':<35} {test_metrics['counts']['ground_truths']:<20} {post_test_metrics['counts']['ground_truths']:<20}")
    print("-"*80)
    
    # Comprehensive Detection
    print(f"{'Spatio-Temporal-Directional Detection':<35}")
    print(f"{'  Precision':<35} {test_metrics['comprehensive']['precision']:<20.3f} {post_test_metrics['comprehensive']['precision']:<20.3f}")
    print(f"{'  Recall':<35} {test_metrics['comprehensive']['recall']:<20.3f} {post_test_metrics['comprehensive']['recall']:<20.3f}")
    print(f"{'  F1 Score':<35} {test_metrics['comprehensive']['f1']:<20.3f} {post_test_metrics['comprehensive']['f1']:<20.3f}")
    print("-"*80)
    
    # Spatial Detection
    print(f"{'Spatial Detection':<35}")
    #print(f"{'  Precision':<35} {test_metrics['spatial']['precision']:<20.3f} {post_test_metrics['spatial']['precision']:<20.3f}")
    #print(f"{'  Recall':<35} {test_metrics['spatial']['recall']:<20.3f} {post_test_metrics['spatial']['recall']:<20.3f}")
    #print(f"{'  F1 Score':<35} {test_metrics['spatial']['f1']:<20.3f} {post_test_metrics['spatial']['f1']:<20.3f}")
    print(f"{'  Mean Error (px)':<35} {test_metrics['spatial']['mean_error']:<20.1f} {post_test_metrics['spatial']['mean_error']:<20.1f}")
    print("-"*80)
    
    # Temporal Detection
    print(f"{'Temporal Detection':<35}")
    print(f"{'  Mean IoU':<35} {test_metrics['temporal']['mean_iou']:<20.3f} {post_test_metrics['temporal']['mean_iou']:<20.3f}")
    print(f"{'  Duration Accuracy':<35} {test_metrics['temporal']['duration_accuracy']:<20.3f} {post_test_metrics['temporal']['duration_accuracy']:<20.3f}")
    print(f"{'  ME Start (frames)':<35} {test_metrics['temporal']['me_start']:<20.2f} {post_test_metrics['temporal']['me_start']:<20.2f}")
    print(f"{'  ME End   (frames)':<35} {test_metrics['temporal']['me_end']:<20.2f} {post_test_metrics['temporal']['me_end']:<20.2f}")
    print(f"{'  MSE Start (frames²)':<35} {test_metrics['temporal']['mse_start']:<20.2f} {post_test_metrics['temporal']['mse_start']:<20.2f}")
    print(f"{'  MSE End   (frames²)':<35} {test_metrics['temporal']['mse_end']:<20.2f} {post_test_metrics['temporal']['mse_end']:<20.2f}")
    print("-"*80)
    
    # Directional Detection
    print(f"{'Directional Detection':<35}")
    print(f"{'  Accuracy':<35} {test_metrics['directional']['accuracy']:<20.3f} {post_test_metrics['directional']['accuracy']:<20.3f}")
    print(f"{'  Mean Angular Error (°)':<35} {test_metrics['directional']['mean_error']:<20.1f} {post_test_metrics['directional']['mean_error']:<20.1f}")
    print("="*80 + "\n")

def get_wandb_log_dict(epoch, test_metrics, post_test_metrics):
    """Build wandb log dict from eval metrics."""
    def _metrics_dict(m, prefix):
        return {
            f'{prefix}/std_f1':              m['comprehensive']['f1'],
            f'{prefix}/std_precision':       m['comprehensive']['precision'],
            f'{prefix}/std_recall':          m['comprehensive']['recall'],
            #f'{prefix}/spatial_f1':          m['spatial']['f1'],
            #f'{prefix}/spatial_precision':   m['spatial']['precision'],
            #f'{prefix}/spatial_recall':      m['spatial']['recall'],
            f'{prefix}/spatial_mean_err_px': m['spatial']['mean_error'],
            f'{prefix}/dir_accuracy':        m['directional']['accuracy'],
            f'{prefix}/dir_mean_err_deg':    m['directional']['mean_error'],
            f'{prefix}/dir_cosine_sim':      m['directional']['mean_cosine_similarity'],
            f'{prefix}/temp_mean_iou':       m['temporal']['mean_iou'],
            f'{prefix}/temp_duration_acc':   m['temporal']['duration_accuracy'],
            f'{prefix}/temp_me_start':       m['temporal']['me_start'],
            f'{prefix}/temp_me_end':         m['temporal']['me_end'],
            f'{prefix}/temp_mse_start':      m['temporal']['mse_start'],
            f'{prefix}/temp_mse_end':        m['temporal']['mse_end'],
            f'{prefix}/n_predictions':       m['counts']['predictions'],
        }

    return {
        'epoch':      epoch,
        'eval/n_ground_truths': test_metrics['counts']['ground_truths'],
        **_metrics_dict(test_metrics,      'eval/pre'),
        **_metrics_dict(post_test_metrics, 'eval/post'),
    }