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
        progress_bar = tqdm(dataloader, desc='Getting Predictions', leave=False)
        
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
        return all_outputs, all_targets, all_starts, all_ends, all_video_names, all_original_res

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

def calculate_position_metrics_comprehensive(preds, gts, pos_thresholds=[5, 10, 15, 20, 25, 30]):
    """
    Calculate position metrics with comprehensive thresholds.
    """
    # handle single threshold input
    if isinstance(pos_thresholds, (int, float)):
        pos_thresholds = [pos_thresholds]
    
    all_metrics = {}
    all_matched_pairs = {}
    
    # calculate for each threshold
    for threshold in pos_thresholds:
        all_distances = []
        matched_pairs = []
        
        for batch_preds, batch_gts in zip(preds, gts):
            # for each GT, find closest prediction within threshold
            for gt in batch_gts:
                min_dist = float('inf')
                best_pred = None
                
                for pred in batch_preds:
                    gt_pos = np.array(gt['position'])
                    pred_pos = np.array(pred['position'])
                    distance = np.linalg.norm(gt_pos - pred_pos)
                    
                    if distance < min_dist and distance <= threshold:
                        min_dist = distance
                        best_pred = pred
                
                if best_pred is not None:
                    all_distances.append(min_dist)
                    matched_pairs.append((gt, best_pred))
        
        total_predictions = sum(len(batch_preds) for batch_preds in preds)
        total_gts = sum(len(batch_gts) for batch_gts in gts)
        
        # calculate precision, recall, and F1
        precision = len(matched_pairs) / total_predictions if total_predictions > 0 else 0.0
        recall = len(matched_pairs) / total_gts if total_gts > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # store metrics for this threshold
        metrics_at_threshold = {
            f'mean_position_error_{threshold}': np.mean(all_distances) if all_distances else float('inf'),
            f'position_precision_{threshold}': precision,
            f'position_recall_{threshold}': recall,
            f'position_f1_{threshold}': f1,
            f'matched_detections_{threshold}': len(matched_pairs),
        }
        
        if all_distances:
            metrics_at_threshold[f'position_error_std_{threshold}'] = np.std(all_distances)
        
        all_metrics.update(metrics_at_threshold)
        all_matched_pairs[threshold] = matched_pairs
    
    # calculate mean metrics across all thresholds
    precisions = [all_metrics[f'position_precision_{t}'] for t in pos_thresholds]
    recalls = [all_metrics[f'position_recall_{t}'] for t in pos_thresholds]
    f1_scores = [all_metrics[f'position_f1_{t}'] for t in pos_thresholds]
    position_errors = [all_metrics[f'mean_position_error_{t}'] for t in pos_thresholds]
    
    summary_metrics = {
        'precision': np.mean(precisions),
        'recall': np.mean(recalls),
        'f1': np.mean(f1_scores),
        'mean_error': np.mean(position_errors),
        'total_predictions': total_predictions,
        'total_ground_truths': total_gts
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

def get_eval_metrics(
    preds, 
    gts, 
    pos_thresholds=None,
    iou_threshold_range=None,
    angular_thresholds=None
):
    """
    Run complete evaluation with hierarchical metrics structure.
    
    Args:
        preds: List of prediction batches
        gts: List of ground truth batches
        pos_thresholds: List of spatial thresholds or single value. 
                       Default: [5, 10, 15, 20, 25, 30]
                       Examples: [10] or [5, 10, 15, 20, 25, 30] or 10
        iou_threshold_range: Tuple for temporal IoU range or single value.
                            Default: (0.25, 0.75)
                            Examples: (0.5,) or (0.25, 0.75) or 0.5
        angular_thresholds: List of angular thresholds or single value.
                           Default: [10, 15, 20]
                           Examples: [10] or [10, 15, 20] or 10
    
    Returns:
        Hierarchical dictionary with metrics containing:
        - comprehensive: Detection metrics across all thresholds
        - spatial: Position-based metrics
        - temporal: Temporal IoU metrics
        - directional: Angular direction metrics
        - counts: Prediction and ground truth counts
        - config: Configuration used for evaluation
    """
    # Set defaults
    if pos_thresholds is None:
        pos_thresholds = [5, 10, 15, 20, 25, 30]
    if iou_threshold_range is None:
        iou_threshold_range = (0.25, 0.75)
    if angular_thresholds is None:
        angular_thresholds = [10, 15, 20]
    
    # Normalize inputs - handle flexible input formats
    # Convert single pos_threshold to list
    if isinstance(pos_thresholds, (int, float)):
        pos_thresholds = [pos_thresholds]
    
    # Handle single iou_threshold_range value
    if isinstance(iou_threshold_range, (int, float)):
        iou_threshold_range = (iou_threshold_range, iou_threshold_range)
    elif len(iou_threshold_range) == 1:
        iou_threshold_range = (iou_threshold_range[0], iou_threshold_range[0])
    
    # Convert single angular_threshold to list
    if isinstance(angular_thresholds, (int, float)):
        angular_thresholds = [angular_thresholds]
    
    # Get matched pairs for detailed metrics
    position_metrics, all_matched_pairs = calculate_position_metrics_comprehensive(
        preds, gts, pos_thresholds
    )
    
    # Use matched pairs from the first threshold for direction/temporal metrics
    primary_matched_pairs = all_matched_pairs.get(pos_thresholds[0], [])
    
    # Calculate all metrics
    comprehensive_metrics = calculate_detection_metrics(
        preds, gts, 
        pos_thresholds,
        iou_threshold_range,
        angular_thresholds
    )
    
    directional_metrics = calculate_direction_metrics_comprehensive(
        primary_matched_pairs, 
        angular_thresholds
    )
    
    temporal_metrics = calculate_temporal_metrics(
        primary_matched_pairs,
        iou_threshold_range
    )
    
    # Combine into hierarchical structure for clean prints and outputs
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