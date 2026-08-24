"""
Optimized (vectorized) versions of eval_utils grid decoding functions.

These are drop-in replacements for `yolo_to_img_space` and `yolo_to_img_space_gt`
from eval_utils.py.  The original versions iterate over every grid cell in pure
Python (4-deep nested loops); these versions use bulk tensor ops and only loop
over the surviving detections (those above the confidence threshold).

Numerical equivalence is verified by test_eval_utils_fast.py.
"""

import torch
import numpy as np
from typing import Dict, List


def yolo_to_img_space_vectorized(
    model_output: torch.Tensor,
    all_starts: List[int],
    all_ends: List[int],
    window_size: int = 16,
    confidence_threshold: float = 0.001,
    original_size: tuple = (960, 540),
    max_dets: int = 10,
) -> List[List[Dict]]:
    """Vectorized version of yolo_to_img_space.

    Replaces the 4-deep Python loop (batch × grid_h × grid_w × max_det) with
    bulk tensor operations.  Only the detections that survive the confidence
    threshold are touched in a Python loop (to build dicts).

    Args / Returns: identical to the original ``yolo_to_img_space``.
    """
    batch_size, grid_size, _, n_anchors, _ = model_output.shape
    orig_w, orig_h = original_size
    cell_width = orig_w / grid_size
    cell_height = orig_h / grid_size

    # ── 1. Vectorised confidence filter ──────────────────────────────────
    # Detach + move to CPU once for the entire tensor
    data = model_output.detach().cpu()                    # (B, G, G, K, 7)
    all_confs = torch.sigmoid(data[..., 0])               # (B, G, G, K)
    mask = all_confs > confidence_threshold                # bool

    # Indices of surviving cells: each is a 1-D tensor of length N_surviving
    b_idx, i_idx, j_idx, k_idx = torch.where(mask)

    # ── 2. Vectorised coordinate transform ───────────────────────────────
    # Gather the 7-vectors for surviving cells
    surviving = data[b_idx, i_idx, j_idx, k_idx]          # (N, 7)
    confs = all_confs[b_idx, i_idx, j_idx, k_idx]         # (N,)

    norm_x = surviving[:, 1]
    norm_y = surviving[:, 2]
    dir_x  = surviving[:, 3]
    dir_y  = surviving[:, 4]
    t_start_off = surviving[:, 5]
    t_end_off   = surviving[:, 6]

    # Grid cell → image coords  (note: j → x, i → y, matching original)
    pos_x = (j_idx.float() + norm_x) * cell_width
    pos_y = (i_idx.float() + norm_y) * cell_height

    # Clamp to image boundaries
    pos_x = pos_x.clamp(0.0, orig_w - 1)
    pos_y = pos_y.clamp(0.0, orig_h - 1)

    # Temporal offsets → absolute frames
    # Use float64 to match original's .item() precision (prevents ±1 frame rounding)
    w_starts = torch.tensor(np.asarray(all_starts), dtype=torch.float64)[b_idx]
    w_ends   = torch.tensor(np.asarray(all_ends),   dtype=torch.float64)[b_idx]

    abs_start = (t_start_off.double() * window_size + w_starts).to(torch.int64)
    abs_end   = (t_end_off.double()   * window_size + w_starts).to(torch.int64)

    # Clamp temporal
    abs_start = abs_start.clamp(w_starts.to(torch.int64), w_ends.to(torch.int64))
    abs_end   = torch.max(abs_start, abs_end.clamp(min=None, max=w_ends.to(torch.int64)))

    # ── 3. Group by batch and build dicts ────────────────────────────────
    # Pre-convert to Python scalars in bulk (much faster than per-element .item())
    confs_list     = confs.tolist()
    pos_x_list     = pos_x.tolist()
    pos_y_list     = pos_y.tolist()
    dir_x_list     = dir_x.tolist()
    dir_y_list     = dir_y.tolist()
    abs_start_list = abs_start.tolist()
    abs_end_list   = abs_end.tolist()
    b_list         = b_idx.tolist()
    i_list         = i_idx.tolist()
    j_list         = j_idx.tolist()
    k_list         = k_idx.tolist()

    # Initialise empty per-sample lists
    all_detections: List[List[Dict]] = [[] for _ in range(batch_size)]

    for idx in range(len(b_list)):
        all_detections[b_list[idx]].append({
            "confidence":       confs_list[idx],
            "position":         [pos_x_list[idx], pos_y_list[idx]],
            "direction":        [dir_x_list[idx], dir_y_list[idx]],
            "grid_cell":        [i_list[idx], j_list[idx], k_list[idx]],
            "temporal_offsets":  [int(abs_start_list[idx]), int(abs_end_list[idx])],
        })

    # ── 4. Per-sample sort + truncate (same as original) ─────────────────
    for dets in all_detections:
        dets.sort(key=lambda x: x['confidence'], reverse=True)
        del dets[max_dets:]  # in-place truncation

    return all_detections


def yolo_to_img_space_gt_vectorized(
    model_output: torch.Tensor,
    all_starts: List[int],
    all_ends: List[int],
    window_size: int = 16,
    original_size: tuple = (960, 540),
) -> List[List[Dict]]:
    """Vectorized version of yolo_to_img_space_gt.

    GT channel 0 is binary (0 or 1), not a logit — no sigmoid needed.

    Args / Returns: identical to the original ``yolo_to_img_space_gt``.
    """
    batch_size, grid_size, _, n_anchors, _ = model_output.shape
    orig_w, orig_h = original_size
    cell_width = orig_w / grid_size
    cell_height = orig_h / grid_size

    data = model_output.detach().cpu()
    conf_vals = data[..., 0]           # (B, G, G, K) — raw, 0 or 1
    mask = conf_vals > 0

    b_idx, i_idx, j_idx, k_idx = torch.where(mask)

    surviving = data[b_idx, i_idx, j_idx, k_idx]
    confs = conf_vals[b_idx, i_idx, j_idx, k_idx]

    norm_x      = surviving[:, 1]
    norm_y      = surviving[:, 2]
    dir_x       = surviving[:, 3]
    dir_y       = surviving[:, 4]
    t_start_off = surviving[:, 5]
    t_end_off   = surviving[:, 6]

    pos_x = (j_idx.float() + norm_x) * cell_width
    pos_y = (i_idx.float() + norm_y) * cell_height
    pos_x = pos_x.clamp(0.0, orig_w - 1)
    pos_y = pos_y.clamp(0.0, orig_h - 1)

    w_starts = torch.tensor(np.asarray(all_starts), dtype=torch.float64)[b_idx]
    w_ends   = torch.tensor(np.asarray(all_ends),   dtype=torch.float64)[b_idx]

    abs_start = (t_start_off.double() * window_size + w_starts).to(torch.int64)
    abs_end   = (t_end_off.double()   * window_size + w_starts).to(torch.int64)
    abs_start = abs_start.clamp(w_starts.to(torch.int64), w_ends.to(torch.int64))
    abs_end   = torch.max(abs_start, abs_end.clamp(min=None, max=w_ends.to(torch.int64)))

    # Bulk convert to Python scalars
    confs_list     = confs.tolist()
    pos_x_list     = pos_x.tolist()
    pos_y_list     = pos_y.tolist()
    dir_x_list     = dir_x.tolist()
    dir_y_list     = dir_y.tolist()
    abs_start_list = abs_start.tolist()
    abs_end_list   = abs_end.tolist()
    b_list         = b_idx.tolist()
    i_list         = i_idx.tolist()
    j_list         = j_idx.tolist()
    k_list         = k_idx.tolist()

    all_detections: List[List[Dict]] = [[] for _ in range(batch_size)]

    for idx in range(len(b_list)):
        b = b_list[idx]
        all_detections[b].append({
            "confidence":       float(confs_list[idx]),
            "position":         [pos_x_list[idx], pos_y_list[idx]],
            "direction":        [float(dir_x_list[idx]), float(dir_y_list[idx])],
            "grid_cell":        [i_list[idx], j_list[idx], k_list[idx]],
            "temporal_offsets":  [int(abs_start_list[idx]), int(abs_end_list[idx])],
            "is_ground_truth":  True,
            "window_idx":       b,
        })

    return all_detections


# ---------------------------------------------------------------------------
# Parallelised detection metrics
# ---------------------------------------------------------------------------

def _evaluate_single_combo(args):
    """Evaluate a single (pos, iou, angular) threshold combo.

    This is a module-level function (not a closure) so it can be pickled
    for ProcessPoolExecutor.

    Args:
        args: tuple of (combo_idx, pos_threshold, iou_threshold, angular_threshold,
              all_preds_flat, gts, total_gts, match_pairs)

    Returns:
        (combo_idx, ap, precision, recall, f1, best_conf, matched_pairs)
    """
    from src.utils.eval_utils import _compute_pair_validity, hungarian_match

    (combo_idx, pos_threshold, iou_threshold, angular_threshold,
     all_preds_flat, gts, total_gts, match_pairs) = args

    # greedy confidence sweep
    matched_gts = {i: set() for i in range(len(gts))}
    tp_flags = []
    matched_pairs_list = []

    for _, sample_idx, pred in all_preds_flat:
        batch_gts = gts[sample_idx]
        best_cost = float('inf')
        best_gt_idx = None
        best_gt = None

        for gt_idx, gt in enumerate(batch_gts):
            if gt_idx in matched_gts[sample_idx]:
                continue
            valid, cost = _compute_pair_validity(
                pred, gt, pos_threshold, iou_threshold, angular_threshold
            )
            if valid and cost < best_cost:
                best_cost = cost
                best_gt_idx = gt_idx
                best_gt = gt

        if best_gt_idx is not None:
            matched_gts[sample_idx].add(best_gt_idx)
            tp_flags.append(1)
            matched_pairs_list.append((best_gt, pred))
        else:
            tp_flags.append(0)

    # early exit
    if len(tp_flags) == 0 or total_gts == 0:
        return (combo_idx, 0.0, 0.0, 0.0, 0.0, 0.0, [])

    tp_cumsum = np.cumsum(tp_flags)
    n_preds = np.arange(1, len(tp_flags) + 1)
    precisions = tp_cumsum / n_preds
    recalls = tp_cumsum / total_gts

    # monotonic precision envelope
    precisions = np.maximum.accumulate(precisions[::-1])[::-1]

    # F1-optimal confidence threshold
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-8)
    best_idx = np.argmax(f1s)
    best_conf = float(all_preds_flat[best_idx][0])

    if match_pairs == 'greedy':
        precision_val = float(precisions[best_idx])
        recall_val = float(recalls[best_idx])
        f1_val = float(f1s[best_idx])
        final_pairs = matched_pairs_list
    else:  # hungarian
        total_tp = 0
        total_fp = 0
        total_fn = 0
        final_pairs = []

        preds_at_threshold = {}
        for conf, sample_idx, pred in all_preds_flat:
            if conf < best_conf:
                break
            if sample_idx not in preds_at_threshold:
                preds_at_threshold[sample_idx] = []
            preds_at_threshold[sample_idx].append(pred)

        for sample_idx in range(len(gts)):
            sample_preds_filtered = preds_at_threshold.get(sample_idx, [])
            pairs, fp, fn = hungarian_match(
                sample_preds_filtered, gts[sample_idx],
                pos_threshold, iou_threshold, angular_threshold
            )
            final_pairs.extend(pairs)
            total_tp += len(pairs)
            total_fp += fp
            total_fn += fn

        precision_val = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall_val = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1_val = (2 * precision_val * recall_val / (precision_val + recall_val + 1e-8)
                  if (precision_val + recall_val) > 0 else 0.0)

    # AP from greedy sweep
    precisions_curve = np.concatenate([[1.0], precisions])
    recalls_curve = np.concatenate([[0.0], recalls])
    ap = float(np.trapezoid(precisions_curve, recalls_curve))

    return (combo_idx, ap, precision_val, recall_val, f1_val, best_conf, final_pairs)


def calculate_detection_metrics_parallel(
    preds, gts,
    pos_thresholds=[5, 10, 15, 20, 25, 30],
    iou_threshold_range=(0.25, 0.75),
    angular_thresholds=[10, 15, 20],
    match_pairs='hungarian',
    max_workers=None,
):
    """Parallelized version of calculate_detection_metrics.

    Dispatches each threshold combo to a process pool worker.
    Returns identical output to the original.

    Args:
        max_workers: Number of worker processes. Defaults to min(cpu_count, 8).
                     Set to 1 to run serially (useful for debugging).
    """
    import os
    from concurrent.futures import ProcessPoolExecutor, as_completed

    # normalise threshold inputs (same as original)
    if isinstance(pos_thresholds, (int, float)):
        pos_thresholds = [pos_thresholds]
    if isinstance(iou_threshold_range, (int, float)):
        iou_threshold_range = (iou_threshold_range, iou_threshold_range)
    elif len(iou_threshold_range) == 1:
        iou_threshold_range = (iou_threshold_range[0], iou_threshold_range[0])
    if isinstance(angular_thresholds, (int, float)):
        angular_thresholds = [angular_thresholds]

    if match_pairs not in ('greedy', 'hungarian'):
        raise ValueError(f"Unknown matching: '{match_pairs}'. Use 'greedy' or 'hungarian'.")

    iou_thresholds = [round(t, 2) for t in
                      np.arange(iou_threshold_range[0], iou_threshold_range[1] + 0.1, 0.1)]

    total_gts = sum(len(g) for g in gts)

    # flatten predictions once (shared across all combos)
    all_preds_flat = []
    for sample_idx, sample_preds in enumerate(preds):
        for pred in sample_preds:
            all_preds_flat.append((pred['confidence'], sample_idx, pred))
    all_preds_flat.sort(key=lambda x: x[0], reverse=True)

    # Build list of combos with indices to maintain ordering
    combos = []
    combo_idx = 0
    for pos_threshold in pos_thresholds:
        for iou_threshold in iou_thresholds:
            for angular_threshold in angular_thresholds:
                combos.append((
                    combo_idx, pos_threshold, iou_threshold, angular_threshold,
                    all_preds_flat, gts, total_gts, match_pairs
                ))
                combo_idx += 1

    n_combos = len(combos)

    if max_workers is None:
        max_workers = min(os.cpu_count() or 1, 8)

    # For small numbers of combos or single worker, run serially to avoid
    # process spawn overhead
    if n_combos <= 3 or max_workers <= 1:
        results = [_evaluate_single_combo(c) for c in combos]
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_evaluate_single_combo, c): c[0]
                       for c in combos}
            results = []
            for future in as_completed(futures):
                results.append(future.result())

    # Sort by combo_idx to restore original order
    results.sort(key=lambda x: x[0])

    # Unpack into original output format
    ap_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    conf_scores = []
    all_matched_pairs_per_combo = []

    for (_, ap, prec, rec, f1, conf, pairs) in results:
        ap_scores.append(ap)
        precision_scores.append(prec)
        recall_scores.append(rec)
        f1_scores.append(f1)
        conf_scores.append(conf)
        all_matched_pairs_per_combo.append(pairs)

    return {
        'map':             float(np.mean(ap_scores)),
        'mean_precision':  float(np.mean(precision_scores)),
        'mean_recall':     float(np.mean(recall_scores)),
        'mean_f1':         float(np.mean(f1_scores)),
        'mean_best_conf':  float(np.mean(conf_scores)),
        'matching':        match_pairs,
        'ap_per_combo':    ap_scores,
        'matched_pairs_per_combo': all_matched_pairs_per_combo,
    }


# ---------------------------------------------------------------------------
# Fast get_eval_metrics — drop-in replacement using parallel metrics
# ---------------------------------------------------------------------------

def get_eval_metrics_fast(
    preds,
    gts,
    pos_thresholds=None,
    iou_threshold_range=None,
    angular_thresholds=None,
    match_pairs='greedy',
    waggle_run_ids=None,
    video_names=None,
    max_workers=None,
):
    """Drop-in replacement for get_eval_metrics using parallelized metric computation.

    Args / Returns: identical to the original ``get_eval_metrics``,
    with an additional ``max_workers`` parameter for the process pool.
    """
    from src.utils.eval_utils import (
        calculate_position_metrics_comprehensive,
        calculate_direction_metrics_comprehensive,
        calculate_temporal_metrics,
    )

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

    # Use parallelized detection metrics
    comprehensive_metrics = calculate_detection_metrics_parallel(
        preds, gts,
        pos_thresholds,
        iou_threshold_range,
        angular_thresholds,
        match_pairs=match_pairs,
        max_workers=max_workers,
    )
    matched_pairs_per_combo = comprehensive_metrics.pop('matched_pairs_per_combo')

    # Everything below is identical to original get_eval_metrics
    total_predictions = sum(len(b) for b in preds)
    total_gts_count = sum(len(b) for b in gts)

    position_metrics = calculate_position_metrics_comprehensive(
        matched_pairs_per_combo, total_predictions, total_gts_count
    )

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
        'mean_cosine_similarity': np.mean(dir_cosines) if dir_cosines else 0.0,
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

    # Detection coverage (same logic as original)
    mid_idx = len(matched_pairs_per_combo) // 2
    mid_pairs = matched_pairs_per_combo[mid_idx] if matched_pairs_per_combo else []

    if waggle_run_ids is not None and video_names is not None:
        matched_window_idxs = set()
        for gt, pred in mid_pairs:
            if 'window_idx' in gt:
                matched_window_idxs.add(gt['window_idx'])

        all_dance_ids = set()
        detected_dance_ids = set()
        for window_idx, run_id in enumerate(waggle_run_ids):
            if run_id is not None and not (isinstance(run_id, float) and np.isnan(run_id)):
                dance_key = (video_names[window_idx], int(run_id))
                all_dance_ids.add(dance_key)
                if window_idx in matched_window_idxs:
                    detected_dance_ids.add(dance_key)

        n_unique_dances = len(all_dance_ids)
        n_detected = len(detected_dance_ids)
        n_missed = n_unique_dances - n_detected
        total_preds_clustered = sum(len(b) for b in preds)

        coverage = {
            'level': 'waggle_run',
            'unique_gt_dances': n_unique_dances,
            'detected_dances': n_detected,
            'missed_dances': n_missed,
            'total_predictions': total_preds_clustered,
            'gt_recall': n_detected / n_unique_dances if n_unique_dances > 0 else 0.0,
            'pred_precision': n_detected / total_preds_clustered if total_preds_clustered > 0 else 0.0,
        }
    else:
        total_preds = sum(len(b) for b in preds)
        total_gts_val = sum(len(b) for b in gts)
        tp = len(mid_pairs)
        coverage = {
            'level': 'window',
            'unique_gt_dances': total_gts_val,
            'detected_dances': tp,
            'missed_dances': total_gts_val - tp,
            'total_predictions': total_preds,
            'gt_recall': tp / total_gts_val if total_gts_val > 0 else 0.0,
            'pred_precision': tp / total_preds if total_preds > 0 else 0.0,
        }

    return {
        'comprehensive': comprehensive_metrics,
        'spatial': position_metrics,
        'temporal': temporal_metrics,
        'directional': directional_metrics,
        'detection_coverage': coverage,
        'counts': {
            'predictions': position_metrics['total_predictions'],
            'ground_truths': position_metrics['total_ground_truths'],
        },
        'config': {
            'pos_thresholds': pos_thresholds,
            'iou_threshold_range': iou_threshold_range,
            'angular_thresholds': angular_thresholds,
        },
    }

