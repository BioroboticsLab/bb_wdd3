"""
Cross-window waggle-run level evaluation.

Maps per-window crop-space predictions to frame-space, clusters across
overlapping windows per video, deduplicates GT from annotations, and
computes dance-level metrics (AP, P/R/F1, position error, temporal IoU,
angular error).
"""

import numpy as np
from collections import defaultdict
from sklearn.cluster import DBSCAN
from scipy.optimize import linear_sum_assignment


# ─── Crop origin computation ─────────────────────────────────────────────────

def compute_crop_origins(test_df, crop_w=224, crop_h=224, all_original_res=None):
    """
    Compute the (x_min, y_min) crop origin for each window in test_df.
    
    IMPORTANT: VideoYoloDataset.__getitem__ applies a deterministic random
    offset (seeded by sample index) even when is_training=False. We must
    replicate that exact offset here so that decoded crop-space positions
    map back to the correct frame-space coordinates.
    
    The offset logic (dataset.py L142-154):
        rng = RandomState(seed=idx)
        margin = int(crop_w * 0.2)
        max_offset = (crop_w // 2) - margin
        offset_x = rng.randint(-max_offset, max_offset + 1)
        offset_y = rng.randint(-max_offset, max_offset + 1)
        x_min = clamp(gt_x - crop_w/2 + offset_x, 0, orig_w - crop_w)
    
    Args:
        test_df: DataFrame with x1, y1 columns (GT pixel positions)
        crop_w, crop_h: crop dimensions (default 224)
        all_original_res: list of (H, W) tuples per window, or None
                          if None, uses a large default so clamping rarely triggers
    
    Returns:
        list of (x_min, y_min) tuples, one per window
    """
    # Replicate dataset's deterministic random offset
    margin_x = int(crop_w * 0.2)
    margin_y = int(crop_h * 0.2)
    max_offset_x = (crop_w // 2) - margin_x
    max_offset_y = (crop_h // 2) - margin_y
    
    origins = []
    for i in range(len(test_df)):
        row = test_df.iloc[i]
        gt_x, gt_y = float(row['x1']), float(row['y1'])
        
        if all_original_res is not None:
            orig_h, orig_w = all_original_res[i]
        else:
            orig_h, orig_w = 9999, 9999
        
        # Same RNG seed as VideoYoloDataset.__getitem__ (dataset.py L143)
        rng = np.random.RandomState(seed=i)
        offset_x = rng.randint(-max_offset_x, max_offset_x + 1)
        offset_y = rng.randint(-max_offset_y, max_offset_y + 1)
        
        x_min_ideal = int(gt_x - crop_w / 2) + offset_x
        y_min_ideal = int(gt_y - crop_h / 2) + offset_y
        x_min = max(0, min(int(orig_w - crop_w), x_min_ideal))
        y_min = max(0, min(int(orig_h - crop_h), y_min_ideal))
        origins.append((x_min, y_min))
    
    return origins


# ─── GT deduplication ─────────────────────────────────────────────────────────

def deduplicate_gt_dances(test_df):
    """
    Group annotations by (video_name, waggle_run_id) to get unique dances.
    
    Uses x1/y1 (frame-space pixel coords) for position, direction_x/y for
    direction, and waggle_start/waggle_end for temporal extent.
    
    Returns:
        dict: {video_name: [dance_dict, ...]}
        where dance_dict has keys:
            position: [x1, y1]  in frame-space pixels (NOT normalized)
            direction: [dx, dy]  (unit vector)
            temporal_offsets: [waggle_start, waggle_end]
            waggle_run_id: int
            n_windows: int
    """
    waggle_df = test_df[test_df['waggle'] == 1].copy()
    
    gt_dances = defaultdict(list)
    
    grouped = waggle_df.groupby(['video_name', 'waggle_run_id'])
    for (vname, rid), group in grouped:
        # Position: mean of x1, y1 across windows (consistent within a video)
        px = group['x1'].mean()
        py = group['y1'].mean()
        
        # Direction: mean of unit vectors, re-normalize
        dx = group['direction_x'].mean()
        dy = group['direction_y'].mean()
        d_norm = np.sqrt(dx**2 + dy**2) + 1e-8
        dx, dy = dx / d_norm, dy / d_norm
        
        # Temporal: full extent from waggle_start/waggle_end 
        t_start = group['waggle_start'].min()
        t_end = group['waggle_end'].max()
        
        gt_dances[vname].append({
            'position': [float(px), float(py)],
            'direction': [float(dx), float(dy)],
            'temporal_offsets': [float(t_start), float(t_end)],
            'waggle_run_id': int(rid),
            'n_windows': len(group),
        })
    
    return dict(gt_dances)

# ─── Cross-window overlap averaging ──────────────────────────────────────────

def average_overlapping_predictions(per_window_preds, crop_origins, video_names,
                                     window_ranges=None, video_fps=None,
                                     max_temporal_gap_sec=1.0):
    """
    Average predictions across overlapping temporal windows that share the
    same spatial crop.
    
    Windows are grouped by (video_name, crop_origin). Within each group,
    predictions are matched by grid_cell index. For each cell, confidence
    is averaged using a **temporal-overlap-aware denominator**: only windows
    whose temporal range overlaps the detection's temporal extent contribute
    to the denominator.
    
    Before averaging, detections at the same grid cell are split into
    temporally coherent sub-groups (gap > max_temporal_gap_sec starts a
    new group). This prevents merging detections from different bees that
    happen to occupy the same cell at different times — critical when many
    temporally distant windows share one crop origin (low-res videos).
    
    Args:
        per_window_preds: list of lists of prediction dicts, one list per window.
            Each prediction dict has keys: confidence, position, direction,
            temporal_offsets, grid_cell (list [i,j,k]).
        crop_origins: list of (x_min, y_min) per window
        video_names: list of video name strings per window
        window_ranges: optional list of (start_frame, end_frame) per window.
            If None, inferred from the detections' temporal_offsets within
            each window (min start, max end). Falls back to group-size
            denominator for windows with no detections.
        video_fps: dict {video_name: fps} or None (defaults to 15fps).
            Used to convert max_temporal_gap_sec to frames.
        max_temporal_gap_sec: maximum temporal gap in seconds between
            consecutive detections at the same grid cell before they are
            split into separate sub-groups. Default 1.0s.
    
    Returns:
        list of lists of averaged prediction dicts (same outer length as input,
        but predictions are reassigned to the first window of each group;
        other windows in the group get empty lists).
    """
    if not per_window_preds:
        return []
    
    # ── 0. Build window temporal ranges ──
    # Each window needs a [start, end] range so we can count temporal overlaps.
    if window_ranges is not None:
        w_ranges = list(window_ranges)
    else:
        # Infer from detections: use min/max of temporal_offsets per window.
        # Windows with no detections get None (they can't contribute to overlap
        # counts anyway since they have no detections to average).
        w_ranges = []
        for preds in per_window_preds:
            if preds:
                t_starts = [p['temporal_offsets'][0] for p in preds]
                t_ends = [p['temporal_offsets'][1] for p in preds]
                w_ranges.append((min(t_starts), max(t_ends)))
            else:
                w_ranges.append(None)
    
    # ── 1. Group windows by (video_name, crop_origin) ──
    groups = defaultdict(list)  # key → list of (window_index, preds)
    for w_idx, (preds, origin, vname) in enumerate(
        zip(per_window_preds, crop_origins, video_names)
    ):
        key = (vname, tuple(origin))
        groups[key].append((w_idx, preds))
    
    # ── 2. Average within each group ──
    result = [[] for _ in range(len(per_window_preds))]
    
    for key, window_list in groups.items():
        n_windows = len(window_list)
        
        # Precompute temporal ranges for all windows in this group
        group_ranges = []  # (start, end) or None per window in group
        for w_idx, _ in window_list:
            group_ranges.append(w_ranges[w_idx])
        
        # Collect all detections by grid cell
        # cell_key → list of prediction dicts (one per window that detected it)
        cell_preds = defaultdict(list)
        for w_idx, preds in window_list:
            for pred in preds:
                gc = pred.get('grid_cell', [0, 0, 0])
                cell_key = tuple(gc)
                cell_preds[cell_key].append(pred)
        
        # Average each cell — but first split by temporal coherence.
        # The same grid cell can fire for DIFFERENT bees at different times
        # when many temporally distant windows share one crop origin (e.g.
        # 67 windows from 8 runs at a low-res 480×270 video).  Without
        # splitting, detections from run #4 (frame 158) and run #9 (frame
        # 403) at the same cell get merged into one detection spanning
        # frames 158–419, crushing confidence and creating a bogus extent.
        vname = key[0]
        fps = (video_fps or {}).get(vname, 15.0)
        max_gap_frames = max_temporal_gap_sec * fps
        
        averaged = []
        for cell_key, preds_for_cell in cell_preds.items():
            # Split into temporally coherent sub-groups:
            # sort by temporal midpoint, then cut where consecutive
            # detections are more than max_gap_frames apart.
            preds_for_cell.sort(
                key=lambda p: (p['temporal_offsets'][0] + p['temporal_offsets'][1]) / 2
            )
            sub_groups = [[preds_for_cell[0]]]
            for pred in preds_for_cell[1:]:
                prev_end = sub_groups[-1][-1]['temporal_offsets'][1]
                curr_start = pred['temporal_offsets'][0]
                if curr_start - prev_end > max_gap_frames:
                    sub_groups.append([pred])
                else:
                    sub_groups[-1].append(pred)
            
            for sub_preds in sub_groups:
                n_detections = len(sub_preds)
                
                # Detection's temporal extent (union within this sub-group)
                det_t_start = min(p['temporal_offsets'][0] for p in sub_preds)
                det_t_end = max(p['temporal_offsets'][1] for p in sub_preds)
                
                # Count windows in this group whose temporal range overlaps
                # the detection's temporal extent
                n_overlapping = 0
                for wr in group_ranges:
                    if wr is None:
                        n_overlapping += 1
                    else:
                        w_start, w_end = wr
                        if w_start <= det_t_end and w_end >= det_t_start:
                            n_overlapping += 1
                
                denom = max(n_overlapping, 1)
                
                avg_conf = sum(p['confidence'] for p in sub_preds) / denom
                
                avg_pos = [
                    sum(p['position'][0] for p in sub_preds) / n_detections,
                    sum(p['position'][1] for p in sub_preds) / n_detections,
                ]
                
                avg_dir = [
                    sum(p['direction'][0] for p in sub_preds) / n_detections,
                    sum(p['direction'][1] for p in sub_preds) / n_detections,
                ]
                norm = (avg_dir[0]**2 + avg_dir[1]**2) ** 0.5
                if norm > 1e-8:
                    avg_dir = [avg_dir[0] / norm, avg_dir[1] / norm]
                
                averaged.append({
                    'confidence': avg_conf,
                    'position': avg_pos,
                    'direction': avg_dir,
                    'temporal_offsets': [det_t_start, det_t_end],
                    'grid_cell': list(cell_key),
                    'n_detections': n_detections,
                    'n_overlapping': n_overlapping,
                    'n_windows': n_windows,
                })
        
        # Assign averaged predictions to the first window in the group
        first_w_idx = window_list[0][0]
        result[first_w_idx] = averaged
    
    return result


# ─── Cross-window prediction clustering ──────────────────────────────────────

def cross_window_cluster_predictions(
    per_window_preds,
    crop_origins,
    video_names,
    video_resolutions,
    video_fps=None,
    spatial_threshold=30.0,
    temporal_threshold_sec=0.3,
    confidence_threshold=0.5,
    min_samples=1,
    mode='mean',
):
    """
    Map per-window crop-space predictions to frame-space, pool per video,
    and cluster to produce consolidated waggle run predictions.
    
    Uses normalized [0,1] coordinates for resolution-invariant spatial
    clustering, and seconds for fps-invariant temporal clustering.
    
    Args:
        per_window_preds: list of lists, per-window predictions in crop space
        crop_origins: list of (x_min, y_min) per window
        video_names: list/array of video names per window
        video_resolutions: list of (H, W) per window
        video_fps: dict {video_name: fps} or None (defaults to 15fps)
        spatial_threshold: DBSCAN eps in pixels (at reference 1000px width)
        temporal_threshold_sec: temporal proximity in seconds for clustering
        confidence_threshold: minimum confidence to include
        min_samples: DBSCAN min_samples
        mode: 'mean' or 'median' for consolidation
    
    Returns:
        dict: {video_name: [predicted_run_dict, ...]}
        where predicted_run_dict has keys:
            position: [x, y]  normalized [0,1]
            direction: [dx, dy]  (unit vector)
            temporal_offsets: [start_frame, end_frame]  (original frame numbers)
            confidence: float
            n_detections: int
    """
    
    # Collect all predictions per video, transformed to frame space
    video_preds = defaultdict(list)
    # Track fps per video for temporal normalization
    _video_fps = {}
    
    for w_idx, (window_preds, (x_min, y_min), vname, (orig_h, orig_w)) in enumerate(
        zip(per_window_preds, crop_origins, video_names, video_resolutions)
    ):
        # Determine fps for this video
        if vname not in _video_fps:
            if video_fps and vname in video_fps:
                _video_fps[vname] = video_fps[vname]
            else:
                _video_fps[vname] = 15.0  # fallback
        fps = _video_fps[vname]
        
        for pred in window_preds:
            if pred['confidence'] < confidence_threshold:
                continue
            
            # Crop-space → frame-space
            crop_x, crop_y = pred['position']
            frame_x = crop_x + x_min
            frame_y = crop_y + y_min
            
            # Frame-space → normalized [0,1] (resolution-invariant)
            norm_x = frame_x / orig_w
            norm_y = frame_y / orig_h
            
            # Temporal: convert frame numbers to seconds (fps-invariant)
            t_start, t_end = pred['temporal_offsets']
            t_mid_sec = (t_start + t_end) / (2.0 * fps)
            
            video_preds[vname].append({
                'position_norm': [norm_x, norm_y],
                'direction': list(pred['direction']),
                'temporal_offsets': list(pred['temporal_offsets']),
                't_mid_sec': t_mid_sec,
                'confidence': pred['confidence'],
            })
    
    # Cluster per video
    result = {}
    for vname, preds in video_preds.items():
        if not preds:
            result[vname] = []
            continue
        
        # Build feature matrix: [norm_x, norm_y, time_seconds]
        features = []
        for p in preds:
            nx, ny = p['position_norm']
            features.append([nx, ny, p['t_mid_sec']])
        features = np.array(features)
        
        # Spatial eps in normalized space
        ref_size = 1000.0
        norm_spatial_eps = spatial_threshold / ref_size
        
        # Scale temporal (seconds) so that temporal_threshold_sec ≈ norm_spatial_eps
        temporal_scale = norm_spatial_eps / temporal_threshold_sec if temporal_threshold_sec > 0 else 1.0
        
        scaled_features = features.copy()
        scaled_features[:, 2] *= temporal_scale  # scale temporal dimension
        
        clustering = DBSCAN(
            eps=norm_spatial_eps,
            min_samples=min_samples,
        ).fit(scaled_features)
        
        labels = clustering.labels_
        
        # Consolidate clusters
        consolidated = []
        unique_labels = set(labels)
        
        for label in sorted(unique_labels):
            if label == -1:
                continue  # skip noise
            
            cluster_mask = labels == label
            cluster_preds = [p for p, m in zip(preds, cluster_mask) if m]
            
            if not cluster_preds:
                continue
            
            agg = np.median if mode == 'median' else np.mean
            
            positions = np.array([p['position_norm'] for p in cluster_preds])
            directions = np.array([p['direction'] for p in cluster_preds])
            temporals = np.array([p['temporal_offsets'] for p in cluster_preds])
            confidences = np.array([p['confidence'] for p in cluster_preds])
            
            # Position: aggregate
            pos = [float(agg(positions[:, 0])), float(agg(positions[:, 1]))]
            
            # Direction: circular mean (average then normalize)
            dir_mean = [float(np.mean(directions[:, 0])), float(np.mean(directions[:, 1]))]
            d_norm = np.sqrt(dir_mean[0]**2 + dir_mean[1]**2) + 1e-8
            dir_mean = [dir_mean[0] / d_norm, dir_mean[1] / d_norm]
            
            # Temporal: use median start/end (more robust than min/max)
            if mode == 'median':
                t_start = float(np.median(temporals[:, 0]))
                t_end = float(np.median(temporals[:, 1]))
            else:
                t_start = float(np.mean(temporals[:, 0]))
                t_end = float(np.mean(temporals[:, 1]))
            
            # Confidence: max
            conf = float(np.max(confidences))
            
            consolidated.append({
                'position': pos,  # normalized [0,1]
                'direction': dir_mean,
                'temporal_offsets': [t_start, t_end],
                'confidence': conf,
                'n_detections': len(cluster_preds),
            })
        
        # Also add noise points as individual predictions (confidence-sorted)
        noise_preds = [p for p, l in zip(preds, labels) if l == -1]
        for p in noise_preds:
            consolidated.append({
                'position': p['position_norm'],
                'direction': p['direction'],
                'temporal_offsets': p['temporal_offsets'],
                'confidence': p['confidence'],
                'n_detections': 1,
            })
        
        # Sort by confidence descending
        consolidated.sort(key=lambda x: x['confidence'], reverse=True)
        result[vname] = consolidated
    
    return result


# ─── Dance-level matching and metrics ─────────────────────────────────────────

def compute_dance_level_metrics(
    predicted_runs,
    gt_dances,
    gt_video_resolutions,
    pos_thresholds=None,
    iou_threshold_range=None,
    angular_thresholds=None,
):
    """
    Match predicted waggle runs against GT dances per video, using STD
    thresholds.
    
    Predictions use normalized [0,1] positions.
    GTs use origin-space positions → also normalize to [0,1] using resolution.
    
    Args:
        predicted_runs: dict {video_name: [pred_dict, ...]}
        gt_dances: dict {video_name: [gt_dict, ...]} from deduplicate_gt_dances()
        gt_video_resolutions: dict {video_name: (H, W)} for normalizing GT positions
        pos_thresholds: list of position thresholds (in normalized [0,1] space)
        iou_threshold_range: (min, max) for temporal IoU
        angular_thresholds: list of angular thresholds in degrees
    
    Returns:
        dict with keys: map, precision, recall, f1, coverage, spatial, temporal, directional
    """
    if pos_thresholds is None:
        pos_thresholds = [0.02, 0.04, 0.06, 0.08, 0.10]  # ~20-100px at 1000px width
    if iou_threshold_range is None:
        iou_threshold_range = (0.1, 0.5)  # more lenient for dance-level
    if angular_thresholds is None:
        angular_thresholds = [15, 20, 30]
    
    iou_thresholds = [round(t, 2) for t in
                      np.arange(iou_threshold_range[0], iou_threshold_range[1] + 0.05, 0.05)]
    
    # Normalize GT positions from frame-pixel space to [0,1]
    gt_norm = {}
    for vname, dances in gt_dances.items():
        res = gt_video_resolutions.get(vname)
        if res is None:
            continue
        orig_h, orig_w = res
        gt_norm[vname] = []
        for d in dances:
            gt_norm[vname].append({
                **d,
                'position': [d['position'][0] / orig_w, d['position'][1] / orig_h],
            })
    
    # Collect all predictions and GTs as flat lists for AP computation
    all_preds_flat = []  # (confidence, video_name, pred_dict)
    total_gt = 0
    
    all_videos = set(list(predicted_runs.keys()) + list(gt_norm.keys()))
    
    for vname in all_videos:
        vpreds = predicted_runs.get(vname, [])
        vgts = gt_norm.get(vname, [])
        total_gt += len(vgts)
        
        for pred in vpreds:
            all_preds_flat.append((pred['confidence'], vname, pred))
    
    all_preds_flat.sort(key=lambda x: x[0], reverse=True)
    
    if total_gt == 0 or not all_preds_flat:
        return _empty_dance_metrics()
    
    # Sweep through threshold combos
    ap_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    all_matched_pairs = []
    
    for pos_threshold in pos_thresholds:
        for iou_threshold in iou_thresholds:
            for angular_threshold in angular_thresholds:
                # Greedy confidence sweep
                matched_gts = {vname: set() for vname in all_videos}
                tp_flags = []
                matched_pairs = []
                
                for conf, vname, pred in all_preds_flat:
                    vgts = gt_norm.get(vname, [])
                    
                    best_cost = float('inf')
                    best_gt_idx = None
                    best_gt = None
                    
                    for gt_idx, gt in enumerate(vgts):
                        if gt_idx in matched_gts[vname]:
                            continue
                        
                        # Position distance: both in normalized [0,1] space
                        gt_x_norm = gt['position'][0]
                        gt_y_norm = gt['position'][1]
                        pos_dist = np.sqrt(
                            (pred['position'][0] - gt_x_norm)**2 +
                            (pred['position'][1] - gt_y_norm)**2
                        )
                        
                        # Temporal IoU
                        gt_s, gt_e = gt['temporal_offsets']
                        pr_s, pr_e = pred['temporal_offsets']
                        inter = max(0, min(gt_e, pr_e) - max(gt_s, pr_s))
                        union = max(gt_e, pr_e) - min(gt_s, pr_s)
                        tiou = inter / union if union > 0 else 0
                        
                        # Angular error
                        gt_d = np.array(gt['direction'])
                        pr_d = np.array(pred['direction'])
                        gt_d = gt_d / (np.linalg.norm(gt_d) + 1e-8)
                        pr_d = pr_d / (np.linalg.norm(pr_d) + 1e-8)
                        ang_err = np.degrees(np.arccos(
                            np.clip(np.dot(gt_d, pr_d), -1.0, 1.0)
                        ))
                        
                        if (pos_dist <= pos_threshold and
                                tiou >= iou_threshold and
                                ang_err <= angular_threshold):
                            cost = (pos_dist / pos_threshold +
                                    (1 - tiou) +
                                    ang_err / angular_threshold)
                            if cost < best_cost:
                                best_cost = cost
                                best_gt_idx = gt_idx
                                best_gt = gt
                    
                    if best_gt_idx is not None:
                        matched_gts[vname].add(best_gt_idx)
                        tp_flags.append(1)
                        matched_pairs.append((best_gt, pred))
                    else:
                        tp_flags.append(0)
                
                if not tp_flags:
                    ap_scores.append(0.0)
                    precision_scores.append(0.0)
                    recall_scores.append(0.0)
                    f1_scores.append(0.0)
                    all_matched_pairs.append([])
                    continue
                
                tp_cumsum = np.cumsum(tp_flags)
                n_preds = np.arange(1, len(tp_flags) + 1)
                precisions = tp_cumsum / n_preds
                recalls = tp_cumsum / total_gt
                
                precisions = np.maximum.accumulate(precisions[::-1])[::-1]
                
                f1s = 2 * precisions * recalls / (precisions + recalls + 1e-8)
                best_idx = np.argmax(f1s)
                
                precision_scores.append(float(precisions[best_idx]))
                recall_scores.append(float(recalls[best_idx]))
                f1_scores.append(float(f1s[best_idx]))
                
                precisions_curve = np.concatenate([[1.0], precisions])
                recalls_curve = np.concatenate([[0.0], recalls])
                ap = float(np.trapezoid(precisions_curve, recalls_curve))
                ap_scores.append(ap)
                
                all_matched_pairs.append(matched_pairs)
    
    # Use mid-threshold matched pairs for detailed metrics
    mid_idx = len(all_matched_pairs) // 2
    mid_pairs = all_matched_pairs[mid_idx] if all_matched_pairs else []
    
    # Spatial metrics on matched pairs
    if mid_pairs:
        pos_errors = []
        temporal_ious = []
        angular_errors = []
        
        for gt, pred in mid_pairs:
            # Position error in normalized space
            pos_err = np.sqrt(
                (pred['position'][0] - gt['position'][0])**2 +
                (pred['position'][1] - gt['position'][1])**2
            )
            pos_errors.append(pos_err)
            
            # Temporal IoU
            gt_s, gt_e = gt['temporal_offsets']
            pr_s, pr_e = pred['temporal_offsets']
            inter = max(0, min(gt_e, pr_e) - max(gt_s, pr_s))
            union = max(gt_e, pr_e) - min(gt_s, pr_s)
            temporal_ious.append(inter / union if union > 0 else 0)
            
            # Angular error  
            gt_d = np.array(gt['direction'])
            pr_d = np.array(pred['direction'])
            gt_d = gt_d / (np.linalg.norm(gt_d) + 1e-8)
            pr_d = pr_d / (np.linalg.norm(pr_d) + 1e-8)
            angular_errors.append(
                np.degrees(np.arccos(np.clip(np.dot(gt_d, pr_d), -1, 1)))
            )
        
        spatial = {
            'mean_error': float(np.mean(pos_errors)),
            'median_error': float(np.median(pos_errors)),
        }
        temporal = {
            'mean_iou': float(np.mean(temporal_ious)),
            'median_iou': float(np.median(temporal_ious)),
        }
        directional = {
            'mean_error': float(np.mean(angular_errors)),
            'median_error': float(np.median(angular_errors)),
        }
    else:
        spatial = {'mean_error': 0, 'median_error': 0}
        temporal = {'mean_iou': 0, 'median_iou': 0}
        directional = {'mean_error': 0, 'median_error': 0}
    
    # Coverage
    total_preds = sum(len(v) for v in predicted_runs.values())
    n_detected = len(mid_pairs)
    
    return {
        'level': 'waggle_run',
        'comprehensive': {
            'map': float(np.mean(ap_scores)),
            'mean_precision': float(np.mean(precision_scores)),
            'mean_recall': float(np.mean(recall_scores)),
            'mean_f1': float(np.mean(f1_scores)),
        },
        'spatial': spatial,
        'temporal': temporal,
        'directional': directional,
        'coverage': {
            'total_gt_dances': total_gt,
            'total_predicted_runs': total_preds,
            'detected_dances': n_detected,
            'missed_dances': total_gt - n_detected,
            'recall': n_detected / total_gt if total_gt > 0 else 0,
            'precision': n_detected / total_preds if total_preds > 0 else 0,
        },
        'counts': {
            'n_videos': len(all_videos),
            'n_gt_dances': total_gt,
            'n_predicted_runs': total_preds,
            'n_matched': n_detected,
        },
    }


def _empty_dance_metrics():
    return {
        'level': 'waggle_run',
        'comprehensive': {'map': 0, 'mean_precision': 0, 'mean_recall': 0, 'mean_f1': 0},
        'spatial': {'mean_error': 0, 'median_error': 0},
        'temporal': {'mean_iou': 0, 'median_iou': 0},
        'directional': {'mean_error': 0, 'median_error': 0},
        'coverage': {
            'total_gt_dances': 0, 'total_predicted_runs': 0,
            'detected_dances': 0, 'missed_dances': 0,
            'recall': 0, 'precision': 0,
        },
        'counts': {'n_videos': 0, 'n_gt_dances': 0, 'n_predicted_runs': 0, 'n_matched': 0},
    }
