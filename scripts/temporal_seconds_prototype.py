"""
prototyping: convert temporal_offsets from frame numbers to seconds so
dance-level eval is fps-agnostic like position is resolution-agnostic,
instead of just patching the fps-fallback bug in average_overlapping_predictions.

building this piece by piece, testing each bit before wiring it into the
next. not touching dance_eval.py yet, just prototyping here first.
"""
import pickle
import yaml
import pandas as pd
import numpy as np

from collections import defaultdict

from sklearn.cluster import DBSCAN

from src.utils.video_utils import train_val_split_videos
from src.utils.dance_eval import (
    deduplicate_gt_dances, average_overlapping_predictions,
    lab_source, BEE_LENGTH_FRACTION,
)

CACHE_PATH = "viewer/.eval_cache.pkl"
SEED = 42


# step 1: frames -> seconds for per-window preds
def preds_frames_to_seconds(per_window_preds, video_names, video_fps, default_fps=15.0):
    # raises loud instead of silently defaulting like the current code does
    out = []
    missing_fps = set()
    for preds, vname in zip(per_window_preds, video_names):
        fps = video_fps.get(vname)
        if fps is None:
            missing_fps.add(vname)
            fps = default_fps
        converted = []
        for p in preds:
            p2 = dict(p)
            t0, t1 = p['temporal_offsets']
            p2['temporal_offsets'] = [t0 / fps, t1 / fps]
            converted.append(p2)
        out.append(converted)
    return out, missing_fps


def test_step_1():
    # tiny hand-checkable example, two different fps
    per_window_preds = [
        [{'temporal_offsets': [30, 45], 'confidence': 0.9}],  # 30fps: 1.0-1.5s
        [{'temporal_offsets': [15, 22], 'confidence': 0.8}],  # 15fps: 1.0-1.467s
    ]
    video_names = ['A', 'B']
    video_fps = {'A': 30.0, 'B': 15.0}

    out, missing = preds_frames_to_seconds(per_window_preds, video_names, video_fps)
    assert missing == set(), missing
    assert out[0][0]['temporal_offsets'] == [1.0, 1.5]
    assert abs(out[1][0]['temporal_offsets'][1] - 22 / 15) < 1e-9
    assert out[0][0]['confidence'] == 0.9  # untouched
    print("step 1 ok:", out[0][0]['temporal_offsets'], out[1][0]['temporal_offsets'])


# step 2: same conversion but for GT dances (dict of video -> list of dance dicts)
def gt_dances_frames_to_seconds(gt_dances, video_fps, default_fps=15.0):
    out = {}
    missing_fps = set()
    for vname, dances in gt_dances.items():
        fps = video_fps.get(vname)
        if fps is None:
            missing_fps.add(vname)
            fps = default_fps
        converted = []
        for d in dances:
            d2 = dict(d)
            t0, t1 = d['temporal_offsets']
            d2['temporal_offsets'] = [t0 / fps, t1 / fps]
            converted.append(d2)
        out[vname] = converted
    return out, missing_fps


def test_step_2():
    gt_dances = {
        'A': [{'temporal_offsets': [100, 130], 'waggle_run_id': 1}],  # 30fps -> 3.33-4.33s
    }
    video_fps = {'A': 30.0}
    out, missing = gt_dances_frames_to_seconds(gt_dances, video_fps)
    assert missing == set(), missing
    s, e = out['A'][0]['temporal_offsets']
    assert abs(s - 100 / 30) < 1e-9 and abs(e - 130 / 30) < 1e-9
    assert out['A'][0]['waggle_run_id'] == 1  # untouched
    print("step 2 ok:", out['A'][0]['temporal_offsets'])


# step 3: run against real cached data, see what GT durations look like
# in seconds, and how many videos are actually missing fps info
def step_3_real_data():
    with open(CACHE_PATH, "rb") as f:
        cache = pickle.load(f)
    video_fps = cache.get("video_fps") or {}

    with open("configs/config.yaml") as f:
        live_config = yaml.safe_load(f)
    data = pd.read_csv(live_config["data"]["annotations"])
    train_videos, _ = train_val_split_videos(
        data["video_name"].unique(), train_ratio=live_config["data"]["train_ratio"], seed=SEED
    )
    test_df = data[~data["video_name"].isin(train_videos)].reset_index(drop=True)
    gt_dances = deduplicate_gt_dances(test_df)

    print(f"\nvideo_fps dict has {len(video_fps)} entries")
    print(f"gt_dances covers {len(gt_dances)} videos")

    gt_sec, missing = gt_dances_frames_to_seconds(gt_dances, video_fps)
    print(f"videos missing fps (fell back to default): {len(missing)}")
    if missing:
        print("  e.g.:", list(missing)[:5])

    durs = np.array([d["temporal_offsets"][1] - d["temporal_offsets"][0]
                      for dances in gt_sec.values() for d in dances])
    print(f"\nGT dance duration in seconds: n={len(durs)}  mean={durs.mean():.3f}  median={np.median(durs):.3f}")
    for p in [5, 25, 50, 75, 95]:
        print(f"  p{p}: {np.percentile(durs, p):.3f}s")


# step 4: seconds-based version of average_overlapping_predictions.
# same logic as the original, just no fps/frames conversion needed since
# temporal_offsets are already seconds -- so video_fps param goes away
# entirely for this function.
def average_overlapping_predictions_sec(per_window_preds, crop_origins, video_names,
                                         window_ranges=None, max_temporal_gap_sec=1.0):
    if not per_window_preds:
        return []

    if window_ranges is not None:
        w_ranges = list(window_ranges)
    else:
        w_ranges = []
        for preds in per_window_preds:
            if preds:
                t_starts = [p['temporal_offsets'][0] for p in preds]
                t_ends = [p['temporal_offsets'][1] for p in preds]
                w_ranges.append((min(t_starts), max(t_ends)))
            else:
                w_ranges.append(None)

    groups = defaultdict(list)
    for w_idx, (preds, origin, vname) in enumerate(zip(per_window_preds, crop_origins, video_names)):
        key = (vname, tuple(origin))
        groups[key].append((w_idx, preds))

    result = [[] for _ in range(len(per_window_preds))]

    for key, window_list in groups.items():
        n_windows = len(window_list)
        group_ranges = [w_ranges[w_idx] for w_idx, _ in window_list]

        cell_preds = defaultdict(list)
        for w_idx, preds in window_list:
            for pred in preds:
                gc = pred.get('grid_cell', [0, 0, 0])
                cell_preds[tuple(gc)].append(pred)

        averaged = []
        for cell_key, preds_for_cell in cell_preds.items():
            preds_for_cell.sort(key=lambda p: (p['temporal_offsets'][0] + p['temporal_offsets'][1]) / 2)
            sub_groups = [[preds_for_cell[0]]]
            for pred in preds_for_cell[1:]:
                prev_end = sub_groups[-1][-1]['temporal_offsets'][1]
                curr_start = pred['temporal_offsets'][0]
                if curr_start - prev_end > max_temporal_gap_sec:  # no more * fps needed
                    sub_groups.append([pred])
                else:
                    sub_groups[-1].append(pred)

            for sub_preds in sub_groups:
                n_detections = len(sub_preds)
                det_t_start = min(p['temporal_offsets'][0] for p in sub_preds)
                det_t_end = max(p['temporal_offsets'][1] for p in sub_preds)

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
                norm = (avg_dir[0] ** 2 + avg_dir[1] ** 2) ** 0.5
                if norm > 1e-8:
                    avg_dir = [avg_dir[0] / norm, avg_dir[1] / norm]

                averaged.append({
                    'confidence': avg_conf, 'position': avg_pos, 'direction': avg_dir,
                    'temporal_offsets': [det_t_start, det_t_end],
                    'grid_cell': list(cell_key), 'n_detections': n_detections,
                    'n_overlapping': n_overlapping, 'n_windows': n_windows,
                })

        result[window_list[0][0]] = averaged

    return result


def test_step_4():
    # two detections same cell, 1.5s apart -> should split at gap=1.0
    per_window_preds = [
        [{'temporal_offsets': [0.0, 0.2], 'confidence': 0.9, 'position': [1, 1],
          'direction': [1, 0], 'grid_cell': [0, 0, 0]}],
        [{'temporal_offsets': [1.7, 1.9], 'confidence': 0.8, 'position': [1, 1],
          'direction': [1, 0], 'grid_cell': [0, 0, 0]}],
    ]
    crop_origins = [(0, 0), (0, 0)]
    video_names = ['A', 'A']
    out = average_overlapping_predictions_sec(per_window_preds, crop_origins, video_names,
                                               max_temporal_gap_sec=1.0)
    n_results = sum(len(w) for w in out)
    assert n_results == 2, f"expected 2 separate sub-groups, got {n_results}"

    # same but gap=2.0 -> should merge into one
    out2 = average_overlapping_predictions_sec(per_window_preds, crop_origins, video_names,
                                                max_temporal_gap_sec=2.0)
    n_results2 = sum(len(w) for w in out2)
    assert n_results2 == 1, f"expected 1 merged group, got {n_results2}"
    print("step 4 ok: split at gap=1.0 ->", n_results, " merged at gap=2.0 ->", n_results2)


# step 5: real data, three-way comparison
#   A) original fn, video_fps=None      <- current production bug
#   B) original fn, video_fps=<correct> <- same logic, fps actually threaded through
#   C) new sec fn, on seconds-converted input <- should match B
def step_5_real_data():
    with open(CACHE_PATH, "rb") as f:
        cache = pickle.load(f)
    preds = cache["preds"]
    crop_origins = cache["crop_origins"]
    video_names = cache["video_names"]
    video_fps = cache.get("video_fps") or {}

    out_a = average_overlapping_predictions(preds, crop_origins, video_names)  # no video_fps passed
    out_b = average_overlapping_predictions(preds, crop_origins, video_names, video_fps=video_fps)

    preds_sec, missing = preds_frames_to_seconds(preds, video_names, video_fps)
    assert not missing
    out_c = average_overlapping_predictions_sec(preds_sec, crop_origins, video_names)

    n_a = sum(len(w) for w in out_a)
    n_b = sum(len(w) for w in out_b)
    n_c = sum(len(w) for w in out_c)
    print(f"\nA) current prod (no fps):     {n_a} averaged detections")
    print(f"B) original fn, correct fps: {n_b} averaged detections")
    print(f"C) new sec fn (should = B):  {n_c} averaged detections")

    # B and C should match in COUNT -- confirms the sec conversion is faithful.
    # can't compare temporal_offsets directly since B is in frames, C is in seconds.
    print(f"\nB vs C count match: {n_b == n_c}")
    print(f"A vs B differ by: {n_a - n_b} detections  (this is the bug's real impact)")


# step 6/7: same clustering as cross_window_cluster_predictions, but each
# consolidated run also carries single_best_offsets (just the top-confidence
# member, no aggregation at all) and mean_offsets (mean start, mean end across
# members -- same idea as how position already gets aggregated in the real
# function, just never applied to the temporal boundary). lets us compare all
# three on the exact same clusters.
def cluster_with_single_best(per_window_preds, crop_origins, video_names, video_resolutions,
                              video_fps, spatial_threshold, temporal_threshold_sec,
                              confidence_threshold, min_samples, mode, direction_threshold_deg,
                              bee_size_multiplier):
    video_preds = defaultdict(list)
    _video_fps = {}
    for window_preds, (x_min, y_min), vname, (orig_h, orig_w) in zip(
        per_window_preds, crop_origins, video_names, video_resolutions
    ):
        if vname not in _video_fps:
            _video_fps[vname] = (video_fps or {}).get(vname, 15.0)
        fps = _video_fps[vname]
        for pred in window_preds:
            if pred['confidence'] < confidence_threshold:
                continue
            crop_x, crop_y = pred['position']
            frame_x, frame_y = crop_x + x_min, crop_y + y_min
            norm_x, norm_y = frame_x / orig_w, frame_y / orig_h
            t_start, t_end = pred['temporal_offsets']
            t_mid_sec = (t_start + t_end) / (2.0 * fps)
            video_preds[vname].append({
                'position_norm': [norm_x, norm_y], 'direction': list(pred['direction']),
                'temporal_offsets': list(pred['temporal_offsets']), 't_mid_sec': t_mid_sec,
                'confidence': pred['confidence'],
            })

    result = {}
    for vname, preds in video_preds.items():
        if not preds:
            result[vname] = []
            continue
        use_direction = direction_threshold_deg > 0
        features = []
        for p in preds:
            nx, ny = p['position_norm']
            row = [nx, ny, p['t_mid_sec']]
            if use_direction:
                row.extend(p['direction'])
            features.append(row)
        features = np.array(features)

        ref_size = 1000.0
        if bee_size_multiplier is not None:
            lab = lab_source(vname)
            norm_spatial_eps = (bee_size_multiplier * BEE_LENGTH_FRACTION[lab]
                                 if lab in BEE_LENGTH_FRACTION else spatial_threshold / ref_size)
        else:
            norm_spatial_eps = spatial_threshold / ref_size

        temporal_scale = norm_spatial_eps / temporal_threshold_sec if temporal_threshold_sec > 0 else 1.0
        scaled_features = features.copy()
        scaled_features[:, 2] *= temporal_scale
        if use_direction:
            max_chord = 2.0 * np.sin(np.radians(direction_threshold_deg) / 2.0)
            dir_scale = norm_spatial_eps / max_chord if max_chord > 1e-8 else 0.0
            scaled_features[:, 3] *= dir_scale
            scaled_features[:, 4] *= dir_scale

        labels = DBSCAN(eps=norm_spatial_eps, min_samples=min_samples).fit(scaled_features).labels_

        consolidated = []
        for label in sorted(set(labels)):
            if label == -1:
                continue
            cluster_preds = [p for p, m in zip(preds, labels == label) if m]
            if not cluster_preds:
                continue
            agg = np.median if mode == 'median' else np.mean
            positions = np.array([p['position_norm'] for p in cluster_preds])
            directions = np.array([p['direction'] for p in cluster_preds])
            temporals = np.array([p['temporal_offsets'] for p in cluster_preds])
            confidences = np.array([p['confidence'] for p in cluster_preds])

            pos = [float(agg(positions[:, 0])), float(agg(positions[:, 1]))]
            dir_mean = [float(np.mean(directions[:, 0])), float(np.mean(directions[:, 1]))]
            d_norm = np.sqrt(dir_mean[0] ** 2 + dir_mean[1] ** 2) + 1e-8
            dir_mean = [dir_mean[0] / d_norm, dir_mean[1] / d_norm]

            t_start, t_end = float(np.min(temporals[:, 0])), float(np.max(temporals[:, 1]))
            best_idx = int(np.argmax(confidences))
            single_best = list(cluster_preds[best_idx]['temporal_offsets'])
            conf = float(confidences[best_idx])
            # mean start / mean end across members -- same agg() already used
            # for position, just never applied to the temporal boundary
            mean_offsets = [float(np.mean(temporals[:, 0])), float(np.mean(temporals[:, 1]))]

            consolidated.append({
                'position': pos, 'direction': dir_mean,
                'temporal_offsets': [t_start, t_end],
                'single_best_offsets': single_best,
                'mean_offsets': mean_offsets,
                'confidence': conf, 'n_detections': len(cluster_preds),
            })

        noise_preds = [p for p, l in zip(preds, labels) if l == -1]
        for p in noise_preds:
            consolidated.append({
                'position': p['position_norm'], 'direction': p['direction'],
                'temporal_offsets': p['temporal_offsets'],
                'single_best_offsets': p['temporal_offsets'],  # noise = single detection already
                'mean_offsets': p['temporal_offsets'],
                'confidence': p['confidence'], 'n_detections': 1,
            })

        consolidated.sort(key=lambda x: x['confidence'], reverse=True)
        result[vname] = consolidated

    return result


def step_6_isolate_cause():
    with open(CACHE_PATH, "rb") as f:
        cache = pickle.load(f)
    preds = cache["preds"]
    crop_origins = cache["crop_origins"]
    video_names = cache["video_names"]
    original_res = cache["original_res"]
    video_fps = cache.get("video_fps") or {}
    pp = cache["config"]["post_process"]

    with open("configs/config.yaml") as f:
        live_config = yaml.safe_load(f)
    data = pd.read_csv(live_config["data"]["annotations"])
    train_videos, _ = train_val_split_videos(
        data["video_name"].unique(), train_ratio=live_config["data"]["train_ratio"], seed=SEED
    )
    test_df = data[~data["video_name"].isin(train_videos)].reset_index(drop=True)
    gt_dances = deduplicate_gt_dances(test_df)

    averaged = average_overlapping_predictions(preds, crop_origins, video_names)
    predicted_runs = cluster_with_single_best(
        averaged, crop_origins, video_names, original_res, video_fps,
        spatial_threshold=pp["spatial_threshold"],
        temporal_threshold_sec=pp.get("temporal_threshold_ms", 300) / 1000.0,
        confidence_threshold=pp["confidence_threshold"],
        min_samples=pp.get("min_samples", 1),
        mode=pp.get("mode", "mean"),
        direction_threshold_deg=pp.get("direction_threshold_deg", 30.0),
        bee_size_multiplier=pp.get("bee_size_multiplier"),
    )

    video_res = {vn: res for vn, res in zip(video_names, original_res)}
    gt_norm = {}
    for vname, dances in gt_dances.items():
        res = video_res.get(vname)
        if res is None:
            continue
        oh, ow = res
        gt_norm[vname] = [{**d, "position": [d["position"][0] / ow, d["position"][1] / oh]} for d in dances]

    all_preds_flat = []
    for vname in sorted(set(list(predicted_runs.keys()) + list(gt_norm.keys()))):
        for idx, pred in enumerate(predicted_runs.get(vname, [])):
            all_preds_flat.append((pred["confidence"], vname, idx, pred))
    all_preds_flat.sort(key=lambda x: (-x[0], x[1], x[2]))

    pos_threshold, iou_threshold, angular_threshold = 0.06, 0.1, 20.0
    matched_gts = {vname: set() for vname in gt_norm}
    matched = []  # (gt, pred) using union temporal_offsets for the match decision
    for conf, vname, _idx, pred in all_preds_flat:
        vgts = gt_norm.get(vname, [])
        best_cost, best_idx_, best_gt = float("inf"), None, None
        for gidx, gt in enumerate(vgts):
            if gidx in matched_gts[vname]:
                continue
            pos_dist = np.sqrt((pred["position"][0] - gt["position"][0]) ** 2 +
                                (pred["position"][1] - gt["position"][1]) ** 2)
            gs, ge = gt["temporal_offsets"]
            ps, pe = pred["temporal_offsets"]
            inter = max(0, min(ge, pe) - max(gs, ps))
            union = max(ge, pe) - min(gs, ps)
            tiou = inter / union if union > 0 else 0
            gd = np.array(gt["direction"]); gd = gd / (np.linalg.norm(gd) + 1e-8)
            pd_ = np.array(pred["direction"]); pd_ = pd_ / (np.linalg.norm(pd_) + 1e-8)
            ang = np.degrees(np.arccos(np.clip(np.dot(gd, pd_), -1, 1)))
            if pos_dist <= pos_threshold and tiou >= iou_threshold and ang <= angular_threshold:
                cost = pos_dist / pos_threshold + (1 - tiou) + ang / angular_threshold
                if cost < best_cost:
                    best_cost, best_idx_, best_gt = cost, gidx, gt
        if best_idx_ is not None:
            matched_gts[vname].add(best_idx_)
            matched.append((best_gt, pred))

    def bias_and_iou(key):
        starts = np.array([m[1][key][0] - m[0]["temporal_offsets"][0] for m in matched])
        ends = np.array([m[1][key][1] - m[0]["temporal_offsets"][1] for m in matched])
        ious = []
        for gt, pred in matched:
            gs, ge = gt["temporal_offsets"]
            ps, pe = pred[key]
            inter = max(0, min(ge, pe) - max(gs, ps))
            union = max(ge, pe) - min(gs, ps)
            ious.append(inter / union if union > 0 else 0)
        return starts.mean(), ends.mean(), np.mean(ious)

    print(f"\n=== isolating the cause, n={len(matched)} matched pairs ===")
    for key, label in [("temporal_offsets", "union (current)"),
                        ("single_best_offsets", "single best det"),
                        ("mean_offsets", "mean start/end")]:
        s, e, iou = bias_and_iou(key)
        print(f"{label:18s}: start bias={s:+.2f}f  end bias={e:+.2f}f  mean IoU={iou:.3f}")


# step 8: does the min/max in average_overlapping_predictions ALSO hurt, on
# top of the DBSCAN-stage one we already fixed? same func as
# average_overlapping_predictions_sec but det_t_start/det_t_end use mean
# instead of min/max.
def average_overlapping_predictions_mean(per_window_preds, crop_origins, video_names,
                                          window_ranges=None, max_temporal_gap_sec=1.0):
    if not per_window_preds:
        return []

    if window_ranges is not None:
        w_ranges = list(window_ranges)
    else:
        w_ranges = []
        for preds in per_window_preds:
            if preds:
                t_starts = [p['temporal_offsets'][0] for p in preds]
                t_ends = [p['temporal_offsets'][1] for p in preds]
                w_ranges.append((min(t_starts), max(t_ends)))
            else:
                w_ranges.append(None)

    groups = defaultdict(list)
    for w_idx, (preds, origin, vname) in enumerate(zip(per_window_preds, crop_origins, video_names)):
        key = (vname, tuple(origin))
        groups[key].append((w_idx, preds))

    result = [[] for _ in range(len(per_window_preds))]

    for key, window_list in groups.items():
        n_windows = len(window_list)
        group_ranges = [w_ranges[w_idx] for w_idx, _ in window_list]

        cell_preds = defaultdict(list)
        for w_idx, preds in window_list:
            for pred in preds:
                gc = pred.get('grid_cell', [0, 0, 0])
                cell_preds[tuple(gc)].append(pred)

        averaged = []
        for cell_key, preds_for_cell in cell_preds.items():
            preds_for_cell.sort(key=lambda p: (p['temporal_offsets'][0] + p['temporal_offsets'][1]) / 2)
            sub_groups = [[preds_for_cell[0]]]
            for pred in preds_for_cell[1:]:
                prev_end = sub_groups[-1][-1]['temporal_offsets'][1]
                curr_start = pred['temporal_offsets'][0]
                if curr_start - prev_end > max_temporal_gap_sec:
                    sub_groups.append([pred])
                else:
                    sub_groups[-1].append(pred)

            for sub_preds in sub_groups:
                n_detections = len(sub_preds)
                # mean instead of min/max here -- that's the only change vs _sec
                det_t_start = sum(p['temporal_offsets'][0] for p in sub_preds) / n_detections
                det_t_end = sum(p['temporal_offsets'][1] for p in sub_preds) / n_detections

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
                norm = (avg_dir[0] ** 2 + avg_dir[1] ** 2) ** 0.5
                if norm > 1e-8:
                    avg_dir = [avg_dir[0] / norm, avg_dir[1] / norm]

                averaged.append({
                    'confidence': avg_conf, 'position': avg_pos, 'direction': avg_dir,
                    'temporal_offsets': [det_t_start, det_t_end],
                    'grid_cell': list(cell_key), 'n_detections': n_detections,
                    'n_overlapping': n_overlapping, 'n_windows': n_windows,
                })

        result[window_list[0][0]] = averaged

    return result


def test_step_8():
    # 3 detections same cell, spread [0,1], [0.3,1.2], [0.1,0.9]
    # min/max union would give [0, 1.2] -- wide. mean should land in the middle.
    per_window_preds = [[
        {'temporal_offsets': [0.0, 1.0], 'confidence': 0.9, 'position': [1, 1],
         'direction': [1, 0], 'grid_cell': [0, 0, 0]},
        {'temporal_offsets': [0.3, 1.2], 'confidence': 0.8, 'position': [1, 1],
         'direction': [1, 0], 'grid_cell': [0, 0, 0]},
        {'temporal_offsets': [0.1, 0.9], 'confidence': 0.7, 'position': [1, 1],
         'direction': [1, 0], 'grid_cell': [0, 0, 0]},
    ]]
    out = average_overlapping_predictions_mean(per_window_preds, [(0, 0)], ['A'])
    start, end = out[0][0]['temporal_offsets']
    assert abs(start - (0.0 + 0.3 + 0.1) / 3) < 1e-9
    assert abs(end - (1.0 + 1.2 + 0.9) / 3) < 1e-9
    print("step 8 ok: mean boundary =", [round(start, 3), round(end, 3)], "(union would've been [0.0, 1.2])")


# step 9: full pipeline with BOTH fixes stacked -- mean at the averaging
# stage AND mean at the DBSCAN clustering stage -- see if it compounds on
# top of the 0.510 -> 0.604 IoU jump we already measured with just the
# clustering-stage fix.
def step_9_stacked_fix():
    with open(CACHE_PATH, "rb") as f:
        cache = pickle.load(f)
    preds = cache["preds"]
    crop_origins = cache["crop_origins"]
    video_names = cache["video_names"]
    original_res = cache["original_res"]
    video_fps = cache.get("video_fps") or {}
    pp = cache["config"]["post_process"]

    with open("configs/config.yaml") as f:
        live_config = yaml.safe_load(f)
    data = pd.read_csv(live_config["data"]["annotations"])
    train_videos, _ = train_val_split_videos(
        data["video_name"].unique(), train_ratio=live_config["data"]["train_ratio"], seed=SEED
    )
    test_df = data[~data["video_name"].isin(train_videos)].reset_index(drop=True)
    gt_dances = deduplicate_gt_dances(test_df)

    # only diff from step 6/7: mean-based averaging instead of the original
    averaged = average_overlapping_predictions_mean(preds, crop_origins, video_names)
    predicted_runs = cluster_with_single_best(
        averaged, crop_origins, video_names, original_res, video_fps,
        spatial_threshold=pp["spatial_threshold"],
        temporal_threshold_sec=pp.get("temporal_threshold_ms", 300) / 1000.0,
        confidence_threshold=pp["confidence_threshold"],
        min_samples=pp.get("min_samples", 1),
        mode=pp.get("mode", "mean"),
        direction_threshold_deg=pp.get("direction_threshold_deg", 30.0),
        bee_size_multiplier=pp.get("bee_size_multiplier"),
    )

    video_res = {vn: res for vn, res in zip(video_names, original_res)}
    gt_norm = {}
    for vname, dances in gt_dances.items():
        res = video_res.get(vname)
        if res is None:
            continue
        oh, ow = res
        gt_norm[vname] = [{**d, "position": [d["position"][0] / ow, d["position"][1] / oh]} for d in dances]

    all_preds_flat = []
    for vname in sorted(set(list(predicted_runs.keys()) + list(gt_norm.keys()))):
        for idx, pred in enumerate(predicted_runs.get(vname, [])):
            all_preds_flat.append((pred["confidence"], vname, idx, pred))
    all_preds_flat.sort(key=lambda x: (-x[0], x[1], x[2]))

    pos_threshold, iou_threshold, angular_threshold = 0.06, 0.1, 20.0
    matched_gts = {vname: set() for vname in gt_norm}
    matched = []
    for conf, vname, _idx, pred in all_preds_flat:
        vgts = gt_norm.get(vname, [])
        best_cost, best_idx_, best_gt = float("inf"), None, None
        for gidx, gt in enumerate(vgts):
            if gidx in matched_gts[vname]:
                continue
            pos_dist = np.sqrt((pred["position"][0] - gt["position"][0]) ** 2 +
                                (pred["position"][1] - gt["position"][1]) ** 2)
            gs, ge = gt["temporal_offsets"]
            ps, pe = pred["temporal_offsets"]
            inter = max(0, min(ge, pe) - max(gs, ps))
            union = max(ge, pe) - min(gs, ps)
            tiou = inter / union if union > 0 else 0
            gd = np.array(gt["direction"]); gd = gd / (np.linalg.norm(gd) + 1e-8)
            pd_ = np.array(pred["direction"]); pd_ = pd_ / (np.linalg.norm(pd_) + 1e-8)
            ang = np.degrees(np.arccos(np.clip(np.dot(gd, pd_), -1, 1)))
            if pos_dist <= pos_threshold and tiou >= iou_threshold and ang <= angular_threshold:
                cost = pos_dist / pos_threshold + (1 - tiou) + ang / angular_threshold
                if cost < best_cost:
                    best_cost, best_idx_, best_gt = cost, gidx, gt
        if best_idx_ is not None:
            matched_gts[vname].add(best_idx_)
            matched.append((best_gt, pred))

    starts = np.array([m[1]["temporal_offsets"][0] - m[0]["temporal_offsets"][0] for m in matched])
    ends = np.array([m[1]["temporal_offsets"][1] - m[0]["temporal_offsets"][1] for m in matched])
    ious = []
    for gt, pred in matched:
        gs, ge = gt["temporal_offsets"]
        ps, pe = pred["temporal_offsets"]
        inter = max(0, min(ge, pe) - max(gs, ps))
        union = max(ge, pe) - min(gs, ps)
        ious.append(inter / union if union > 0 else 0)

    print(f"\n=== both stages mean-fixed, n={len(matched)} matched pairs ===")
    print(f"start bias={starts.mean():+.2f}f  end bias={ends.mean():+.2f}f  mean IoU={np.mean(ious):.3f}")
    print("(compare: union both stages=0.510, mean at clustering only=0.604)")


if __name__ == "__main__":
    test_step_1()
    test_step_2()
    step_3_real_data()
    test_step_4()
    step_5_real_data()
    step_6_isolate_cause()
    test_step_8()
    step_9_stacked_fix()
