"""
Investigate why dance-level temporal IoU is low despite a modest mean
start/end bias. Reuses the cached predictions from viewer/.eval_cache.pkl
(no GPU inference needed) to look at the actual distributions -- not just
means -- of GT dance duration and matched-pair temporal IoU.
"""
import pickle
import yaml
import pandas as pd
import numpy as np

from src.utils.video_utils import train_val_split_videos
from src.utils.dance_eval import (
    average_overlapping_predictions,
    cross_window_cluster_predictions,
    deduplicate_gt_dances,
)

CACHE_PATH = "viewer/.eval_cache.pkl"
SEED = 42


def main():
    with open(CACHE_PATH, "rb") as f:
        cache = pickle.load(f)

    preds = cache["preds"]
    crop_origins = cache["crop_origins"]
    video_names = cache["video_names"]
    original_res = cache["original_res"]
    video_fps = cache.get("video_fps") or {}
    config = cache["config"]
    pp = config["post_process"]

    with open("configs/config.yaml") as f:
        live_config = yaml.safe_load(f)
    data = pd.read_csv(live_config["data"]["annotations"])
    train_videos, _ = train_val_split_videos(
        data["video_name"].unique(), train_ratio=live_config["data"]["train_ratio"], seed=SEED
    )
    test_df = data[~data["video_name"].isin(train_videos)].reset_index(drop=True)
    gt_dances = deduplicate_gt_dances(test_df)

    averaged = average_overlapping_predictions(
        preds, crop_origins, video_names, video_fps=video_fps,
    )
    predicted_runs = cross_window_cluster_predictions(
        averaged, crop_origins, video_names, original_res,
        video_fps=video_fps,
        spatial_threshold=pp["spatial_threshold"],
        temporal_threshold_sec=pp.get("temporal_threshold_ms", 300) / 1000.0,
        confidence_threshold=pp["confidence_threshold"],
        min_samples=pp.get("min_samples", 1),
        mode=pp.get("mode", "mean"),
        direction_threshold_deg=pp.get("direction_threshold_deg", 30.0),
        cluster_radius_mm=pp.get("cluster_radius_mm"),
    )

    video_res = {}
    for vn, res in zip(video_names, original_res):
        video_res[vn] = res

    # GT duration distribution (all dances, frames)
    all_gt_durs = []
    for vname, dances in gt_dances.items():
        for d in dances:
            s, e = d["temporal_offsets"]
            all_gt_durs.append(e - s)
    all_gt_durs = np.array(all_gt_durs)

    print("=== GT dance duration (frames) ===")
    print(f"n={len(all_gt_durs)}  mean={all_gt_durs.mean():.1f}  median={np.median(all_gt_durs):.1f}")
    for p in [5, 10, 25, 50, 75, 90, 95]:
        print(f"  p{p}: {np.percentile(all_gt_durs, p):.1f}")

    # Greedy match at ONE representative threshold combo (mid of the
    # standard sweep: pos=0.06, iou>=0.1, angular=20) to get per-pair IoU.
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
    matched_pairs = []
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
            gs, ge = best_gt["temporal_offsets"]
            ps, pe = pred["temporal_offsets"]
            inter = max(0, min(ge, pe) - max(gs, ps))
            union = max(ge, pe) - min(gs, ps)
            tiou = inter / union if union > 0 else 0
            matched_pairs.append({
                "gt_dur": ge - gs, "pred_dur": pe - ps, "iou": tiou,
                "start_err": ps - gs, "end_err": pe - ge,
            })

    ious = np.array([m["iou"] for m in matched_pairs])
    gt_durs = np.array([m["gt_dur"] for m in matched_pairs])

    print(f"\n=== Matched pairs (pos<=0.06, iou>=0.1, angular<=20) n={len(matched_pairs)} ===")
    print(f"IoU:  mean={ious.mean():.3f}  median={np.median(ious):.3f}  std={ious.std():.3f}")
    for p in [5, 10, 25, 50, 75, 90, 95]:
        print(f"  IoU p{p}: {np.percentile(ious, p):.3f}")

    print(f"\n=== Matched-pair GT duration (frames) ===")
    print(f"mean={gt_durs.mean():.1f}  median={np.median(gt_durs):.1f}")
    for p in [5, 10, 25, 50, 75, 90, 95]:
        print(f"  p{p}: {np.percentile(gt_durs, p):.1f}")

    # Correlation: does IoU depend on GT duration?
    corr = np.corrcoef(gt_durs, ious)[0, 1]
    print(f"\ncorrelation(gt_duration, iou) = {corr:.3f}")

    # Split matched pairs into short vs long GT dances, compare IoU
    median_dur = np.median(gt_durs)
    short_mask = gt_durs <= median_dur
    print(f"\nShort dances (<= median {median_dur:.0f}f): n={short_mask.sum()}  mean IoU={ious[short_mask].mean():.3f}")
    print(f"Long dances  (>  median {median_dur:.0f}f): n={(~short_mask).sum()}  mean IoU={ious[~short_mask].mean():.3f}")


if __name__ == "__main__":
    main()
