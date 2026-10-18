"""
prototyping: bee-length-relative pos_thresholds instead of one fixed
normalized fraction for every video. same idea as bee_size_multiplier --
one shared N, applied on top of each lab's own measured BEE_LENGTH_FRACTION,
so the threshold means "N bee-lengths away" consistently across labs
instead of a fixed distance that's tight for one lab and loose for another.

testing piece by piece against the cache before touching dance_eval.py /
diagnostic.py for real.
"""
import pickle
import yaml
import pandas as pd
import numpy as np

from src.utils.video_utils import train_val_split_videos
from src.utils.dance_eval import (
    deduplicate_gt_dances, average_overlapping_predictions,
    cross_window_cluster_predictions, BEE_LENGTH_FRACTION, lab_source,
)

CACHE_PATH = "viewer/.eval_cache.pkl"
SEED = 42
FALLBACK_FRACTION = 0.046  # rough weighted-avg bee length across labs, used when lab unrecognized


# step 1: the core threshold-per-pair helper
def bee_relative_pos_threshold(n, vname, fallback=FALLBACK_FRACTION):
    lab = lab_source(vname)
    if lab in BEE_LENGTH_FRACTION:
        return n * BEE_LENGTH_FRACTION[lab]
    return n * fallback


def test_step_1():
    # berlin video (0-prefix), known bee length 0.0224
    assert abs(bee_relative_pos_threshold(1.0, '033_576_440.mp4') - 0.0224) < 1e-9
    # nieh video (T-prefix), known bee length 0.0775
    assert abs(bee_relative_pos_threshold(0.5, 'T1-D1-B8-V4-C_960_540.mp4') - 0.0775 * 0.5) < 1e-9
    # sharoni video (C-prefix), known bee length 0.033
    assert abs(bee_relative_pos_threshold(2.0, 'C_some_video.mp4') - 0.033 * 2.0) < 1e-9
    # unrecognized lab -> fallback
    assert abs(bee_relative_pos_threshold(1.0, 'weird_video_name.mp4') - FALLBACK_FRACTION) < 1e-9
    print("step 1 ok:", bee_relative_pos_threshold(1.0, '033_576_440.mp4'),
          bee_relative_pos_threshold(0.5, 'T1-D1-B8-V4-C_960_540.mp4'),
          bee_relative_pos_threshold(2.0, 'C_some_video.mp4'),
          bee_relative_pos_threshold(1.0, 'weird_video_name.mp4'))


# step 2: real data. match with bee-relative thresholds vs. an equivalent
# fixed global threshold, compare per-lab pass rate -- does bee-relative
# actually balance things out across labs, or not?
def step_2_real_data():
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
    predicted_runs = cross_window_cluster_predictions(
        averaged, crop_origins, video_names, original_res, video_fps=video_fps,
        spatial_threshold=pp["spatial_threshold"],
        temporal_threshold_sec=pp.get("temporal_threshold_ms", 300) / 1000.0,
        confidence_threshold=pp["confidence_threshold"],
        min_samples=pp.get("min_samples", 1), mode=pp.get("mode", "mean"),
        direction_threshold_deg=pp.get("direction_threshold_deg", 30.0),
        bee_size_multiplier=pp.get("bee_size_multiplier"),
        temporal_boundary="mean",
    )

    video_res = {vn: res for vn, res in zip(video_names, original_res)}
    gt_norm = {}
    for vname, dances in gt_dances.items():
        res = video_res.get(vname)
        if res is None:
            continue
        oh, ow = res
        gt_norm[vname] = [
            {**d, "position": [d["position"][0] / ow, d["position"][1] / oh], "lab": lab_source(vname)}
            for d in dances
        ]

    all_preds_flat = []
    for vname in sorted(set(list(predicted_runs.keys()) + list(gt_norm.keys()))):
        for idx, pred in enumerate(predicted_runs.get(vname, [])):
            all_preds_flat.append((pred["confidence"], vname, idx, pred))
    all_preds_flat.sort(key=lambda x: (-x[0], x[1], x[2]))

    def run_match(pos_threshold_fn, iou_threshold=0.1, angular_threshold=30.0):
        matched_gts = {vname: set() for vname in gt_norm}
        matched_by_lab = {}
        total_by_lab = {}
        for vname, dances in gt_norm.items():
            for d in dances:
                total_by_lab[d["lab"]] = total_by_lab.get(d["lab"], 0) + 1

        for conf, vname, _idx, pred in all_preds_flat:
            vgts = gt_norm.get(vname, [])
            pos_threshold = pos_threshold_fn(vname)
            best_cost, best_idx_, best_lab = float("inf"), None, None
            for gidx, gt in enumerate(vgts):
                if gidx in matched_gts[vname]:
                    continue
                pos_dist = np.sqrt((pred["position"][0] - gt["position"][0]) ** 2 +
                                    (pred["position"][1] - gt["position"][1]) ** 2)
                gs, ge = gt["temporal_offsets"]; ps, pe = pred["temporal_offsets"]
                inter = max(0, min(ge, pe) - max(gs, ps)); union = max(ge, pe) - min(gs, ps)
                tiou = inter / union if union > 0 else 0
                gd = np.array(gt["direction"]); gd = gd / (np.linalg.norm(gd) + 1e-8)
                pdv = np.array(pred["direction"]); pdv = pdv / (np.linalg.norm(pdv) + 1e-8)
                ang = np.degrees(np.arccos(np.clip(np.dot(gd, pdv), -1, 1)))
                if pos_dist <= pos_threshold and tiou >= iou_threshold and ang <= angular_threshold:
                    cost = pos_dist / pos_threshold + (1 - tiou) + ang / angular_threshold
                    if cost < best_cost:
                        best_cost, best_idx_, best_lab = cost, gidx, gt["lab"]
            if best_idx_ is not None:
                matched_gts[vname].add(best_idx_)
                matched_by_lab[best_lab] = matched_by_lab.get(best_lab, 0) + 1

        return matched_by_lab, total_by_lab

    N = 0.5
    equiv_global = N * FALLBACK_FRACTION  # same "average" scale, but fixed instead of per-lab

    bee_rel_matched, total_by_lab = run_match(lambda vname: bee_relative_pos_threshold(N, vname))
    fixed_matched, _ = run_match(lambda vname: equiv_global)

    print(f"\n=== N={N} bee-lengths vs equivalent fixed threshold ({equiv_global:.4f}) ===")
    print(f"{'lab':10s} {'total_gt':>9s} {'fixed_match':>12s} {'fixed_%':>8s} {'bee_match':>10s} {'bee_%':>8s}")
    for lab in total_by_lab:
        tot = total_by_lab[lab]
        fm = fixed_matched.get(lab, 0)
        bm = bee_rel_matched.get(lab, 0)
        print(f"{lab:10s} {tot:9d} {fm:12d} {100*fm/tot:7.1f}% {bm:10d} {100*bm/tot:7.1f}%")


if __name__ == "__main__":
    test_step_1()
    step_2_real_data()
