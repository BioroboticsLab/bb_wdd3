import numpy as np 
import torch 
import matplotlib.pyplot as plt
import os 

def compute_pass_rates(test_preds, test_gts, 
                       pos_threshold=20, 
                       iou_threshold=0.5, 
                       angular_threshold=15):
    n_spatial_pass = 0
    n_temporal_pass = 0
    n_angular_pass = 0
    n_all_pass = 0
    n_clips = 0

    for sample_preds, sample_gts in zip(test_preds, test_gts):
        '''
        Diagnostic functiont hat computes the number of passes in % from the total preds
        that pass through the matching thresholds
        '''
        if not sample_preds or not sample_gts:
            continue
        n_clips += 1
        pred = sample_preds[0]
        gt   = sample_gts[0]

        pos_dist = np.linalg.norm(np.array(gt['position']) - np.array(pred['position']))
        gt_s, gt_e = gt['temporal_offsets']
        pr_s, pr_e = pred['temporal_offsets']
        inter = max(0, min(gt_e, pr_e) - max(gt_s, pr_s))
        union = max(gt_e, pr_e) - min(gt_s, pr_s)
        tiou  = inter / union if union > 0 else 0
        gt_d  = np.array(gt['direction']); gt_d /= np.linalg.norm(gt_d) + 1e-8
        pr_d  = np.array(pred['direction']); pr_d /= np.linalg.norm(pr_d) + 1e-8
        ang_err = np.degrees(np.arccos(np.clip(np.dot(gt_d, pr_d), -1, 1)))

        if pos_dist  <= pos_threshold:   n_spatial_pass  += 1
        if tiou      >= iou_threshold:   n_temporal_pass += 1
        if ang_err   <= angular_threshold: n_angular_pass += 1
        if pos_dist <= pos_threshold and tiou >= iou_threshold and ang_err <= angular_threshold:
            n_all_pass += 1

    print(f"Clips with predictions: {n_clips}")
    print(f"Pass spatial  (<={pos_threshold}px):  {n_spatial_pass} ({100*n_spatial_pass/n_clips:.1f}%)")
    print(f"Pass temporal (>={iou_threshold}):    {n_temporal_pass} ({100*n_temporal_pass/n_clips:.1f}%)")
    print(f"Pass angular  (<={angular_threshold}°): {n_angular_pass} ({100*n_angular_pass/n_clips:.1f}%)")
    print(f"Pass ALL three:          {n_all_pass} ({100*n_all_pass/n_clips:.1f}%)")


def compute_dance_pass_rates(predicted_runs, gt_dances, video_res,
                             pos_threshold=0.06,
                             iou_threshold=0.3,
                             angular_threshold=15):
    """
    Dance-level equivalent of compute_pass_rates(). Unlike the window-level
    version (which just checks the top prediction against the first GT per
    clip), this uses the real confidence-sorted greedy matching heuristic --
    same joint AND validity check, same cost-minimization among valid
    candidates -- so the angular/spatial/temporal figures are a fair,
    apples-to-apples comparison against window-level's own pass rates.
    """
    gt_norm = {}
    for vname, dances in gt_dances.items():
        res = video_res.get(vname)
        if res is None:
            continue
        orig_h, orig_w = res
        gt_norm[vname] = [
            {**d, 'position': [d['position'][0] / orig_w, d['position'][1] / orig_h]}
            for d in dances
        ]

    all_preds_flat = []
    for vname, preds in predicted_runs.items():
        for pred in preds:
            all_preds_flat.append((pred['confidence'], vname, pred))
    all_preds_flat.sort(key=lambda x: x[0], reverse=True)

    total_gt = sum(len(v) for v in gt_norm.values())
    n_spatial_pass = n_temporal_pass = n_angular_pass = n_all_pass = 0
    n_pairs_considered = 0
    matched_gts = {vname: set() for vname in gt_norm}

    for conf, vname, pred in all_preds_flat:
        vgts = gt_norm.get(vname, [])
        best_cost, best_idx = float('inf'), None
        for gidx, gt in enumerate(vgts):
            if gidx in matched_gts[vname]:
                continue
            n_pairs_considered += 1
            pos_dist = np.sqrt((pred['position'][0] - gt['position'][0]) ** 2 +
                                (pred['position'][1] - gt['position'][1]) ** 2)
            gt_s, gt_e = gt['temporal_offsets']
            pr_s, pr_e = pred['temporal_offsets']
            inter = max(0, min(gt_e, pr_e) - max(gt_s, pr_s))
            union = max(gt_e, pr_e) - min(gt_s, pr_s)
            tiou = inter / union if union > 0 else 0
            gd = np.array(gt['direction']); gd = gd / (np.linalg.norm(gd) + 1e-8)
            pd_ = np.array(pred['direction']); pd_ = pd_ / (np.linalg.norm(pd_) + 1e-8)
            ang_err = np.degrees(np.arccos(np.clip(np.dot(gd, pd_), -1.0, 1.0)))

            if pos_dist <= pos_threshold:
                n_spatial_pass += 1
            if tiou >= iou_threshold:
                n_temporal_pass += 1
            if ang_err <= angular_threshold:
                n_angular_pass += 1
            if pos_dist <= pos_threshold and tiou >= iou_threshold and ang_err <= angular_threshold:
                n_all_pass += 1
                cost = pos_dist / pos_threshold + (1 - tiou) + ang_err / angular_threshold
                if cost < best_cost:
                    best_cost, best_idx = cost, gidx
        if best_idx is not None:
            matched_gts[vname].add(best_idx)

    n_detected = sum(len(v) for v in matched_gts.values())

    print(f"\n=== Dance-level pass rates (pos<={pos_threshold}, iou>={iou_threshold}, angular<={angular_threshold}°) ===")
    print(f"GT dances: {total_gt} | Predicted runs: {len(all_preds_flat)} | Candidate pairs considered: {n_pairs_considered}")
    if n_pairs_considered > 0:
        print(f"Pass spatial:  {n_spatial_pass} ({100*n_spatial_pass/n_pairs_considered:.1f}%)")
        print(f"Pass temporal: {n_temporal_pass} ({100*n_temporal_pass/n_pairs_considered:.1f}%)")
        print(f"Pass angular:  {n_angular_pass} ({100*n_angular_pass/n_pairs_considered:.1f}%)")
        print(f"Pass ALL three (candidate pairs): {n_all_pass} ({100*n_all_pass/n_pairs_considered:.1f}%)")
    if total_gt > 0:
        print(f"Dances actually matched (post-greedy-assignment): {n_detected}/{total_gt} "
              f"({100*n_detected/total_gt:.1f}%)")


def plot_confidence_histogram(test_preds_raw,
                              test_preds, 
                              test_gts, 
                              config, 
                              save_path='./outputs/confidence_histogram.png'):
    """
    Plots 1. confidence histogram across all cells/predictionshistogram
    2. of high confidence region 0.9 to 1.0 and
    3. after postprocessing/filtering.
    Args:
        test_preds_raw: Raw model output tensor (B, grid_h, grid_w, max_det, 7)
        test_preds: List of filtered prediction dicts (after max_dets filtering)
        test_gts: List of ground truth dicts
        config: Config dict
        save_path: Path to save the figure
    """
    raw_confs = torch.sigmoid(test_preds_raw[..., 0]).flatten().cpu().numpy()
    filtered_confs = np.array([
        pred['confidence']
        for sample in test_preds
        for pred in sample
    ])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].hist(raw_confs, bins=50, edgecolor='black')
    axes[0].set_title('Confidence across all cells per clip')
    axes[0].set_xlabel('Confidence')
    axes[0].set_ylabel('Count')
    axes[0].axvline(raw_confs.mean(), color='red', linestyle='--',
                    label=f'mean={raw_confs.mean():.3f}')
    axes[0].legend()

    axes[1].hist(raw_confs, bins=50, range=(0.9, 1.0), edgecolor='black')
    axes[1].set_title('Confidences at 0.9 to 1.0 region')
    axes[1].set_xlabel('Confidence')
    axes[1].set_ylabel('Count')

    axes[2].hist(filtered_confs, bins=50, edgecolor='black')
    axes[2].set_title(f'After max_dets={config["eval"]["max_dets"]} filter')
    axes[2].set_xlabel('Confidence')
    axes[2].set_ylabel('Count')
    axes[2].axvline(filtered_confs.mean(), color='red', linestyle='--',
                    label=f'mean={filtered_confs.mean():.3f}')
    axes[2].legend()

    n_raw = len(raw_confs)
    n_filtered = len(filtered_confs)
    n_gt = sum(len(g) for g in test_gts)
    plt.suptitle(
        f'Predictions: {n_raw} | After filter: {n_filtered} | GTs: {n_gt} | '
        f'Ratio GT vs. predictions: {100*n_gt/n_raw:.3f}% | After filter: {100*n_gt/n_filtered:.3f}%'
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
    
    # after histogram print some stats about the confidence and predictions and gt 
    # for clearer diagnostic purposes
    print_detection_diagnostics(test_gts, raw_confs, filtered_confs)


def print_detection_diagnostics(test_gts, raw_confs, filtered_confs):
    """
    Prints dataset and confidence statistics.

    Args:
        test_gts: List of ground truth lists per clip
        raw_confs: numpy array of raw confidences (flattened)
        filtered_confs: numpy array of filtered confidences
    """

    # GT stats
    n_empty_gts = sum(1 for g in test_gts if len(g) == 0)
    n_nonempty_gts = sum(1 for g in test_gts if len(g) > 0)
    n_gt = sum(len(g) for g in test_gts)

    # Prediction stats
    n_raw = len(raw_confs)
    n_filtered = len(filtered_confs)

    print(f"Clips with GT:    {n_nonempty_gts}")
    print(f"Clips without GT: {n_empty_gts}")
    print(f"Total GTs:        {n_gt}")

    print(f"\nRaw cells:        {n_raw}")
    print(f"Filtered cells:   {n_filtered}")
    print(f"Ground truths:    {n_gt}")
    print(f"Positive rate:    {100 * n_gt / n_raw:.3f}%")

    #  Raw confidence stats
    print(f"\nRaw confidence:")
    print(f"  mean:    {raw_confs.mean():.4f}")
    print(f"  std:     {raw_confs.std():.4f}")
    print(f"  >0.99:   {(raw_confs > 0.99).mean():.4f}")
    print(f"  >0.5:    {(raw_confs > 0.5).mean():.4f}")
    print(f"  >0.1:    {(raw_confs > 0.1).mean():.4f}")

    # Filtered confidence stats
    print(f"\nFiltered confidence:")
    print(f"  mean:    {filtered_confs.mean():.4f}")
    print(f"  std:     {filtered_confs.std():.4f}")
    print(f"  >0.99:   {(filtered_confs > 0.99).mean():.4f}")