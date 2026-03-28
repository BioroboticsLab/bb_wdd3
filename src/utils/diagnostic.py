import numpy as np 

def compute_pass_rates(test_preds, test_gts, pos_threshold=20, iou_threshold=0.5, angular_threshold=15):
    n_spatial_pass = 0
    n_temporal_pass = 0
    n_angular_pass = 0
    n_all_pass = 0
    n_clips = 0

    for sample_preds, sample_gts in zip(test_preds, test_gts):
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