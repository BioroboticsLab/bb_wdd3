"""
Check whether dance-level detection depends on how many windows a dance had.

Reuses the same eval pipeline as ckpt_eval.py up through predicted_runs/gt_dances,
then does a single-threshold-combo match (instead of the full sweep) so we can
track per-dance detected/not-detected status, joined with n_windows (already
computed by deduplicate_gt_dances). Bins dances by window count and reports
recall per bin.
"""
import os
import argparse
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
import cv2

from src.utils.data_utils import load_config
from src.utils.model_utils import load_pretrained_model
from src.utils.video_utils import train_val_split_videos
from src.data.dataset import VideoYoloDataset, TemporalWaggleCollator
from src.utils.eval_utils import get_preds_gt, yolo_to_img_space, yolo_to_img_space_gt
from src.utils.dance_eval import (
    compute_crop_origins,
    deduplicate_gt_dances,
    average_overlapping_predictions,
    cross_window_cluster_predictions,
)

SEED = 42

def main(args):
    config = load_config(args.config_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data = pd.read_csv(config['data']['annotations'])
    train_videos, _ = train_val_split_videos(
        data['video_name'].unique(), train_ratio=config['data']['train_ratio'], seed=SEED
    )
    test_df = data[~data['video_name'].isin(train_videos)].reset_index(drop=True)

    test_transform = T.Compose([
        T.ToPILImage(),
        T.Resize((config['augmentations']['width'], config['augmentations']['height'])),
        T.Grayscale(num_output_channels=3),
        T.ToTensor(),
        T.Normalize(mean=config['augmentations']['mean'], std=config['augmentations']['std']),
    ])

    test_dataset = VideoYoloDataset(
        test_df, config['data']['data_dir'], test_transform,
        width=config['data']['width'], height=config['data']['height'],
        window_size=config['data']['window_size'], grid_size=config['model']['grid_size'],
        max_detections_per_cell=config['model']['max_detections_per_cell'],
        n_classes=config['model']['n_classes'], augment=None, is_training=False,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config['eval']['batch_size'],
        collate_fn=TemporalWaggleCollator(), shuffle=False,
        num_workers=config['train']['num_workers'], persistent_workers=False, pin_memory=True,
    )

    model, _ = load_pretrained_model(args.ckpt_path, config, device)
    model.eval()

    print("Running inference...")
    (test_preds_raw, test_gt_raw, all_starts, all_ends,
     all_video_names, _, all_original_res) = get_preds_gt(model, test_loader, device)

    test_gts = yolo_to_img_space_gt(
        test_gt_raw, all_starts=all_starts, all_ends=all_ends,
        window_size=config['data']['window_size'],
        original_size=(config['data']['width'], config['data']['height']),
    )
    test_preds = yolo_to_img_space(
        test_preds_raw, all_starts=all_starts, all_ends=all_ends,
        confidence_threshold=config['eval']['confidence_threshold'],
        window_size=config['data']['window_size'],
        original_size=(config['data']['width'], config['data']['height']),
        max_dets=config['eval']['max_dets'],
    )

    crop_origins = compute_crop_origins(
        test_df, crop_w=config['data']['width'], crop_h=config['data']['height'],
        all_original_res=all_original_res,
    )

    video_res, video_fps = {}, {}
    for vn, res in zip(all_video_names, all_original_res):
        video_res[vn] = res
        if vn not in video_fps:
            vpath = os.path.join(config['data']['data_dir'], vn)
            cap = cv2.VideoCapture(vpath)
            video_fps[vn] = float(cap.get(cv2.CAP_PROP_FPS)) or 15.0
            cap.release()

    gt_dances = deduplicate_gt_dances(test_df)

    averaged = average_overlapping_predictions(
        test_preds, crop_origins, all_video_names,
        window_ranges=list(zip(all_starts, all_ends)), video_fps=video_fps,
    )
    predicted_runs = cross_window_cluster_predictions(
        averaged, crop_origins, all_video_names, all_original_res,
        video_fps=video_fps,
        spatial_threshold=config['post_process']['spatial_threshold'],
        temporal_threshold_sec=config['post_process'].get('temporal_threshold_ms', 300) / 1000.0,
        confidence_threshold=config['post_process']['confidence_threshold'],
        min_samples=config['post_process'].get('min_samples', 1),
        mode=config['post_process'].get('mode', 'mean'),
        direction_threshold_deg=config['post_process'].get('direction_threshold_deg', 30.0),
    )

    # Single representative threshold combo (middle of the dance-level defaults)
    POS_T, IOU_T, ANG_T = 0.06, 0.3, 20.0
    print(f"Matching at a single representative combo: pos<={POS_T}, iou>={IOU_T}, ang<={ANG_T} deg")

    rows = []
    for vname, dances in gt_dances.items():
        res = video_res.get(vname)
        if res is None:
            continue
        orig_h, orig_w = res
        preds = sorted(predicted_runs.get(vname, []), key=lambda p: p['confidence'], reverse=True)
        matched_gt_idx = set()

        for pred in preds:
            best_idx, best_cost = None, float('inf')
            for gidx, gt in enumerate(dances):
                if gidx in matched_gt_idx:
                    continue
                gt_x, gt_y = gt['position'][0] / orig_w, gt['position'][1] / orig_h
                pos_dist = np.sqrt((pred['position'][0] - gt_x) ** 2 + (pred['position'][1] - gt_y) ** 2)

                gt_s, gt_e = gt['temporal_offsets']
                pr_s, pr_e = pred['temporal_offsets']
                inter = max(0, min(gt_e, pr_e) - max(gt_s, pr_s))
                union = max(gt_e, pr_e) - min(gt_s, pr_s)
                tiou = inter / union if union > 0 else 0

                gd = np.array(gt['direction']); gd = gd / (np.linalg.norm(gd) + 1e-8)
                pd_ = np.array(pred['direction']); pd_ = pd_ / (np.linalg.norm(pd_) + 1e-8)
                ang_err = np.degrees(np.arccos(np.clip(np.dot(gd, pd_), -1.0, 1.0)))

                if pos_dist <= POS_T and tiou >= IOU_T and ang_err <= ANG_T:
                    cost = pos_dist / POS_T + (1 - tiou) + ang_err / ANG_T
                    if cost < best_cost:
                        best_cost, best_idx = cost, gidx
            if best_idx is not None:
                matched_gt_idx.add(best_idx)

        for gidx, gt in enumerate(dances):
            # Count predicted runs that overlap this dance's true temporal span
            # at all (regardless of pos/angular match) -- fragmentation shows up
            # as multiple partial-overlap predictions instead of one full-span one.
            gt_s, gt_e = gt['temporal_offsets']
            overlapping = []
            for pred in preds:
                pr_s, pr_e = pred['temporal_offsets']
                inter = max(0, min(gt_e, pr_e) - max(gt_s, pr_s))
                if inter > 0:
                    overlapping.append(pred)

            rows.append({
                'video_name': vname,
                'waggle_run_id': gt['waggle_run_id'],
                'n_windows': gt['n_windows'],
                'duration_frames': gt['temporal_offsets'][1] - gt['temporal_offsets'][0],
                'detected': gidx in matched_gt_idx,
                'n_overlapping_preds': len(overlapping),
            })

    df = pd.DataFrame(rows)
    df.to_csv('outputs/diagnostics/stratified_recall.csv', index=False)
    print(f"\nTotal GT dances: {len(df)}, overall recall: {df['detected'].mean():.4f}")

    bins = [0, 2, 4, 7, 15, 1000]
    labels = ['1-2', '3-4', '5-7', '8-15', '16+']
    df['window_bin'] = pd.cut(df['n_windows'], bins=bins, labels=labels)

    print("\n=== Recall stratified by number of contributing windows ===")
    summary = df.groupby('window_bin', observed=True).agg(
        n_dances=('detected', 'size'), recall=('detected', 'mean')
    )
    print(summary.to_string())

    print("\n=== Failure mode breakdown per bin (among NOT detected dances) ===")
    print("clean_miss = 0 overlapping predictions | fragmented = 2+ overlapping "
          "predictions (split into pieces) | single_overlap_failed_match = exactly "
          "1 overlapping prediction that still failed pos/angular thresholds")
    missed = df[~df['detected']].copy()
    missed['failure_mode'] = missed['n_overlapping_preds'].apply(
        lambda n: 'clean_miss' if n == 0 else ('single_overlap_failed_match' if n == 1 else 'fragmented')
    )
    breakdown = missed.groupby(['window_bin', 'failure_mode'], observed=True).size().unstack(fill_value=0)
    print(breakdown.to_string())


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config_path", type=str, default='./configs/config.yaml')
    p.add_argument("--ckpt_path", type=str, default='./ckpt/baseline/best.pth')
    return p.parse_args()


if __name__ == '__main__':
    main(get_args())
