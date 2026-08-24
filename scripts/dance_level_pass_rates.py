"""
Dance-level equivalent of diagnostic.compute_pass_rates(), but using the real
matching heuristic (confidence-sorted greedy, joint AND validity, cost-minimized
among valid candidates) instead of the naive top-pred-vs-first-gt shortcut --
so the angular pass rate is directly comparable to window-level's.
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

# Single representative combo. Angular matches window-level's most recent test
# (15 deg) for a direct, apples-to-apples comparison on the axis in question.
POS_T = 0.06     # normalized position threshold (dance-level's own scale)
IOU_T = 0.3      # temporal IoU threshold
ANG_T = 15.0     # angular threshold, deg -- matches window-level's Pass angular test


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

    # Normalize GT positions to [0,1], same as compute_dance_level_metrics does
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

    # Flatten all predictions across videos, sorted by confidence descending
    all_preds_flat = []
    for vname, preds in predicted_runs.items():
        for pred in preds:
            all_preds_flat.append((pred['confidence'], vname, pred))
    all_preds_flat.sort(key=lambda x: x[0], reverse=True)

    total_gt = sum(len(v) for v in gt_norm.values())
    n_spatial_pass = n_temporal_pass = n_angular_pass = n_all_pass = 0
    matched_gts = {vname: set() for vname in gt_norm}

    for conf, vname, pred in all_preds_flat:
        vgts = gt_norm.get(vname, [])
        best_cost, best_idx = float('inf'), None
        for gidx, gt in enumerate(vgts):
            if gidx in matched_gts[vname]:
                continue
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

            # per-axis pass counters, over every candidate pair considered
            # (mirrors window-level's per-pair check, just within real matching)
            if pos_dist <= POS_T:
                n_spatial_pass += 1
            if tiou >= IOU_T:
                n_temporal_pass += 1
            if ang_err <= ANG_T:
                n_angular_pass += 1
            if pos_dist <= POS_T and tiou >= IOU_T and ang_err <= ANG_T:
                n_all_pass += 1
                cost = pos_dist / POS_T + (1 - tiou) + ang_err / ANG_T
                if cost < best_cost:
                    best_cost, best_idx = cost, gidx
        if best_idx is not None:
            matched_gts[vname].add(best_idx)

    n_pairs_considered = sum(len(gt_norm.get(vname, [])) for _, vname, _ in all_preds_flat)
    print(f"\nCombo: pos<={POS_T} (normalized), iou>={IOU_T}, angular<={ANG_T} deg")
    print(f"GT dances: {total_gt} | Predicted runs: {len(all_preds_flat)}")
    print(f"Candidate pairs considered: {n_pairs_considered}")
    print(f"Pass spatial:  {n_spatial_pass} ({100*n_spatial_pass/n_pairs_considered:.1f}%)")
    print(f"Pass temporal: {n_temporal_pass} ({100*n_temporal_pass/n_pairs_considered:.1f}%)")
    print(f"Pass angular:  {n_angular_pass} ({100*n_angular_pass/n_pairs_considered:.1f}%)")
    print(f"Pass ALL three (candidate pairs): {n_all_pass} ({100*n_all_pass/n_pairs_considered:.1f}%)")
    n_detected = sum(len(v) for v in matched_gts.values())
    print(f"Dances actually matched (post-greedy-assignment): {n_detected}/{total_gt} "
          f"({100*n_detected/total_gt:.1f}%)")


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config_path", type=str, default='./configs/config.yaml')
    p.add_argument("--ckpt_path", type=str, default='./ckpt/baseline/best.pth')
    return p.parse_args()


if __name__ == '__main__':
    main(get_args())
