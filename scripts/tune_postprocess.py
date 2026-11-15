"""
Finds better post_process thresholds for a checkpoint using Optuna.

The viewer dashboard can already do this (the Optimize button), but it
loads validation data one sample at a time with no parallel workers, so it
takes about 14 hours on our current val set. This script does the same
search, just using a normal DataLoader with multiple workers like training
uses, which only takes about 3 hours for the same amount of data.

Usage:
    python scripts/tune_postprocess.py --ckpt_path ./ckpt/balanced_300k/latest.pth
"""
import argparse
import os
import random
import time

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

from src.data.dataset import VideoYoloDataset, TemporalWaggleCollator
from src.utils.data_utils import load_config
from src.utils.video_utils import train_val_split_videos
from src.utils.model_utils import load_pretrained_model, EMA
from src.utils.eval_utils import get_preds_gt, yolo_to_img_space, yolo_to_img_space_gt
from src.utils.dance_eval import (
    compute_crop_origins,
    deduplicate_gt_dances,
    average_overlapping_predictions,
    cross_window_cluster_predictions,
    compute_dance_level_metrics,
)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


def get_args():
    parser = argparse.ArgumentParser(description="Search for better post_process thresholds.")
    parser.add_argument("--config_path", type=str, default="./configs/config_balanced_300k.yaml")
    parser.add_argument("--ckpt_path", type=str, default="./ckpt/balanced_300k/latest.pth")
    parser.add_argument("--n_trials", type=int, default=100)
    parser.add_argument("--metric", type=str, default="map", choices=["map", "f1", "recall", "precision"])
    parser.add_argument("--bee_size_aware", action="store_true",
                         help="Search bee_size_multiplier instead of a fixed spatial_threshold.")
    parser.add_argument("--max_videos", type=int, default=None,
                         help="Only use this many val videos, for quickly testing the script works.")
    parser.add_argument("--profile", action="store_true",
                         help="Profile a single trial (one score() call) with cProfile and print the "
                              "top time-consuming functions, instead of running the full Optuna search. "
                              "Use this to find the real bottleneck before trying to speed things up.")
    return parser.parse_args()


def build_val_loader(config, test_df):
    test_transform = T.Compose([
        T.ToPILImage(),
        T.Resize((config['augmentations']['width'], config['augmentations']['height'])),
        T.Grayscale(num_output_channels=3),
        T.ToTensor(),
        T.Normalize(mean=config['augmentations']['mean'], std=config['augmentations']['std']),
    ])
    test_dataset = VideoYoloDataset(
        test_df,
        config['data']['data_dir'],
        test_transform,
        width=config['data']['width'],
        height=config['data']['height'],
        window_size=config['data']['window_size'],
        grid_size=config['model']['grid_size'],
        max_detections_per_cell=config['model']['max_detections_per_cell'],
        n_classes=config['model']['n_classes'],
        augment=None,
        is_training=False,
    )
    collator = TemporalWaggleCollator()
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['eval']['batch_size'],
        collate_fn=collator,
        shuffle=False,
        num_workers=config['train']['num_workers'],
        persistent_workers=(config['train']['num_workers'] > 0),
        pin_memory=True,
    )
    return test_dataset, test_loader


def main():
    args = get_args()
    config = load_config(args.config_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data = pd.read_csv(config['data']['annotations'])
    train_videos, _ = train_val_split_videos(
        data['video_name'].unique(),
        train_ratio=config['data']['train_ratio'],
        seed=SEED,
    )
    test_df = data[~data['video_name'].isin(train_videos)].reset_index(drop=True)

    if args.max_videos:
        keep = test_df['video_name'].unique()[:args.max_videos]
        test_df = test_df[test_df['video_name'].isin(keep)].reset_index(drop=True)
        print(f"  --max_videos set, only using {len(keep)} val videos (for quick testing)")

    test_dataset, test_loader = build_val_loader(config, test_df)

    print(f"Loading checkpoint: {args.ckpt_path}")
    model, checkpoint = load_pretrained_model(args.ckpt_path, config, device)
    ema = EMA(model, decay=config['train']['ema_decay'], device=device)
    if 'ema_state_dict' in checkpoint:
        ema.load_state_dict(checkpoint['ema_state_dict'])
        print(f"  Loaded EMA weights (updates: {ema.updates})")
    ema.apply_shadow()

    print(f"Running inference on {len(test_dataset)} val windows "
          f"({len(test_loader)} batches, bs={config['eval']['batch_size']}, "
          f"num_workers={config['train']['num_workers']}). This is the slow part, "
          f"everything after this is fast.")
    test_preds_raw, test_gt_raw, test_all_starts, test_all_ends, \
        test_all_video_names, _frames, test_all_original_res = get_preds_gt(model, test_loader, device)
    ema.restore()

    test_gts = yolo_to_img_space_gt(
        test_gt_raw, all_starts=test_all_starts, all_ends=test_all_ends,
        window_size=config['data']['window_size'],
        original_size=(config['data']['width'], config['data']['height']),
    )
    test_preds = yolo_to_img_space(
        test_preds_raw, all_starts=test_all_starts, all_ends=test_all_ends,
        confidence_threshold=config['eval']['confidence_threshold'],
        window_size=config['data']['window_size'],
        original_size=(config['data']['width'], config['data']['height']),
        max_dets=config['eval']['max_dets'],
    )

    print("Getting dance level eval ready (crop origins, gt dances, overlap averaging)...")
    crop_origins = compute_crop_origins(
        test_df, crop_w=config['data']['width'], crop_h=config['data']['height'],
        all_original_res=test_all_original_res,
    )
    video_res = {}
    video_fps = {}
    for vn, res in zip(test_all_video_names, test_all_original_res):
        video_res[vn] = res
        if vn not in video_fps:
            vpath = os.path.join(config['data']['data_dir'], vn)
            cap = cv2.VideoCapture(vpath)
            video_fps[vn] = float(cap.get(cv2.CAP_PROP_FPS)) or 15.0
            cap.release()
    gt_dances = deduplicate_gt_dances(test_df)

    # overlap averaging does not depend on the thresholds we are searching
    # over, so we only need to do it once and reuse it for every trial
    averaged_preds = average_overlapping_predictions(
        test_preds, crop_origins, test_all_video_names,
        window_ranges=list(zip(test_all_starts, test_all_ends)),
        video_fps=video_fps,
    )

    # use the same thresholds as training logs for dance_map, so this number
    # is actually comparable to what we have been tracking so far
    dl_cfg = config['eval']['dance_level']
    pos_thresholds = dl_cfg['pos_thresholds']
    iou_threshold_range = tuple(dl_cfg['iou_thresholds'])
    angular_thresholds = dl_cfg['angular_thresholds']

    def score(spatial, temporal_ms, conf, min_samp, dir_deg, bee_n, mode):
        predicted_runs = cross_window_cluster_predictions(
            averaged_preds, crop_origins, test_all_video_names, test_all_original_res,
            video_fps=video_fps,
            spatial_threshold=spatial,
            temporal_threshold_sec=temporal_ms / 1000.0,
            confidence_threshold=conf,
            min_samples=min_samp,
            mode=mode,
            direction_threshold_deg=dir_deg,
            bee_size_multiplier=bee_n,
        )
        return compute_dance_level_metrics(
            predicted_runs, gt_dances, video_res,
            pos_thresholds=pos_thresholds,
            iou_threshold_range=iou_threshold_range,
            angular_thresholds=angular_thresholds,
        )

    pp_cfg = config['post_process']
    baseline_metrics = score(
        pp_cfg['spatial_threshold'],
        pp_cfg.get('temporal_threshold_ms', 300),
        pp_cfg['confidence_threshold'],
        pp_cfg.get('min_samples', 1),
        pp_cfg.get('direction_threshold_deg', 30.0),
        pp_cfg.get('bee_size_multiplier'),
        pp_cfg.get('mode', 'mean'),
    )
    print(f"\nBaseline (current config values) dance mAP: {baseline_metrics['comprehensive']['map']:.4f}  "
          f"recall={baseline_metrics['coverage']['recall']:.4f}  "
          f"precision={baseline_metrics['coverage']['precision']:.4f}")

    if args.profile:
        import cProfile
        import pstats

        print("\n--- profiling a single trial (one score() call) ---")
        profiler = cProfile.Profile()
        profiler.enable()
        score(
            pp_cfg['spatial_threshold'],
            pp_cfg.get('temporal_threshold_ms', 300),
            pp_cfg['confidence_threshold'],
            pp_cfg.get('min_samples', 1),
            pp_cfg.get('direction_threshold_deg', 30.0),
            pp_cfg.get('bee_size_multiplier'),
            pp_cfg.get('mode', 'mean'),
        )
        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats('cumulative')
        stats.print_stats(25)
        return

    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        spatial = trial.suggest_float('spatial_threshold', 5, 200)
        temporal_ms = trial.suggest_float('temporal_threshold_ms', 50, 5000, log=True)
        conf = trial.suggest_float('confidence_threshold', 0.0, 0.5)
        min_samp = trial.suggest_int('min_samples', 1, 5)
        dir_deg = trial.suggest_float('direction_threshold_deg', 0, 90, step=5)
        bee_n = trial.suggest_float('bee_size_multiplier', 0.5, 20.0) if args.bee_size_aware else None
        mode = trial.suggest_categorical('mode', ['mean', 'median'])

        dance_metrics = score(spatial, temporal_ms, conf, min_samp, dir_deg, bee_n, mode)
        cov = dance_metrics['coverage']
        comp = dance_metrics['comprehensive']
        if args.metric == 'f1':
            recall, precision = cov['recall'], cov['precision']
            return 2 * precision * recall / (precision + recall + 1e-9)
        elif args.metric == 'recall':
            return cov['recall']
        elif args.metric == 'precision':
            return cov['precision']
        return comp['map']

    print(f"\nRunning Optuna search ({args.n_trials} trials, metric={args.metric}, "
          f"bee_size_aware={args.bee_size_aware})...")

    # optuna's own logging is turned off above, so without this we would not
    # see anything printed until all trials are done. This just prints a
    # short line every few trials so we can tell it is still moving and see
    # how the best score is changing.
    search_start_time = time.time()
    PRINT_EVERY_N_TRIALS = 5

    def print_progress(study, trial):
        if trial.number % PRINT_EVERY_N_TRIALS == 0 or trial.number == args.n_trials - 1:
            minutes_elapsed = (time.time() - search_start_time) / 60
            print(f"  trial {trial.number + 1}/{args.n_trials}  "
                  f"({minutes_elapsed:.1f} min elapsed)  "
                  f"this trial: {trial.value:.4f}  best so far: {study.best_value:.4f}")

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=args.n_trials, callbacks=[print_progress])

    print("\n=== Best params found ===")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
    print(f"  {args.metric}: {study.best_value:.4f}  "
          f"(baseline dance mAP was {baseline_metrics['comprehensive']['map']:.4f})")


if __name__ == '__main__':
    main()
