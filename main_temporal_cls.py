import os
import random
import numpy as np
import torch
import torchvision.transforms as T
import pandas as pd
from src.train.train import train
from src.eval.eval import eval
import torch.optim as optim
from torch.utils.data import DataLoader
from src.data.dataset_temporal_cls import TemporalWaggleCollator, VideoYoloDatasetTemporalCls
from src.models.model_temporal_cls import R2Plus1D_YOLO_MultiHead
from src.loss.loss_temporal_cls import WaggleDetectionLoss
from src.data.augmentation import WaggleAugmentations
import torch.nn as nn
from src.utils.data_utils import load_config, balance_sample
import datetime
import wandb
# aliased to _cls versions -- everything below this line reads "yolo_to_img_space"
# but is actually calling the per-frame-classification decode, so the rest of
# this file can stay identical to main.py
from src.utils.eval_utils import (
    get_preds_gt,
    yolo_to_img_space_cls as yolo_to_img_space,
    yolo_to_img_space_gt_cls as yolo_to_img_space_gt,
    get_eval_metrics,
    get_wandb_log_dict,
)
from src.utils.postprocess import batch_postprocess_predictions
from src.utils.dance_eval import (
    compute_crop_origins,
    deduplicate_gt_dances,
    average_overlapping_predictions,
    cross_window_cluster_predictions,
    compute_dance_level_metrics,
)
import cv2
import argparse
from src.utils.model_utils import EMA

# This is the "per-frame temporal classification" experiment from the plan
# to fix dance-level temporal IoU -- keep this file's diff against main.py
# minimal, so it stays easy to tell which parts actually differ. Doesn't
# support the temporal_jitter dataset variant (VideoYoloDatasetTemporalJitter)
# -- combining that with this is a separate experiment nobody asked for yet.

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def main(args):
    config = load_config(args.config_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    run_name = args.run_name if args.run_name else f"run_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    ckpt_dir = f"./ckpt/{run_name}"
    os.makedirs(ckpt_dir, exist_ok=True)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs.")
        use_multi_gpu = True
    else:
        print("Using single GPU.")
        use_multi_gpu = False

    data = pd.read_csv(config['data']['annotations'])
    full_data_size = len(data)

    from src.utils.video_utils import train_val_split_videos
    train_videos, _ = train_val_split_videos(
        data['video_name'].unique(),
        train_ratio=config['data']['train_ratio'],
        seed=SEED
    )

    train_df = data[data['video_name'].isin(train_videos)].reset_index(drop=True)
    test_df  = data[~data['video_name'].isin(train_videos)].reset_index(drop=True)

    if config['data']['data_fraction_divisor'] > 1:
        train_df = balance_sample(train_df, config['data']['data_fraction_divisor'])

    n_unique_videos = data['video_name'].nunique()
    print(f"Using: {len(train_df)} train / {len(test_df)} test samples (from {full_data_size} total).")
    print(f"Train videos: {len(train_videos)} | Test videos: {n_unique_videos - len(train_videos)}")

    transforms = T.Compose([
        T.ToPILImage(),
        T.Resize((config['augmentations']['width'],
                  config['augmentations']['height'])),
        T.Grayscale(num_output_channels=3),
        T.ToTensor(),
    ])

    test_transform = T.Compose([
        T.ToPILImage(),
        T.Resize((config['augmentations']['width'],
                  config['augmentations']['height'])),
        T.Grayscale(num_output_channels=3),
        T.ToTensor(),
        T.Normalize(mean=config['augmentations']['mean'],
                    std=config['augmentations']['std'])])

    train_augmentation = WaggleAugmentations(
        width=config['augmentations']['width'],
        height=config['augmentations']['height'],
        prob_flip_h=config['augmentations']['prob_flip_h'],
        prob_flip_v=config['augmentations']['prob_flip_v'],
        prob_rotate=config['augmentations']['prob_rotate'],
        rotate_range=config['augmentations']['rotate_range'],
        prob_scale=config['augmentations']['prob_scale'],
        scale_range=config['augmentations']['scale_range'],
        prob_translate=config['augmentations']['prob_translate'],
        translate_range=config['augmentations']['translate_range'],
        prob_hsv=config['augmentations']['prob_hsv'],
        hsv_hue=config['augmentations']['hsv_hue'],
        hsv_saturation=config['augmentations']['hsv_saturation'],
        hsv_value=config['augmentations']['hsv_value'],
        prob_brightness=config['augmentations']['prob_brightness'],
        brightness_range=config['augmentations']['brightness_range'],
        prob_contrast=config['augmentations']['prob_contrast'],
        contrast_range=config['augmentations']['contrast_range'],
        prob_gamma=config['augmentations']['prob_gamma'],
        gamma_range=config['augmentations']['gamma_range'],
        prob_blur=config['augmentations']['prob_blur'],
        blur_range=config['augmentations']['blur_range'],
        prob_clahe=config['augmentations']['prob_clahe'],
        clahe_clip_limit=config['augmentations']['clahe_clip_limit'],
        clahe_tile_grid_size=config['augmentations']['clahe_tile_grid_size'],
        prob_color_shuffle=config['augmentations']['prob_color_shuffle'],
        prob_posterize=config['augmentations']['prob_posterize'],
        posterize_bits=config['augmentations']['posterize_bits'],
        prob_greyscale=config['augmentations']['prob_greyscale'],
        normalize=config['augmentations']['normalize'],
        mean=config['augmentations']['mean'],
        std=config['augmentations']['std'],
        )

    train_dataset = VideoYoloDatasetTemporalCls(
        train_df,
        config['data']['data_dir'],
        transforms,
        width=config['data']['width'],
        height=config['data']['height'],
        window_size=config['data']['window_size'],
        grid_size=config['model']['grid_size'],
        max_detections_per_cell=config['model']['max_detections_per_cell'],
        n_classes=config['model']['n_classes'],
        augment=train_augmentation,
        is_training=True
    )

    test_dataset = VideoYoloDatasetTemporalCls(
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
        is_training=False
    )

    collator = TemporalWaggleCollator()

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['train']['batch_size'],
        collate_fn=collator,
        shuffle=True,
        num_workers=config['train']['num_workers'],
        persistent_workers=(config['train']['num_workers'] > 0),
        pin_memory=True,
        drop_last=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['val']['batch_size'],
        collate_fn=collator,
        shuffle=False,
        num_workers=config['train']['num_workers'],
        persistent_workers=(config['train']['num_workers'] > 0),
        pin_memory=True
    )

    train_dataset.count_categories()

    print(f'Training on {len(train_loader)} batches with {config["train"]["batch_size"]}.')
    print(f'{len(train_loader) * config["train"]["batch_size"]} videos in total.')
    print(f'Evaluating on {len(test_loader)} batches with {config["val"]["batch_size"]}.')
    print(f'{len(test_loader) * config["val"]["batch_size"]} videos in total.')

    model = R2Plus1D_YOLO_MultiHead(n_classes=config['model']['n_classes'],
                                    max_detections_per_cell=config['model']['max_detections_per_cell'],
                                    grid_size=config['model']['grid_size'],
                                    transformer_heads=config['model']['transformer_heads'],
                                    transformer_layers=config['model']['transformer_layers'],
                                    self_attention=config['model']['self_attention'],
                                    cross_attention=config['model']['cross_attention'],
                                    dropout_rate=config['model']['dropout'],
                                    window_size=config['data']['window_size'],
                                    )

    model = model.to(device)

    if use_multi_gpu:
        model = nn.DataParallel(model)
        print("Using Distributed Data Parallel (DDP).")

    optimizer = optim.AdamW(model.parameters(),
                            lr=config['train']['lr'],
                            weight_decay=config['train']['weight_decay'],
                            betas=config['train']['betas'])

    ema = EMA(model, decay=config['train']['ema_decay'], device=device)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config['train']['lr'],
        steps_per_epoch=len(train_loader),
        epochs=config['train']['epochs'],
        anneal_strategy='cos',
        pct_start=config['train']['warmup_ratio']
    )
    yolocriteria = WaggleDetectionLoss(lambda_obj=config["loss"]["lambda_obj"],
                                       lambda_coord=config["loss"]["lambda_coord"],
                                       lambda_noobj=config["loss"]["lambda_noobj"],
                                       lambda_direction=config["loss"]["lambda_direction"],
                                       lambda_temporal=config["loss"]["lambda_temporal"],
                                       use_varifocal=config['loss'].get('use_varifocal', False),
                                       gamma=config["loss"]["varifocal_gamma"],
                                       quality_decay=config["loss"]["quality_decay"],
                                       grid_size=config["loss"]["grid_size"],
                                       input_size=config["loss"]["input_size"])

    scaler = torch.amp.GradScaler()

    start_epoch = 0
    monitor_metric = config['eval']['metric']
    best_score = float('inf') if monitor_metric == 'loss' else 0.0

    if monitor_metric == 'map' and not config['eval']['window_level'].get('enabled', True):
        raise ValueError(
            "eval.window_level.enabled is False, but eval.metric is "
            f"'{monitor_metric}' -- checkpoint selection needs window-level "
            "mAP to pick the best model. Set eval.metric to 'loss' or "
            "'dance_map', or re-enable window_level."
        )
    if monitor_metric == 'dance_map' and not config['eval']['dance_level'].get('enabled', True):
        raise ValueError(
            "eval.dance_level.enabled is False, but eval.metric is "
            f"'{monitor_metric}' -- checkpoint selection needs dance-level "
            "mAP to pick the best model. Set eval.metric to 'loss' or "
            "'map', or re-enable dance_level."
        )

    # wandb resume is keyed by run id, not run name -- name alone (below)
    # would silently start a brand new run every time. Grab the id the
    # aborted run saved into its own checkpoint so this run's graphs
    # actually continue instead of restarting from step 0. Older
    # checkpoints saved before this fix won't have it, hence the
    # WANDB_RUN_ID env var escape hatch for reconnecting to those by hand.
    resumed_wandb_id = None

    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        if isinstance(model, torch.nn.DataParallel):
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch']
        best_score = checkpoint['best_score']
        resumed_wandb_id = checkpoint.get('wandb_run_id')

        if args.reset_scheduler:
            remaining_epochs = config['train']['epochs'] - start_epoch
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=config['train']['lr'],
                steps_per_epoch=len(train_loader),
                epochs=remaining_epochs,
                anneal_strategy='cos',
                pct_start=config['train']['warmup_ratio']
            )
            print(f"Scheduler RESET for {remaining_epochs} remaining epochs (new steps_per_epoch={len(train_loader)})")
        else:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if 'ema_state_dict' in checkpoint:
            ema.load_state_dict(checkpoint['ema_state_dict'])
            print(f"Loaded EMA state (updates: {ema.updates})")

        print(f"Resumed at epoch {start_epoch+1}, best_score so far: {best_score:.4f}")
    elif args.resume:
        print(f"Warning: checkpoint path '{args.resume}' not found, starting from scratch.")

    dance_level_cache = {}

    # explicit env var wins over whatever (if anything) came from the
    # checkpoint -- lets you manually reconnect to a run that predates this
    # fix, e.g. WANDB_RUN_ID=<id from the wandb URL> when resuming.
    resumed_wandb_id = os.environ.get('WANDB_RUN_ID') or resumed_wandb_id

    wandb.init(
    project=config['logging']['wandb_project'],
    config={
        **config,
        "seed": SEED,
        "temporal_head": "per_frame_classification",
    },
    name=run_name,
    id=resumed_wandb_id,
    resume="must" if resumed_wandb_id else None,
)
    for epoch in range(start_epoch, config['train']['epochs']):
        print(f'\n Epoch {epoch+1}/{config["train"]["epochs"]}')
        train_loss, train_log = train(
            model, device, optimizer, yolocriteria, scheduler, train_loader, epoch, scaler, ema)

        ema.apply_shadow()
        val_loss, val_log = eval(model, device, yolocriteria, test_loader, epoch)

        map_freq = config['eval'].get('map_freq', 1)
        compute_map = (epoch % map_freq == 0) or (epoch == config['train']['epochs'] - 1)

        test_metrics = None
        post_test_metrics = None
        dance_metrics = None

        if compute_map:
            test_preds_raw, test_gt_raw, test_all_starts, test_all_ends, test_all_video_names, _, test_all_original_res = get_preds_gt(model, test_loader, device)

            test_gts = yolo_to_img_space_gt(test_gt_raw,
                                            all_starts=test_all_starts,
                                            all_ends=test_all_ends,
                                            window_size = config['data']['window_size'],
                                            original_size=(config['data']['width'],
                                                           config['data']['height']))

            test_preds  = yolo_to_img_space(test_preds_raw,
                                            all_starts=test_all_starts,
                                            all_ends=test_all_ends,
                                            confidence_threshold=config['eval']['confidence_threshold'],
                                            window_size = config['data']['window_size'],
                                            original_size=(config['data']['width'],
                                                           config['data']['height']),
                                                           max_dets=config['eval']['max_dets'])

            dl_cfg = config['eval']['dance_level']
            dance_wandb_dict = {}
            if dl_cfg.get('enabled', True):
                if 'crop_origins' not in dance_level_cache:
                    dance_level_cache['crop_origins'] = compute_crop_origins(
                        test_df,
                        crop_w=config['data']['width'],
                        crop_h=config['data']['height'],
                        all_original_res=test_all_original_res,
                    )
                    dance_level_cache['gt_dances'] = deduplicate_gt_dances(test_df)
                    video_res = {}
                    video_fps = {}
                    for vn, res in zip(test_all_video_names, test_all_original_res):
                        video_res[vn] = res
                        if vn not in video_fps:
                            vpath = os.path.join(config['data']['data_dir'], vn)
                            cap = cv2.VideoCapture(vpath)
                            video_fps[vn] = float(cap.get(cv2.CAP_PROP_FPS)) or 15.0
                            cap.release()
                    dance_level_cache['video_res'] = video_res
                    dance_level_cache['video_fps'] = video_fps

                crop_origins = dance_level_cache['crop_origins']
                gt_dances = dance_level_cache['gt_dances']
                video_res = dance_level_cache['video_res']
                video_fps = dance_level_cache['video_fps']

                averaged_preds = average_overlapping_predictions(
                    test_preds, crop_origins, test_all_video_names,
                    window_ranges=list(zip(test_all_starts, test_all_ends)),
                    video_fps=video_fps,
                )
                predicted_runs = cross_window_cluster_predictions(
                    averaged_preds, crop_origins, test_all_video_names, test_all_original_res,
                    video_fps=video_fps,
                    spatial_threshold=config['post_process']['spatial_threshold'],
                    temporal_threshold_sec=config['post_process'].get('temporal_threshold_ms', 300) / 1000.0,
                    confidence_threshold=config['post_process']['confidence_threshold'],
                    min_samples=config['post_process'].get('min_samples', 1),
                    mode=config['post_process'].get('mode', 'mean'),
                    direction_threshold_deg=config['post_process'].get('direction_threshold_deg', 30.0),
                    bee_size_multiplier=config['post_process'].get('bee_size_multiplier'),
                    temporal_boundary=config['post_process'].get('temporal_boundary', 'mean'),
                )
                dance_metrics = compute_dance_level_metrics(
                    predicted_runs, gt_dances, video_res,
                    pos_thresholds=dl_cfg['pos_thresholds'],
                    iou_threshold_range=tuple(dl_cfg['iou_thresholds']),
                    angular_thresholds=dl_cfg['angular_thresholds'],
                )
                print(f"Epoch {epoch+1}/{config['train']['epochs']} | Dance mAP: {dance_metrics['comprehensive']['map']:.4f} "
                      f"| Recall: {dance_metrics['coverage']['recall']:.4f} | Precision: {dance_metrics['coverage']['precision']:.4f}")
                dance_wandb_dict = {
                    'dance/map': dance_metrics['comprehensive']['map'],
                    'dance/mean_precision': dance_metrics['comprehensive']['mean_precision'],
                    'dance/mean_recall': dance_metrics['comprehensive']['mean_recall'],
                    'dance/mean_f1': dance_metrics['comprehensive']['mean_f1'],
                    'dance/coverage_recall': dance_metrics['coverage']['recall'],
                    'dance/coverage_precision': dance_metrics['coverage']['precision'],
                    'dance/temporal_iou': dance_metrics['temporal']['mean_iou'],
                    'dance/duration_accuracy': dance_metrics['temporal']['duration_accuracy'],
                }

            wl_cfg = config['eval']['window_level']
            if wl_cfg.get('enabled', True):
                post_test_preds = batch_postprocess_predictions(test_preds,
                                                                spatial_threshold=config['post_process']['spatial_threshold'],
                                                                temporal_threshold=config['post_process']['temporal_threshold'],
                                                                confidence_threshold=config['post_process']['confidence_threshold'],
                                                                strategy=config['post_process']['strategy'],
                                                                mode=config['post_process']['mode'],
                                                                remove_outliers=False,
                                                                min_samples=config['post_process'].get('min_samples', 1))

                test_metrics = get_eval_metrics(test_preds, test_gts,
                                                pos_thresholds=wl_cfg['pos_thresholds'],
                                                iou_threshold_range=wl_cfg['iou_thresholds'],
                                                angular_thresholds=wl_cfg['angular_thresholds'],
                                                match_pairs=wl_cfg['match_pairs'])

                post_test_metrics = get_eval_metrics(post_test_preds, test_gts,
                                                pos_thresholds=wl_cfg['pos_thresholds'],
                                                iou_threshold_range=wl_cfg['iou_thresholds'],
                                                angular_thresholds=wl_cfg['angular_thresholds'],
                                                match_pairs=wl_cfg['match_pairs'])

                print(f"Epoch {epoch+1}/{config['train']['epochs']} | STD-mAP pre: {test_metrics['comprehensive']['map']:.4f} | post: {post_test_metrics['comprehensive']['map']:.4f}")

                wandb.log({
                    **train_log,
                    **val_log,
                    **get_wandb_log_dict(epoch, test_metrics, post_test_metrics),
                    **dance_wandb_dict,
                    })
            else:
                print(f"Epoch {epoch+1}/{config['train']['epochs']} | val_loss: {val_loss:.4f} (window-level metrics disabled via config)")
                wandb.log({
                    **train_log,
                    **val_log,
                    **dance_wandb_dict,
                    })
        else:
            print(f"Epoch {epoch+1}/{config['train']['epochs']} | val_loss: {val_loss:.4f} (mAP skipped, next at epoch {((epoch // map_freq) + 1) * map_freq + 1})")
            wandb.log({
                **train_log,
                **val_log,
                })
        ema.restore()

        if monitor_metric == 'loss':
            current_score = val_loss
            is_best = current_score < best_score
            score_str = f"val_loss: {current_score:.4f}"
        elif monitor_metric == 'dance_map' and compute_map:
            current_score = dance_metrics['comprehensive']['map']
            is_best = current_score > best_score
            score_str = f"Dance mAP: {current_score:.4f}"
        elif compute_map:
            current_score = post_test_metrics['comprehensive']['map']
            is_best = current_score > best_score
            score_str = f"STD-mAP (post-proc): {current_score:.4f}"
        else:
            current_score = None
            is_best = False
            score_str = "(mAP not computed)"

        std_map_val = post_test_metrics['comprehensive']['map'] if post_test_metrics else 0.0
        dance_map_val = dance_metrics['comprehensive']['map'] if dance_metrics else 0.0

        if epoch % config['train']['val_freq'] == 0 or epoch == config['train']['epochs'] - 1:
            if is_best:
                best_score = current_score
                if config['train'].get('save_model', True):
                    torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict': model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'scaler_state_dict': scaler.state_dict(),
                        'ema_state_dict': ema.state_dict(),
                        'best_score': best_score,
                        'val_loss': val_loss,
                        'std_map': std_map_val,
                        'dance_map': dance_map_val,
                        'wandb_run_id': wandb.run.id,
                    }, os.path.join(ckpt_dir, 'best.pth'))
                    print(f"New best model saved -> {score_str}")
                else:
                    print(f"New best -> {score_str} (model saving disabled)")

            if config['train'].get('save_model', True):
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'ema_state_dict': ema.state_dict(),
                    'best_score': best_score,
                    'val_loss': val_loss,
                    'std_map': std_map_val,
                    'dance_map': dance_map_val,
                    'wandb_run_id': wandb.run.id,
                }, os.path.join(ckpt_dir, 'latest.pth'))
                print(f"Saved latest checkpoint at epoch {epoch+1}/{config['train']['epochs']}")
            else:
                print(f"Evaluated model at epoch {epoch+1}/{config['train']['epochs']} (model saving disabled)")

    print('Training complete.')
    wandb.finish()

def get_args():
    parser = argparse.ArgumentParser(description="Waggle detection training (per-frame temporal classification variant)")

    parser.add_argument("--config_path", type=str, default='./configs/config.yaml', help="Path to the config file.")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training from (e.g. ./ckpt/latest.pth).")
    parser.add_argument("--run_name", type=str, default=None, help="Name for this run. Defaults to datetime if not specified.")
    parser.add_argument("--reset_scheduler", action='store_true', help="Reset LR scheduler instead of loading from checkpoint (use when data size or epochs changed).")
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    main(args)
