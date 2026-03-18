import os 
from src.utils.data_utils import create_video_frames_df
import random
import numpy as np
import torch
import torchvision.transforms as T
import pandas as pd
from src.train.train import train
#from src.train.train_grad_diagnostic import train
from src.eval.eval import eval
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from src.data.dataset import TemporalWaggleCollator, VideoYoloDataset
from src.data.dataset_tempaug import VideoYoloDatasetTemporalJitter
from src.models.model import R2Plus1D_YOLO_MultiHead
from src.loss.loss import WaggleDetectionLoss
from src.data.augmentation import WaggleAugmentations
from src.tests.aug_vis import demo_visualization
import torch.nn  as nn
from src.utils.data_utils import fix_dataframe_with_video_lengths, load_config, save_preds_to_csv, balance_sample
import datetime
import wandb
from src.utils.eval_utils import get_preds_gt, yolo_to_img_space, yolo_to_img_space_gt, get_eval_metrics, print_evaluation_results, get_wandb_log_dict
from src.utils.postprocess import batch_postprocess_predictions
from src.utils.vis_utils import reverse_transform_batch, save_frames
import argparse
from src.utils.model_utils import load_pretrained_model, EMA
from collections import Counter
from src.utils.video_utils import get_video_category


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
    
    # run name either specified or date time so multiple runs dont overwrite checkpoint
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

    # Stratified video-level split:
    # all windows from a given video go entirely to train or test never both
    # split is done per category so each group contributes proportionally to both splits 
    # regardless of how many videos it has
    video_df = pd.DataFrame({'video_name': data['video_name'].unique()})
    video_df['category'] = video_df['video_name'].apply(get_video_category)

    train_videos = set()
    for category, group in video_df.groupby('category'):
        vids = np.random.RandomState(SEED).permutation(group['video_name'].values)
        n_train = int(config['data']['train_ratio'] * len(vids))
        train_videos.update(vids[:n_train])
    
    train_df = data[data['video_name'].isin(train_videos)].reset_index(drop=True)
    test_df  = data[~data['video_name'].isin(train_videos)].reset_index(drop=True)

    # Balance/subsample data
    if config['data']['data_fraction_divisor'] > 1:
        train_df = balance_sample(train_df, config['data']['data_fraction_divisor'])
        test_df  = balance_sample(test_df,  config['data']['data_fraction_divisor'])
    
    total_videos = len(video_df)
    print(f"Using: {len(train_df)} train / {len(test_df)} test samples (from {full_data_size} total).")
    print(f"Train videos: {len(train_videos)} | Test videos: {total_videos - len(train_videos)}")

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
    
    # Create augmentation
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

    if config['augmentations']['temporal_jitter'] is True:
        # applies temporal jitter to entire dataset as done in action detection
        train_dataset = VideoYoloDatasetTemporalJitter(
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
    
        test_dataset = VideoYoloDatasetTemporalJitter(
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

    else:
        # Create dataset without temporal jitter
        train_dataset = VideoYoloDataset(
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
                                    dropout_rate=config['model']['dropout']   
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
                                       quality_scale=config["loss"]["varifocal_quality_scale"])
    
    scaler = torch.amp.GradScaler()

    # Resume from checkpoint if provided
    start_epoch = 0
    monitor_metric = config['eval']['metric']
    best_score = float('inf') if monitor_metric == 'loss' else 0.0

    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        if isinstance(model, torch.nn.DataParallel):
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_score = checkpoint['best_score']

        if 'ema_state_dict' in checkpoint:
            ema.load_state_dict(checkpoint['ema_state_dict'])
            print(f"Loaded EMA state (updates: {ema.updates})")

        print(f"Resumed at epoch {start_epoch}, best_score so far: {best_score:.4f}")
    elif args.resume:
        print(f"Warning: checkpoint path '{args.resume}' not found, starting from scratch.")

    # init wandb for logging
    wandb.init(
    project="waggle-detection",  # Project name
    config={
        **config,  # Log entire config
        "seed": SEED,
    },
    name=run_name,
    resume="allow" if args.resume else None
)
    
    for epoch in range(start_epoch, config['train']['epochs']):
        print(f'\n Epoch {epoch+1}/{config["train"]["epochs"]}')
        # Train one epoch
        train_loss, train_log = train(
            model, device, optimizer, yolocriteria, scheduler, train_loader, epoch, scaler, ema)
        
        # Validate
        ema.apply_shadow()
        val_loss, val_log = eval(model, device, yolocriteria, test_loader, epoch)

        # fetch gt and preds
        test_preds_raw, test_gt_raw, test_all_starts, test_all_ends, _ , _, _ = get_preds_gt(model, test_loader, device)
       
        # Transform yolo coordinates onto image domain for both gt and predicted values
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
                                                       config['data']['height']))
        
        # Post Process all predictions
        post_test_preds = batch_postprocess_predictions(test_preds, 
                                                        spatial_threshold=config['post_process']['spatial_threshold'], 
                                                        temporal_threshold=config['post_process']['temporal_threshold'], 
                                                        confidence_threshold=config['post_process']['confidence_threshold'], 
                                                        strategy=config['post_process']['strategy'], 
                                                        mode=config['post_process']['mode'],
                                                        remove_outliers=config['post_process']['outlier_detection'],
                                                        outlier_method='isolation_forest')
        
        test_metrics = get_eval_metrics(test_preds, test_gts, 
                                        pos_thresholds=config['eval']['pos_thresholds'],
                                        iou_threshold_range=config['eval']['iou_thresholds'],
                                        angular_thresholds=config['eval']['angular_thresholds'])
        
        post_test_metrics = get_eval_metrics(post_test_preds, test_gts, 
                                        pos_thresholds=config['eval']['pos_thresholds'],
                                        iou_threshold_range=config['eval']['iou_thresholds'],
                                        angular_thresholds=config['eval']['angular_thresholds'])
        
        #print_evaluation_results(test_metrics, post_test_metrics)

        print(f"Epoch {epoch+1}/{config['train']['epochs']} | STD-F1 pre: {test_metrics['comprehensive']['f1']:.4f} | post: {post_test_metrics['comprehensive']['f1']:.4f}")       
        

        wandb.log({
            **train_log,
            **val_log,
            **get_wandb_log_dict(epoch, test_metrics, post_test_metrics)
            })

        # Restore original parameters after metrics
        ema.restore()

        if monitor_metric == 'loss':
            current_score = val_loss
            is_best = current_score < best_score
            score_str = f"val_loss: {current_score:.4f}"
        # else is std map    
        else:  
            # use post-processed comprehensive F1 as the primary score (higher is better)
            current_score = post_test_metrics['comprehensive']['f1']
            is_best = current_score > best_score
            score_str = f"STD-F1 (post-proc): {current_score:.4f}"

        if epoch % config['train']['val_freq'] == 0 or epoch == config['train']['epochs'] - 1:
            if is_best:
                best_score = current_score
                if config['train'].get('save_model', True):
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'scaler_state_dict': scaler.state_dict(),
                        'ema_state_dict': ema.state_dict(),
                        'best_score': best_score,
                        'val_loss': val_loss,
                        'std_f1': post_test_metrics['comprehensive']['f1'], 
                    }, os.path.join(ckpt_dir, 'best.pth'))
                    print(f"New best model saved → {score_str}")
                else:
                    print(f"New best → {score_str} (model saving disabled)")

            if config['train'].get('save_model', True):
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'ema_state_dict': ema.state_dict(),
                    'best_score': best_score,
                    'val_loss': val_loss,
                    'std_f1': post_test_metrics['comprehensive']['f1'], 
                }, os.path.join(ckpt_dir, 'latest.pth'))
                print(f"Saved latest checkpoint at epoch {epoch}/{config['train']['epochs']}")
            else:
                print(f"Evaluated model at epoch {epoch}/{config['train']['epochs']} (model saving disabled)")
            
    print('Training complete.')
    wandb.finish()

def get_args():
    parser = argparse.ArgumentParser(description="Waggle detection training")

    parser.add_argument("--config_path", type=str, default='./configs/config.yaml', help="Path to the config file.")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training from (e.g. ./ckpt/latest.pth).")
    parser.add_argument("--run_name", type=str, default=None, help="Name for this run. Defaults to datetime if not specified.")
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    main(args)