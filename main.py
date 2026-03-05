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
from src.data.dataset import TemporalWaggleCollator #, VideoYoloDataset
from src.data.dataset_tempaug import VideoYoloDataset
from src.models.model import R2Plus1D_YOLO_MultiHead
from src.loss.loss import WaggleDetectionLoss
from src.data.augmentation import WaggleAugmentations
from src.tests.aug_vis import demo_visualization
import torch.nn  as nn
from src.utils.data_utils import fix_dataframe_with_video_lengths, load_config, save_preds_to_csv
import datetime
import wandb
from src.utils.eval_utils import get_preds_gt, yolo_to_img_space, yolo_to_img_space_gt, get_eval_metrics, print_evaluation_results
from src.utils.nms import batch_postprocess_predictions
from src.utils.vis_utils import reverse_transform, save_frames
import argparse
from src.utils.model_utils import load_pretrained_model, EMA


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


    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        use_multi_gpu = True
    else:
        print("Using single GPU")
        use_multi_gpu = False

    data = pd.read_csv(config['data']['annotations'])

    print(f"Original dataset length: {len(data)}")
    # 1/8 of original data for fine-tuning
    data = data.iloc[:len(data)//16].reset_index(drop=True)
    #data = data.iloc[:100].reset_index(drop=True)
    print(f"After subsetting dataset: {len(data)} samples")
    
    video_frames_dict = create_video_frames_df(data["video_name"].unique())

    data = fix_dataframe_with_video_lengths(data, video_frames_dict)

    data = data.sample(frac=1).reset_index(drop=True)

    transforms = T.Compose([
        T.ToPILImage(),
        T.Resize((224, 224)),
        T.ToTensor(),
    ])

    test_transform = T.Compose([
        T.ToPILImage(),
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225])])
    
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
    
    total_len = len(data)
    train_len = int(0.8 * total_len)
    test_len = total_len - train_len

    train_indices = list(range(train_len))
    test_indices = list(range(train_len, total_len))

    train_df = data.iloc[train_indices].reset_index(drop=True)
    test_df = data.iloc[test_indices].reset_index(drop=True)

    # Create datasets
    train_dataset = VideoYoloDataset(
        train_df,
        config['data']['data_dir'],
        transforms,
        width=config['data']['width'],
        height=config['data']['height'],
        clip_len=config['data']['clip_len'],
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
        clip_len=config['data']['clip_len'],
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
        batch_size=config['train']['batch_size'],
        collate_fn=collator, 
        shuffle=False, 
        num_workers=config['train']['num_workers'],
        persistent_workers=(config['train']['num_workers'] > 0),
        pin_memory=True
    )

    print('Len train loader:', len(train_loader))

    model = R2Plus1D_YOLO_MultiHead(n_classes=config['model']['n_classes'],
                                    max_detections_per_cell=config['model']['max_detections_per_cell'], 
                                    grid_size=config['model']['grid_size'],
                                    transformer_heads=config['model']['transformer_heads'],
                                    transformer_layers=config['model']['transformer_layers']
                                    self_attention=config['model']['self_attention'],
                                    cross_attention=config['model']['cross_attention'],
                                    dropout_rate=config['model']['dropout']   
                                    )

    os.makedirs('./ckpt', exist_ok=True)

    model = model.to(device)

    if use_multi_gpu:
        model = nn.DataParallel(model)
        print("Using Distributed Data Parallel (DDP).")

    optimizer = optim.AdamW(model.parameters(), 
                            lr=config['train']['lr'], 
                            weight_decay=0.config['train']['weight_decay'], 
                            betas=config['train']['betas'])
    
    ema = EMA(model, decay=0.9999, device=device)

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
    best_val_loss = float('inf')

    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        if isinstance(model, torch.nn.DataParallel):
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']

        if 'ema_state_dict' in checkpoint:
            ema.load_state_dict(checkpoint['ema_state_dict'])
            print(f"Loaded EMA state (updates: {ema.updates})")

        print(f"Resumed at epoch {start_epoch}, best_val_loss so far: {best_val_loss:.4f}")
    elif args.resume:
        print(f"Warning: checkpoint path '{args.resume}' not found, starting from scratch.")

    # init wandb for logging
    wandb.init(
    project="waggle-detection",  # Project name
    config={
        **config,  # Log entire config
        "seed": SEED,
    },
    name=f"run_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}",
    resume="allow" if args.resume else None
)
    
    for epoch in range(start_epoch, config['train']['epochs']):
        print(f'\nEpoch {epoch+1}/{config["train"]["epochs"]}')
        # Train one epoch
        train_loss = train(
            model, device, optimizer, yolocriteria, scheduler, train_loader, epoch, scaler, ema)
        
        # Validate
        val_loss = eval(model, device, yolocriteria, test_loader, epoch, ema)

        """
        # fetch all raw logits
        test_preds_raw, test_gt_raw, test_all_starts, test_all_ends, _, test_frames = get_preds_gt(model, test_loader, device, return_frames=False)
        # transform yolo gt annotations to image domain
        test_gts = yolo_to_img_space_gt(test_gt_raw, all_starts=test_all_starts, all_ends=test_all_ends)
        # transform raw logits to img space and filter by confidence
        test_preds = yolo_to_img_space(test_preds_raw, all_starts=test_all_starts, all_ends=test_all_ends, confidence_threshold=config['eval']['confidence_threshold'], window_size = 16, original_size=(224,224))
        # get eval metrics
        test_metrics = get_eval_metrics(test_preds, test_gts, 
                                        pos_thresholds=config['eval']['pos_thresholds'],
                                        iou_threshold_range=config['eval']['iou_thresholds'],
                                        angular_thresholds=config['eval']['angular_thresholds'])
        
        # post process test preds
        post_test_preds = batch_postprocess_predictions(test_preds, 
                                                        spatial_threshold=config['post_process']['spatial_threshold'], 
                                                        temporal_threshold=config['post_process']['temporal_threshold'], 
                                                        confidence_threshold=config['post_process']['confidence_threshold'], 
                                                        strategy=config['post_process']['strategy'], 
                                                        mode=config['post_process']['mode'])
        
        post_test_metrics = get_eval_metrics(post_test_preds, test_gts, 
                                        pos_thresholds=config['eval']['pos_thresholds'],
                                        iou_threshold_range=config['eval']['iou_thresholds'],
                                        angular_thresholds=config['eval']['angular_thresholds'])

        save_preds_to_csv(post_test_preds, f'postprocessed_predictions_epoch_{epoch}.csv', 'postprocessed', './outputs/preds_csv')

        # Print unique clusters identifies
        unique_clusters_test_gt = len({det['cluster_id'] for seq in test_preds for det in seq})

        unique_clusters_test = len({det['cluster_id'] for seq in test_preds for det in seq})
        unique_clusters_test_post = len({det['cluster_id'] for seq in post_test_preds for det in seq})
        print('Number of clusterns - Ground Truth:', unique_clusters_test_gt)
        print('Number of clusterns - Before Postprocessing:', unique_clusters_test)
        print('Number of clusterns - After Postprocessing:', unique_clusters_test_post)

        print_evaluation_results(test_metrics, post_test_metrics)
        """

        # Restore original parameters after metrics
        ema.restore()

        if epoch % config['train']['val_freq'] == 0 or epoch == config['train']['epochs'] - 1:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if config['train'].get('save_model', True):
                    # Save best model
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'scaler_state_dict': scaler.state_dict(),
                        'ema_state_dict': ema.state_dict(),
                        'best_val_loss': best_val_loss,
                    }, './ckpt/best_model.pth')
                    print(f"New best model saved with val_loss: {val_loss:.4f}")
                else:
                    print(f"New best val_loss: {val_loss:.4f} (model saving disabled)")
            
            if config['train'].get('save_model', True):
                # Save latest checkpoint
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'ema_state_dict': ema.state_dict(),
                    'best_val_loss': best_val_loss,
                }, './ckpt/latest.pth')
                print(f'Saved and evaluated model at Epoch {epoch}/{config["train"]["epochs"]}')
            else:
                print(f'Evaluated model at Epoch {epoch}/{config["train"]["epochs"]} (model saving disabled)')
            
    print('Training complete.')
    wandb.finish()

def get_args():
    parser = argparse.ArgumentParser(description="Waggle detection training")

    parser.add_argument("--config_path", type=str, default='./configs/config.yaml', help="Path to the config file.")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training from (e.g. ./ckpt/latest.pth).")

    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    main(args)