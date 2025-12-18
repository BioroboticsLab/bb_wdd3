import os
import yaml
import torch
import torch.nn as nn
import pandas as pd
import torchvision.transforms as T
from torch.utils.data import DataLoader
import json
from datetime import datetime
import numpy as np

# Import from your modular structure
from src.models import R2Plus1D_YOLO
from src.losses import WaggleDetectionLoss
from src.data import VideoYoloDataset, TemporalWaggleCollator, WaggleAugmentations
from src.training import WaggleTrainer
from src.utils import set_seed, load_checkpoint, split_by_video_groups


def load_config(config_path='config.yaml'):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_data_loaders(train_df, val_df, config):
    """Create train and validation data loaders from DataFrames."""
    
    # Create transforms
    transforms = T.Compose([
        T.ToPILImage(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create augmentation
    train_augmentation = WaggleAugmentations(
        width=224,
        height=224,
        prob=config['augmentation']['prob'],
        scale_range=tuple(config['augmentation']['scale_range']),
        mutual_exclusive=True
    )
    
    # Create datasets
    train_dataset = VideoYoloDataset(
        train_df,
        config['data']['data_dir'],
        transforms,
        width=224,
        height=224,
        clip_len=16,
        grid_size=28,
        max_detections_per_cell=1,
        num_classes=1,
        augment=train_augmentation if config['augmentation']['prob'] > 0 else None,
        is_training=True
    )
    
    val_dataset = VideoYoloDataset(
        val_df,
        config['data']['data_dir'],
        transforms,
        width=224,
        height=224,
        clip_len=16,
        grid_size=28,
        max_detections_per_cell=1,
        num_classes=1,
        augment=None,
        is_training=False 
    )
    
    # Create data loaders
    collator = TemporalWaggleCollator()
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        collate_fn=collator,
        shuffle=True,
        num_workers=config['system']['num_workers'],
        persistent_workers=True if config['system']['num_workers'] > 0 else False,
        prefetch_factor=2 if config['system']['num_workers'] > 0 else None,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        collate_fn=collator,
        shuffle=False,
        num_workers=config['system']['num_workers'],
        pin_memory=True,
        persistent_workers=True if config['system']['num_workers'] > 0 else False,
        prefetch_factor=2 if config['system']['num_workers'] > 0 else None,
        drop_last=False
    )
    
    return train_loader, val_loader


def train(train_df, val_df, config, device):
    """
    Train model on a single train/val split.
    
    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        config: Configuration dictionary
        device: Torch device
        
    Returns:
        dict with training results
    """
    print(f"\n{'='*60}")
    print(f"Training Single Split")
    print(f"{'='*60}\n")
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(train_df, val_df, config)

    print("Train videos:", train_df['video_name'].unique())
    print("Val videos:", val_df["video_name"].unique())
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches:   {len(val_loader)}")
    
    # Create model
    model = R2Plus1D_YOLO(
        pretrained=config['model']['pretrained']
    )

    use_multi_gpu = False
    # Multi-GPU support
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
        use_multi_gpu = True

    # Move model to device
    model = model.to(device)

    # Create optimizer - use learning rate from config
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['lr'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Initialize training state
    start_epoch = 0
    scheduler = None
    
    # Resume from checkpoint if specified
    if config['resume']['checkpoint_path'] is not None:
        print(f"\n{'='*60}")
        print(f"Resume training from checkpoint.")
        print(f"{'='*60}")
        print(f"Checkpoint: {config['resume']['checkpoint_path']}")
        
        # Load checkpoint
        metadata = load_checkpoint(
            config['resume']['checkpoint_path'], 
            model, 
            optimizer, 
            scheduler=None,  # create scheduler below
            device=device, 
            load_scheduler=config['resume']['load_scheduler']
        )
        start_epoch = metadata['epoch']
        
        # Warn if no epochs remaining
        remaining_epochs = config['training']['epochs'] - start_epoch
        if remaining_epochs <= 0 and not config['evaluation']['only_eval']:
            print(f"\n Warning: No epochs remaining to train.")
            print(f"   Start epoch: {start_epoch}, Target epochs: {config['training']['epochs']}")
            print(f"   Please set epochs to a value > {start_epoch}")
    
    # Create OneCycleLR scheduler for training (warmup + cosine annealing)
    if not config['evaluation']['only_eval']:

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config['training']['lr'],
            steps_per_epoch=len(train_loader),
            epochs=config['training']['epochs'],
            pct_start=config['training']['warmup_ratio'],
            div_factor=25,
            final_div_factor=100,
            anneal_strategy='cos'
        )
        
        print(f"\nUsing OneCycleLR Scheduler (Warmup + Cosine Annealing):")
        print(f"  Max LR: {config['training']['lr']:.6f}")
        print(f"  Initial LR: {config['training']['lr']/25.0:.6f}")
        print(f"  Final LR: {config['training']['lr']/2500.0:.6f}")  # 25.0 * 100.0
        print(f"  Warmup ratio: {config['training']['warmup_ratio']:.2%}")
        print(f"  Total epochs: {config['training']['epochs']}")
        print(f"  Starting from epoch: {start_epoch + 1}")
    
    # Create loss with adjusted weights if specified
    loss_weights = config.get('loss_weights', {})
    if any(loss_weights.values()):
        criterion = WaggleDetectionLoss(
            lambda_obj=loss_weights.get('lambda_obj'),
            lambda_noobj=loss_weights.get('lambda_noobj'),
            lambda_pos=loss_weights.get('lambda_pos'),
            lambda_dir=loss_weights.get('lambda_dir'),
            lambda_temporal=loss_weights.get('lambda_temporal')
        )
        print(f"Loss weights:")
        for key, value in loss_weights.items():
            if value is not None:
                print(f"  {key}: {value}")

    else:
        # Use default weights
        criterion = WaggleDetectionLoss()
        print(f"Using default loss weights\n")
    
    # Create evaluation config if enabled
    eval_config = None
    if config['evaluation']['enabled']:
        eval_config = {
            'pos_thresholds': config['evaluation']['pos_thresholds'],
            'iou_threshold_range': tuple(config['evaluation']['iou_thresholds']),
            'angular_thresholds': config['evaluation']['angular_thresholds'],
            'confidence_threshold': config['evaluation']['confidence_threshold'],
            'strategy': config['evaluation']['consolidation_strategy']
        }
    
    # Create trainer
    trainer = WaggleTrainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        eval_config=eval_config,
        scheduler=scheduler,
        device=device,
        log_dir=config['logging']['log_dir'],
        checkpoint_dir=config['logging']['checkpoint_dir'],
        experiment_name=config['logging']['experiment_name'],
        use_amp=config['training']['use_amp']
    )
    
    # Train or evaluate
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config['training']['epochs'],
        only_eval=config['evaluation']['only_eval'],
        start_epoch=start_epoch,
        early_stopping_patience=config['early_stopping'].get('patience') if 'early_stopping' in config else None,
        val_freq=config['training']['val_freq']
    )
    
    # Extract best results
    best_val_loss = None
    best_epoch = None
    
    if history['val_loss']:
        best_val_loss = min(history['val_loss'])
        best_idx = history['val_loss'].index(best_val_loss)
        best_epoch = history['val_epochs'][best_idx] if history['val_epochs'] else None

    return {
        'best_val_loss': best_val_loss,
        'best_epoch': best_epoch,
        'train_videos': len(train_df['base_video'].unique()),
        'val_videos': len(val_df['base_video'].unique()),
        'train_clips': len(train_df),
        'val_clips': len(val_df),
        'history': history,
        'resumed_from_epoch': start_epoch if config['resume']['checkpoint_path'] else None
    }


def main(config_path='config.yaml'):
    """Main training function."""
    # Load configuration
    config = load_config(config_path)
    
    # Set random seed
    set_seed(config['system']['seed'])
    
    # Setup device
    device_str = config['system']['device'] if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_str)
    print(f"\nUsing device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load data
    annotations_path = config['data']['annotations']
    print(f"\nLoading annotations from: {annotations_path}")
    data = pd.read_csv(annotations_path)
    print(f"Loaded {len(data)} annotations")
    
    # Create base directories
    os.makedirs(config['logging']['log_dir'], exist_ok=True)
    os.makedirs(config['logging']['checkpoint_dir'], exist_ok=True)
    
    # Single split mode
    print(f"\n{'='*60}")
    print(f"Running Single Train/Val Split")
    print(f"{'='*60}")
    
    # Create single split
    train_df, val_df = split_by_video_groups(
        data,     
        subset_size=config['data']['subset_size'],                                  
        train_ratio=config['data']['train_ratio'],
        seed=config['system']['seed']
    )
    
    # Train
    result = train(
        train_df=train_df,
        val_df=val_df,
        config=config,
        device=device
    )
    
    # Save results
    results_path = os.path.join(config['logging']['checkpoint_dir'], 'training_results.json')
    with open(results_path, 'w') as f:
        json.dump({
            'experiment_name': config['logging']['experiment_name'],
            'timestamp': datetime.now().isoformat(),
            'result': result,
            'config': config
        }, f, indent=2)
    
    print(f"\n Training complete!")
    print(f" Best validation loss: {result['best_val_loss']:.4f} (epoch {result['best_epoch']})")
    print(f" Results saved to: {results_path}")
    
    # Save config used for this run
    config_copy = config.copy()
    if config_copy['resume']['checkpoint_path']:
        config_copy['resume']['checkpoint_path'] = str(config_copy['resume']['checkpoint_path'])
    
    config_path = os.path.join(config['logging']['checkpoint_dir'], 'config_used.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config_copy, f, default_flow_style=False)
    
    print(f"✓ Configuration saved to: {config_path}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Waggle Dance Detection Model')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to configuration YAML file')
    args = parser.parse_args()
    
    main(config_path=args.config)