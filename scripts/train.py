import os
import argparse
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
from src.utils import set_seed, load_checkpoint, split_by_video_groups, create_video_based_folds


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train Waggle Dance Detection Model')
    
    # Data
    parser.add_argument('--data-dir', type=str, default='./data/resvideos',
                       help='Directory containing video data')
    parser.add_argument('--annotations', type=str, 
                       default='./data/annotations/fps_multires_full_data.csv',
                       help='Path to annotations CSV')

    # Cross-Validation
    parser.add_argument('--k-folds', type=int, default=None,
                       help='Number of folds for cross-validation (None for single split)')
    parser.add_argument('--train-ratio', type=float, default=0.75,
                       help='Train/val split ratio (used if k-folds is None)')
    
    # Model
    parser.add_argument('--pretrained', action='store_true', default=True,
                       help='Use pretrained backbone')
    
    # Training
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=200,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=3e-4,
                       help='Initial learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-3,
                       help='Weight decay')
    parser.add_argument('--val-freq', type=int, default=5,
                       help='Validation Frequency')
    parser.add_argument('--subset-size', type=float, default=1,
                       help='Subset size 0.1 means 10%')
    
    # Augmentation
    parser.add_argument('--aug-prob', type=float, default=0.5,
                       help='Augmentation probability')
    parser.add_argument('--scale-range', nargs=2, type=float, default=[0.8, 1.2],
                       help='Scale augmentation range')
    
    # System
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--num-workers', type=int, default=8,
                       help='Number of data loading workers')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use for training')
    
    # Logging
    parser.add_argument('--log-dir', type=str, default='./logs',
                       help='Directory for logs')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints',
                       help='Directory for checkpoints')
    parser.add_argument('--experiment-name', type=str, default=None,
                       help='Name of the experiment')
    

    # Evaluation
    parser.add_argument('--eval',action='store_true',default=False,
                       help='Enable evaluation with thresholds')
    parser.add_argument('--pos-thresholds',nargs='+', type=float, default=[20],
                       help='List of positional thresholds')
    parser.add_argument('--iou-thresholds',nargs='+', type=float, default=[0.5],
                       help='List of IoU thresholds')
    parser.add_argument('--angular-thresholds',nargs='+', type=float, default=[15],
                       help='List of angular thresholds')
    parser.add_argument('--confidence-threshold', type=float, default=0.5,
                       help='Confidence threshold to filter predictions')
    parser.add_argument('--only-eval',action='store_true',default=False,
                       help='Enable evaluation with thresholds without training')
    parser.add_argument('--consolidation-strategy', type=str, 
                        help='cluster_consolidate_v2, cluster_consolidate, nms_spatiotemporal, nms, max, weighted, cluster_max, dbscan')
    
    # Resume
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    return parser.parse_args()


def create_data_loaders(train_df, val_df, args):
    """Create train and validation data loaders from DataFrames."""
    
    # Create transforms
    transforms = T.Compose([
        T.ToPILImage(),
        T.ToTensor(),
        # T.Resize((224,224)),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create augmentation
    train_augmentation = WaggleAugmentations(
        width=224,
        height=224,
        prob=args.aug_prob,
        scale_range=tuple(args.scale_range),
        mutual_exclusive=True
    )
    
    # Create datasets
    train_dataset = VideoYoloDataset(
        train_df,
        args.data_dir,
        transforms,
        width=224,
        height=224,
        clip_len=16,
        grid_size=28,
        max_detections_per_cell=1,
        num_classes=1,
        augment=train_augmentation if args.aug_prob > 0 else None,
        is_training=True
    )
    
    val_dataset = VideoYoloDataset(
        val_df,
        args.data_dir,
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
        batch_size=args.batch_size,
        collate_fn=collator,
        shuffle=True,
        num_workers=args.num_workers,
        persistent_workers=True if args.num_workers > 0 else False,
        prefetch_factor=2 if args.num_workers > 0 else None,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        collate_fn=collator,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False,
        prefetch_factor=2 if args.num_workers > 0 else None,
        drop_last=False
    )
    
    return train_loader, val_loader

def train_single_fold(fold_idx, train_df, val_df, args, device, eval_config):
    """
    Train model on a single fold.
    
    Args:
        fold_idx: Current fold index (0-indexed, None for single split)
        train_df: Training DataFrame
        val_df: Validation DataFrame
        args: Command line arguments
        device: Torch device
        
    Returns:
        dict with fold results
    """
    # Determine fold name for logging
    if fold_idx is not None:
        fold_name = f"fold_{fold_idx + 1}"
        print(f"\n{'='*60}")
        print(f"Training Fold {fold_idx + 1}/{args.k_folds}")
        print(f"{'='*60}\n")
    else:
        fold_name = "single_split"
        print(f"\n{'='*60}")
        print(f"Training Single Split")
        print(f"{'='*60}\n")
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(train_df, val_df, args)

    print(train_df['video_name'].unique(), "val", val_df["video_name"].unique())
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches:   {len(val_loader)}")
    
    # Create model
    model = R2Plus1D_YOLO(
        pretrained=args.pretrained
    )

    use_multi_gpu = False
    # Multi-GPU support
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
        use_multi_gpu = True

    # Move model to device BEFORE creating optimizer
    model = model.to(device)

    # Calculate learning rate
    if use_multi_gpu:
        scaled_max_lr = args.lr * (args.batch_size / 8)
    else:
        scaled_max_lr = args.lr * (args.batch_size / 8)

    print(f"Using learning rate: {scaled_max_lr}")
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=scaled_max_lr / 25,
        weight_decay=args.weight_decay
    )
    
    # Initialize scheduler (will be recreated if resuming)
    scheduler = None
    start_epoch = 0
    resumed_val_loss = None
    
    # Resume from checkpoint if specified (only for single split)
    if args.resume is not None and fold_idx is None and not args.eval:
        print(f"\n{'='*60}")
        print(f"RESUMING FROM CHECKPOINT")
        print(f"{'='*60}")
        print(f"Checkpoint: {args.resume}")
        
        # Load checkpoint (model and optimizer only, not scheduler)
        metadata = load_checkpoint(
            args.resume, 
            model, 
            optimizer, 
            scheduler=None, 
            device=device, 
            load_scheduler=False
        )
        start_epoch = metadata['epoch']
        resumed_val_loss = metadata['metric_value']
        
        # Calculate remaining epochs
        remaining_epochs = args.epochs - start_epoch
        
        if remaining_epochs <= 0:
            print(f"\n⚠ Warning: No epochs remaining to train!")
            print(f"   Start epoch: {start_epoch}, Target epochs: {args.epochs}")
            print(f"   Please set --epochs to a value > {start_epoch}")
            # Return early or handle as needed
        
        # FINE-TUNING MODE: Create new scheduler for remaining epochs
        print(f"\n🎯 Fine-tuning Configuration:")
        print(f"   Starting from epoch: {start_epoch + 1}")
        print(f"   Target epoch: {args.epochs}")
        print(f"   Remaining epochs: {remaining_epochs}")
        
        # Use lower learning rate for fine-tuning
        finetune_max_lr = scaled_max_lr  # 30% of original
        print(f"   Original max_lr: {scaled_max_lr:.6f}")
        print(f"   Fine-tune max_lr: {finetune_max_lr:.6f}")
        
        # Create new scheduler optimized for fine-tuning
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=finetune_max_lr,
            steps_per_epoch=len(train_loader),
            epochs=remaining_epochs,
            pct_start=0.2,      
            div_factor=10,     
            final_div_factor=100,  
            anneal_strategy='cos'
        )
        print(f"   Initial LR: {finetune_max_lr/10:.6f}")
        print(f"   Peak LR: {finetune_max_lr:.6f}")
        print(f"   Final LR: {finetune_max_lr/100:.6f}")
        print(f"{'='*60}\n")
        
    elif not args.eval:
        # FRESH TRAINING: Create normal scheduler
        print(f"\n{'='*60}")
        print(f"STARTING FRESH TRAINING")
        print(f"{'='*60}\n")
        
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=scaled_max_lr,
            steps_per_epoch=len(train_loader),
            epochs=args.epochs,
            pct_start=0.3,
            div_factor=25,
            final_div_factor=1000,
            anneal_strategy='cos'
        )
    else: 
        print(f"Loaded Checkpoint : {args.resume} for evaluation")
        print(f"{'='*60}\n")
        metadata = load_checkpoint(
            args.resume, 
            model, 
            optimizer, 
            scheduler=None, 
            device=device, 
            load_scheduler=False
        )

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=scaled_max_lr,
            steps_per_epoch=len(train_loader),
            epochs=args.epochs,
            pct_start=0.3,
            div_factor=25,
            final_div_factor=1000,
            anneal_strategy='cos'
        )
    
    # Create loss with adjusted weights if specified
    if hasattr(args, 'lambda_obj'):
        criterion = WaggleDetectionLoss(
            lambda_obj=args.lambda_obj,
            lambda_noobj=args.lambda_noobj,
            lambda_pos=args.lambda_pos,
            lambda_dir=args.lambda_dir,
            lambda_temporal=args.lambda_temporal
        )
        print(f"Loss weights:")
        print(f"  lambda_obj: {args.lambda_obj}")
        print(f"  lambda_noobj: {args.lambda_noobj}")
        print(f"  lambda_pos: {args.lambda_pos}")
        print(f"  lambda_dir: {args.lambda_dir}")
        print(f"  lambda_temporal: {args.lambda_temporal}\n")
    else:
        # Use default weights
        criterion = WaggleDetectionLoss()
        print(f"Using default loss weights\n")
    
    # Create fold-specific directories
    if fold_idx is not None:
        log_dir = os.path.join(args.log_dir, fold_name)
        checkpoint_dir = os.path.join(args.checkpoint_dir, fold_name)
    else:
        log_dir = args.log_dir
        checkpoint_dir = args.checkpoint_dir
    
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Create trainer
    trainer = WaggleTrainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        eval_config=eval_config,
        scheduler=scheduler,
        device=device,
        log_dir=log_dir,
        checkpoint_dir=checkpoint_dir,
        experiment_name=args.experiment_name or fold_name,
        use_amp=True
    )
    
    # Train
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        only_eval = args.only_eval,
        start_epoch=start_epoch,
        early_stopping_patience=args.early_stopping if hasattr(args, 'early_stopping') else None,
        val_freq=args.val_freq
    )
    
    # Extract best results
    if history['val_loss']:  # If we have validation losses from this run
        best_val_loss = min(history['val_loss'])
        # Find the actual epoch number (accounting for start_epoch)
        best_idx = history['val_loss'].index(best_val_loss)
        best_epoch = history['val_epochs'][best_idx] if history['val_epochs'] else None
        
        # If we resumed, compare with previous best
        if resumed_val_loss is not None:
            if resumed_val_loss < best_val_loss:
                best_val_loss = resumed_val_loss
                best_epoch = start_epoch
    else:
        # No validation in this run, use resumed value
        best_val_loss = resumed_val_loss
        best_epoch = start_epoch if resumed_val_loss is not None else None

    return {
        'fold_idx': fold_idx,
        'fold_name': fold_name,
        'best_val_loss': best_val_loss,
        'best_epoch': best_epoch,
        'train_videos': len(train_df['base_video'].unique()),
        'val_videos': len(val_df['base_video'].unique()),
        'train_clips': len(train_df),
        'val_clips': len(val_df),
        'history': history,
        'resumed_from_epoch': start_epoch if args.resume else None
    }


def run_cross_validation(data, args, device, eval_config):
    """
    Run K-fold cross-validation.
    
    Args:
        data: Full DataFrame
        args: Command line arguments
        device: Torch device
        
    Returns:
        dict with aggregated results
    """
    # Create folds
    fold_splits = create_video_based_folds(
        data,
        n_splits=args.k_folds,
        shuffle=True,
        random_seed=args.seed
    )
    
    # Train each fold
    all_results = []
    
    for fold_idx, (train_df, val_df) in enumerate(fold_splits):
        fold_result = train_single_fold(
            fold_idx=fold_idx,
            train_df=train_df,
            val_df=val_df,
            args=args,
            device=device
        )
        all_results.append(fold_result)
        
        # Save intermediate results
        results_path = os.path.join(args.checkpoint_dir, 'cv_intermediate_results.json')
        with open(results_path, 'w') as f:
            json.dump({
                'completed_folds': fold_idx + 1,
                'total_folds': args.k_folds,
                'results': all_results
            }, f, indent=2)
    
    # Compute summary statistics
    val_losses = [r['best_val_loss'] for r in all_results if r['best_val_loss'] is not None]
    
    summary = {
        'experiment_name': args.experiment_name or f'cv_{args.k_folds}fold',
        'timestamp': datetime.now().isoformat(),
        'k_folds': args.k_folds,
        'completed_folds': len(all_results),
        'statistics': {
            'mean_val_loss': float(np.mean(val_losses)),
            'std_val_loss': float(np.std(val_losses)),
            'min_val_loss': float(np.min(val_losses)),
            'max_val_loss': float(np.max(val_losses)),
            'median_val_loss': float(np.median(val_losses))
        },
        'fold_results': all_results,
        'config': vars(args)
    }
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"CROSS-VALIDATION SUMMARY")
    print(f"{'='*60}")
    print(f"Folds completed: {len(all_results)}/{args.k_folds}")
    print(f"\nValidation Loss Statistics:")
    print(f"  Mean:   {summary['statistics']['mean_val_loss']:.4f} ± {summary['statistics']['std_val_loss']:.4f}")
    print(f"  Median: {summary['statistics']['median_val_loss']:.4f}")
    print(f"  Best:   {summary['statistics']['min_val_loss']:.4f}")
    print(f"  Worst:  {summary['statistics']['max_val_loss']:.4f}")
    print(f"\nPer-Fold Results:")
    for result in all_results:
        print(f"  Fold {result['fold_idx'] + 1}: {result['best_val_loss']:.4f} (epoch {result['best_epoch']})")
    print(f"{'='*60}\n")
    
    # Save final summary
    summary_path = os.path.join(args.checkpoint_dir, 'cv_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✓ Results saved to: {summary_path}")
    
    return summary


def main():
    """Main training function with optional cross-validation."""
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load data
    print(f"\nLoading annotations from: {args.annotations}")
    data = pd.read_csv(args.annotations)
    print(f"Loaded {len(data)} annotations")
    
    # Create base directories
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    eval_config=None

    def parse_list(value):
        return [float(v.strip()) for v in value.split(',')]

    if args.eval:
        eval_config = {
        'pos_thresholds': args.pos_thresholds,
        'iou_threshold_range': tuple(args.iou_thresholds),
        'angular_thresholds': args.angular_thresholds,
        'confidence_threshold': args.confidence_threshold,
        'strategy': args.consolidation_strategy
        }
    
    # Determine training mode
    if args.k_folds is not None and args.k_folds > 1:
        # Cross-validation mode
        print(f"\n{'='*60}")
        print(f"Running {args.k_folds}-Fold Cross-Validation")
        print(f"{'='*60}")
        
        summary = run_cross_validation(data, args, device, eval_config)
        
    else:
        # Single split mode
        print(f"\n{'='*60}")
        print(f"Running Single Train/Val Split")
        print(f"{'='*60}")
        
        # Create single split
        train_df, val_df = split_by_video_groups(
            data,     
            subset_size=args.subset_size,                                  
            train_ratio=args.train_ratio,
            seed=args.seed
        )
        
        # Train
        result = train_single_fold(
            fold_idx=None,
            train_df=train_df,
            val_df=val_df,
            args=args,
            device=device,
            eval_config=eval_config
        )
        
        # Save results
        results_path = os.path.join(args.checkpoint_dir, 'training_results.json')
        with open(results_path, 'w') as f:
            json.dump({
                'experiment_name': args.experiment_name or 'single_split',
                'timestamp': datetime.now().isoformat(),
                'result': result,
                'config': vars(args)
            }, f, indent=2)
        
        print(f"\n✓ Training complete!")
        print(f"✓ Best validation loss: {result['best_val_loss']:.4f} (epoch {result['best_epoch']})")
        print(f"✓ Results saved to: {results_path}")


if __name__ == '__main__':
    main()