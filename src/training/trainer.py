import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Optional, Dict, Any
import numpy as np

from .callbacks import EarlyStopping, ModelCheckpoint, EvaluateModel
from .metrics_logger import MetricsLogger
from src.utils import get_base_video_name, get_video_category

class WaggleTrainer:
    """
    Trainer class for waggle dance detection model.
    
    Args:
        model: PyTorch model
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler (optional)
        device: Device to train on
        log_dir: Directory for logs
        checkpoint_dir: Directory for checkpoints
    """
    
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        eval_config: dict,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = 'cuda',
        log_dir: str = './logs',
        checkpoint_dir: str = './checkpoints',
        experiment_name: Optional[str] = None,
        use_amp: bool = True
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = torch.device(device)
        self.use_amp = use_amp
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Initialize metrics logger
        self.logger = MetricsLogger(log_dir, experiment_name)
        
        # Initialize checkpoint manager
        self.checkpoint = ModelCheckpoint(
            checkpoint_dir,
            monitor='val_loss',
            mode='min',
            save_best_only=False,
            save_freq=10
        )
        self.eval_config = eval_config
        if self.eval_config is not None:
            self.eval = EvaluateModel(eval_config=eval_config)

        print(self.eval_config)
        
        # AMP scaler
        self.scaler = torch.amp.GradScaler() if use_amp else None
        
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
            Dictionary of average training metrics
        """
        self.model.train()
        torch.cuda.empty_cache()
        
        metrics = {
            'total_loss': 0.0,
            'obj_loss': 0.0,
            'noobj_loss': 0.0,
            'position_loss': 0.0,
            'direction_loss': 0.0,
            'temporal_loss': 0.0
        }
        
        progress_bar = tqdm(
            train_loader, 
            desc=f'Epoch {epoch}', 
            leave=False
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            inputs = batch['video'].to(self.device)
            targets = batch['targets'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.use_amp:
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    outputs = self.model(inputs)
                    loss_tuple = self.criterion(outputs, targets)
                
                total_loss = loss_tuple[0]
                self.scaler.scale(total_loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(inputs)
                loss_tuple = self.criterion(outputs, targets)
                total_loss = loss_tuple[0]
                total_loss.backward()
                self.optimizer.step()
            
            # Update scheduler
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Accumulate metrics
            metrics['total_loss'] += total_loss.item()
            metrics['obj_loss'] += loss_tuple[1].item()
            metrics['noobj_loss'] += loss_tuple[2].item()
            metrics['position_loss'] += loss_tuple[3].item()
            metrics['direction_loss'] += loss_tuple[4].item()
            metrics['temporal_loss'] += loss_tuple[5].item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f"{metrics['total_loss'] / (batch_idx + 1):.4f}"
            })
        
        # Average metrics
        num_batches = len(train_loader)
        for key in metrics:
            metrics[key] /= num_batches
        
        return metrics
    
    def validate_epoch(self, val_loader: DataLoader, epoch: int, run_evaluation: bool = False) -> Dict[str, float]:
        """
        Validate for one epoch.
        
        Returns:
            Dictionary of average validation metrics
        """
        self.model.eval()
        
        metrics = {
            'total_loss': 0.0,
            'obj_loss': 0.0,
            'noobj_loss': 0.0,
            'position_loss': 0.0,
            'direction_loss': 0.0,
            'temporal_loss': 0.0
        }

        if run_evaluation:
            all_outputs = []
            all_targets = []
            all_starts = []
            all_ends = []
            all_video_names = []
            all_original_res = []
            all_fps = []
        
        progress_bar = tqdm(
            val_loader,
            desc=f'Validation Epoch {epoch}',
            leave=False
        )
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(progress_bar):
                inputs = batch['video'].to(self.device)
                targets = batch['targets'].to(self.device)
                
                if self.use_amp:
                    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                        outputs = self.model(inputs)
                        loss_tuple = self.criterion(outputs, targets)
                else:
                    outputs = self.model(inputs)
                    loss_tuple = self.criterion(outputs, targets)
                
                # Accumulate metrics
                metrics['total_loss'] += loss_tuple[0].item()
                metrics['obj_loss'] += loss_tuple[1].item()
                metrics['noobj_loss'] += loss_tuple[2].item()
                metrics['position_loss'] += loss_tuple[3].item()
                metrics['direction_loss'] += loss_tuple[4].item()
                metrics['temporal_loss'] += loss_tuple[5].item()

                if run_evaluation:
                    all_outputs.append(outputs.cpu())
                    all_targets.append(targets.cpu())
                    
                    metadata = batch['metadata']
                    for meta in metadata:
                        all_starts.append(meta['start_frame'])
                        all_ends.append(meta['end_frame'])
                        all_video_names.append(meta['video_name'])
                        all_original_res.append(meta['res'])
                        all_fps.append(meta['fps'])
                
                progress_bar.set_postfix({
                    'Loss': f"{metrics['total_loss'] / (batch_idx + 1):.4f}"
                })
        
        # Average metrics
        num_batches = len(val_loader)
        for key in metrics:
            metrics[key] /= num_batches


        if run_evaluation and self.eval_config is not None:
            # Convert lists to tensors/arrays
            all_outputs = torch.cat(all_outputs, dim=0)
            all_targets = torch.cat(all_targets, dim=0)
            all_starts = np.array(all_starts)
            all_ends = np.array(all_ends)
            all_original_res = np.array(all_original_res)
            all_fps = np.array(all_fps)
            
            # Run evaluation pipeline with collected data
            self.eval.evaluatepipeline(
                all_outputs, all_targets, all_starts, all_ends, all_original_res, all_fps
            )
        
        return metrics
    
    def run_evaluation_on_val_set(self, val_loader):
        """
        Run evaluation on validation set with per-category breakdown.
        """
        self.model.eval()
        
        # Store results by category
        category_data = {
            '0': {'outputs': [], 'targets': [], 'starts': [], 'ends': [], 'video_names': [], 'res': [], 'fps': []},
            'T': {'outputs': [], 'targets': [], 'starts': [], 'ends': [], 'video_names': [], 'res': [], 'fps': []},
            'C': {'outputs': [], 'targets': [], 'starts': [], 'ends': [], 'video_names': [], 'res': [], 'fps': []},
            'other': {'outputs': [], 'targets': [], 'starts': [], 'ends': [], 'video_names': [], 'res': [], 'fps': []}
        }
        
        # Also store all data for overall evaluation
        all_outputs = []
        all_targets = []
        all_starts = []
        all_ends = []
        all_video_names = []
        all_original_res = []
        all_fps = []
        
        progress_bar = tqdm(
            val_loader,
            desc=f'Evaluation run on validation set',
            leave=False
        )
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(progress_bar):
                inputs = batch['video'].to(self.device)
                targets = batch['targets'].to(self.device)
                
                if self.use_amp:
                    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                        outputs = self.model(inputs)
                else:
                    outputs = self.model(inputs)

                # Store overall data
                all_outputs.append(outputs.cpu())
                all_targets.append(targets.cpu())
                
                # Process each sample in batch and categorize
                metadata = batch['metadata']
                for i, meta in enumerate(metadata):
                    video_name = meta['video_name']
                    
                    # Get base video name and category
                    base_video = get_base_video_name(video_name)
                    category = get_video_category(base_video)

                    
                    # Store in category-specific lists
                    category_data[category]['outputs'].append(outputs[i].cpu().unsqueeze(0))
                    category_data[category]['targets'].append(targets[i].cpu().unsqueeze(0))
                    category_data[category]['starts'].append(meta['start_frame'])
                    category_data[category]['ends'].append(meta['end_frame'])
                    category_data[category]['video_names'].append(video_name)
                    category_data[category]['res'].append(meta['res'])
                    category_data[category]['fps'].append(meta['fps'])
                    
                    # Store in overall lists
                    all_starts.append(meta['start_frame'])
                    all_ends.append(meta['end_frame'])
                    all_video_names.append(video_name)
                    all_original_res.append(meta['res'])
                    all_fps.append(meta['fps'])

        # Convert overall lists to tensors/arrays
        all_outputs = torch.cat(all_outputs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        all_starts = np.array(all_starts)
        all_ends = np.array(all_ends)
        all_original_res = np.array(all_original_res)
        all_fps = np.array(all_fps)
        
        print(f"\n{'='*70}")
        print(f"Overall Evaluation (All Categories)")
        print(f"{'='*70}")
        
        # Run overall evaluation
        overall_metrics = self.eval.evaluatepipeline(
            all_outputs, all_targets, all_starts, all_ends, all_original_res, all_fps
        )
        
        # Evaluate each category separately
        category_metrics = {}
        
        for category in ['0', 'T', 'C', 'other']:
            cat_data = category_data[category]
            
            # Skip if category has no samples
            if len(cat_data['outputs']) == 0:
                print(f"\nCategory '{category}': No samples in validation set")
                continue
            
            # Convert category data to tensors/arrays
            cat_outputs = torch.cat(cat_data['outputs'], dim=0)
            cat_targets = torch.cat(cat_data['targets'], dim=0)
            cat_starts = np.array(cat_data['starts'])
            cat_ends = np.array(cat_data['ends'])
            cat_res = np.array(cat_data['res'])
            cat_fps = np.array(cat_data['fps'])
            
            print(f"\n{'='*70}")
            print(f"Category '{category}' Evaluation")
            print(f"{'='*70}")
            print(f"Number of samples: {len(cat_outputs)}")
            print(f"Number of unique videos: {len(set(cat_data['video_names']))}")
            
            # Run evaluation for this category
            cat_metrics = self.eval.evaluatepipeline(
                cat_outputs, cat_targets, cat_starts, cat_ends, cat_res, cat_fps
            )
            
            category_metrics[category] = cat_metrics
        
        # Print summary comparison
        print(f"\n{'='*70}")
        print(f"Category-wise Summary")
        print(f"{'='*70}")
        
        for category, metrics in category_metrics.items():
            cat_data = category_data[category]
            print(f"\n{category:10s}: {len(cat_data['outputs']):4d} samples, "
                f"{len(set(cat_data['video_names'])):3d} videos")
            
            if metrics and 'comprehensive' in metrics:
                print(f"  Precision: {metrics['comprehensive']['precision']:.4f}")
                print(f"  Recall: {metrics['comprehensive']['recall']:.4f}")
                print(f"  F1-Score: {metrics['comprehensive']['f1']:.4f}")
        
        return {
            'overall': overall_metrics,
            'by_category': category_metrics
        }
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 100,
        only_eval: bool = False,
        start_epoch: int = 0,
        early_stopping_patience: Optional[int] = None,
        val_freq: int = 1
    ):
        """
        Train the model for multiple epochs.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            start_epoch: Epoch to start/resume from (0-indexed)
            num_epochs: Number of epochs to train
            early_stopping_patience: Enable early stopping (None to disable)
            val_freq: Validate every N epochs
        """
        # Setup early stopping
        early_stopper = None
        if early_stopping_patience is not None:
            early_stopper = EarlyStopping(
                patience=early_stopping_patience,
                min_delta=0.001
            )
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_epochs': [] 
        }

        if only_eval:
            results = self.run_evaluation_on_val_set(val_loader)

            # Access results
            overall_metrics = results['overall']
            category_0_metrics = results['by_category'].get('0')
            t_metrics = results['by_category'].get('T')

            import json
            with open('category_evaluation_results.json', 'w') as f:
                json.dump({
                    'overall': overall_metrics,
                    'categories': results['by_category']
                }, f, indent=2)
            print("Evaluation Complete!")
            exit(0)
        
        print(f"\n{'='*60}")
        print(f"Starting Training")
        print(f"{'='*60}")
        print(f"Epochs: {start_epoch + 1} to {num_epochs}")
        print(f"Device: {self.device}")
        print(f"AMP: {self.use_amp}")
        if start_epoch > 0:
            print(f"Resuming from epoch: {start_epoch + 1}")
        print(f"{'='*60}\n")
        
        for epoch in range(start_epoch + 1, num_epochs + 1):
            # Train
            train_metrics = self.train_epoch(train_loader, epoch)
            history['train_loss'].append(train_metrics['total_loss'])
            
            # Log training metrics
            self.logger.log_loss_components(epoch, 'Train', **train_metrics)
            
            if self.scheduler is not None:
                current_lr = self.optimizer.param_groups[0]['lr']
                self.logger.log_learning_rate(epoch, current_lr)
            
            # Print epoch summary
            print(f"Epoch [{epoch}/{num_epochs}] - "
                  f"Train Loss: {train_metrics['total_loss']:.4f}")
            
            # Validate
            if epoch % val_freq == 0  and epoch <= num_epochs:
                run_eval = self.eval_config is not None
                val_metrics = self.validate_epoch(val_loader, epoch, run_evaluation=run_eval)

                history['val_loss'].append(val_metrics['total_loss'])
                history['val_epochs'].append(epoch)
                
                # Log validation metrics
                self.logger.log_loss_components(epoch, 'Validation', **val_metrics)
                
                # Print validation summary
                print(f"  Val Loss: {val_metrics['total_loss']:.4f} | "
                      f"Obj: {val_metrics['obj_loss']:.4f} | "
                      f"NoObj: {val_metrics['noobj_loss']:.4f}")
                
                # Save checkpoint
                self.checkpoint.save(
                    self.model,
                    self.optimizer,
                    self.scheduler,
                    epoch,
                    val_metrics['total_loss']
                )
                
                # # Check early stopping
                if early_stopper is not None:
                    if early_stopper(val_metrics['total_loss']):
                        print(f"\n⚠ Early stopping triggered at epoch {epoch}")
                        break
        
        print(f"\n{'='*60}")
        print("Training Completed!")
        print(f"{'='*60}\n")
        
        self.logger.close()
        return history