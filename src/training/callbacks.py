from typing import Optional
import torch
import torch.nn as nn
import os
from ..evaluation import batch_postprocess_predictions,get_eval_metrics, print_evaluation_results, get_detailed_metrics, print_comparison_summary, get_detailed_metrics_silent, print_clean_comparison
from ..utils import reverse_transform_batch, yolo_to_img_space,yolo_to_img_space_gt

class EarlyStopping:
    """
    Early stopping callback to stop training when validation loss stops improving.
    
    Args:
        patience: Number of epochs to wait before stopping
        min_delta: Minimum change in loss to qualify as improvement
        mode: 'min' for loss, 'max' for accuracy
    """
    
    def __init__(self, patience: int = 5, min_delta: float = 0.0, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.early_stop = False
        
        if mode == 'min':
            self.best_score = float('inf')
            self.is_better = lambda new, best: new < best - min_delta
        else:
            self.best_score = float('-inf')
            self.is_better = lambda new, best: new > best + min_delta
    
    def __call__(self, metric: float) -> bool:
        """
        Update early stopping state.
        
        Args:
            metric: Current metric value
            
        Returns:
            True if training should stop
        """
        if self.is_better(metric, self.best_score):
            self.best_score = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


class ModelCheckpoint:
    """
    Save model checkpoints based on validation metrics.
    
    Args:
        checkpoint_dir: Directory to save checkpoints
        monitor: Metric to monitor ('val_loss', 'val_acc', etc.)
        mode: 'min' or 'max'
        save_best_only: Only save when metric improves
        save_freq: Save every N epochs (if save_best_only=False)
        
    """
    
    def __init__(
        self,
        checkpoint_dir: str,
        monitor: str = 'val_loss',
        mode: str = 'min',
        save_best_only: bool = True,
        save_freq: int = 10
    ):
        self.checkpoint_dir = checkpoint_dir
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_freq = save_freq
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        if mode == 'min':
            self.best_score = float('inf')
            self.is_better = lambda new, best: new < best
        else:
            self.best_score = float('-inf')
            self.is_better = lambda new, best: new > best
    
    def save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        epoch: int,
        metric_value: float,
        **extra_state
    ):
        """Save checkpoint if conditions are met."""
        
        # Extract state dict (handle DataParallel)
        if isinstance(model, nn.DataParallel):
            model_state = model.module.state_dict()
        else:
            model_state = model.state_dict()
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_state,
            'optimizer_state_dict': optimizer.state_dict(),
            'metric_value': metric_value,
            **extra_state
        }
        
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        # Save best model
        if self.is_better(metric_value, self.best_score):
            self.best_score = metric_value
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f"✓ Saved best model (epoch {epoch}, {self.monitor}={metric_value:.4f})")
        
        # Save periodic checkpoints
        if not self.save_best_only and epoch % self.save_freq == 0:
            epoch_path = os.path.join(self.checkpoint_dir, f'model_epoch_{epoch}.pth')
            torch.save(checkpoint, epoch_path)
            print(f"✓ Saved checkpoint at epoch {epoch}")

class EvaluateModel:
    def __init__(self, eval_config):
        self.eval_config = eval_config

    def evaluatepipeline(self, test_preds_raw, test_gt_raw, 
                        test_all_starts, test_all_ends, test_res, test_fps, category="Overall"):
        
        # Transform yolo coordinates onto image domain for both gt and predicted values
        test_gts = yolo_to_img_space_gt(test_gt_raw, all_starts=test_all_starts, 
                                        all_ends=test_all_ends, window_size=16, 
                                        all_original_size=test_res, all_fps=test_fps)
        test_preds = yolo_to_img_space(test_preds_raw, all_starts=test_all_starts, 
                                       all_ends=test_all_ends, 
                                       confidence_threshold=self.eval_config['confidence_threshold'], 
                                       window_size=16, all_original_size=test_res, all_fps=test_fps)

        # Post Process all predictions
        post_test_preds = batch_postprocess_predictions(
            test_preds, spatial_threshold=30, temporal_threshold=8, 
            confidence_threshold=self.eval_config['confidence_threshold'], strategy=self.eval_config['strategy']
        )  

        # unique_clusters_all = len({det['cluster_id'] for seq in post_test_preds for det in seq})
        # print(f'Number of clusters - {category}: {unique_clusters_all}')

        test_metrics = get_detailed_metrics_silent(test_preds, test_gts, self.eval_config)
        post_test_metrics = get_detailed_metrics_silent(post_test_preds, test_gts, self.eval_config)
        
        print_clean_comparison(test_metrics, post_test_metrics, category)
        
        return {
            'before': test_metrics,
            'after': post_test_metrics,
            'category': category
        }


