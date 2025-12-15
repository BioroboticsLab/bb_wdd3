from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Optional
import os
import datetime


class MetricsLogger:
    """
    Wrapper for TensorBoard logging with convenience methods.
    
    Args:
        log_dir: Base directory for logs (auto-creates timestamped subdir)
        experiment_name: Optional experiment name
        
    """
    
    def __init__(self, log_dir: str = './logs', experiment_name: Optional[str] = None):
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        
        if experiment_name:
            run_name = f"{experiment_name}_{timestamp}"
        else:
            run_name = timestamp
        
        self.log_dir = os.path.join(log_dir, run_name)
        self.writer = SummaryWriter(log_dir=self.log_dir)
        print(f"TensorBoard logs: {self.log_dir}")
    
    def log_train_metrics(self, epoch: int, **metrics):
        """Log training metrics for an epoch."""
        for name, value in metrics.items():
            self.writer.add_scalar(f'Train/{name}', value, epoch)
    
    def log_val_metrics(self, epoch: int, **metrics):
        """Log validation metrics for an epoch."""
        for name, value in metrics.items():
            self.writer.add_scalar(f'Validation/{name}', value, epoch)
    
    def log_learning_rate(self, epoch: int, lr: float):
        """Log current learning rate."""
        self.writer.add_scalar('Learning_Rate', lr, epoch)
    
    def log_loss_components(
        self, 
        epoch: int, 
        split: str,  # 'Train' or 'Validation'
        **loss_components
    ):
        """
        Log individual loss components.
    
        """
        for name, value in loss_components.items():
            self.writer.add_scalar(f'Loss/{split}/{name}', value, epoch)
    
    def close(self):
        """Close the TensorBoard writer."""
        self.writer.close()