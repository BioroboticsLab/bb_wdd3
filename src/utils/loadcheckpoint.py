import torch
import torch.nn as nn
from typing import Optional, Dict, Any


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    device: str = 'cuda',
    load_scheduler: bool = False
) -> Dict[str, Any]:
    """
    Load model checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optimizer to load state into (optional)
        scheduler: Scheduler to load state into (optional)
        device: Device to load checkpoint to
        
    Returns:
        Dictionary with checkpoint metadata (epoch, metrics, etc.)
        
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model state
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        # Check if checkpoint was saved with DataParallel
        state_dict = checkpoint['model_state_dict']
        if list(state_dict.keys())[0].startswith('module.'):
            # Remove 'module.' prefix
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
    
    # Load optimizer state
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)
        
        print("Optimizer state loaded and moved to", device)
    
    # Load scheduler state
    if scheduler is not None and 'scheduler_state_dict' in checkpoint and load_scheduler:
        try:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print("Scheduler state loaded")
        except Exception as e:
            print(f"Warning: Could not load scheduler state: {e}")
            print("Scheduler will start from initial configuration")

    # Extract metadata
    metadata = {
        'epoch': checkpoint.get('epoch', 0),
        'metric_value': checkpoint.get('metric_value', None)
    }
    
    print(f"Loaded checkpoint from {checkpoint_path}")
    if 'epoch' in checkpoint:
        print(f"  Epoch: {checkpoint['epoch']}")
    if 'metric_value' in checkpoint:
        print(f"  Metric: {checkpoint['metric_value']:.4f}")
    
    return metadata