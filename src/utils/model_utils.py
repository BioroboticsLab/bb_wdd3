import torch 
from src.models.model import R2Plus1D_YOLO

def load_pretrained_model(checkpoint_path, max_detections_per_cell=1, grid_size=28, 
                          device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
    """Load a pretrained model for training or evaluation"""
    model = R2Plus1D_YOLO(max_detections_per_cell=1, grid_size=grid_size)
    
    print(f"Loading pretrained weights from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        # Comprehensive checkpoint format
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Loaded model from comprehensive checkpoint")
    else:
        # Simple model state dict
        model.load_state_dict(checkpoint)
        print("Loaded model from state dict")
    
    model = model.to(device)
    return model