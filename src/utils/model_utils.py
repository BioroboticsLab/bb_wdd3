import torch 
import torch.nn as nn
import math
from copy import deepcopy
from src.models.model import R2Plus1D_YOLO

#from src.models.model_multihead import R2Plus1D_YOLO_MultiHead
#from src.models.model_multihead_standard import R2Plus1D_YOLO_MultiHead
#from src.models.model_multihead_deeper_heads import R2Plus1D_YOLO_MultiHead
#from src.models.model_multihead_deeper_heads_tempstack_dirdial import R2Plus1D_YOLO_MultiHead
from src.models.model_multihead_deeper_heads_transformer_dir import R2Plus1D_YOLO_MultiHead
#from src.models.model_multihead_deeper_heads_transformer import R2Plus1D_YOLO_MultiHead
#from src.models.model_multihead_deeper_heads_transformer_dirvit import R2Plus1D_YOLO_MultiHead


def load_pretrained_model(checkpoint_path, max_detections_per_cell=1, grid_size=28, 
                          device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
    """Load a pretrained model for training or evaluation"""
    #model = R2Plus1D_YOLO(max_detections_per_cell=1, grid_size=grid_size)
    model = R2Plus1D_YOLO_MultiHead(max_detections_per_cell=1, grid_size=grid_size)
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


def de_parallel(model):
    """De-parallelize a model"""
    return model.module if hasattr(model, 'module') else model

class EMA:
    """Memory-efficient in-place EMA that doesn't duplicate the model"""
    def __init__(self, model, decay=0.9999, tau=2000, updates=0, device=None):
        """Initialize EMA with parameter-level shadow values"""
        self.model = model
        self.updates = updates
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))  # decay exponential ramp
        self.device = device
        
        # Store shadow parameters instead of full model copy
        self.shadow = {}
        self.backup = {}
        self._register_shadow_params()
        
    def _register_shadow_params(self):
        """Initialize shadow parameters from current model"""
        model = de_parallel(self.model)
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad and param.dtype.is_floating_point:
                    self.shadow[name] = param.data.clone()
        
    def update(self, model=None):
        """Update EMA parameters in-place"""
        if model is None:
            model = self.model
            
        self.updates += 1
        d = self.decay(self.updates)

        model = de_parallel(model)
        msd = model.state_dict()  # model state_dict
        
        with torch.no_grad():
            for k, v in msd.items():
                if k in self.shadow and v.dtype.is_floating_point:
                    self.shadow[k] = self.shadow[k] * d + v.detach() * (1 - d)

    def apply_shadow(self):
        """Swap model parameters with EMA shadow parameters for evaluation"""
        model = de_parallel(self.model)
        self.backup = {}
        
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in self.shadow and param.requires_grad:
                    self.backup[name] = param.data.clone()
                    param.data.copy_(self.shadow[name])

    def restore(self):
        """Restore original model parameters after evaluation"""
        model = de_parallel(self.model)
        
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in self.backup and param.requires_grad:
                    param.data.copy_(self.backup[name])
            self.backup = {}

    def state_dict(self):
        return {
            'updates': self.updates,
            'shadow_params': self.shadow,
        }

    def load_state_dict(self, state_dict):
        self.updates = state_dict['updates']
        self.shadow = state_dict['shadow_params']