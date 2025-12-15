import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.video import r2plus1d_18
from torch.utils.checkpoint import checkpoint

from .components import TemporalAttention

class R2Plus1D_YOLO(nn.Module):
    """
    Fully optimized R2+1D YOLO model with:
    - Depthwise separable convolutions in detection head
    - Lighter feature projection
    - Gradient checkpointing support
    - GroupNorm instead of BatchNorm (more stable)
    
    This is an optional upgrade. Your existing model will still work,
    but this version is faster and uses less memory.
    """
    
    def __init__(
        self,
        num_classes: int = 1,
        grid_size: int = 28,
        pretrained: bool = True,
        max_detections_per_cell: int = 1,
        dropout_rate: float = 0.3,
        use_gradient_checkpointing: bool = True
    ):
        super().__init__()
        self.grid_size = grid_size
        self.num_classes = num_classes
        self.max_detections_per_cell = max_detections_per_cell
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        self.detection_params = 7
        self.total_outputs_per_cell = self.max_detections_per_cell * self.detection_params
        
        # Backbone: R(2+1)D-18 without final classification layers
        base_r2p1d = r2plus1d_18(pretrained=pretrained)
        self.backbone = nn.Sequential(*list(base_r2p1d.children())[:-3])
        
        # Optimized temporal aggregation
        self.temporal_pool = TemporalAttention(channels=256, reduction=4)
        
        # Lighter feature projection (256 -> 64 instead of 256 -> 128)
        self.feature_projection = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate * 0.5)
        )
        
        # Detection head with depthwise separable convs
        self.detection_head = nn.Sequential(
            self._depthwise_block(64, 64, dropout_rate),
            self._depthwise_block(64, 32, dropout_rate),
            nn.Conv2d(32, self.total_outputs_per_cell, kernel_size=1)
        )
        
        self._initialize_weights()
    
    def _depthwise_block(self, in_channels, out_channels, dropout_rate):
        """
        Depthwise separable convolution block.
        Much faster than regular convolution (8x fewer parameters).
        """
        return nn.Sequential(
            # Depthwise: each channel processed separately
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, 
                     groups=in_channels, bias=False),
            nn.GroupNorm(8, in_channels),
            nn.ReLU(inplace=True),
            # Pointwise: mix channels
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate)
        )
    
    def _initialize_weights(self):
        """Initialize weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        # Special initialization for final detection layer
        final_conv = self.detection_head[-1]
        nn.init.normal_(final_conv.weight, 0, 0.01)
        if final_conv.bias is not None:
            nn.init.constant_(final_conv.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with optional gradient checkpointing.
        
        Args:
            x: Input video tensor (B, C, T, H, W)
            
        Returns:
            Detections tensor (B, grid_h, grid_w, max_det, 7)
            where last dimension is [confidence, x, y, dir_x, dir_y, t_start, t_end]
        """
        # Backbone feature extraction with optional gradient checkpointing
        if self.use_gradient_checkpointing and self.training:
            backbone_feat = torch.utils.checkpoint.checkpoint(
                self.backbone, x, use_reentrant=False
            )
        else:
            backbone_feat = self.backbone(x)
        
        # Temporal pooling: (B, C, T, H, W) -> (B, C, H, W)
        pooled_feat = self.temporal_pool(backbone_feat)
        
        # Feature projection and detection head
        projected_feat = self.feature_projection(pooled_feat)
        detection_output = self.detection_head(projected_feat)
        
        # Reshape to detection format
        actual_grid_h, actual_grid_w = detection_output.shape[-2:]
        batch_size = detection_output.shape[0]
        
        detections = detection_output.view(
            batch_size, 
            self.total_outputs_per_cell, 
            actual_grid_h, 
            actual_grid_w
        ).permute(0, 2, 3, 1)  # (B, grid_h, grid_w, total_outputs)
        
        detections = detections.view(
            batch_size,
            actual_grid_h,
            actual_grid_w,
            self.max_detections_per_cell,
            self.detection_params
        )
        
        # Apply activations to normalize outputs
        detections = self.apply_output_activations(detections)
        
        return detections
    
    def apply_output_activations(self, detections: torch.Tensor) -> torch.Tensor:
        """
        Apply activation functions to raw detection outputs.
        
        Args:
            detections: Raw detections (..., 7)
            
        Returns:
            Activated detections with normalized ranges
        """
        confidence = detections[..., 0:1]
        position = detections[..., 1:3]
        direction = detections[..., 3:5]
        offsets = detections[..., 5:7]
        
        position = torch.sigmoid(position)
        direction = F.normalize(direction, p=2, dim=-1)
        offsets = torch.sigmoid(offsets)
        
        activated_detections = torch.cat(
            [confidence, position, direction, offsets], dim=-1
        )
        
        return activated_detections