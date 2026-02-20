import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.video import r2plus1d_18
from torch.utils.checkpoint import checkpoint

from src.models.temporal import Soft_Temporal_Pool

class R2Plus1D_YOLO_MultiHead(nn.Module):
    """
    Multi-head R2+1D YOLO model with separate prediction heads:
    - Spatial-Confidence Head: confidence + x,y position
    - Direction Head: dir_x, dir_y (waggle direction)
    - Temporal Head: t_start, t_end (with richer temporal features)
    
    Each task to optimize independently with with own representations so 
    that they do not compete for capacity in pred heads.
    
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
        
        # Temporal aggregation for spatial features
        self.temporal_pool = Soft_Temporal_Pool(channels=256, reduction=4)
        
        # Shared feature projection (256 to 64)
        self.feature_projection = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate * 0.5)
        )
        
        # Shared feature processing
        self.shared_features = self._depthwise_block(64, 64, dropout_rate)
        
        # Spatial confidence head
        # Predicts: [confidence, x, y] per detection
        self.spatial_head = nn.Sequential(
            self._depthwise_block(64, 32, dropout_rate),
            nn.Conv2d(32, 3 * max_detections_per_cell, kernel_size=1)
        )
        
        # Direction head
        # Predicts: [dir_x, dir_y] per detection
        self.direction_head = nn.Sequential(
            self._depthwise_block(64, 32, dropout_rate),
            nn.Conv2d(32, 2 * max_detections_per_cell, kernel_size=1)
        )
        
        # Temporal head
        # Processes temporal features before pooling for better temporal prediction
        # Predicts: [t_start, t_end] per detection
        self.temporal_features = nn.Sequential(
            nn.Conv3d(256, 64, kernel_size=(3, 1, 1), padding=(1, 0, 0), bias=False),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((1, None, None))  # Pool only temporal dimension
        )
        
        self.temporal_head = nn.Sequential(
            self._depthwise_block(64, 32, dropout_rate),
            nn.Conv2d(32, 2 * max_detections_per_cell, kernel_size=1)
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
        
        # Special initialization for final detection layers
        for head in [self.spatial_head, self.direction_head, self.temporal_head]:
            final_conv = head[-1]
            nn.init.normal_(final_conv.weight, 0, 0.01)
            if final_conv.bias is not None:
                nn.init.constant_(final_conv.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with multi-head predictions.
        
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
        
        # Spatial confidence and direction
        # Temporal pooling: (B, C, T, H, W) to (B, C, H, W)
        pooled_feat = self.temporal_pool(backbone_feat)
        
        # Feature projection and shared processing
        projected_feat = self.feature_projection(pooled_feat)
        shared_feat = self.shared_features(projected_feat)
        
        # Spatial-confidence predictions: [conf, x, y]
        spatial_output = self.spatial_head(shared_feat)
        
        # Direction predictions: [dir_x, dir_y]
        direction_output = self.direction_head(shared_feat)
        
        # Temporal 
        # Process temporal features before aggressive pooling
        temporal_feat = self.temporal_features(backbone_feat)
        temporal_feat = temporal_feat.squeeze(2)  # Remove temporal dim after pooling
        
        # Temporal predictions: [t_start, t_end]
        temporal_output = self.temporal_head(temporal_feat)
        
        # ==== COMBINE ALL PREDICTIONS ====
        batch_size = x.shape[0]
        actual_grid_h, actual_grid_w = spatial_output.shape[-2:]
        
        # Reshape each head output
        spatial_preds = self._reshape_head_output(
            spatial_output, batch_size, actual_grid_h, actual_grid_w, 3
        )
        direction_preds = self._reshape_head_output(
            direction_output, batch_size, actual_grid_h, actual_grid_w, 2
        )
        temporal_preds = self._reshape_head_output(
            temporal_output, batch_size, actual_grid_h, actual_grid_w, 2
        )
        
        # Concatenate: [conf, x, y, dir_x, dir_y, t_start, t_end]
        detections = torch.cat([spatial_preds, direction_preds, temporal_preds], dim=-1)
        
        # Apply activations to normalize outputs
        detections = self.apply_output_activations(detections)
        
        return detections
    
    def _reshape_head_output(self, output, batch_size, grid_h, grid_w, num_params):
        """
        Reshape head output to (B, grid_h, grid_w, max_det, num_params).
        
        Args:
            output: Raw head output (B, max_det * num_params, H, W)
            batch_size: Batch size
            grid_h, grid_w: Grid dimensions
            num_params: Number of parameters per detection for this head
            
        Returns:
            Reshaped tensor (B, grid_h, grid_w, max_det, num_params)
        """
        output = output.view(
            batch_size,
            self.max_detections_per_cell * num_params,
            grid_h,
            grid_w
        ).permute(0, 2, 3, 1)  # (B, grid_h, grid_w, max_det * num_params)
        
        output = output.view(
            batch_size,
            grid_h,
            grid_w,
            self.max_detections_per_cell,
            num_params
        )
        
        return output
    
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