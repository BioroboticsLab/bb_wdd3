import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.video import r2plus1d_18
from torch.utils.checkpoint import checkpoint

from src.models.temporal import Soft_Temporal_Pool

class R2Plus1D_YOLO_MultiHead(nn.Module):
    """
    Multi-head R2+1D YOLO model with temporal transformers for ALL heads.
    
    All three heads use transformer encoders to attend across temporal features,
    to capture spatial position, motion, and temporal boundary patterns.
    
    Heads:
    - Spatial-Confidence Head: confidence + x,y position with transformer
    - Direction Head: dir_x, dir_y with  transformer
    - Temporal Head: t_start, t_end with transformer
    """
    
    def __init__(
        self,
        num_classes: int = 1,
        grid_size: int = 28,
        pretrained: bool = True,
        max_detections_per_cell: int = 1,
        dropout_rate: float = 0.3,
        use_gradient_checkpointing: bool = True,
        transformer_layers: int = 4,
        transformer_heads: int = 8
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
        
        # Spatial confidence head with transformer
        # Project temporal features for spatial head
        self.spatial_temporal_proj = nn.Sequential(
            nn.Conv3d(256, 256, kernel_size=(1, 1, 1), bias=False),
            nn.GroupNorm(16, 256),
            nn.ReLU(inplace=True)
        )
        
        # Transformer encoder to attend across time for spatial predictions
        spatial_encoder_layer = nn.TransformerEncoderLayer(
            d_model=256,
            nhead=transformer_heads,
            dim_feedforward=512,
            dropout=dropout_rate,
            activation='relu',
            batch_first=True
        )
        self.spatial_transformer = nn.TransformerEncoder(
            spatial_encoder_layer,
            num_layers=transformer_layers
        )
        
        # After transformer, predict spatial confidence
        self.spatial_head = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
            
            self._depthwise_block(128, 64, dropout_rate),
            
            nn.Conv2d(64, 3 * max_detections_per_cell, kernel_size=1)
        )
        
        # direction head with transformer
        # Project temporal features
        self.direction_temporal_proj = nn.Sequential(
            nn.Conv3d(256, 256, kernel_size=(1, 1, 1), bias=False),
            nn.GroupNorm(16, 256),
            nn.ReLU(inplace=True)
        )
        
        # Transformer encoder to attend across time
        direction_encoder_layer = nn.TransformerEncoderLayer(
            d_model=256,
            nhead=transformer_heads,
            dim_feedforward=512,
            dropout=dropout_rate,
            activation='relu',
            batch_first=True
        )
        self.direction_transformer = nn.TransformerEncoder(
            direction_encoder_layer,
            num_layers=transformer_layers
        )
        
        # After transformer, predict direction
        self.direction_head = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
            
            self._depthwise_block(128, 64, dropout_rate),
            
            nn.Conv2d(64, 2 * max_detections_per_cell, kernel_size=1)
        )
        
        # Temporal head with transformer
        # Project temporal features for temporal head
        self.temporal_temporal_proj = nn.Sequential(
            nn.Conv3d(256, 256, kernel_size=(1, 1, 1), bias=False),
            nn.GroupNorm(16, 256),
            nn.ReLU(inplace=True)
        )
        
        # Transformer encoder to attend across time for temporal boundaries
        temporal_encoder_layer = nn.TransformerEncoderLayer(
            d_model=256,
            nhead=transformer_heads,
            dim_feedforward=512,
            dropout=dropout_rate,
            activation='relu',
            batch_first=True
        )
        self.temporal_transformer = nn.TransformerEncoder(
            temporal_encoder_layer,
            num_layers=transformer_layers
        )
        
        # After transformer, predict temporal boundaries
        self.temporal_head = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
            
            self._depthwise_block(128, 64, dropout_rate),
            
            nn.Conv2d(64, 2 * max_detections_per_cell, kernel_size=1)
        )
        
        self._initialize_weights()
    
    def _depthwise_block(self, in_channels, out_channels, dropout_rate):
        """Depthwise separable convolution block."""
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, 
                     groups=in_channels, bias=False),
            nn.GroupNorm(8, in_channels),
            nn.ReLU(inplace=True),
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
        
        for head in [self.spatial_head, self.direction_head, self.temporal_head]:
            final_conv = head[-1]
            nn.init.normal_(final_conv.weight, 0, 0.01)
            if final_conv.bias is not None:
                nn.init.constant_(final_conv.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with temporal transformers for ALL heads.
        
        Args:
            x: Input video tensor (B, C, T, H, W)
            
        Returns:
            Detections tensor (B, grid_h, grid_w, max_det, 7)
            where last dimension is [confidence, x, y, dir_x, dir_y, t_start, t_end]
        """
        # Backbone feature extraction
        if self.use_gradient_checkpointing and self.training:
            backbone_feat = torch.utils.checkpoint.checkpoint(
                self.backbone, x, use_reentrant=False
            )
        else:
            backbone_feat = self.backbone(x)
        
        B, C, T, H, W = backbone_feat.shape
        
        # Spatial head with transformer
        # Project temporal features
        spatial_feat = self.spatial_temporal_proj(backbone_feat)  # (B, 256, T, H, W)
        
        # Reshape for transformer: (B, C, T, H, W) to (B*H*W, T, C)
        # Each spatial location gets a sequence of T temporal features
        spatial_feat = spatial_feat.permute(0, 3, 4, 2, 1)  # (B, H, W, T, C)
        spatial_feat = spatial_feat.reshape(B * H * W, T, C)  # (B*H*W, T, C)
        
        # Apply transformer: let each spatial location attend across time
        spatial_feat = self.spatial_transformer(spatial_feat)  # (B*H*W, T, C)
        
        # Aggregate temporal information (mean pooling across time)
        spatial_feat = spatial_feat.mean(dim=1)  # (B*H*W, C)
        
        # Reshape back to spatial: (B*H*W, C) to (B, C, H, W)
        spatial_feat = spatial_feat.view(B, H, W, C).permute(0, 3, 1, 2)  # (B, C, H, W)
        
        # Predict spatial confidence and position
        spatial_output = self.spatial_head(spatial_feat)
        
        # Direction head with transformer
        # Project temporal features
        direction_feat = self.direction_temporal_proj(backbone_feat)  # (B, 256, T, H, W)
        
        # Reshape for transformer: (B, C, T, H, W) to (B*H*W, T, C)
        # Each spatial location gets a sequence of T temporal features
        direction_feat = direction_feat.permute(0, 3, 4, 2, 1)  # (B, H, W, T, C)
        direction_feat = direction_feat.reshape(B * H * W, T, C)  # (B*H*W, T, C)
        
        # Apply transformer: let each spatial location attend across time
        direction_feat = self.direction_transformer(direction_feat)  # (B*H*W, T, C)
        
        # Aggregate temporal information (mean pooling across time)
        direction_feat = direction_feat.mean(dim=1)  # (B*H*W, C)
        
        # Reshape back to spatial: (B*H*W, C) to (B, C, H, W)
        direction_feat = direction_feat.view(B, H, W, C).permute(0, 3, 1, 2)  # (B, C, H, W)
        
        # Predict direction
        direction_output = self.direction_head(direction_feat)
        
        # Temporal head with transformer
        # Project temporal features
        temporal_feat = self.temporal_temporal_proj(backbone_feat)  # (B, 256, T, H, W)
        
        # Reshape for transformer: (B, C, T, H, W) to (B*H*W, T, C)
        # Each spatial location gets a sequence of T temporal features
        temporal_feat = temporal_feat.permute(0, 3, 4, 2, 1)  # (B, H, W, T, C)
        temporal_feat = temporal_feat.reshape(B * H * W, T, C)  # (B*H*W, T, C)
        
        # Apply transformer: let each spatial location attend across time
        temporal_feat = self.temporal_transformer(temporal_feat)  # (B*H*W, T, C)
        
        # Aggregate temporal information (mean pooling across time)
        temporal_feat = temporal_feat.mean(dim=1)  # (B*H*W, C)
        
        # Reshape back to spatial: (B*H*W, C) to (B, C, H, W)
        temporal_feat = temporal_feat.view(B, H, W, C).permute(0, 3, 1, 2)  # (B, C, H, W)
        
        # Predict temporal boundaries
        temporal_output = self.temporal_head(temporal_feat)
        
        # Combine all predictions
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