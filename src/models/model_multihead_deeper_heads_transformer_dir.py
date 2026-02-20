import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.video import r2plus1d_18
from torch.utils.checkpoint import checkpoint

from src.models.temporal import Soft_Temporal_Pool

class R2Plus1D_YOLO_MultiHead(nn.Module):
    """
    Multi-head R2+1D YOLO model with temporal transformer for direction prediction.
    
    The direction head now uses a transformer encoder to attend across temporal features. 
    
    Heads:
    - Spatial-Confidence Head: confidence + x,y position (uses temporal pooling)
    - Direction Head: dir_x, dir_y with transformer
    - Temporal Head: t_start, t_end (with 3D conv)
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
        transformer_heads: int = 4 #8
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
        
        # Temporal aggregation for spatial features (used by spatial head)
        self.temporal_pool = Soft_Temporal_Pool(channels=256, reduction=4)
        
        # Shared feature projection (256 to 256)
        self.feature_projection = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1, bias=False),
            nn.GroupNorm(16, 256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate * 0.5)
        )
        
        # Shared feature processing - 256 channels
        self.shared_features = self._depthwise_block(256, 256, dropout_rate)
        
        # Spatial confidence head - predicts [confidence, x, y] per detection
        self.spatial_head = nn.Sequential(
            self._depthwise_block(256, 128, dropout_rate),
            self._depthwise_block(128, 64, dropout_rate),
            nn.Conv2d(64, 3 * max_detections_per_cell, kernel_size=1)
        )
        
        # Direction head with temp transformer
        # First project temporal features
        self.direction_temporal_proj = nn.Sequential(
            nn.Conv3d(256, 256, kernel_size=(1, 1, 1), bias=False),
            nn.GroupNorm(16, 256),
            nn.ReLU(inplace=True)
        )
        
        # Transformer encoder to attend across time
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256,
            nhead=transformer_heads,
            dim_feedforward=512,
            dropout=dropout_rate,
            activation='relu',
            batch_first=True
        )
        self.direction_transformer = nn.TransformerEncoder(
            encoder_layer,
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
        
        # Temporal head with 3D conv
        self.temporal_features = nn.Sequential(
            nn.Conv3d(256, 256, kernel_size=(3, 1, 1), padding=(1, 0, 0), bias=False),
            nn.GroupNorm(16, 256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((1, None, None))
        )
        
        self.temporal_head = nn.Sequential(
            self._depthwise_block(256, 128, dropout_rate),
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
        Forward pass with temporal transformer for direction prediction.
        
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
        
        # Spatial - conf heads
        # Uses temporal pooling
        pooled_feat = self.temporal_pool(backbone_feat)
        projected_feat = self.feature_projection(pooled_feat)
        shared_feat = self.shared_features(projected_feat)
        spatial_output = self.spatial_head(shared_feat)
        
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
        
        # === TEMPORAL HEAD ===
        temporal_feat = self.temporal_features(backbone_feat)
        temporal_feat = temporal_feat.squeeze(2)
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