import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.video import r2plus1d_18
from torch.utils.checkpoint import checkpoint

from src.models.temporal import Soft_Temporal_Pool


class R2Plus1D_YOLO_MultiHead(nn.Module):
    """
    Multi-head R2+1D YOLO model with temporal transformers for ALL heads.

    Spatial and Direction heads use self-attention transformers.
    Temporal head uses cross-attention: queries from its own temporal features,
    keys/values from the spatial head's transformer output. Idea is for the temporal head:
    given where the spatial head sees a waggle, when does it start and end ? 

    Heads:
    - Spatial-Confidence Head: confidence + x,y position (self-attention transformer)
    - Direction Head: dir_x, dir_y (self-attention transformer)
    - Temporal Head: t_start, t_end (cross-attention on spatial features)
    """

    def __init__(
        self,
        num_classes: int = 1,
        grid_size: int = 28,
        pretrained: bool = True,
        max_detections_per_cell: int = 1,
        dropout_rate: float = 0.15,
        use_gradient_checkpointing: bool = True,
        transformer_layers: int = 4,
        transformer_heads: int = 4,
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

        # Spatial-Confidence Head self-attention
        self.spatial_temporal_proj = nn.Sequential(
            nn.Conv3d(256, 256, kernel_size=(1, 1, 1), bias=False),
            nn.GroupNorm(16, 256),
            nn.ReLU(inplace=True)
        )

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

        self.spatial_head = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
            self._depthwise_block(128, 64, dropout_rate),
            nn.Conv2d(64, 3 * max_detections_per_cell, kernel_size=1)
        )

        # Direction Head self-attention
        self.direction_temporal_proj = nn.Sequential(
            nn.Conv3d(256, 256, kernel_size=(1, 1, 1), bias=False),
            nn.GroupNorm(16, 256),
            nn.ReLU(inplace=True)
        )

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

        self.direction_head = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
            self._depthwise_block(128, 64, dropout_rate),
            nn.Conv2d(64, 2 * max_detections_per_cell, kernel_size=1)
        )

        # Temporal Head cross-attention: temporal queries, spatial keys/values
        self.temporal_temporal_proj = nn.Sequential(
            nn.Conv3d(256, 256, kernel_size=(1, 1, 1), bias=False),
            nn.GroupNorm(16, 256),
            nn.ReLU(inplace=True)
        )

        # Self-attention first to let temporal features resolve their own sequence
        temporal_self_encoder_layer = nn.TransformerEncoderLayer(
            d_model=256,
            nhead=transformer_heads,
            dim_feedforward=512,
            dropout=dropout_rate,
            activation='relu',
            batch_first=True
        )
        self.temporal_self_transformer = nn.TransformerEncoder(
            temporal_self_encoder_layer,
            num_layers=transformer_layers
        )

        # Cross-attention: temporal queries attend to spatial keys/values
        # Query  = temporal self-attended features  (B*H*W, T, C)
        # Key/Value = spatial transformer output (B*H*W, T, C)
        self.temporal_cross_attention = nn.MultiheadAttention(
            embed_dim=256,
            num_heads=transformer_heads,
            dropout=dropout_rate,
            batch_first=True
        )
        self.temporal_cross_attn_norm = nn.LayerNorm(256)

        # Optional feedforward after cross-attention
        self.temporal_cross_ffn = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.Dropout(dropout_rate)
        )
        self.temporal_cross_ffn_norm = nn.LayerNorm(256)

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

    def _run_self_attention(self, backbone_feat, proj, transformer):
        """
        Shared helper: project to reshape to self-attention to mean pool to reshape back.

        Args:
            backbone_feat: (B, C, T, H, W)
            proj:          Conv3d projection module
            transformer:   TransformerEncoder module

        Returns:
            attended_seq:  (B*H*W, T, C)  — pre-pooled, for cross-attention reuse
            pooled_feat:   (B, C, H, W)   — mean-pooled over time
        """
        B, C, T, H, W = backbone_feat.shape
        feat = proj(backbone_feat)                              # (B, C, T, H, W)
        feat = feat.permute(0, 3, 4, 2, 1)                     # (B, H, W, T, C)
        feat = feat.reshape(B * H * W, T, C)                   # (B*H*W, T, C)
        attended = transformer(feat)                            # (B*H*W, T, C)
        pooled = attended.mean(dim=1)                           # (B*H*W, C)
        pooled = pooled.view(B, H, W, C).permute(0, 3, 1, 2)   # (B, C, H, W)
        return attended, pooled

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input video tensor (B, C, T, H, W)

        Returns:
            Detections tensor (B, grid_h, grid_w, max_det, 7)
            [confidence, x, y, dir_x, dir_y, t_start, t_end]
        """
        # Backbone
        if self.use_gradient_checkpointing and self.training:
            backbone_feat = torch.utils.checkpoint.checkpoint(
                self.backbone, x, use_reentrant=False
            )
        else:
            backbone_feat = self.backbone(x)

        B, C, T, H, W = backbone_feat.shape

        # Spatial head (self-attention)
        # spatial_seq kept for cross-attention use by temporal head
        spatial_seq, spatial_pooled = self._run_self_attention(
            backbone_feat, self.spatial_temporal_proj, self.spatial_transformer
        )
        spatial_output = self.spatial_head(spatial_pooled)

        # Direction head (self-attention, independent)
        _, direction_pooled = self._run_self_attention(
            backbone_feat, self.direction_temporal_proj, self.direction_transformer
        )
        direction_output = self.direction_head(direction_pooled)

        # Temporal head (self-attention to cross-attention on spatial_seq)

        # Step 1: temporal self-attention
        temporal_feat = self.temporal_temporal_proj(backbone_feat)  # (B, C, T, H, W)
        temporal_feat = temporal_feat.permute(0, 3, 4, 2, 1)        # (B, H, W, T, C)
        temporal_feat = temporal_feat.reshape(B * H * W, T, C)       # (B*H*W, T, C)
        temporal_self_out = self.temporal_self_transformer(temporal_feat)  # (B*H*W, T, C)

        # Step 2: cross-attention
        # Query:     temporal self-attended features  (B*H*W, T, C)
        # Key/Value: spatial transformer sequence     (B*H*W, T, C)
        cross_out, _ = self.temporal_cross_attention(
            query=temporal_self_out,
            key=spatial_seq,
            value=spatial_seq
        )  # (B*H*W, T, C)

        # Residual + norm
        cross_out = self.temporal_cross_attn_norm(temporal_self_out + cross_out)

        # Feedforward + residual + norm
        ffn_out = self.temporal_cross_ffn(cross_out)
        cross_out = self.temporal_cross_ffn_norm(cross_out + ffn_out)

        # Mean pool over time and reshape back to spatial
        temporal_pooled = cross_out.mean(dim=1)                          # (B*H*W, C)
        temporal_pooled = temporal_pooled.view(B, H, W, C).permute(0, 3, 1, 2)  # (B, C, H, W)

        temporal_output = self.temporal_head(temporal_pooled)

        # Combine predictions
        batch_size = x.shape[0]
        actual_grid_h, actual_grid_w = spatial_output.shape[-2:]

        spatial_preds = self._reshape_head_output(
            spatial_output, batch_size, actual_grid_h, actual_grid_w, 3
        )
        direction_preds = self._reshape_head_output(
            direction_output, batch_size, actual_grid_h, actual_grid_w, 2
        )
        temporal_preds = self._reshape_head_output(
            temporal_output, batch_size, actual_grid_h, actual_grid_w, 2
        )

        # [conf, x, y, dir_x, dir_y, t_start, t_end]
        detections = torch.cat([spatial_preds, direction_preds, temporal_preds], dim=-1)
        detections = self.apply_output_activations(detections)

        return detections

    def _reshape_head_output(self, output, batch_size, grid_h, grid_w, num_params):
        """
        Reshape head output to (B, grid_h, grid_w, max_det, num_params).
        """
        output = output.view(
            batch_size,
            self.max_detections_per_cell * num_params,
            grid_h,
            grid_w
        ).permute(0, 2, 3, 1)

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
        """
        confidence = detections[..., 0:1]
        position = detections[..., 1:3]
        direction = detections[..., 3:5]
        offsets = detections[..., 5:7]

        position = torch.sigmoid(position)
        direction = F.normalize(direction, p=2, dim=-1)
        offsets = torch.sigmoid(offsets)

        return torch.cat([confidence, position, direction, offsets], dim=-1)