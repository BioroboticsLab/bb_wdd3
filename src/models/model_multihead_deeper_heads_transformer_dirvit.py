import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.video import r2plus1d_18
from torch.utils.checkpoint import checkpoint

from src.models.temporal import Soft_Temporal_Pool


class WindowedSpatialAttention(nn.Module):
    """
    ViT-style spatial attention over local windows like Swim Transformer.
    
    Instead of attending over the full H*W grid, we partition
    the spatial map into non-overlapping windows and attend within each.
    
    Sequence length per attention call = window_size^2 (e.g. 49 for 7x7)
    instead of H*W (e.g. 784 for 28x28).
    """

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        window_size: int = 7,
        dropout: float = 0.1,
        num_layers: int = 2,
    ):
        super().__init__()
        self.window_size = window_size
        self.d_model = d_model

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            activation="relu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Learned positional embeddings for tokens within a window
        self.pos_embed = nn.Parameter(torch.zeros(1, window_size * window_size, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def _pad_to_window(self, x: torch.Tensor, H: int, W: int):
        """Pad H and W so they are divisible by window_size."""
        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))  # (N, Hp, Wp, C)
        return x, H + pad_h, W + pad_w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) spatial feature map

        Returns:
            (B, C, H, W) with direction-aware spatial attention applied
        """
        B, C, H, W = x.shape
        ws = self.window_size

        # (B, C, H, W) -> (B, H, W, C) for window partitioning
        x = x.permute(0, 2, 3, 1)

        # Pad so H, W divisible by window_size
        x, Hp, Wp = self._pad_to_window(x, H, W)

        num_win_h = Hp // ws
        num_win_w = Wp // ws
        num_windows = num_win_h * num_win_w

        # Partition into windows: (B, num_win_h, ws, num_win_w, ws, C)
        x = x.view(B, num_win_h, ws, num_win_w, ws, C)

        # -> (B, num_windows, ws*ws, C) — each window is a token sequence
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(B * num_windows, ws * ws, C)

        # Add positional embeddings within each window
        x = x + self.pos_embed

        # Self-attention within each window independently
        # Sequence length = ws^2 = 49 for ws=7, cost O(49^2) not O((H*W)^2)
        x = self.transformer(x)  # (B*num_windows, ws*ws, C)

        # Reverse window partition: (B*num_windows, ws*ws, C) -> (B, Hp, Wp, C)
        x = x.view(B, num_win_h, num_win_w, ws, ws, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(B, Hp, Wp, C)

        # Remove padding
        x = x[:, :H, :W, :].contiguous()

        # -> (B, C, H, W)
        return x.permute(0, 3, 1, 2)


class R2Plus1D_YOLO_MultiHead(nn.Module):
    """
    Multi-head R2+1D YOLO model with temporal transformers for ALL heads,
    and additional windowed ViT-style spatial attention for the direction head.

    Heads:
    - Spatial-Confidence Head: confidence + x,y position — temporal transformer
    - Direction Head:          dir_x, dir_y              — temporal transformer
                                                           + windowed spatial ViT
    - Temporal Head:           t_start, t_end            — temporal transformer

    The direction head gets spatial attention.
    The windowed approach keeps the sequence length at window_size^2 (e.g. 49)
    rather than the full H*W (e.g. 784), hence computationally more feasible.
    """

    def __init__(
        self,
        num_classes: int = 1,
        grid_size: int = 28,
        pretrained: bool = True,
        max_detections_per_cell: int = 1,
        dropout_rate: float = 0.1,
        use_gradient_checkpointing: bool = True,
        transformer_layers: int = 4,
        transformer_heads: int = 4,
        # Direction head spatial ViT params
        spatial_window_size: int = 7,
        spatial_vit_layers: int = 4,
        spatial_vit_heads: int = 4,
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

        # Spatial-Confidence Head (transformer over temoral dim)
        self.spatial_temporal_proj = nn.Sequential(
            nn.Conv3d(256, 256, kernel_size=(1, 1, 1), bias=False),
            nn.GroupNorm(16, 256),
            nn.ReLU(inplace=True),
        )
        spatial_encoder_layer = nn.TransformerEncoderLayer(
            d_model=256,
            nhead=transformer_heads,
            dim_feedforward=512,
            dropout=dropout_rate,
            activation="relu",
            batch_first=True,
        )
        self.spatial_transformer = nn.TransformerEncoder(
            spatial_encoder_layer, num_layers=transformer_layers
        )
        self.spatial_head = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
            self._depthwise_block(128, 64, dropout_rate),
            nn.Conv2d(64, 3 * max_detections_per_cell, kernel_size=1),
        )

        # Direction Head (transformer over temporal head + windowed spatial ViT)
        self.direction_temporal_proj = nn.Sequential(
            nn.Conv3d(256, 256, kernel_size=(1, 1, 1), bias=False),
            nn.GroupNorm(16, 256),
            nn.ReLU(inplace=True),
        )
        direction_encoder_layer = nn.TransformerEncoderLayer(
            d_model=256,
            nhead=transformer_heads,
            dim_feedforward=512,
            dropout=dropout_rate,
            activation="relu",
            batch_first=True,
        )
        self.direction_transformer = nn.TransformerEncoder(
            direction_encoder_layer, num_layers=transformer_layers
        )

        # Windowed spatial ViT — applied after temporal aggregation
        # Input: (B, 256, H, W) — the temporally-pooled direction feature map
        # Attends within 7x7 local windows so each spatial location can
        # gather directional context from its immediate neighborhood
        self.direction_spatial_vit = WindowedSpatialAttention(
            d_model=256,
            nhead=spatial_vit_heads,
            window_size=spatial_window_size,
            dropout=dropout_rate,
            num_layers=spatial_vit_layers,
        )

        self.direction_head = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
            self._depthwise_block(128, 64, dropout_rate),
            nn.Conv2d(64, 2 * max_detections_per_cell, kernel_size=1),
        )

        # Transformer on temporal Temporal Head 
        self.temporal_temporal_proj = nn.Sequential(
            nn.Conv3d(256, 256, kernel_size=(1, 1, 1), bias=False),
            nn.GroupNorm(16, 256),
            nn.ReLU(inplace=True),
        )
        temporal_encoder_layer = nn.TransformerEncoderLayer(
            d_model=256,
            nhead=transformer_heads,
            dim_feedforward=512,
            dropout=dropout_rate,
            activation="relu",
            batch_first=True,
        )
        self.temporal_transformer = nn.TransformerEncoder(
            temporal_encoder_layer, num_layers=transformer_layers
        )
        self.temporal_head = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
            self._depthwise_block(128, 64, dropout_rate),
            nn.Conv2d(64, 2 * max_detections_per_cell, kernel_size=1),
        )

        self._initialize_weights()

    def _depthwise_block(self, in_channels, out_channels, dropout_rate):
        """Depthwise separable convolution block."""
        return nn.Sequential(
            nn.Conv2d(
                in_channels, in_channels, kernel_size=3, padding=1,
                groups=in_channels, bias=False,
            ),
            nn.GroupNorm(8, in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
        )

    def _initialize_weights(self):
        """Initialize weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
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

    def _apply_temporal_transformer(
        self,
        feat: torch.Tensor,
        transformer: nn.TransformerEncoder,
        B: int, C: int, T: int, H: int, W: int,
    ) -> torch.Tensor:
        """
        Shared helper: reshape -> temporal attention -> mean pool -> reshape back.

        Args:
            feat: (B, C, T, H, W)
            transformer: temporal TransformerEncoder

        Returns:
            (B, C, H, W) temporally-aggregated feature map
        """
        # Each spatial location (h, w) attends across its T temporal features
        feat = feat.permute(0, 3, 4, 2, 1)          # (B, H, W, T, C)
        feat = feat.reshape(B * H * W, T, C)          # (B*H*W, T, C)
        feat = transformer(feat)                       # (B*H*W, T, C)
        feat = feat.mean(dim=1)                        # (B*H*W, C)
        feat = feat.view(B, H, W, C).permute(0, 3, 1, 2)  # (B, C, H, W)
        return feat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input video tensor (B, C, T, H, W)

        Returns:
            Detections tensor (B, grid_h, grid_w, max_det, 7)
            where last dim is [confidence, x, y, dir_x, dir_y, t_start, t_end]
        """
        # Backbone
        if self.use_gradient_checkpointing and self.training:
            backbone_feat = torch.utils.checkpoint.checkpoint(
                self.backbone, x, use_reentrant=False
            )
        else:
            backbone_feat = self.backbone(x)

        B, C, T, H, W = backbone_feat.shape

        # Spatial-Confidence Head
        spatial_feat = self.spatial_temporal_proj(backbone_feat)
        spatial_feat = self._apply_temporal_transformer(
            spatial_feat, self.spatial_transformer, B, C, T, H, W
        )
        spatial_output = self.spatial_head(spatial_feat)

        # Direction Head
        # Temporal attention first (what has moved and when),
        # then windowed spatial ViT (where relative to local neighbors).
        direction_feat = self.direction_temporal_proj(backbone_feat)

        # Step 1 — temporal attention: (B, C, H, W)
        direction_feat = self._apply_temporal_transformer(
            direction_feat, self.direction_transformer, B, C, T, H, W
        )

        # Step 2 — windowed spatial ViT: (B, C, H, W) to (B, C, H, W)
        # Each spatial location now attends to its local window neighbors,
        # giving the direction head awareness of surrounding spatial context
        direction_feat = self.direction_spatial_vit(direction_feat)

        direction_output = self.direction_head(direction_feat)

        # Temporal Head
        temporal_feat = self.temporal_temporal_proj(backbone_feat)
        temporal_feat = self._apply_temporal_transformer(
            temporal_feat, self.temporal_transformer, B, C, T, H, W
        )
        temporal_output = self.temporal_head(temporal_feat)

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

        Args:
            output: (B, max_det * num_params, H, W)

        Returns:
            (B, grid_h, grid_w, max_det, num_params)
        """
        output = output.view(
            batch_size,
            self.max_detections_per_cell * num_params,
            grid_h,
            grid_w,
        ).permute(0, 2, 3, 1)  # (B, grid_h, grid_w, max_det * num_params)

        output = output.view(
            batch_size,
            grid_h,
            grid_w,
            self.max_detections_per_cell,
            num_params,
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
        position   = detections[..., 1:3]
        direction  = detections[..., 3:5]
        offsets    = detections[..., 5:7]

        position  = torch.sigmoid(position)
        direction = F.normalize(direction, p=2, dim=-1)
        offsets   = torch.sigmoid(offsets)

        return torch.cat([confidence, position, direction, offsets], dim=-1)