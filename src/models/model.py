import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.video import r2plus1d_18
from torch.utils.checkpoint import checkpoint

from src.models.temporal import Soft_Temporal_Pool


class R2Plus1D_YOLO_MultiHead(nn.Module):
    """
    Multi-head R2+1D YOLO model with configurable prediction heads.

    Three modes controlled by self_attention and cross_attention flags:

    - self_attention=False, cross_attention=False (default):
        All three heads use 3D conv + soft temporal pooling.
        Fastest, least memory.

    - self_attention=True, cross_attention=False:
        Each head independently runs a TransformerEncoder over the T temporal
        frames at each spatial location (B*H*W, T, C), then mean-pools.
        Captures temporal context per head.

    - cross_attention=True (self_attention must be False):
        Spatial and Direction heads use self-attention transformers.
        Temporal head uses self-attention followed by cross-attention,
        where queries come from temporal features and keys/values come from
        the spatial head's transformer output. The intuition: given *where*
        the spatial head sees a waggle, *when* does it start and end?

    Both flags True simultaneously is invalid and raises a ValueError.

    All heads predict per-grid-cell detections:
        [confidence, x, y, dir_x, dir_y, t_start, t_end]
    """

    def __init__(
        self,
        n_classes: int = 1,
        grid_size: int = 28,
        pretrained: bool = True,
        max_detections_per_cell: int = 1,
        dropout_rate: float = 0.1,
        use_gradient_checkpointing: bool = True,
        self_attention: bool = False,
        cross_attention: bool = False,
        transformer_layers: int = 4,
        transformer_heads: int = 4,
    ):
        if self_attention and cross_attention:
            raise ValueError(
                "self_attention and cross_attention cannot both be True. "
                "Set exactly one (or neither for 3D-conv mode)."
            )

        super().__init__()
        self.grid_size = grid_size
        self.n_classes = n_classes
        self.max_detections_per_cell = max_detections_per_cell
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_self_attention = self_attention
        self.use_cross_attention = cross_attention

        self.detection_params = 7
        self.total_outputs_per_cell = self.max_detections_per_cell * self.detection_params

        # Backbone: R(2+1)D-18 without final classification layers
        base_r2p1d = r2plus1d_18(pretrained=pretrained)
        self.backbone = nn.Sequential(*list(base_r2p1d.children())[:-3])

        # Mode baseline 3D conv: 3D-conv heads no attention effectively baseline model
        if not self_attention and not cross_attention:
            # Temporal aggregation for spatial/direction features
            self.temporal_pool = Soft_Temporal_Pool(channels=256, reduction=4)

            self.feature_projection = nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=1, bias=False),
                nn.GroupNorm(16, 256),
                nn.ReLU(inplace=True),
                nn.Dropout2d(dropout_rate)
            )
            self.shared_features = self._depthwise_block(256, 256, dropout_rate)

            # Spatial-Confidence head for [conf, x, y]
            self.spatial_head = nn.Sequential(
                self._depthwise_block(256, 128, dropout_rate),
                self._depthwise_block(128, 64, dropout_rate),
                nn.Conv2d(64, 3 * max_detections_per_cell, kernel_size=1)
            )

            # Direction head for [dir_x, dir_y]
            self.direction_head = nn.Sequential(
                self._depthwise_block(256, 128, dropout_rate),
                self._depthwise_block(128, 64, dropout_rate),
                nn.Conv2d(64, 2 * max_detections_per_cell, kernel_size=1)
            )

            # Temporal head for [t_start, t_end]
            self.temporal_features = nn.Sequential(
                nn.Conv3d(256, 256, kernel_size=(3, 1, 1), padding=(1, 0, 0), bias=False),
                nn.GroupNorm(16, 256),
                nn.SiLU(inplace=True),
                nn.AdaptiveAvgPool3d((1, None, None))  # collapse T only
            )
            self.temporal_head = nn.Sequential(
                self._depthwise_block(256, 128, dropout_rate),
                self._depthwise_block(128, 64, dropout_rate),
                nn.Conv2d(64, 2 * max_detections_per_cell, kernel_size=1)
            )

        # Mode transformer in pred heads: independent self-attention per head
        elif self_attention:
            # Spatial head
            self.spatial_temporal_proj = self._conv3d_proj(256)
            self.spatial_transformer = self._make_transformer(
                256, transformer_heads, transformer_layers, dropout_rate
            )
            self.spatial_head = self._make_conv_head(256, 3, max_detections_per_cell, dropout_rate)

            # Direction head
            self.direction_temporal_proj = self._conv3d_proj(256)
            self.direction_transformer = self._make_transformer(
                256, transformer_heads, transformer_layers, dropout_rate
            )
            self.direction_head = self._make_conv_head(256, 2, max_detections_per_cell, dropout_rate)

            # Temporal head
            self.temporal_temporal_proj = self._conv3d_proj(256)
            self.temporal_transformer = self._make_transformer(
                256, transformer_heads, transformer_layers, dropout_rate
            )
            self.temporal_head = self._make_conv_head(256, 2, max_detections_per_cell, dropout_rate)

        # Mode cross attention pos and temporal component in heads: cross-attention (temporal queries <- spatial keys/values)

        else:  # cross_attention=True
            # Spatial head (self-attention; output sequence reused by temporal)
            self.spatial_temporal_proj = self._conv3d_proj(256)
            self.spatial_transformer = self._make_transformer(
                256, transformer_heads, transformer_layers, dropout_rate
            )
            self.spatial_head = self._make_conv_head(256, 3, max_detections_per_cell, dropout_rate)

            # Direction head (independent self-attention)
            self.direction_temporal_proj = self._conv3d_proj(256)
            self.direction_transformer = self._make_transformer(
                256, transformer_heads, transformer_layers, dropout_rate
            )
            self.direction_head = self._make_conv_head(256, 2, max_detections_per_cell, dropout_rate)

            # Temporal head: self-attention then cross-attention onto spatial seq
            self.temporal_temporal_proj = self._conv3d_proj(256)
            self.temporal_self_transformer = self._make_transformer(
                256, transformer_heads, transformer_layers, dropout_rate
            )

            # Cross-attention: temporal queries, spatial keys/values
            self.temporal_cross_attention = nn.MultiheadAttention(
                embed_dim=256,
                num_heads=transformer_heads,
                dropout=dropout_rate,
                batch_first=True
            )
            self.temporal_cross_attn_norm = nn.LayerNorm(256)

            self.temporal_cross_ffn = nn.Sequential(
                nn.Linear(256, 512),
                nn.GELU(),
                nn.Dropout(dropout_rate),
                nn.Linear(512, 256),
                nn.Dropout(dropout_rate)
            )
            self.temporal_cross_ffn_norm = nn.LayerNorm(256)

            self.temporal_head = self._make_conv_head(256, 2, max_detections_per_cell, dropout_rate)

        self._initialize_weights()

    # Builder helpers
    def _conv3d_proj(self, channels: int) -> nn.Sequential:
        """1 x 1 x 1 Conv3d projection with GroupNorm + ReLU."""
        return nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=(1, 1, 1), bias=False),
            nn.GroupNorm(16, channels),
            nn.SiLU(inplace=True)
        )

    def _make_transformer(
        self, d_model: int, nhead: int, num_layers: int, dropout: float
    ) -> nn.TransformerEncoder:
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=512,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        return nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def _make_conv_head(
        self, in_channels: int, num_params: int, max_det: int, dropout_rate: float
    ) -> nn.Sequential:
        """Standard 3-stage conv head: 256 -> 128 -> 64 -> output."""
        return nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 128),
            nn.SiLU(inplace=True),
            nn.Dropout2d(dropout_rate),
            self._depthwise_block(128, 64, dropout_rate),
            nn.Conv2d(64, num_params * max_det, kernel_size=1)
        )

    def _depthwise_block(self, in_channels: int, out_channels: int, dropout_rate: float) -> nn.Sequential:
        """Depthwise separable convolution block."""
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1,
                      groups=in_channels, bias=False),
            nn.GroupNorm(8, in_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(inplace=True),
            nn.Dropout2d(dropout_rate)
        )

    def _initialize_weights(self):
        """Kaiming init for conv layers; special small-weight init for final detection convs."""
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

        # confidence bias only on spatial head (index 0 = confidence logit)
        # sigmoid(-4.6) approx 0.01, matches true prior of approx 0.128% positive cells
        # avoids starting at sigmoid(0)=0.5 for all cells
        # only apply it to spatial head bcs only spatial head has confidence
        spatial_final = self.spatial_head[-1]
        if spatial_final.bias is not None:
            spatial_final.bias.data[0] = -4.6

    # Attention helper func
    def _run_self_attention(self, backbone_feat, proj, transformer):
        """
        Project -> reshape -> self-attention -> return (sequence, mean-pooled spatial map).

        Args:
            backbone_feat: (B, C, T, H, W)
            proj:          Conv3d projection module
            transformer:   TransformerEncoder module

        Returns:
            attended_seq:  (B*H*W, T, C)  — kept for optional cross-attention reuse
            pooled_feat:   (B, C, H, W)   — mean-pooled over T
        """
        B, C, T, H, W = backbone_feat.shape
        feat = proj(backbone_feat) # (B, C, T, H, W)
        feat = feat.permute(0, 3, 4, 2, 1) # (B, H, W, T, C)
        feat = feat.reshape(B * H * W, T, C) # (B*H*W, T, C)
        attended = transformer(feat) # (B*H*W, T, C)
        pooled = attended.mean(dim=1) # (B*H*W, C)
        pooled = pooled.view(B, H, W, C).permute(0, 3, 1, 2) # (B, C, H, W)
        return attended, pooled

    # Forward pass
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input video tensor (B, C, T, H, W)

        Returns:
            Detections (B, grid_h, grid_w, max_det, 7)
            Last dim: [confidence, x, y, dir_x, dir_y, t_start, t_end]
        """
        # Backbone
        if self.use_gradient_checkpointing and self.training:
            backbone_feat = torch.utils.checkpoint.checkpoint(
                self.backbone, x, use_reentrant=False
            )
        else:
            backbone_feat = self.backbone(x)

        # Mode Baseline Model: 3D-conv
        if not self.use_self_attention and not self.use_cross_attention:
            pooled_feat = self.temporal_pool(backbone_feat) # (B, C, H, W)
            projected = self.feature_projection(pooled_feat)
            shared = self.shared_features(projected)

            spatial_output = self.spatial_head(shared)
            direction_output = self.direction_head(shared)

            temporal_feat = self.temporal_features(backbone_feat) # (B, C, 1, H, W)
            temporal_feat = temporal_feat.squeeze(2) # (B, C, H, W)
            temporal_output = self.temporal_head(temporal_feat)

        # Mode transformer model in pred heads: independent self-attention per head
        elif self.use_self_attention:
            B, C, T, H, W = backbone_feat.shape

            _, spatial_pooled = self._run_self_attention(
                backbone_feat, self.spatial_temporal_proj, self.spatial_transformer
            )
            spatial_output = self.spatial_head(spatial_pooled)

            _, direction_pooled = self._run_self_attention(
                backbone_feat, self.direction_temporal_proj, self.direction_transformer
            )
            direction_output = self.direction_head(direction_pooled)

            _, temporal_pooled = self._run_self_attention(
                backbone_feat, self.temporal_temporal_proj, self.temporal_transformer
            )
            temporal_output = self.temporal_head(temporal_pooled)

        # Mode cross attention pos and temporal in pred heads: cross-attention (temporal -> spatial) 
        else:
            B, C, T, H, W = backbone_feat.shape

            # Spatial head: self-attention; keep sequence for cross-attn
            spatial_seq, spatial_pooled = self._run_self_attention(
                backbone_feat, self.spatial_temporal_proj, self.spatial_transformer
            )
            spatial_output = self.spatial_head(spatial_pooled)

            # Direction head: independent self-attention
            _, direction_pooled = self._run_self_attention(
                backbone_feat, self.direction_temporal_proj, self.direction_transformer
            )
            direction_output = self.direction_head(direction_pooled)

            # Temporal head: self-attention then cross-attention onto spatial_seq
            temporal_feat = self.temporal_temporal_proj(backbone_feat) # (B, C, T, H, W)
            temporal_feat = temporal_feat.permute(0, 3, 4, 2, 1) # (B, H, W, T, C)
            temporal_feat = temporal_feat.reshape(B * H * W, T, C) # (B*H*W, T, C)

            # Self-attention over temporal sequence
            temporal_self_out = self.temporal_self_transformer(temporal_feat)  # (B*H*W, T, C)

            # Cross-attention: query=temporal, key/value=spatial_seq
            cross_out, _ = self.temporal_cross_attention(
                query=temporal_self_out,
                key=spatial_seq,
                value=spatial_seq
            ) # (B*H*W, T, C)

            # Residual + norm
            cross_out = self.temporal_cross_attn_norm(temporal_self_out + cross_out)

            # FFN + residual + norm
            ffn_out = self.temporal_cross_ffn(cross_out)
            cross_out = self.temporal_cross_ffn_norm(cross_out + ffn_out)

            # Mean-pool and reshape back to spatial map
            temporal_pooled = cross_out.mean(dim=1) # (B*H*W, C)
            temporal_pooled = temporal_pooled.view(B, H, W, C).permute(0, 3, 1, 2)  # (B, C, H, W)
            temporal_output = self.temporal_head(temporal_pooled)

        # Combine all head outputs
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
        return self.apply_output_activations(detections)


    def _reshape_head_output(self, output, batch_size, grid_h, grid_w, num_params):
        """
        Reshape (B, max_det*num_params, H, W) -> (B, grid_h, grid_w, max_det, num_params).
        """
        output = output.view(
            batch_size,
            self.max_detections_per_cell * num_params,
            grid_h,
            grid_w
        ).permute(0, 2, 3, 1) # (B, grid_h, grid_w, max_det*num_params)

        return output.view(
            batch_size,
            grid_h,
            grid_w,
            self.max_detections_per_cell,
            num_params
        )

    def apply_output_activations(self, detections: torch.Tensor) -> torch.Tensor:
        """
        Normalize raw head outputs:
          - position  -> sigmoid  (0, 1)
          - direction -> L2-norm  (unit vector)
          - t offsets -> sigmoid  (0, 1)
          - confidence left raw (handled by loss)
        """
        confidence = detections[..., 0:1]
        position   = torch.sigmoid(detections[..., 1:3])
        direction  = F.normalize(detections[..., 3:5], p=2, dim=-1)
        offsets    = torch.sigmoid(detections[..., 5:7])

        return torch.cat([confidence, position, direction, offsets], dim=-1)