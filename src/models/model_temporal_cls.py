import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.video import r2plus1d_18
from torch.utils.checkpoint import checkpoint

from src.models.temporal import Soft_Temporal_Pool

# Same model as model.py, except the temporal head no longer collapses the
# time axis before predicting start/end. Instead it keeps T alive and
# predicts one "is this frame part of the waggle" logit per frame, per the
# per-frame classification plan for fixing the dance-level temporal IoU
# weakness. Spatial/direction heads are untouched -- they still pool over T
# the same way as before, since that asymmetry is the whole point (see the
# discussion on why "when does it happen" shouldn't be answered from a
# single pooled summary the way "where"/"which way" can be).
#
# Detection output layout changes from
#   [confidence, x, y, dir_x, dir_y, t_start, t_end]                (7 values)
# to
#   [confidence, x, y, dir_x, dir_y, t_logit_0, ..., t_logit_{T-1}] (5+T values)
# where t_logit_i is a raw (pre-sigmoid) logit for sampled frame i.


class R2Plus1D_YOLO_MultiHead(nn.Module):
    """
    Multi-head R2+1D YOLO model, temporal-head-as-classifier variant.
    See model.py for the original (regression-temporal-head) version and the
    fuller docstring on the three attention modes -- unchanged here.
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
        window_size: int = 16,
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
        self.window_size = window_size

        # 5 non-temporal params (conf, x, y, dir_x, dir_y) + one logit per frame
        self.detection_params = 5 + self.window_size
        self.total_outputs_per_cell = self.max_detections_per_cell * self.detection_params

        # Backbone: R(2+1)D-18 without final classification layers
        base_r2p1d = r2plus1d_18(pretrained=pretrained)
        self.backbone = nn.Sequential(*list(base_r2p1d.children())[:-3])

        # Mode baseline 3D conv: 3D-conv heads no attention effectively baseline model
        if not self_attention and not cross_attention:
            self.temporal_pool = Soft_Temporal_Pool(channels=256, reduction=4)

            self.feature_projection = nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=1, bias=False),
                nn.GroupNorm(16, 256),
                nn.ReLU(inplace=True),
                nn.Dropout2d(dropout_rate)
            )
            self.shared_features = self._depthwise_block(256, 256, dropout_rate)

            self.spatial_head = nn.Sequential(
                self._depthwise_block(256, 128, dropout_rate),
                self._depthwise_block(128, 64, dropout_rate),
                nn.Conv2d(64, 3 * max_detections_per_cell, kernel_size=1)
            )

            self.direction_head = nn.Sequential(
                self._depthwise_block(256, 128, dropout_rate),
                self._depthwise_block(128, 64, dropout_rate),
                nn.Conv2d(64, 2 * max_detections_per_cell, kernel_size=1)
            )

            # Temporal features: dropped the AdaptiveAvgPool3d that used to
            # collapse T here -- we need T alive for the per-frame head below
            self.temporal_features = nn.Sequential(
                nn.Conv3d(256, 256, kernel_size=(3, 1, 1), padding=(1, 0, 0), bias=False),
                nn.GroupNorm(16, 256),
                nn.SiLU(inplace=True),
            )
            # outputs 1 logit per detection slot (not 2), applied per-frame
            self.temporal_head = nn.Sequential(
                self._depthwise_block(256, 128, dropout_rate),
                self._depthwise_block(128, 64, dropout_rate),
                nn.Conv2d(64, 1 * max_detections_per_cell, kernel_size=1)
            )

        elif self_attention:
            self.spatial_temporal_proj = self._conv3d_proj(256)
            self.spatial_transformer = self._make_transformer(
                256, transformer_heads, transformer_layers, dropout_rate
            )
            self.spatial_head = self._make_conv_head(256, 3, max_detections_per_cell, dropout_rate)

            self.direction_temporal_proj = self._conv3d_proj(256)
            self.direction_transformer = self._make_transformer(
                256, transformer_heads, transformer_layers, dropout_rate
            )
            self.direction_head = self._make_conv_head(256, 2, max_detections_per_cell, dropout_rate)

            self.temporal_temporal_proj = self._conv3d_proj(256)
            self.temporal_transformer = self._make_transformer(
                256, transformer_heads, transformer_layers, dropout_rate
            )
            # 1 logit per frame per detection slot, not 2 params total
            self.temporal_head = self._make_conv_head(256, 1, max_detections_per_cell, dropout_rate)

        else:  # cross_attention=True
            self.spatial_temporal_proj = self._conv3d_proj(256)
            self.spatial_transformer = self._make_transformer(
                256, transformer_heads, transformer_layers, dropout_rate
            )
            self.spatial_head = self._make_conv_head(256, 3, max_detections_per_cell, dropout_rate)

            self.direction_temporal_proj = self._conv3d_proj(256)
            self.direction_transformer = self._make_transformer(
                256, transformer_heads, transformer_layers, dropout_rate
            )
            self.direction_head = self._make_conv_head(256, 2, max_detections_per_cell, dropout_rate)

            self.temporal_temporal_proj = self._conv3d_proj(256)
            self.temporal_self_transformer = self._make_transformer(
                256, transformer_heads, transformer_layers, dropout_rate
            )

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

            # 1 logit per frame per detection slot -- head runs once per
            # frame now instead of once per window (see forward())
            self.temporal_head = self._make_conv_head(256, 1, max_detections_per_cell, dropout_rate)

        self._initialize_weights()

    def _conv3d_proj(self, channels: int) -> nn.Sequential:
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
        return nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 128),
            nn.SiLU(inplace=True),
            nn.Dropout2d(dropout_rate),
            self._depthwise_block(128, 64, dropout_rate),
            nn.Conv2d(64, num_params * max_det, kernel_size=1)
        )

    def _depthwise_block(self, in_channels: int, out_channels: int, dropout_rate: float) -> nn.Sequential:
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
        spatial_final = self.spatial_head[-1]
        if spatial_final.bias is not None:
            spatial_final.bias.data[0] = -4.6

    def _upsample_temporal(self, x, target_T):
        """The R(2+1)D backbone temporally downsamples the input (16 frames
        in -> only 4 steps left in backbone_feat), so "keep T alive" by
        itself isn't enough -- there aren't window_size distinct backbone
        steps to begin with. Nearest-neighbor upsample back to window_size
        so every real input frame gets a logit (frames sharing a backbone
        step just share its label, which is the best we can do without
        redesigning the backbone's temporal stride)."""
        return F.interpolate(x, size=(target_T, x.shape[-2], x.shape[-1]), mode='nearest')

    def _run_self_attention(self, backbone_feat, proj, transformer):
        """
        Project -> reshape -> self-attention -> return (sequence, mean-pooled spatial map).
        Unchanged from model.py -- the temporal branch below just uses the
        returned sequence instead of the pooled map now.
        """
        B, C, T, H, W = backbone_feat.shape
        feat = proj(backbone_feat)
        feat = feat.permute(0, 3, 4, 2, 1)
        feat = feat.reshape(B * H * W, T, C)
        attended = transformer(feat)
        pooled = attended.mean(dim=1)
        pooled = pooled.view(B, H, W, C).permute(0, 3, 1, 2)
        return attended, pooled

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input video tensor (B, C, T, H, W)
        Returns:
            Detections (B, grid_h, grid_w, max_det, 5+window_size)
            Last dim: [confidence, x, y, dir_x, dir_y, t_logit_0..t_logit_{T-1}]
        """
        if self.use_gradient_checkpointing and self.training:
            backbone_feat = torch.utils.checkpoint.checkpoint(
                self.backbone, x, use_reentrant=False
            )
        else:
            backbone_feat = self.backbone(x)

        if not self.use_self_attention and not self.use_cross_attention:
            pooled_feat = self.temporal_pool(backbone_feat)
            projected = self.feature_projection(pooled_feat)
            shared = self.shared_features(projected)

            spatial_output = self.spatial_head(shared)
            direction_output = self.direction_head(shared)

            # temporal_feat keeps T alive: (B, C, T_backbone, H, W)
            temporal_feat = self.temporal_features(backbone_feat)
            temporal_feat = self._upsample_temporal(temporal_feat, self.window_size)  # (B, C, window_size, H, W)
            B_, C_, T_, H_, W_ = temporal_feat.shape
            # fold T into the batch dim so the existing 2D conv head can run
            # once per frame instead of once per window
            temporal_feat = temporal_feat.permute(0, 2, 1, 3, 4).reshape(B_ * T_, C_, H_, W_)
            temporal_output = self.temporal_head(temporal_feat)  # (B*T, max_det, H, W)
            temporal_output = temporal_output.view(B_, T_, self.max_detections_per_cell, H_, W_)
            temporal_output = temporal_output.permute(0, 3, 4, 2, 1)  # (B, H, W, max_det, T)

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

            # use the full attended sequence here instead of the pooled map
            temporal_seq, _ = self._run_self_attention(
                backbone_feat, self.temporal_temporal_proj, self.temporal_transformer
            )  # (B*H*W, T_backbone, C)
            temporal_seq = temporal_seq.view(B, H, W, T, C).permute(0, 4, 3, 1, 2)  # (B, C, T_backbone, H, W)
            temporal_seq = self._upsample_temporal(temporal_seq, self.window_size)  # (B, C, window_size, H, W)
            Tw = self.window_size
            temporal_seq = temporal_seq.permute(0, 2, 1, 3, 4).reshape(B * Tw, C, H, W)
            temporal_output = self.temporal_head(temporal_seq)  # (B*Tw, max_det, H, W)
            temporal_output = temporal_output.view(B, Tw, self.max_detections_per_cell, H, W)
            temporal_output = temporal_output.permute(0, 3, 4, 2, 1)  # (B, H, W, max_det, Tw)

        else:  # cross_attention
            B, C, T, H, W = backbone_feat.shape

            spatial_seq, spatial_pooled = self._run_self_attention(
                backbone_feat, self.spatial_temporal_proj, self.spatial_transformer
            )
            spatial_output = self.spatial_head(spatial_pooled)

            _, direction_pooled = self._run_self_attention(
                backbone_feat, self.direction_temporal_proj, self.direction_transformer
            )
            direction_output = self.direction_head(direction_pooled)

            temporal_feat = self.temporal_temporal_proj(backbone_feat)
            temporal_feat = temporal_feat.permute(0, 3, 4, 2, 1)
            temporal_feat = temporal_feat.reshape(B * H * W, T, C)

            temporal_self_out = self.temporal_self_transformer(temporal_feat)

            cross_out, _ = self.temporal_cross_attention(
                query=temporal_self_out,
                key=spatial_seq,
                value=spatial_seq
            )
            cross_out = self.temporal_cross_attn_norm(temporal_self_out + cross_out)

            ffn_out = self.temporal_cross_ffn(cross_out)
            cross_out = self.temporal_cross_ffn_norm(cross_out + ffn_out)  # (B*H*W, T, C)

            # this is the actual fix: keep T alive instead of cross_out.mean(dim=1),
            # then upsample T_backbone -> window_size (see _upsample_temporal)
            temporal_seq = cross_out.view(B, H, W, T, C).permute(0, 4, 3, 1, 2)  # (B, C, T_backbone, H, W)
            temporal_seq = self._upsample_temporal(temporal_seq, self.window_size)  # (B, C, window_size, H, W)
            Tw = self.window_size
            temporal_seq = temporal_seq.permute(0, 2, 1, 3, 4).reshape(B * Tw, C, H, W)
            temporal_output = self.temporal_head(temporal_seq)  # (B*Tw, max_det, H, W)
            temporal_output = temporal_output.view(B, Tw, self.max_detections_per_cell, H, W)
            temporal_output = temporal_output.permute(0, 3, 4, 2, 1)  # (B, H, W, max_det, Tw)

        batch_size = x.shape[0]
        actual_grid_h, actual_grid_w = spatial_output.shape[-2:]

        spatial_preds = self._reshape_head_output(
            spatial_output, batch_size, actual_grid_h, actual_grid_w, 3
        )
        direction_preds = self._reshape_head_output(
            direction_output, batch_size, actual_grid_h, actual_grid_w, 2
        )
        # temporal_output is already (B, grid_h, grid_w, max_det, T) from the
        # branches above -- no _reshape_head_output needed here, unlike
        # spatial/direction which still come out of their heads flattened.
        temporal_preds = temporal_output

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
        ).permute(0, 2, 3, 1)

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
          - temporal  -> left as raw logits (sigmoid happens at decode time,
            in eval_utils.yolo_to_img_space_cls) since BCEWithLogits in the
            loss wants raw logits for numerical stability
          - confidence left raw (handled by loss)
        """
        confidence = detections[..., 0:1]
        position   = torch.sigmoid(detections[..., 1:3])
        direction  = F.normalize(detections[..., 3:5], p=2, dim=-1)
        temporal_logits = detections[..., 5:5 + self.window_size]

        return torch.cat([confidence, position, direction, temporal_logits], dim=-1)
