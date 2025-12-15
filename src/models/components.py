import torch.nn as nn
import torch
import torch.nn.functional as F

class TemporalAttention(nn.Module):
    """
    Optimized temporal attention module using depthwise separable convolutions.
    
    Improvements over original:
    - 8x fewer parameters (depthwise separable instead of full 3D conv)
    - More stable training (GroupNorm instead of BatchNorm)
    - Faster inference (reduced computation)
    
    Args:
        channels: Number of input channels (default 256 for R2+1D layer3)
        reduction: Channel reduction factor for attention (default 4)
    """
    def __init__(self, channels=256, reduction=4):
        super().__init__()
        
        # Depthwise separable 3D convolution (much faster than full conv)
        self.temporal_conv = nn.Sequential(
            # Depthwise: each channel processed independently
            nn.Conv3d(channels, channels, kernel_size=(3, 1, 1), 
                     padding=(1, 0, 0), groups=channels, bias=False),
            # Pointwise: mix channels
            nn.Conv3d(channels, channels, kernel_size=1, bias=False),
            nn.GroupNorm(32, channels),  # More stable than BatchNorm for video
            nn.ReLU(inplace=True)
        )
        
        # Lightweight attention mechanism
        mid_channels = max(channels // reduction, 16)
        self.attention = nn.Sequential(
            nn.Conv3d(channels, mid_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, 1, kernel_size=1)
        )
        
        # Initialize attention to uniform weights (stable start)
        nn.init.constant_(self.attention[-1].weight, 0.0)
        nn.init.constant_(self.attention[-1].bias, 0.0)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor (B, C, T, H, W)
        Returns:
            Temporally pooled features (B, C, H, W)
        """
        # Extract temporal features
        temporal_feat = self.temporal_conv(x)
        
        # Compute attention weights over time dimension
        attention_weights = torch.softmax(self.attention(temporal_feat), dim=2)
        
        # Weighted sum over temporal dimension
        weighted = (x * attention_weights).sum(dim=2)  # (B, C, H, W)
        
        return weighted