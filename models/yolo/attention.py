"""
Attention mechanism modules for YOLO-based defect detection.

Attention mechanisms help the model focus on defect regions,
improving detection performance especially for small defects.

Implemented modules:
- SE (Squeeze-and-Excitation) Attention
- CBAM (Convolutional Block Attention Module)
- CA (Coordinate Attention)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SEAttention(nn.Module):
    """
    Squeeze-and-Excitation Attention Module.

    Performs channel-wise attention by:
    1. Squeeze: Global average pooling to get channel-wise statistics
    2. Excitation: Two FC layers to learn channel-wise attention weights

    Reference:
        Squeeze-and-Excitation Networks (Hu et al., 2018)
    """

    def __init__(self, channel: int, reduction: int = 16):
        """
        Args:
            channel: Number of input channels
            reduction: Channel reduction ratio
        """
        super(SEAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ChannelAttention(nn.Module):
    """
    Channel Attention Module for CBAM.

    Uses both average and max pooling to compute channel attention.
    """

    def __init__(self, in_channels: int, reduction: int = 16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module for CBAM.

    Computes spatial attention map from channel-wise statistics.
    """

    def __init__(self, kernel_size: int = 7):
        super(SpatialAttention, self).__init__()

        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(x_cat)
        return self.sigmoid(out)


class CBAMAttention(nn.Module):
    """
    CBAM (Convolutional Block Attention Module).

    Combines channel attention and spatial attention sequentially.
    Applied after a convolutional block to refine features.

    Reference:
        CBAM: Convolutional Block Attention Module (Woo et al., 2018)
    """

    def __init__(self, in_channels: int, reduction: int = 16,
                 kernel_size: int = 7):
        """
        Args:
            in_channels: Number of input channels
            reduction: Channel reduction ratio for channel attention
            kernel_size: Kernel size for spatial attention
        """
        super(CBAMAttention, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        # Channel attention first
        x = x * self.channel_attention(x)
        # Then spatial attention
        x = x * self.spatial_attention(x)
        return x


class CAAttention(nn.Module):
    """
    CA (Coordinate Attention) Module.

    Captures long-range dependencies along two spatial directions
    (height and width) separately, encoding positional information.

    Particularly effective for detecting elongated defects like scratches.

    Reference:
        Coordinate Attention for Efficient Mobile Network Design
        (Hou et al., 2021)
    """

    def __init__(self, in_channels: int, out_channels: int = None,
                 reduction: int = 32):
        """
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels (same as input if None)
            reduction: Channel reduction ratio
        """
        super(CAAttention, self).__init__()

        if out_channels is None:
            out_channels = in_channels

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.reduction = reduction

        hidden_channels = max(in_channels // reduction, 8)

        # Horizontal pooling (X-direction)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        # Vertical pooling (Y-direction)
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        # Convolution to learn attention
        self.conv1 = nn.Conv2d(in_channels, hidden_channels,
                               kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.act = nn.Hardswish(inplace=True)

        # Attention weights for height direction
        self.conv_h = nn.Conv2d(hidden_channels, out_channels,
                                kernel_size=1, stride=1, padding=0)
        # Attention weights for width direction
        self.conv_w = nn.Conv2d(hidden_channels, out_channels,
                                kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()

        # Horizontal pooling: (N, C, H, 1)
        x_h = self.pool_h(x)
        # Vertical pooling: (N, C, 1, W) -> (N, C, W, 1)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        # Concatenate along spatial dimension
        y = torch.cat([x_h, x_w], dim=2)  # (N, C, H+W, 1)

        # Learn attention
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        # Split back into height and width
        x_h, x_w = torch.split(y, [h, w], dim=2)

        # Restore shape
        x_w = x_w.permute(0, 1, 3, 2)  # (N, C, 1, W)

        # Compute attention weights
        a_h = self.conv_h(x_h).sigmoid()  # (N, C, H, 1)
        a_w = self.conv_w(x_w).sigmoid()  # (N, C, 1, W)

        # Apply attention
        out = identity * a_h * a_w

        return out


class MultiScaleFusion(nn.Module):
    """
    Multi-scale Feature Fusion Module.

    Fuses features from different scales to improve detection
    of defects with varying sizes.

    Uses lateral connections and top-down/bottom-up fusion
    similar to FPN/PAN structures.
    """

    def __init__(self, in_channels_list: list, out_channels: int = 256):
        """
        Args:
            in_channels_list: List of input channel sizes from different scales
            out_channels: Number of output channels for fused features
        """
        super(MultiScaleFusion, self).__init__()

        self.num_scales = len(in_channels_list)
        self.out_channels = out_channels

        # 1x1 convolutions to unify channel dimensions
        self.lateral_convs = nn.ModuleList()
        for in_channels in in_channels_list:
            self.lateral_convs.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )

        # Fusion convolutions
        self.fuse_convs = nn.ModuleList()
        for _ in range(self.num_scales):
            self.fuse_convs.append(
                nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )

    def forward(self, features):
        """
        Args:
            features: List of feature maps from different scales
                      (ordered from high resolution to low resolution)

        Returns:
            List of fused feature maps at each scale
        """
        # Project to common channel dimension
        lateral_features = []
        for i, feat in enumerate(features):
            lateral_features.append(self.lateral_convs[i](feat))

        # Top-down fusion (from low res to high res)
        fused_features = [None] * self.num_scales
        fused_features[-1] = lateral_features[-1]

        for i in range(self.num_scales - 2, -1, -1):
            # Upsample lower resolution feature
            upsampled = F.interpolate(
                fused_features[i + 1],
                size=lateral_features[i].shape[2:],
                mode='nearest'
            )
            # Add lateral connection
            fused_features[i] = lateral_features[i] + upsampled

        # Apply fusion convolutions
        for i in range(self.num_scales):
            fused_features[i] = self.fuse_convs[i](fused_features[i])

        return fused_features


def get_attention_module(attention_type: str, channels: int, **kwargs):
    """
    Factory function to get attention module by name.

    Args:
        attention_type: Type of attention ("SE", "CBAM", "CA", or None)
        channels: Number of input channels
        **kwargs: Additional arguments for the attention module

    Returns:
        Attention module or nn.Identity() if attention_type is None
    """
    if attention_type is None or attention_type.lower() == "none":
        return nn.Identity()
    elif attention_type.upper() == "SE":
        reduction = kwargs.get('reduction', 16)
        return SEAttention(channels, reduction=reduction)
    elif attention_type.upper() == "CBAM":
        reduction = kwargs.get('reduction', 16)
        kernel_size = kwargs.get('kernel_size', 7)
        return CBAMAttention(channels, reduction=reduction, kernel_size=kernel_size)
    elif attention_type.upper() == "CA":
        reduction = kwargs.get('reduction', 32)
        return CAAttention(channels, reduction=reduction)
    else:
        raise ValueError(f"Unknown attention type: {attention_type}. "
                         f"Supported: SE, CBAM, CA")
