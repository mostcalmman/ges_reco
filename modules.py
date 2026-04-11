"""
Reusable neural network modules for video gesture recognition.

Contains building blocks shared across model architectures:
- temporal_shift: Zero-parameter temporal modeling via channel shifting (TSM)
- ConvGRUCell / ConvGRU: Convolutional GRU for spatiotemporal sequence modeling
- TSMResBlock: Residual block with integrated temporal shift
- TSMMEResBlock: TSM + Motion Excitation (ME after TSM)
- MEBeforeTSMResBlock: Motion Excitation BEFORE TSM (Scheme A)
- MELiteBeforeTSMResBlock: Motion Excitation Lite BEFORE TSM (Scheme A)
- ParallelMETSMResBlock: ME and TSM in PARALLEL (Scheme B)
- ParallelMELiteTSMResBlock: ME Lite and TSM in PARALLEL (Scheme B, Lite)

Reference:
    TSM: Lin et al., "TSM: Temporal Shift Module for Efficient Video
         Understanding", ICCV 2019. arXiv:1811.08383
    ConvGRU: Ballas et al., "Delving Deeper into Convolutional Networks
             for Learning Video Representations", ICLR 2016. arXiv:1511.06432
    ACTION-Net: Wang et al., "ACTION-Net: Multipath Excitation for Action
                Recognition", CVPR 2021.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------
# MARK: Temporal Shift Module
# --------------------------

def temporal_shift(x, n_segment, fold_div=8):
    """Temporal Shift Module (TSM) — zero-parameter temporal modeling.

    Shifts a fraction of channels forward/backward along the time axis to
    enable information exchange between adjacent frames. This is a faithful
    re-implementation of the non-inplace variant from the original TSM repo
    (ops/temporal_shift.py :: TemporalShift.shift).

    Shift strategy (default fold_div=8):
        - 1/8 channels: shifted forward  (t+1 -> t)
        - 1/8 channels: shifted backward (t-1 -> t)
        - 6/8 channels: unchanged

    Boundary handling:
        - First frame has no predecessor  -> backward-shifted channels are zero
        - Last  frame has no successor    -> forward-shifted  channels are zero

    Args:
        x: Input tensor, shape (B*T, C, H, W).
        n_segment: Number of temporal segments T. B*T must be divisible by T.
        fold_div: Fraction denominator for shift width. Default 8.

    Returns:
        Tensor of same shape (B*T, C, H, W) with temporal shift applied.
    """
    nt, c, h, w = x.size()
    n_batch = nt // n_segment

    x = x.view(n_batch, n_segment, c, h, w)

    fold = c // fold_div
    out = torch.zeros_like(x)

    # Forward shift: copy frame t+1 into position t
    out[:, :-1, :fold] = x[:, 1:, :fold]
    # Backward shift: copy frame t-1 into position t
    out[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]
    # Identity: remaining channels pass through unchanged
    out[:, :, 2 * fold:] = x[:, :, 2 * fold:]

    return out.view(nt, c, h, w)


# --------------------------
# MARK: Convolutional GRU
# --------------------------

class ConvGRUCell(nn.Module):
    """Single-step Convolutional GRU cell.

    Replaces fully-connected GRU gates with 2D convolutions to preserve
    spatial structure in feature maps.

    Gate equations (following Ballas et al. 2016):
        z_t = sigma(W_z * [x_t, h_{t-1}])          -- update gate
        r_t = sigma(W_r * [x_t, h_{t-1}])          -- reset gate
        n_t = tanh(W_n * [x_t, r_t . h_{t-1}])     -- candidate state
        h_t = (1 - z_t) . h_{t-1} + z_t . n_t      -- new hidden state

    where * denotes 2D convolution and . denotes element-wise product.
    When z_t -> 1, the cell adopts the new candidate; when z_t -> 0,
    it retains the previous hidden state.

    Args:
        input_channels: Number of channels in the input feature map.
        hidden_channels: Number of channels in the hidden state.
        kernel_size: Convolution kernel size for all gates. Default 3.
    """

    def __init__(self, input_channels, hidden_channels, kernel_size=3):
        super(ConvGRUCell, self).__init__()
        self.hidden_channels = hidden_channels
        padding = kernel_size // 2

        # Update gate: input is [x_t, h_{t-1}] concatenated along channels
        self.conv_z = nn.Conv2d(
            input_channels + hidden_channels, hidden_channels,
            kernel_size=kernel_size, padding=padding
        )
        # Reset gate
        self.conv_r = nn.Conv2d(
            input_channels + hidden_channels, hidden_channels,
            kernel_size=kernel_size, padding=padding
        )
        # Candidate state: input is [x_t, r_t * h_{t-1}]
        self.conv_n = nn.Conv2d(
            input_channels + hidden_channels, hidden_channels,
            kernel_size=kernel_size, padding=padding
        )

    def forward(self, x, hidden):
        """
        Args:
            x: Current input, shape (B, C_in, H, W).
            hidden: Previous hidden state, shape (B, C_hidden, H, W).

        Returns:
            New hidden state, shape (B, C_hidden, H, W).
        """
        combined = torch.cat([x, hidden], dim=1)         # (B, C_in + C_hidden, H, W)
        z = torch.sigmoid(self.conv_z(combined))          # update gate
        r = torch.sigmoid(self.conv_r(combined))          # reset gate
        combined_reset = torch.cat([x, r * hidden], dim=1)
        n = torch.tanh(self.conv_n(combined_reset))       # candidate state

        # Standard GRU update: z=1 -> adopt candidate, z=0 -> keep old state
        h_new = (1 - z) * hidden + z * n
        return h_new


class ConvGRU(nn.Module):
    """Convolutional GRU sequence processor.

    Iterates a ConvGRUCell over the temporal dimension of a 5D tensor and
    returns the final hidden state. Hidden state is zero-initialized.

    Args:
        input_channels: Channel count of each input frame's feature map.
        hidden_channels: Channel count of the hidden state.
        kernel_size: Convolution kernel size for all gates. Default 3.
    """

    def __init__(self, input_channels, hidden_channels, kernel_size=3):
        super(ConvGRU, self).__init__()
        self.hidden_channels = hidden_channels
        self.cell = ConvGRUCell(input_channels, hidden_channels, kernel_size)

    def forward(self, x):
        """
        Args:
            x: Input sequence, shape (B, T, C, H, W).

        Returns:
            Final hidden state, shape (B, C_hidden, H, W).
        """
        b, t, c, h, w = x.size()
        hidden = torch.zeros(b, self.hidden_channels, h, w, device=x.device)
        for i in range(t):
            hidden = self.cell(x[:, i], hidden)
        return hidden


# --------------------------
# MARK: TSM Residual Block
# --------------------------

class TSMResBlock(nn.Module):
    """Residual block with temporal shift inserted before the first convolution.

    Architecture:
        identity = shortcut(x)
        out = temporal_shift(x)
        out = ReLU(BN(Conv2d(out)))    -- 3x3, may downsample via stride
        out = BN(Conv2d(out))          -- 3x3, stride=1
        out = ReLU(out + identity)

    When input/output channels differ or stride > 1, a 1x1 convolution
    shortcut is used for the residual connection.

    Args:
        in_channels: Input channel count.
        out_channels: Output channel count.
        stride: Stride for the first convolution (spatial downsampling).
        n_segment: Number of temporal segments for TSM.
    """

    def __init__(self, in_channels, out_channels, stride=1, n_segment=8):
        super(TSMResBlock, self).__init__()
        self.n_segment = n_segment

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Projection shortcut when dimensions change
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)

        # Temporal shift before first conv (zero extra params / FLOPs)
        out = temporal_shift(x, self.n_segment)
        out = F.relu(self.bn1(self.conv1(out)))
        out = self.bn2(self.conv2(out))

        out += identity
        out = F.relu(out)
        return out  # (B*T, C_out, H_out, W_out)


# --------------------------
# MARK: Motion Excitation Modules
# --------------------------

class MotionExcitation(nn.Module):
    """Motion Excitation module for temporal motion feature extraction.

    Captures motion information by computing temporal differences between
    adjacent frames in the feature space, then applies channel-wise excitation
    to highlight motion-sensitive channels.

    Architecture (per ACTION-Net paper, CVPR 2021):
        x: (B*T, C, H, W)
        ↓ squeeze: Conv2d(C, C/r, 1) → (B*T, C/r, H, W)
        ↓ bn → reshape to (B, T, C/r, H, W)
        ↓ conv: depthwise 3x3 on each frame
        ↓ split+diff: (conv(F[t+1]) - F[t]) for t=0..T-2
        ↓ pad: add zero frame at end → (B, T, C/r, H, W)
        ↓ gap + expand + sigmoid → attention mask M
        ↓ output: x * M + x

    Args:
        channels: Input/output channel count (C).
        n_segment: Number of temporal segments (T).
        reduction: Channel reduction ratio for squeeze (default 4).
                   Reduced channels = channels // reduction.

    Reference:
        Wang et al., "ACTION-Net: Multipath Excitation for Action Recognition",
        CVPR 2021, Section 3.3.
    """

    def __init__(self, channels, n_segment, reduction=4):
        super(MotionExcitation, self).__init__()
        self.n_segment = n_segment
        reduced_channels = channels // reduction

        self.squeeze = nn.Conv2d(channels, reduced_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(reduced_channels)
        self.conv = nn.Conv2d(reduced_channels, reduced_channels, kernel_size=3,
                              padding=1, groups=reduced_channels, bias=False)  # depthwise
        self.expand = nn.Conv2d(reduced_channels, channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.pad = (0, 0, 0, 0, 0, 0, 0, 1)  # temporal padding at end

    def forward(self, x):
        # x: (B*T, C, H, W)
        if self.n_segment <= 1:
            return x

        nt, c, h, w = x.size()
        n_batch = nt // self.n_segment

        # Squeeze
        x3 = self.squeeze(x)  # (B*T, C/r, H, W)
        x3 = self.bn(x3)

        # Reshape for temporal operations
        x3 = x3.view(n_batch, self.n_segment, -1, h, w)  # (B, T, C/r, H, W)

        # Conv on temporal sequence
        x3_reshaped = x3.view(nt, -1, h, w)  # (B*T, C/r, H, W)
        x3_conv = self.conv(x3_reshaped)
        x3_conv = x3_conv.view(n_batch, self.n_segment, -1, h, w)  # (B, T, C/r, H, W)

        # Split and diff
        x3_plus0 = x3[:, :-1]  # (B, T-1, C/r, H, W) - original first T-1 frames
        x3_plus1 = x3_conv[:, 1:]  # (B, T-1, C/r, H, W) - convolved last T-1 frames
        x_p3 = x3_plus1 - x3_plus0  # (B, T-1, C/r, H, W) - motion features

        # Pad temporal dimension
        x_p3 = F.pad(x_p3, self.pad, mode="constant", value=0)  # (B, T, C/r, H, W)

        # Reshape, pool, expand, excite
        x_p3 = x_p3.view(nt, -1, h, w)  # (B*T, C/r, H, W)
        x_p3 = self.avg_pool(x_p3)  # (B*T, C/r, 1, 1)
        x_p3 = self.expand(x_p3)  # (B*T, C, 1, 1)
        x_p3 = self.sigmoid(x_p3)

        # Residual excitation
        return x * x_p3 + x  # (B*T, C, H, W)


# --------------------------
# MARK: TMF
# --------------------------

class TMFResBlock(nn.Module):
    """Residual block with PARALLEL Motion Excitation and TSM (Scheme B).

    In this scheme, ME and TSM happen in PARALLEL on the original input x:
    - ME extracts sigmoid attention weights (B*T, C, 1, 1) from x
    - TSM performs temporal shift on x → shifted
    - Fuse: output = shifted * me_weight + shifted (i.e., shifted * (1 + weight))
    - Then pass to conv pipeline

    This differs from TSMMEResBlock (Scheme C) where ME is applied AFTER TSM.

    Architecture:
        identity = shortcut(x)
        me_weight = _get_me_attention(x)    # (B*T, C, 1, 1) in [0,1]
        shifted = temporal_shift(x)        # (B*T, C, H, W)
        fused = shifted * me_weight + shifted
        out = ReLU(BN(Conv2d(fused)))       -- 3x3, may downsample via stride
        out = BN(Conv2d(out))               -- 3x3, stride=1
        out = ReLU(out + identity)

    Args:
        in_channels: Input channel count.
        out_channels: Output channel count.
        stride: Stride for the first convolution (spatial downsampling).
        n_segment: Number of temporal segments for TSM.
        reduction: Channel reduction ratio for ME module (default 4).

    Reference:
        Wang et al., "ACTION-Net: Multipath Excitation for Action Recognition",
        CVPR 2021, Section 3.4.
    """

    def __init__(self, in_channels, out_channels, stride=1, n_segment=8, reduction=4):
        super(TMFResBlock, self).__init__()
        self.n_segment = n_segment

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Projection shortcut when dimensions change
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        # ME components (inline, not as separate module)
        reduced_channels = in_channels // reduction
        self.squeeze = nn.Conv2d(in_channels, reduced_channels, kernel_size=1, bias=False)
        self.me_bn = nn.BatchNorm2d(reduced_channels)
        self.me_conv = nn.Conv2d(reduced_channels, reduced_channels, kernel_size=3,
                                  padding=1, groups=reduced_channels, bias=False)  # depthwise
        self.expand = nn.Conv2d(reduced_channels, in_channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.pad = (0, 0, 0, 0, 0, 0, 0, 1)  # temporal padding at end

    def _get_me_attention(self, x):
        """Extract motion attention weights from input x.

        Returns:
            Tensor of shape (B*T, C, 1, 1) with values in [0, 1].
        """
        if self.n_segment <= 1:
            # Keep fusion as identity when there is no temporal axis.
            return torch.zeros_like(x)[:, :, :1, :1]

        nt, c, h, w = x.size()
        n_batch = nt // self.n_segment

        # Squeeze
        x3 = self.squeeze(x)  # (B*T, C/r, H, W)
        x3 = self.me_bn(x3)

        # Reshape for temporal operations
        x3 = x3.view(n_batch, self.n_segment, -1, h, w)  # (B, T, C/r, H, W)

        # Conv on temporal sequence
        x3_reshaped = x3.view(nt, -1, h, w)  # (B*T, C/r, H, W)
        x3_conv = self.me_conv(x3_reshaped)
        x3_conv = x3_conv.view(n_batch, self.n_segment, -1, h, w)  # (B, T, C/r, H, W)

        # Split and diff
        x3_plus0 = x3[:, :-1]  # (B, T-1, C/r, H, W) - original first T-1 frames
        x3_plus1 = x3_conv[:, 1:]  # (B, T-1, C/r, H, W) - convolved last T-1 frames
        x_p3 = x3_plus1 - x3_plus0  # (B, T-1, C/r, H, W) - motion features

        # Pad temporal dimension
        x_p3 = F.pad(x_p3, self.pad, mode="constant", value=0)  # (B, T, C/r, H, W)

        # Reshape, pool, expand
        x_p3 = x_p3.view(nt, -1, h, w)  # (B*T, C/r, H, W)
        x_p3 = self.avg_pool(x_p3)  # (B*T, C/r, 1, 1)
        x_p3 = self.expand(x_p3)  # (B*T, C, 1, 1)
        x_p3 = self.sigmoid(x_p3)

        return x_p3

    def forward(self, x):
        # x: (B*T, C, H, W)
        identity = self.shortcut(x)

        # Parallel ME + TSM fusion
        me_weight = self._get_me_attention(x)  # (B*T, C, 1, 1)
        shifted = temporal_shift(x, self.n_segment)
        out = x * me_weight + shifted  # broadcasting: (B*T,C,H,W) * (B*T,C,1,1)

        # Conv layers
        out = F.relu(self.bn1(self.conv1(out)))
        out = self.bn2(self.conv2(out))

        out += identity
        out = F.relu(out)
        return out  # (B*T, C_out, H_out, W_out)
    

class TMF2ResBlock(nn.Module):
    """
     MultiFrame Spatial ME + TSM 并联融合的残差块。

     核心改动：
     1) 差分 D 由两部分组成（各占 reduced_channels 的一半）：
         - 短程: conv(X_t) - X_{t-1}
         - 长程: conv(X_{t+1}) - X_{t-1}
     2) D 进入双分支：
         - Channel Path: GAP -> 1x1 expand -> sigmoid -> (B*T, C, 1, 1)
         - Spatial Path: 通道均值 -> 7x7 conv -> sigmoid -> (B*T, 1, H, W)
     3) 融合：x * channel_weight * spatial_weight + temporal_shift(x)

    """
    def __init__(self, in_channels, out_channels, stride=1, n_segment=8, reduction=4):
        super(TMF2ResBlock, self).__init__()
        self.n_segment = n_segment

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Projection shortcut when dimensions change
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        # Multi-Frame Motion Excitation
        reduced_channels = in_channels // reduction
        self.short_channels = reduced_channels // 2
        self.long_channels = reduced_channels - self.short_channels

        self.squeeze = nn.Conv2d(in_channels, reduced_channels, kernel_size=1, bias=False)
        self.me_bn = nn.BatchNorm2d(reduced_channels)
        self.me_conv = nn.Conv2d(reduced_channels, reduced_channels, kernel_size=3,
                                  padding=1, groups=reduced_channels, bias=False)  # depthwise

        # Channel path
        self.expand = nn.Conv2d(reduced_channels, in_channels, kernel_size=1, bias=False)
        self.channel_sigmoid = nn.Sigmoid()

        # Spatial path: single-channel map -> 7x7 conv -> sigmoid
        self.spatial_conv = nn.Conv2d(1, 1, kernel_size=7, padding=3, bias=False)
        self.spatial_sigmoid = nn.Sigmoid()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def _get_me_attention(self, x):
        """Extract channel and spatial attention from multi-frame motion diff.

        Returns:
            channel_weight: (B*T, C, 1, 1), values in [0, 1]
            spatial_weight: (B*T, 1, H, W), values in [0, 1]
        """
        if self.n_segment <= 1:
            # handle single-frame, not mature way, but this situation shouldn't happen in practice
            channel_weight = torch.ones(nt, c, 1, 1, device=x.device, dtype=x.dtype)
            spatial_weight = torch.ones(nt, 1, h, w, device=x.device, dtype=x.dtype)
            return channel_weight, spatial_weight

        nt, c, h, w = x.size()
        n_batch = nt // self.n_segment

        # Squeeze
        x_squeeze_4d = self.me_bn(self.squeeze(x))         # (B*T, C/r, H, W)
        x_conv_4d = self.me_conv(x_squeeze_4d)             # (B*T, C/r, H, W)

        # 统一转换为 5D 以便进行时序差分
        x3 = x_squeeze_4d.view(n_batch, self.n_segment, -1, h, w)      # (B, T, C/r, H, W)
        x3_conv = x_conv_4d.view(n_batch, self.n_segment, -1, h, w)    # (B, T, C/r, H, W)

        # Split channels: short/long each takes half reduced channels.
        d_short = x3.new_zeros(n_batch, self.n_segment, self.short_channels, h, w)
        if self.short_channels > 0 and self.n_segment > 1:
            # 短程: conv(X_t) - X_{t-1}; t=0 置零
            d_short[:, 1:] = x3_conv[:, 1:, :self.short_channels] - x3[:, :-1, :self.short_channels]

        d_long = x3.new_zeros(n_batch, self.n_segment, self.long_channels, h, w)
        if self.long_channels > 0 and self.n_segment > 2:
            # 长程: conv(X_{t+1}) - X_{t-1}; t=0 和 t=T-1 置零
            d_long[:, 1:-1] = x3_conv[:, 2:, self.short_channels:] - x3[:, :-2, self.short_channels:]

        d = torch.cat([d_short, d_long], dim=2)  # (B, T, C/r, H, W)

        # Channel Path
        d_channel = d.reshape(nt, -1, h, w)  # (B*T, C/r, H, W)
        channel_weight = self.avg_pool(d_channel)  # (B*T, C/r, 1, 1)
        channel_weight = self.expand(channel_weight)  # (B*T, C, 1, 1)
        channel_weight = self.channel_sigmoid(channel_weight)

        # Spatial Path
        d_spatial = d.mean(dim=2, keepdim=True)  # (B, T, 1, H, W)
        d_spatial = d_spatial.reshape(nt, 1, h, w)  # (B*T, 1, H, W)
        spatial_weight = self.spatial_conv(d_spatial)
        spatial_weight = self.spatial_sigmoid(spatial_weight)

        return channel_weight, spatial_weight

    def forward(self, x):
        # x: (B*T, C, H, W)
        identity = self.shortcut(x)

        # Parallel MultiFrame Spatial ME + TSM fusion
        channel_weight, spatial_weight = self._get_me_attention(x)
        shifted = temporal_shift(x, self.n_segment)
        out = x * channel_weight * spatial_weight + shifted

        # Conv layers
        out = F.relu(self.bn1(self.conv1(out)))
        out = self.bn2(self.conv2(out))

        out += identity
        out = F.relu(out)
        return out  # (B*T, C_out, H_out, W_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (B*T, C, H, W)
        max_out, _ = torch.max(x, dim=1, keepdim=True) # (B*T, 1, H, W)
        avg_out = torch.mean(x, dim=1, keepdim=True)   # (B*T, 1, H, W)
        
        # 拼接并计算空间掩码
        y = torch.cat([max_out, avg_out], dim=1)       # (B*T, 2, H, W)
        y = self.conv(y)                               # (B*T, 1, H, W)
        
        return x * self.sigmoid(y)


# --------------------------
# MARK: Adaptive Channel Soft Shift (ACSS)
# --------------------------

class ACSS_Module(nn.Module):
    """Adaptive Channel Soft Shift — learnable soft temporal shift.

    Replaces the hard channel-partitioned shift of TSM with a per-channel
    soft blending of temporal neighbors.  Each channel independently learns
    a 3-way softmax distribution over {pull-from-t+1, keep-current, pull-from-t-1}.

    At initialization the logits are set to mimic TSM defaults (1/8 left,
    1/8 right, 6/8 static) so the module starts as a differentiable TSM and
    can be fine-tuned from a TSM checkpoint without accuracy loss.

    Args:
        channels: Number of input channels (C).
        n_segment: Number of temporal segments (T).
        fold_div: Denominator for the TSM-style init partition. Default 8.
    """

    def __init__(self, channels, n_segment, fold_div=8):
        super(ACSS_Module, self).__init__()
        self.channels = channels
        self.n_segment = n_segment
        self.fold_div = fold_div
        fold = channels // fold_div  # channels to shift per direction

        # Learnable logits: (C, 3) — columns are [left, static, right]
        self.shift_logits = nn.Parameter(torch.zeros(channels, 3))

        # Initialize to mimic TSM hard-shift (sharp softmax via large logits)
        with torch.no_grad():
            # Channels [0, fold): favor left = pull from t+1
            self.shift_logits[:fold] = torch.tensor([10.0, 0.0, 0.0])
            # Channels [fold, 2*fold): favor right = pull from t-1
            self.shift_logits[fold: 2 * fold] = torch.tensor([0.0, 0.0, 10.0])
            # Channels [2*fold, C): favor static = keep current frame
            self.shift_logits[2 * fold:] = torch.tensor([0.0, 10.0, 0.0])

        # TODO: 约束注入点 — 可在此处添加 clamp 或正则化逻辑限制前后移位通道总数不超过 C/2

    def forward(self, x, n_segment=None):
        """
        Args:
            x: Input tensor, shape (B*T, C, H, W).
            n_segment: Override for temporal segments. Uses self.n_segment if None.

        Returns:
            Tensor of same shape (B*T, C, H, W) with soft temporal shift applied.
        """
        if n_segment is None:
            n_segment = self.n_segment

        # Guard: no temporal axis → identity
        if n_segment <= 1:
            return x

        BT, C, H, W = x.size()
        B = BT // n_segment
        T = n_segment

        # Soft weights via softmax — float32 for numerical stability
        weights = F.softmax(self.shift_logits, dim=-1,
                            dtype=torch.float32)          # (C, 3)
        w_left = weights[:, 0]    # (C,) — weight for x_{t+1} (pull from future)
        w_static = weights[:, 1]  # (C,) — weight for x_t   (keep current)
        w_right = weights[:, 2]   # (C,) — weight for x_{t-1} (pull from past)

        # Reshape to 5-D for temporal indexing
        x_5d = x.view(B, T, C, H, W)                     # (B, T, C, H, W)

        # Build temporal neighbors with zero-padding at boundaries
        x_prev = torch.zeros_like(x_5d)                   # X_{t-1}
        x_prev[:, 1:] = x_5d[:, :-1]                      # first frame is zero

        x_next = torch.zeros_like(x_5d)                   # X_{t+1}
        x_next[:, :-1] = x_5d[:, 1:]                      # last frame is zero

        # Reshape weights for broadcasting: (1, 1, C, 1, 1)
        w_left = w_left.view(1, 1, C, 1, 1)
        w_static = w_static.view(1, 1, C, 1, 1)
        w_right = w_right.view(1, 1, C, 1, 1)

        # Soft temporal blend
        out = (w_left * x_next          # pull from t+1 (future → current)
               + w_static * x_5d        # keep current frame
               + w_right * x_prev)      # pull from t-1 (past → current)
        # out shape: (B, T, C, H, W)

        return out.view(BT, C, H, W)                      # (B*T, C, H, W)


class ACSSResBlock(nn.Module):
    """Residual block with ACSS (Adaptive Channel Soft Shift) before the first conv.

    Drop-in replacement for TSMResBlock: same interface, same residual
    structure, but uses a learnable soft shift (ACSS_Module) instead of the
    hard channel-partitioned temporal_shift.

    Architecture:
        identity = shortcut(x)
        out = acss(x)                      -- learnable soft temporal shift
        out = ReLU(BN(Conv2d(out)))        -- 3x3, may downsample via stride
        out = BN(Conv2d(out))              -- 3x3, stride=1
        out = ReLU(out + identity)

    Args:
        in_channels: Input channel count.
        out_channels: Output channel count.
        stride: Stride for the first convolution (spatial downsampling).
        n_segment: Number of temporal segments for ACSS.
    """

    def __init__(self, in_channels, out_channels, stride=1, n_segment=8):
        super(ACSSResBlock, self).__init__()
        self.n_segment = n_segment

        # Adaptive soft temporal shift (replaces hard temporal_shift)
        self.acss = ACSS_Module(in_channels, n_segment)

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Projection shortcut when dimensions change
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)

        # Learnable soft temporal shift before first conv
        out = self.acss(x)
        out = F.relu(self.bn1(self.conv1(out)))
        out = self.bn2(self.conv2(out))

        out += identity
        out = F.relu(out)
        return out  # (B*T, C_out, H_out, W_out)


# --------------------------
# MARK: Motion-Guided Dynamic Shift (MGDS)
# --------------------------

class MGDS_Module(nn.Module):
    """Motion-Guided Dynamic Shift — instance-adaptive temporal shift.

    Unlike ACSS which learns static per-channel shift weights, MGDS generates
    per-frame, per-channel shift weights dynamically from motion intensity
    features.  A lightweight bottleneck MLP maps the pre-sigmoid channel
    attention vector (from ME's channel path) to a 3-way softmax distribution
    over {pull-from-t+1, keep-current, pull-from-t-1} for every channel.

    This allows the shift pattern to adapt to the actual motion content of
    each video clip at inference time.

    Args:
        channels: Number of input channels (C).
        n_segment: Number of temporal segments (T).
        reduction: Bottleneck reduction ratio for MLP. Default 4.
    """

    def __init__(self, channels, n_segment, reduction=4):
        super(MGDS_Module, self).__init__()
        self.channels = channels
        self.n_segment = n_segment

        # Bottleneck MLP weight generator: v_motion (C) → shift weights (3*C)
        self.fc1 = nn.Linear(channels, channels // reduction)   # squeeze: C → C//r
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channels // reduction, 3 * channels)  # expand: C//r → 3*C

    def forward(self, x, v_motion):
        """
        Args:
            x: Main branch features, shape (B*T, C, H, W).
            v_motion: Pre-sigmoid motion intensity from ME channel path,
                      shape (B*T, C, 1, 1). Raw logits, NOT sigmoid output.

        Returns:
            Tensor of same shape (B*T, C, H, W) with motion-guided
            dynamic temporal shift applied.
        """
        n_segment = self.n_segment

        # Guard: no temporal axis → identity
        if n_segment <= 1:
            return x

        BT, C, H, W = x.size()

        # ---- Step 1: Weight generation via bottleneck MLP ----
        v = v_motion.view(BT, C)                              # (B*T, C)
        v = self.relu(self.fc1(v))                             # (B*T, C//r)
        v = self.fc2(v)                                        # (B*T, 3*C)
        v = v.view(BT, C, 3)                                   # (B*T, C, 3) — per-channel triplet [left, static, right]

        # Normalize to probability distribution per channel
        weights = F.softmax(v, dim=-1, dtype=torch.float32)    # (B*T, C, 3)
        # weights[i, c, :] = [α_c, β_c, γ_c] 表示第 i 帧第 c 通道的时序融合权重

        # ---- Step 2: Temporal neighbor extraction ----
        B = BT // n_segment
        T = n_segment

        x_5d = x.view(B, T, C, H, W)                          # (B, T, C, H, W)

        x_prev = torch.zeros_like(x_5d)                        # X_{t-1}, zero-pad first frame
        x_prev[:, 1:] = x_5d[:, :-1]                           # (B, T, C, H, W)

        x_next = torch.zeros_like(x_5d)                        # X_{t+1}, zero-pad last frame
        x_next[:, :-1] = x_5d[:, 1:]                           # (B, T, C, H, W)

        # ---- Step 3: Dynamic shift fusion ----
        weights_5d = weights.view(B, T, C, 3)                  # (B, T, C, 3)
        w_left, w_static, w_right = weights_5d.unbind(dim=-1)  # each (B, T, C)

        # Add spatial dims for broadcasting: (B, T, C) → (B, T, C, 1, 1)
        w_left = w_left.unsqueeze(-1).unsqueeze(-1)             # (B, T, C, 1, 1)
        w_static = w_static.unsqueeze(-1).unsqueeze(-1)         # (B, T, C, 1, 1)
        w_right = w_right.unsqueeze(-1).unsqueeze(-1)           # (B, T, C, 1, 1)

        # Weighted blend of temporal neighbors
        out = (w_left * x_prev                                  # pull from t-1
               + w_static * x_5d                                # keep current
               + w_right * x_next)                              # pull from t+1
        # out shape: (B, T, C, H, W)

        return out.view(BT, C, H, W)                           # (B*T, C, H, W)


# --------------------------
# MARK: TMF3 — ME + MGDS Fusion Block
# --------------------------

class TMF3ResBlock(nn.Module):
    """Residual block integrating Motion Excitation + MGDS with dual fusion modes.

    Combines the multi-frame spatial ME pipeline (from TMF2ResBlock) with the
    Motion-Guided Dynamic Shift (MGDS) module.  The ME pipeline produces three
    outputs: pre-sigmoid channel logits, channel attention, and spatial attention.
    The pre-sigmoid logits drive MGDS's instance-adaptive temporal shift.

    Fusion Mode A (Oracle, default):
        mgds_out = MGDS(x, v_pre_sigmoid)
        out = mgds_out * spatial_weight + x

    Fusion Mode B (Dual-path):
        mgds_out = MGDS(x, v_pre_sigmoid)
        out = x * channel_weight * spatial_weight + mgds_out * spatial_weight

    Then: out → conv1 → bn1 → relu → conv2 → bn2 → (+identity) → relu

    Args:
        in_channels: Input channel count.
        out_channels: Output channel count.
        stride: Stride for the first convolution (spatial downsampling).
        n_segment: Number of temporal segments (T).
        reduction: Channel reduction ratio for ME and MGDS. Default 4.
        fusion_mode: 'A' (Oracle) or 'B' (Dual-path). Default 'A'.
    """

    def __init__(self, in_channels, out_channels, stride=1, n_segment=8,
                 reduction=4, fusion_mode='A'):
        super(TMF3ResBlock, self).__init__()
        self.n_segment = n_segment
        self.fusion_mode = fusion_mode

        # ---- Conv pipeline (identical to TMF2ResBlock) ----
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Projection shortcut when dimensions change
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        # ---- Motion Excitation components (replicated from TMF2ResBlock) ----
        reduced_channels = in_channels // reduction
        self.short_channels = reduced_channels // 2
        self.long_channels = reduced_channels - self.short_channels

        self.squeeze = nn.Conv2d(in_channels, reduced_channels,
                                 kernel_size=1, bias=False)
        self.me_bn = nn.BatchNorm2d(reduced_channels)
        self.me_conv = nn.Conv2d(reduced_channels, reduced_channels, kernel_size=3,
                                  padding=1, groups=reduced_channels, bias=False)

        # Channel path
        self.expand = nn.Conv2d(reduced_channels, in_channels,
                                kernel_size=1, bias=False)
        self.channel_sigmoid = nn.Sigmoid()

        # Spatial path: single-channel map → 7×7 conv → sigmoid
        self.spatial_conv = nn.Conv2d(1, 1, kernel_size=7, padding=3, bias=False)
        self.spatial_sigmoid = nn.Sigmoid()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # ---- MGDS module ----
        self.mgds = MGDS_Module(in_channels, n_segment, reduction=reduction)

    def _get_me_attention(self, x):
        """Extract motion attention: pre-sigmoid logits, channel weight, spatial weight.

        Returns:
            v_pre_sigmoid: (B*T, C, 1, 1) — raw channel logits before sigmoid
            channel_weight: (B*T, C, 1, 1) — sigmoid of v_pre_sigmoid, values in [0, 1]
            spatial_weight: (B*T, 1, H, W) — spatial attention, values in [0, 1]
        """
        if self.n_segment <= 1:
            # Guard: no temporal axis → zero attention (identity-like)
            nt, c, h, w = x.size()
            v_pre_sigmoid = torch.zeros(nt, c, 1, 1, device=x.device, dtype=x.dtype)
            channel_weight = torch.zeros(nt, c, 1, 1, device=x.device, dtype=x.dtype)
            spatial_weight = torch.zeros(nt, 1, h, w, device=x.device, dtype=x.dtype)
            return v_pre_sigmoid, channel_weight, spatial_weight

        nt, c, h, w = x.size()
        n_batch = nt // self.n_segment

        # Squeeze: reduce channels
        x3 = self.squeeze(x)                                    # (B*T, C/r, H, W)
        x3 = self.me_bn(x3)                                     # (B*T, C/r, H, W)

        # Reshape for temporal operations
        x3 = x3.view(n_batch, self.n_segment, -1, h, w)         # (B, T, C/r, H, W)

        # Depthwise conv on temporal sequence
        x3_reshaped = x3.view(nt, -1, h, w)                     # (B*T, C/r, H, W)
        x3_conv = self.me_conv(x3_reshaped)                      # (B*T, C/r, H, W)
        x3_conv = x3_conv.view(n_batch, self.n_segment, -1, h, w)  # (B, T, C/r, H, W)

        # Split channels: short-range and long-range diffs
        d_short = x3.new_zeros(n_batch, self.n_segment, self.short_channels, h, w)
        if self.short_channels > 0 and self.n_segment > 1:
            # 短程: conv(X_t) - X_{t-1}; t=0 置零
            d_short[:, 1:] = (x3_conv[:, 1:, :self.short_channels]
                              - x3[:, :-1, :self.short_channels])  # (B, T-1, short_ch, H, W)

        d_long = x3.new_zeros(n_batch, self.n_segment, self.long_channels, h, w)
        if self.long_channels > 0 and self.n_segment > 2:
            # 长程: conv(X_{t+1}) - X_{t-1}; t=0 和 t=T-1 置零
            d_long[:, 1:-1] = (x3_conv[:, 2:, self.short_channels:]
                               - x3[:, :-2, self.short_channels:])  # (B, T-2, long_ch, H, W)

        d = torch.cat([d_short, d_long], dim=2)                  # (B, T, C/r, H, W)

        # ---- Channel Path ----
        d_channel = d.view(nt, -1, h, w)                         # (B*T, C/r, H, W)
        channel_pool = self.avg_pool(d_channel)                   # (B*T, C/r, 1, 1)
        v_pre_sigmoid = self.expand(channel_pool)                 # (B*T, C, 1, 1) — raw logits
        channel_weight = self.channel_sigmoid(v_pre_sigmoid)      # (B*T, C, 1, 1) — [0, 1]

        # ---- Spatial Path ----
        d_spatial = d.mean(dim=2, keepdim=True)                   # (B, T, 1, H, W)
        d_spatial = d_spatial.view(nt, 1, h, w)                   # (B*T, 1, H, W)
        spatial_weight = self.spatial_conv(d_spatial)              # (B*T, 1, H, W)
        spatial_weight = self.spatial_sigmoid(spatial_weight)      # (B*T, 1, H, W) — [0, 1]

        return v_pre_sigmoid, channel_weight, spatial_weight

    def forward(self, x):
        """
        Args:
            x: Input tensor, shape (B*T, C, H, W).

        Returns:
            Output tensor, shape (B*T, C_out, H_out, W_out).
        """
        # x: (B*T, C, H, W)
        identity = self.shortcut(x)                               # (B*T, C_out, H_out, W_out)

        # ME attention: three outputs
        v_pre_sigmoid, channel_weight, spatial_weight = self._get_me_attention(x)

        # Fusion
        if self.fusion_mode == 'A':
            # Oracle: MGDS gated by spatial attention + residual
            mgds_out = self.mgds(x, v_pre_sigmoid)                # (B*T, C, H, W)
            out = mgds_out * spatial_weight + x                   # (B*T, C, H, W)
        elif self.fusion_mode == 'B':
            # Dual-path: ME-weighted x + MGDS, both spatially gated
            mgds_out = self.mgds(x, v_pre_sigmoid)                # (B*T, C, H, W)
            out = (x * channel_weight * spatial_weight
                   + mgds_out * spatial_weight)                   # (B*T, C, H, W)
        else:
            raise ValueError(f"Unknown fusion_mode '{self.fusion_mode}', expected 'A' or 'B'")

        # Conv pipeline
        out = F.relu(self.bn1(self.conv1(out)))                   # (B*T, C_out, H_out, W_out)
        out = self.bn2(self.conv2(out))                           # (B*T, C_out, H_out, W_out)

        # Residual connection
        out += identity                                           # (B*T, C_out, H_out, W_out)
        out = F.relu(out)
        return out                                                # (B*T, C_out, H_out, W_out)