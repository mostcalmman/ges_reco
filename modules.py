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


class MotionExcitationLite(nn.Module):
    """Motion Excitation Lite — simplified without depthwise convolution.

    Directly computes temporal differences on squeezed features without
    the intermediate 3x3 depthwise convolution. This reduces parameters
    and computation while still capturing motion information.

    Architecture:
        x: (B*T, C, H, W)
        ↓ squeeze: Conv2d(C, C/r, 1) → (B*T, C/r, H, W)
        ↓ bn → reshape to (B, T, C/r, H, W)
        ↓ diff: F[t+1] - F[t] (NO conv)
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
        super(MotionExcitationLite, self).__init__()
        self.n_segment = n_segment
        reduced_channels = channels // reduction

        self.squeeze = nn.Conv2d(channels, reduced_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(reduced_channels)
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

        # Direct diff (no conv)
        x_p3 = x3[:, 1:] - x3[:, :-1]  # (B, T-1, C/r, H, W) - motion features

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
# MARK: Parallel ME-TSM Residual Blocks (Scheme B)
# --------------------------

class ParallelMETSMResBlock(nn.Module):
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
        super(ParallelMETSMResBlock, self).__init__()
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
            return torch.ones_like(x)[:, :, :1, :1]

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
        out = shifted * me_weight + shifted  # broadcasting: (B*T,C,H,W) * (B*T,C,1,1)

        # Conv layers
        out = F.relu(self.bn1(self.conv1(out)))
        out = self.bn2(self.conv2(out))

        out += identity
        out = F.relu(out)
        return out  # (B*T, C_out, H_out, W_out)

class ParallelMETSMAddResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, n_segment=8, reduction=4):
        super(ParallelMETSMAddResBlock, self).__init__()
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
            return torch.ones_like(x)[:, :, :1, :1]

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

class ParallelMELiteTSMResBlock(nn.Module):
    """Residual block with PARALLEL Motion Excitation Lite and TSM (Scheme B, Lite version).

    In this scheme, ME Lite and TSM happen in PARALLEL on the original input x:
    - ME Lite extracts sigmoid attention weights (B*T, C, 1, 1) from x
    - TSM performs temporal shift on x → shifted
    - Fuse: output = shifted * me_weight + shifted (i.e., shifted * (1 + weight))
    - Then pass to conv pipeline

    ME Lite skips the depthwise 3x3 convolution, computing temporal differences
    directly on squeezed features for reduced parameters.

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
        super(ParallelMELiteTSMResBlock, self).__init__()
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

        # ME Lite components (inline, no depthwise conv)
        reduced_channels = in_channels // reduction
        self.squeeze = nn.Conv2d(in_channels, reduced_channels, kernel_size=1, bias=False)
        self.me_bn = nn.BatchNorm2d(reduced_channels)
        self.expand = nn.Conv2d(reduced_channels, in_channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.pad = (0, 0, 0, 0, 0, 0, 0, 1)  # temporal padding at end

    def _get_me_attention(self, x):
        """Extract motion attention weights from input x (Lite version).

        Returns:
            Tensor of shape (B*T, C, 1, 1) with values in [0, 1].
        """
        if self.n_segment <= 1:
            return torch.ones_like(x)[:, :, :1, :1]

        nt, c, h, w = x.size()
        n_batch = nt // self.n_segment

        # Squeeze
        x3 = self.squeeze(x)  # (B*T, C/r, H, W)
        x3 = self.me_bn(x3)

        # Reshape for temporal operations
        x3 = x3.view(n_batch, self.n_segment, -1, h, w)  # (B, T, C/r, H, W)

        # Direct diff (no depthwise conv)
        x_p3 = x3[:, 1:] - x3[:, :-1]  # (B, T-1, C/r, H, W) - motion features

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

        # Parallel ME Lite + TSM fusion
        me_weight = self._get_me_attention(x)  # (B*T, C, 1, 1)
        shifted = temporal_shift(x, self.n_segment)
        out = shifted * me_weight + shifted  # broadcasting: (B*T,C,H,W) * (B*T,C,1,1)

        # Conv layers
        out = F.relu(self.bn1(self.conv1(out)))
        out = self.bn2(self.conv2(out))

        out += identity
        out = F.relu(out)
        return out  # (B*T, C_out, H_out, W_out)


# --------------------------
# MARK: ME-Before-TSM Residual Blocks (Scheme A)
# --------------------------

class MEBeforeTSMResBlock(nn.Module):
    """Residual block with Motion Excitation BEFORE Temporal Shift (Scheme A).

    Applies Motion Excitation module BEFORE temporal shift to ensure ME sees
    clean channel semantics without TSM-shift interference. This is the
    opposite ordering from TSMMEResBlock.

    Architecture:
        identity = shortcut(x)              # use original x for shortcut
        out = me(x)                        # APPLY ME FIRST on clean features
        out = temporal_shift(out)          # THEN apply TSM on ME output
        out = ReLU(BN(Conv2d(out)))       -- 3x3, may downsample via stride
        out = BN(Conv2d(out))             -- 3x3, stride=1
        out = ReLU(out + identity)

    Rationale:
        ME captures temporal motion via frame differencing. By applying ME
        before TSM, we ensure the motion-sensitive channel excitation is
        computed on unshifted features, then the motion-enhanced features
        are temporally shifted for further processing.

    Args:
        in_channels: Input channel count.
        out_channels: Output channel count.
        stride: Stride for the first convolution (spatial downsampling).
        n_segment: Number of temporal segments for TSM.
        reduction: Channel reduction ratio for ME module (default 4).

    Reference:
        Wang et al., "ACTION-Net: Multipath Excitation for Action Recognition",
        CVPR 2021, Section 3.3.
    """

    def __init__(self, in_channels, out_channels, stride=1, n_segment=8, reduction=4):
        super(MEBeforeTSMResBlock, self).__init__()
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

        # Motion Excitation module (applied BEFORE TSM)
        self.me = MotionExcitation(channels=in_channels, n_segment=n_segment, reduction=reduction)

    def forward(self, x):
        # x: (B*T, C, H, W)
        # Shortcut uses original x (before ME)
        identity = self.shortcut(x)

        # Motion excitation FIRST on clean features
        out = self.me(x)

        # THEN apply temporal shift on ME output
        out = temporal_shift(out, self.n_segment)

        # Conv layers
        out = F.relu(self.bn1(self.conv1(out)))
        out = self.bn2(self.conv2(out))

        out += identity
        out = F.relu(out)
        return out  # (B*T, C_out, H_out, W_out)


class MELiteBeforeTSMResBlock(nn.Module):
    """Residual block with Motion Excitation Lite BEFORE Temporal Shift (Scheme A).

    Applies Motion Excitation Lite module BEFORE temporal shift. This is the
    opposite ordering from TSMMELiteResBlock, with the same rationale:
    ME sees clean channel semantics before TSM shift interference.

    Architecture:
        identity = shortcut(x)              # use original x for shortcut
        out = me_lite(x)                    # APPLY ME LITE FIRST on clean features
        out = temporal_shift(out)           # THEN apply TSM on ME output
        out = ReLU(BN(Conv2d(out)))        -- 3x3, may downsample via stride
        out = BN(Conv2d(out))              -- 3x3, stride=1
        out = ReLU(out + identity)

    Args:
        in_channels: Input channel count.
        out_channels: Output channel count.
        stride: Stride for the first convolution (spatial downsampling).
        n_segment: Number of temporal segments for TSM.
        reduction: Channel reduction ratio for ME module (default 4).

    Reference:
        Wang et al., "ACTION-Net: Multipath Excitation for Action Recognition",
        CVPR 2021, Section 3.3.
    """

    def __init__(self, in_channels, out_channels, stride=1, n_segment=8, reduction=4):
        super(MELiteBeforeTSMResBlock, self).__init__()
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

        # Motion Excitation Lite module (applied BEFORE TSM)
        self.me = MotionExcitationLite(channels=in_channels, n_segment=n_segment, reduction=reduction)

    def forward(self, x):
        # x: (B*T, C, H, W)
        # Shortcut uses original x (before ME)
        identity = self.shortcut(x)

        # Motion excitation Lite FIRST on clean features
        out = self.me(x)

        # THEN apply temporal shift on ME output
        out = temporal_shift(out, self.n_segment)

        # Conv layers
        out = F.relu(self.bn1(self.conv1(out)))
        out = self.bn2(self.conv2(out))

        out += identity
        out = F.relu(out)
        return out  # (B*T, C_out, H_out, W_out)


class TSMMEResBlock(nn.Module):
    """Residual block with TSM and Motion Excitation.

    Combines temporal shift module (TSM) with Motion Excitation for
    capturing motion information.

    Architecture:
        identity = shortcut(x)
        out = temporal_shift(x)
        out = me(out)                   # INSERT ME AFTER TSM
        out = ReLU(BN(Conv2d(out)))     -- 3x3, may downsample via stride
        out = BN(Conv2d(out))           -- 3x3, stride=1
        out = ReLU(out + identity)

    Args:
        in_channels: Input channel count.
        out_channels: Output channel count.
        stride: Stride for the first convolution (spatial downsampling).
        n_segment: Number of temporal segments for TSM.
        reduction: Channel reduction ratio for ME module (default 4).

    Reference:
        Wang et al., "ACTION-Net: Multipath Excitation for Action Recognition",
        CVPR 2021, Section 3.3.
    """

    def __init__(self, in_channels, out_channels, stride=1, n_segment=8, reduction=4):
        super(TSMMEResBlock, self).__init__()
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

        # Motion Excitation module
        self.me = MotionExcitation(channels=in_channels, n_segment=n_segment, reduction=reduction)

    def forward(self, x):
        # x: (B*T, C, H, W)
        identity = self.shortcut(x)

        # Temporal shift
        out = temporal_shift(x, self.n_segment)

        # Motion excitation after TSM
        out = self.me(out)

        # Conv layers
        out = F.relu(self.bn1(self.conv1(out)))
        out = self.bn2(self.conv2(out))

        out += identity
        out = F.relu(out)
        return out  # (B*T, C_out, H_out, W_out)


class TSMMELiteResBlock(nn.Module):
    """Residual block with TSM and Motion Excitation Lite (simplified version).

    Combines temporal shift module (TSM) with Motion Excitation Lite for
    reduced parameters while still capturing motion information.

    Architecture:
        identity = shortcut(x)
        out = temporal_shift(x)
        out = me_lite(out)               # INSERT ME LITE AFTER TSM
        out = ReLU(BN(Conv2d(out)))     -- 3x3, may downsample via stride
        out = BN(Conv2d(out))           -- 3x3, stride=1
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
        super(TSMMELiteResBlock, self).__init__()
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

        # Motion Excitation Lite module
        self.me = MotionExcitationLite(channels=in_channels, n_segment=n_segment, reduction=reduction)

    def forward(self, x):
        # x: (B*T, C, H, W)
        identity = self.shortcut(x)

        # Temporal shift
        out = temporal_shift(x, self.n_segment)

        # Motion excitation Lite after TSM
        out = self.me(out)

        # Conv layers
        out = F.relu(self.bn1(self.conv1(out)))
        out = self.bn2(self.conv2(out))

        out += identity
        out = F.relu(out)
        return out  # (B*T, C_out, H_out, W_out)
