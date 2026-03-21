"""
Reusable neural network modules for video gesture recognition.

Contains building blocks shared across model architectures:
- temporal_shift: Zero-parameter temporal modeling via channel shifting (TSM)
- ConvGRUCell / ConvGRU: Convolutional GRU for spatiotemporal sequence modeling
- TSMResBlock: Residual block with integrated temporal shift

Reference:
    TSM: Lin et al., "TSM: Temporal Shift Module for Efficient Video
         Understanding", ICCV 2019. arXiv:1811.08383
    ConvGRU: Ballas et al., "Delving Deeper into Convolutional Networks
             for Learning Video Representations", ICLR 2016. arXiv:1511.06432
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
        return out
