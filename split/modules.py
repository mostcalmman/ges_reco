import torch
import torch.nn as nn
import torch.nn.functional as F


def temporal_shift(x, n_segment, fold_div=8):
    nt, c, h, w = x.size()
    n_batch = nt // n_segment

    x = x.view(n_batch, n_segment, c, h, w)

    fold = c // fold_div
    out = torch.zeros_like(x)

    out[:, :-1, :fold] = x[:, 1:, :fold]
    out[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]
    out[:, :, 2 * fold:] = x[:, :, 2 * fold:]

    return out.view(nt, c, h, w)


class TSMResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, n_segment=8):
        super().__init__()
        self.n_segment = n_segment

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = self.shortcut(x)

        out = temporal_shift(x, self.n_segment)
        out = F.relu(self.bn1(self.conv1(out)))
        out = self.bn2(self.conv2(out))

        out += identity
        out = F.relu(out)
        return out


class ParallelMETSMResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, n_segment=8, reduction=4):
        super().__init__()
        self.n_segment = n_segment

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        reduced_channels = max(1, in_channels // reduction)
        self.squeeze = nn.Conv2d(in_channels, reduced_channels, kernel_size=1, bias=False)
        self.me_bn = nn.BatchNorm2d(reduced_channels)
        self.me_conv = nn.Conv2d(
            reduced_channels,
            reduced_channels,
            kernel_size=3,
            padding=1,
            groups=reduced_channels,
            bias=False,
        )
        self.expand = nn.Conv2d(reduced_channels, in_channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.pad = (0, 0, 0, 0, 0, 0, 0, 1)

    def _get_me_attention(self, x):
        if self.n_segment <= 1:
            return torch.zeros_like(x)[:, :, :1, :1]

        nt, _, h, w = x.size()
        n_batch = nt // self.n_segment

        x3 = self.squeeze(x)
        x3 = self.me_bn(x3)

        x3 = x3.view(n_batch, self.n_segment, -1, h, w)

        x3_reshaped = x3.view(nt, -1, h, w)
        x3_conv = self.me_conv(x3_reshaped)
        x3_conv = x3_conv.view(n_batch, self.n_segment, -1, h, w)

        x3_plus0 = x3[:, :-1]
        x3_plus1 = x3_conv[:, 1:]
        x_p3 = x3_plus1 - x3_plus0

        x_p3 = F.pad(x_p3, self.pad, mode="constant", value=0)

        x_p3 = x_p3.view(nt, -1, h, w)
        x_p3 = self.avg_pool(x_p3)
        x_p3 = self.expand(x_p3)
        x_p3 = self.sigmoid(x_p3)

        return x_p3

    def forward(self, x):
        identity = self.shortcut(x)

        me_weight = self._get_me_attention(x)
        shifted = temporal_shift(x, self.n_segment)
        out = x * me_weight + shifted

        out = F.relu(self.bn1(self.conv1(out)))
        out = self.bn2(self.conv2(out))

        out += identity
        out = F.relu(out)
        return out
