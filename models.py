"""
Video gesture recognition model architectures.

All models accept input shape (B, T, 3, H, W) and output (B, num_classes).

Model variants:
    ResNetVideoModel             — ResNet18 backbone, temporal mean pooling
    ResNetGRUVideoModel          — ResNet18 backbone + GRU temporal aggregation
    LightweightTSMModel          — Shallow TSM-ResNet, no ConvGRU (< 3M params)
    UltraLightConvGRUModel       — 3-layer TSM-ResNet + ConvGRU (< 2.5M params)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from modules import TSMResBlock, ConvGRU

modelList = ['resnet', 'resnet_gru', 'lightweight_tsm', 'ultralight_convgru']

# --------------------------
# MARK: ResNet18 Baseline
# --------------------------

class ResNetVideoModel(nn.Module):
    """Baseline: pretrained ResNet18 with temporal mean pooling.

    Architecture: ResNet18 (frozen except layer4) -> AvgPool -> FC

    Args:
        num_classes: Number of output classes.
        freeze_backbone: If True, freeze all layers except layer4.
    """

    def __init__(self, num_classes, freeze_backbone=True):
        super(ResNetVideoModel, self).__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        if freeze_backbone:
            for param in resnet.parameters():
                param.requires_grad = False
            for param in resnet.layer4.parameters():
                param.requires_grad = True

        self.cnn = nn.Sequential(*list(resnet.children())[:-1])
        cnn_out_dim = 512

        self.fc = nn.Linear(cnn_out_dim, num_classes)

    def forward(self, x):
        b, t, c, h, w = x.size()
        x = x.view(b * t, c, h, w)
        cnn_features = self.cnn(x)
        cnn_features = cnn_features.view(b, t, -1)
        features = cnn_features.mean(dim=1)
        out = self.fc(features)
        return out


# --------------------------
# MARK: ResNet18 + GRU
# --------------------------

class ResNetGRUVideoModel(nn.Module):
    """ResNet18 backbone with GRU temporal aggregation.

    Architecture: ResNet18 (frozen except layer4) -> GRU -> FC

    Args:
        num_classes: Number of output classes.
        hidden_dim: GRU hidden state dimension.
        freeze_backbone: If True, freeze all layers except layer4.
    """

    def __init__(self, num_classes, hidden_dim, freeze_backbone=True):
        super(ResNetGRUVideoModel, self).__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        if freeze_backbone:
            for param in resnet.parameters():
                param.requires_grad = False
            for param in resnet.layer4.parameters():
                param.requires_grad = True

        self.cnn = nn.Sequential(*list(resnet.children())[:-1])
        cnn_out_dim = 512

        self.rnn = nn.GRU(
            input_size=cnn_out_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        b, t, c, h, w = x.size()
        x = x.view(b * t, c, h, w)
        cnn_features = self.cnn(x)
        cnn_features = cnn_features.view(b, t, -1)
        rnn_out, hidden = self.rnn(cnn_features)
        last_hidden = hidden[-1]
        out = self.fc(last_hidden)
        return out


# --------------------------
# MARK: Lightweight TSM Only
# --------------------------

class LightweightTSMModel(nn.Module):
    """Lightweight TSM-only model without ConvGRU.

    Architecture:
        Conv1 (3->32, stride=2) -> 4x TSMResBlock [32->64->128->256]
        -> temporal mean pooling -> GlobalAvgPool -> FC

    Uses temporal mean pooling instead of ConvGRU for lower parameter count.
    Target: < 3M parameters.

    Args:
        num_classes: Number of output classes (default 27 for Jester).
        n_segment: Number of temporal segments (sampled frames).
    """

    def __init__(self, num_classes=27, n_segment=8):
        super(LightweightTSMModel, self).__init__()
        self.n_segment = n_segment

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.layer1 = TSMResBlock(32, 32, stride=1, n_segment=n_segment)
        self.layer2 = TSMResBlock(32, 64, stride=2, n_segment=n_segment)
        self.layer3 = TSMResBlock(64, 128, stride=2, n_segment=n_segment)
        self.layer4 = TSMResBlock(128, 256, stride=2, n_segment=n_segment)

        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        """
        Args:
            x: (B, T, 3, H, W) video frame sequence.
        Returns:
            (B, num_classes) classification logits.
        """
        b, t, c, h, w = x.size()

        x = x.view(b * t, c, h, w)              # (B*T, 3, H, W)
        x = self.conv1(x)                       # (B*T, 32, H/2, W/2)
        x = self.layer1(x)                      # (B*T, 32, H/2, W/2)
        x = self.layer2(x)                      # (B*T, 64, H/4, W/4)
        x = self.layer3(x)                      # (B*T, 128, H/8, W/8)
        x = self.layer4(x)                      # (B*T, 256, H/16, W/16)

        _, c_out, h_out, w_out = x.size()
        x = x.view(b, t, c_out, h_out, w_out)   # (B, T, 256, H/16, W/16)
        x = x.mean(dim=1)                       # (B, 256, H/16, W/16)

        x = F.adaptive_avg_pool2d(x, (1, 1))    # (B, 256, 1, 1)
        x = x.view(b, -1)                       # (B, 256)
        out = self.fc(x)                        # (B, num_classes)
        return out


# --------------------------
# MARK: Ultra-light TSM + ConvGRU
# --------------------------

class UltraLightConvGRUModel(nn.Module):
    """Ultra-lightweight model: 3-layer TSM-ResNet with early ConvGRU.

    Architecture:
        Conv1 (3->32, stride=2) -> 3x TSMResBlock [32->64->128]
        -> SpatialPool (7x7) -> ConvGRU (128->128) -> GlobalAvgPool -> FC

    Removes layer4 entirely, attaching ConvGRU after layer3 at 128 channels
    for maximum parameter reduction. Spatial pooling before ConvGRU reduces
    the feature map size to limit ConvGRU compute and memory cost.
    Target: < 2.5M parameters.

    Args:
        num_classes: Number of output classes (default 27 for Jester).
        n_segment: Number of temporal segments (sampled frames).
    """

    def __init__(self, num_classes=27, n_segment=8):
        super(UltraLightConvGRUModel, self).__init__()
        self.n_segment = n_segment

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.layer1 = TSMResBlock(32, 32, stride=1, n_segment=n_segment)
        self.layer2 = TSMResBlock(32, 64, stride=2, n_segment=n_segment)
        self.layer3 = TSMResBlock(64, 128, stride=2, n_segment=n_segment)
        # No layer4 — ConvGRU attaches directly after layer3

        self.convgru = ConvGRU(input_channels=128, hidden_channels=128)

        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        """
        Args:
            x: (B, T, 3, H, W) video frame sequence.
        Returns:
            (B, num_classes) classification logits.
        """
        b, t, c, h, w = x.size()

        x = x.view(b * t, c, h, w)              # (B*T, 3, H, W)
        x = self.conv1(x)                       # (B*T, 32, H/2, W/2)
        x = self.layer1(x)                      # (B*T, 32, H/2, W/2)
        x = self.layer2(x)                      # (B*T, 64, H/4, W/4)
        x = self.layer3(x)                      # (B*T, 128, H/8, W/8)

        # 降低空间分辨率, 减少 ConvGRU 计算量和显存占用
        x = F.adaptive_avg_pool2d(x, (7, 7))    # (B*T, 128, 7, 7)

        _, c_out, h_out, w_out = x.size()
        x = x.view(b, t, c_out, h_out, w_out)   # (B, T, 128, 7, 7)
        x = self.convgru(x)                     # (B, 128, 7, 7)

        x = F.adaptive_avg_pool2d(x, (1, 1))    # (B, 128, 1, 1)
        x = x.view(b, -1)                       # (B, 128)
        out = self.fc(x)                        # (B, num_classes)
        return out
