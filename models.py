"""
Video gesture recognition model architectures.
All models accept input shape (B, T, 3, H, W) and output (B, num_classes).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from modules import *

modelList = ['resnet', 'resnet_gru', 'lightweight_tsm',
             'ultralight_gru', 'ultralight_me_gru', 'ultralight_me_lite_gru',
             'ultralight_me_before_gru', 'ultralight_parallel_me_gru',
             'ultralight_me_lite_before_gru', 'ultralight_parallel_me_lite_gru', 'me_before_1',
             'me_before_2', 'me_before_3', 'deeper', 'spatial_attention_1']


# --------------------------
# MARK: ResNet18
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
# ResnetGRU
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
# LightTSM
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
# MARK: UltraLightGRU
# TSM + Standard GRU, current Best
# --------------------------

class UltraLightGRUModel(nn.Module):
    """Ultra-lightweight model: 3-layer TSM-ResNet with standard GRU.

    Architecture:
        Conv1 (3->32, stride=2) -> 3x TSMResBlock [32->64->128]
        -> SpatialPool -> Flatten -> GRU -> FC

    Similar to UltraLightConvGRUModel but replaces ConvGRU with standard GRU.
    Spatial features are pooled before the GRU to reduce sequence dimension.
    Target: < 2.5M parameters.

    Args:
        num_classes: Number of output classes (default 27 for Jester).
        n_segment: Number of temporal segments (sampled frames).
        hidden_dim: GRU hidden state dimension.
    """

    def __init__(self, num_classes=27, n_segment=8, hidden_dim=128):
        super(UltraLightGRUModel, self).__init__()
        self.n_segment = n_segment
        self.hidden_dim = hidden_dim

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.layer1 = TSMResBlock(32, 32, stride=1, n_segment=n_segment)
        self.layer2 = TSMResBlock(32, 64, stride=2, n_segment=n_segment)
        self.layer3 = TSMResBlock(64, 128, stride=2, n_segment=n_segment)
        # No layer4 — GRU attaches directly after layer3

        # GRU input: AdaptiveAvgPool2d(1,1) collapses spatial dims to 128-d vector per frame
        self.gru = nn.GRU(
            input_size=128,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_dim, num_classes)

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

        x = F.adaptive_avg_pool2d(x, (1, 1))    # (B*T, 128, 1, 1)
        x = x.view(b, t, -1)                    # (B, T, 128)

        # GRU temporal aggregation
        rnn_out, hidden = self.gru(x)           # hidden: (1, B, hidden_dim)
        last_hidden = hidden[-1]                # (B, hidden_dim)

        out = self.fc(last_hidden)              # (B, num_classes)
        return out


# --------------------------
# MARK: UltraLightMEGRU
# TSM + ME (Motion Excitation) + Standard GRU
# --------------------------

class UltraLightMEGRUModel(nn.Module):
    """Ultra-lightweight model: TSM-ResNet with Motion Excitation in layer3 + GRU.

    Architecture:
        Conv1 (3->32, stride=2) -> TSMResBlock [32->32] -> TSMResBlock [32->64]
        -> TSMMEResBlock [64->128] (TSM -> ME -> Conv)
        -> SpatialPool -> Flatten -> GRU -> FC

    Based on UltraLightGRUModel with layer3 replaced by TSMMEResBlock,
    which inserts Motion Excitation after temporal shift for motion-aware
    feature excitation before convolution.

    Reference: ACTION-Net (Wang et al., CVPR 2021) Section 3.3

    Args:
        num_classes: Number of output classes (default 27 for Jester).
        n_segment: Number of temporal segments (sampled frames).
        hidden_dim: GRU hidden state dimension.
    """

    def __init__(self, num_classes=27, n_segment=8, hidden_dim=128):
        super(UltraLightMEGRUModel, self).__init__()
        self.n_segment = n_segment
        self.hidden_dim = hidden_dim

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.layer1 = TSMResBlock(32, 32, stride=1, n_segment=n_segment)
        self.layer2 = TSMResBlock(32, 64, stride=2, n_segment=n_segment)
        self.layer3 = TSMMEResBlock(64, 128, stride=2, n_segment=n_segment, reduction=4)

        # GRU input: AdaptiveAvgPool2d(1,1) collapses spatial dims to 128-d vector per frame
        self.gru = nn.GRU(
            input_size=128,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_dim, num_classes)

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

        x = F.adaptive_avg_pool2d(x, (1, 1))    # (B*T, 128, 1, 1)
        x = x.view(b, t, -1)                    # (B, T, 128)

        # GRU temporal aggregation
        rnn_out, hidden = self.gru(x)           # hidden: (1, B, hidden_dim)
        last_hidden = hidden[-1]                # (B, hidden_dim)

        out = self.fc(last_hidden)              # (B, num_classes)
        return out


# --------------------------
# UltraLightMELiteGRU
# TSM + Lite ME (Motion Excitation) + Standard GRU
# --------------------------

class UltraLightMELiteGRUModel(nn.Module):
    """Ultra-lightweight model: TSM-ResNet with Lite ME in layer3 + GRU.

    Architecture:
        Conv1 (3->32, stride=2) -> TSMResBlock [32->32] -> TSMResBlock [32->64]
        -> TSMMELiteResBlock [64->128] (TSM -> Lite ME -> Conv)
        -> SpatialPool -> Flatten -> GRU -> FC

    Based on UltraLightGRUModel with layer3 replaced by TSMMELiteResBlock,
    which uses a lite version of Motion Excitation with fewer parameters.

    Reference: ACTION-Net (Wang et al., CVPR 2021) Section 3.3

    Args:
        num_classes: Number of output classes (default 27 for Jester).
        n_segment: Number of temporal segments (sampled frames).
        hidden_dim: GRU hidden state dimension.
    """

    def __init__(self, num_classes=27, n_segment=8, hidden_dim=128):
        super(UltraLightMELiteGRUModel, self).__init__()
        self.n_segment = n_segment
        self.hidden_dim = hidden_dim

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.layer1 = TSMResBlock(32, 32, stride=1, n_segment=n_segment)
        self.layer2 = TSMResBlock(32, 64, stride=2, n_segment=n_segment)
        self.layer3 = TSMMELiteResBlock(64, 128, stride=2, n_segment=n_segment, reduction=4)

        # GRU input: AdaptiveAvgPool2d(1,1) collapses spatial dims to 128-d vector per frame
        self.gru = nn.GRU(
            input_size=128,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_dim, num_classes)

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

        x = F.adaptive_avg_pool2d(x, (1, 1))    # (B*T, 128, 1, 1)
        x = x.view(b, t, -1)                    # (B, T, 128)

        # GRU temporal aggregation
        rnn_out, hidden = self.gru(x)           # hidden: (1, B, hidden_dim)
        last_hidden = hidden[-1]                # (B, hidden_dim)

        out = self.fc(last_hidden)              # (B, num_classes)
        return out


# --------------------------
# UltraLightMEBeforeGRU
# TSM + ME-Before-TSM (Scheme A) + Standard GRU
# --------------------------

class UltraLightMEBeforeGRUModel(nn.Module):
    """Ultra-lightweight model: TSM-ResNet with ME-Before-TSM in layer3 + GRU.

    Architecture:
        Conv1 (3->32, stride=2) -> TSMResBlock [32->32] -> TSMResBlock [32->64]
        -> MEBeforeTSMResBlock [64->128] (ME first, then TSM -> Conv)
        -> SpatialPool -> Flatten -> GRU -> FC

    Based on UltraLightMEGRUModel but uses Scheme A (ME-before-TSM),
    where Motion Excitation is applied BEFORE temporal shift for 
    motion-aware feature excitation before temporal modeling.

    Reference: ACTION-Net (Wang et al., CVPR 2021) Section 3.3

    Args:
        num_classes: Number of output classes (default 27 for Jester).
        n_segment: Number of temporal segments (sampled frames).
        hidden_dim: GRU hidden state dimension.
    """

    def __init__(self, num_classes=27, n_segment=8, hidden_dim=128):
        super(UltraLightMEBeforeGRUModel, self).__init__()
        self.n_segment = n_segment
        self.hidden_dim = hidden_dim

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.layer1 = TSMResBlock(32, 32, stride=1, n_segment=n_segment)
        self.layer2 = TSMResBlock(32, 64, stride=2, n_segment=n_segment)
        self.layer3 = MEBeforeTSMResBlock(64, 128, stride=2, n_segment=n_segment, reduction=4)

        # GRU input: AdaptiveAvgPool2d(1,1) collapses spatial dims to 128-d vector per frame
        self.gru = nn.GRU(
            input_size=128,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_dim, num_classes)

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

        x = F.adaptive_avg_pool2d(x, (1, 1))    # (B*T, 128, 1, 1)
        x = x.view(b, t, -1)                    # (B, T, 128)

        # GRU temporal aggregation
        rnn_out, hidden = self.gru(x)           # hidden: (1, B, hidden_dim)
        last_hidden = hidden[-1]                # (B, hidden_dim)

        out = self.fc(last_hidden)              # (B, num_classes)
        return out


# MARK: best for now
class UltraLightMEBeforeGRUModel_1(nn.Module):
    def __init__(self, num_classes=27, n_segment=8, hidden_dim=128):
        super(UltraLightMEBeforeGRUModel_1, self).__init__()
        self.n_segment = n_segment
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(0.5)

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.layer1 = TSMResBlock(32, 32, stride=1, n_segment=n_segment)
        self.layer2 = TSMResBlock(32, 64, stride=2, n_segment=n_segment)
        self.layer3 = MEBeforeTSMResBlock(64, 128, stride=2, n_segment=n_segment, reduction=4)

        # GRU input: AdaptiveAvgPool2d(1,1) collapses spatial dims to 128-d vector per frame
        self.gru = nn.GRU(
            input_size=128,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_dim, num_classes)

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

        x = F.adaptive_avg_pool2d(x, (1, 1))    # (B*T, 128, 1, 1)
        x = x.view(b, t, -1)                    # (B, T, 128)

        # GRU temporal aggregation
        rnn_out, hidden = self.gru(x)           # hidden: (1, B, hidden_dim)
        last_hidden = hidden[-1]                # (B, hidden_dim)
        avg_feat = torch.mean(rnn_out, dim=1)   # (B, hidden_dim)

        last_hidden = self.dropout(last_hidden) # change: add dropout
        out = self.fc(last_hidden)              # (B, num_classes)
        return out


class UltraLightMEBeforeGRUModel_1_SpatialAttention(nn.Module):
    """
    加入了空间注意力模块
    """
    def __init__(self, num_classes=27, n_segment=8, hidden_dim=128):
        super(UltraLightMEBeforeGRUModel_1_SpatialAttention, self).__init__()
        self.n_segment = n_segment
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(0.5)

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.layer1 = TSMResBlock(32, 32, stride=1, n_segment=n_segment)
        self.layer2 = TSMResBlock(32, 64, stride=2, n_segment=n_segment)
        self.layer3 = MEBeforeTSMResBlock(64, 128, stride=2, n_segment=n_segment, reduction=4)

        self.spatial_attention = SpatialAttention(kernel_size=7) # change

        # GRU input: AdaptiveAvgPool2d(1,1) collapses spatial dims to 128-d vector per frame
        self.gru = nn.GRU(
            input_size=128,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_dim, num_classes)

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

        x = self.spatial_attention(x)           # (B*T, 128, H/8, W/8) change: add spatial attention module

        x = F.adaptive_avg_pool2d(x, (1, 1))    # (B*T, 128, 1, 1)
        x = x.view(b, t, -1)                    # (B, T, 128)

        # GRU temporal aggregation
        rnn_out, hidden = self.gru(x)           # hidden: (1, B, hidden_dim)
        last_hidden = hidden[-1]                # (B, hidden_dim)
        avg_feat = torch.mean(rnn_out, dim=1)   # (B, hidden_dim)

        last_hidden = self.dropout(last_hidden) # change: add dropout
        out = self.fc(last_hidden)              # (B, num_classes)
        return out


class UltraLightMEBeforeGRUModel_2(nn.Module):
    def __init__(self, num_classes=27, n_segment=8, hidden_dim=128):
        super(UltraLightMEBeforeGRUModel_2, self).__init__()
        self.n_segment = n_segment
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(0.5)

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.layer1 = TSMResBlock(32, 32, stride=1, n_segment=n_segment)
        self.layer2 = MEBeforeTSMResBlock(32, 64, stride=2, n_segment=n_segment, reduction=2)
        self.layer3 = MEBeforeTSMResBlock(64, 128, stride=2, n_segment=n_segment, reduction=4)

        # GRU input: AdaptiveAvgPool2d(1,1) collapses spatial dims to 128-d vector per frame
        self.gru = nn.GRU(
            input_size=128,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_dim, num_classes)

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

        x = F.adaptive_avg_pool2d(x, (1, 1))    # (B*T, 128, 1, 1)
        x = x.view(b, t, -1)                    # (B, T, 128)

        # GRU temporal aggregation
        rnn_out, hidden = self.gru(x)           # hidden: (1, B, hidden_dim)
        last_hidden = hidden[-1]                # (B, hidden_dim)
        avg_feat = torch.mean(rnn_out, dim=1)   # (B, hidden_dim)

        last_hidden = self.dropout(last_hidden) # change: add dropout
        out = self.fc(last_hidden)              # (B, num_classes)
        return out


class UltraLightMEBeforeGRUModel_3(nn.Module):
    def __init__(self, num_classes=27, n_segment=8, hidden_dim=128):
        super(UltraLightMEBeforeGRUModel_3, self).__init__()
        self.n_segment = n_segment
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(0.5)

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.layer1 = MEBeforeTSMResBlock(32, 32, stride=1, n_segment=n_segment, reduction=2)
        self.layer2 = MEBeforeTSMResBlock(32, 64, stride=2, n_segment=n_segment, reduction=2)
        self.layer3 = MEBeforeTSMResBlock(64, 128, stride=2, n_segment=n_segment, reduction=4)

        # GRU input: AdaptiveAvgPool2d(1,1) collapses spatial dims to 128-d vector per frame
        self.gru = nn.GRU(
            input_size=128,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_dim, num_classes)

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

        x = F.adaptive_avg_pool2d(x, (1, 1))    # (B*T, 128, 1, 1)
        x = x.view(b, t, -1)                    # (B, T, 128)

        # GRU temporal aggregation
        rnn_out, hidden = self.gru(x)           # hidden: (1, B, hidden_dim)
        last_hidden = hidden[-1]                # (B, hidden_dim)
        avg_feat = torch.mean(rnn_out, dim=1)   # (B, hidden_dim)

        last_hidden = self.dropout(last_hidden) # change: add dropout
        out = self.fc(last_hidden)              # (B, num_classes)
        return out


class Deeper(nn.Module):
    """
    相对于UltraLightMEBeforeGRUModel_1迭代
    精度倒退, 用做消融实验
    """
    def __init__(self, num_classes=27, n_segment=8, hidden_dim=128):
        super(Deeper, self).__init__()
        self.n_segment = n_segment
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(0.5)

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.layer1 = TSMResBlock(32, 32, stride=1, n_segment=n_segment)
        self.layer2 = TSMResBlock(32, 64, stride=2, n_segment=n_segment)
        self.layer3 = MEBeforeTSMResBlock(64, 128, stride=2, n_segment=n_segment, reduction=4)
        self.layer4 = MEBeforeTSMResBlock(128, 256, stride=2, n_segment=n_segment, reduction=8) # change

        # GRU input: AdaptiveAvgPool2d(1,1) collapses spatial dims to 128-d vector per frame
        self.gru = nn.GRU(
            input_size=256, # change
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_dim, num_classes)

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

        x = F.adaptive_avg_pool2d(x, (1, 1))    # (B*T, 128, 1, 1)
        x = x.view(b, t, -1)                    # (B, T, 128)

        # GRU temporal aggregation
        rnn_out, hidden = self.gru(x)           # hidden: (1, B, hidden_dim)
        last_hidden = hidden[-1]                # (B, hidden_dim)
        avg_feat = torch.mean(rnn_out, dim=1)   # (B, hidden_dim)

        last_hidden = self.dropout(last_hidden) # change: add dropout
        out = self.fc(last_hidden)              # (B, num_classes)
        return out


# --------------------------
# UltraLightParallelMEGRU
# TSM + Parallel ME/TSM + Standard GRU
# MARK: 发表
# --------------------------

class UltraLightParallelMEGRUModel(nn.Module):
    """Ultra-lightweight model: TSM-ResNet with Parallel ME/TSM in layer3 + GRU.

    Architecture:
        Conv1 (3->32, stride=2) -> TSMResBlock [32->32] -> TSMResBlock [32->64]
        -> ParallelMETSMResBlock [64->128] (Parallel ME + TSM -> Conv)
        -> SpatialPool -> Flatten -> GRU -> FC

    Based on UltraLightMEGRUModel but uses Scheme B (Parallel ME/TSM),
    where Motion Excitation and Temporal Shift are applied in parallel
    branches and then fused for motion-aware temporal modeling.

    Reference: ACTION-Net (Wang et al., CVPR 2021) Section 3.3

    Args:
        num_classes: Number of output classes (default 27 for Jester).
        n_segment: Number of temporal segments (sampled frames).
        hidden_dim: GRU hidden state dimension.
    """

    def __init__(self, num_classes=27, n_segment=8, hidden_dim=128):
        super(UltraLightParallelMEGRUModel, self).__init__()
        self.n_segment = n_segment
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(0.5)

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.layer1 = TSMResBlock(32, 32, stride=1, n_segment=n_segment)
        self.layer2 = TSMResBlock(32, 64, stride=2, n_segment=n_segment)
        self.layer3 = ParallelMETSMResBlock(64, 128, stride=2, n_segment=n_segment, reduction=4)

        # GRU input: AdaptiveAvgPool2d(1,1) collapses spatial dims to 128-d vector per frame
        self.gru = nn.GRU(
            input_size=128,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_dim, num_classes)

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

        x = F.adaptive_avg_pool2d(x, (1, 1))    # (B*T, 128, 1, 1)
        x = x.view(b, t, -1)                    # (B, T, 128)

        # GRU temporal aggregation
        rnn_out, hidden = self.gru(x)           # hidden: (1, B, hidden_dim)
        last_hidden = hidden[-1]                # (B, hidden_dim)

        last_hidden = self.dropout(last_hidden)
        out = self.fc(last_hidden)              # (B, num_classes)
        return out


# --------------------------
# UltraLightMELiteBeforeGRU
# TSM + Lite ME-Before-TSM (Scheme A) + Standard GRU
# --------------------------

class UltraLightMELiteBeforeGRUModel(nn.Module):
    """Ultra-lightweight model: TSM-ResNet with Lite ME-Before-TSM in layer3 + GRU.

    Architecture:
        Conv1 (3->32, stride=2) -> TSMResBlock [32->32] -> TSMResBlock [32->64]
        -> MELiteBeforeTSMResBlock [64->128] (Lite ME first, then TSM -> Conv)
        -> SpatialPool -> Flatten -> GRU -> FC

    Based on UltraLightMELiteGRUModel but uses Scheme A (ME-before-TSM),
    where Lite Motion Excitation is applied BEFORE temporal shift for
    lightweight motion-aware feature excitation before temporal modeling.

    Reference: ACTION-Net (Wang et al., CVPR 2021) Section 3.3

    Args:
        num_classes: Number of output classes (default 27 for Jester).
        n_segment: Number of temporal segments (sampled frames).
        hidden_dim: GRU hidden state dimension.
    """

    def __init__(self, num_classes=27, n_segment=8, hidden_dim=128):
        super(UltraLightMELiteBeforeGRUModel, self).__init__()
        self.n_segment = n_segment
        self.hidden_dim = hidden_dim

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.layer1 = TSMResBlock(32, 32, stride=1, n_segment=n_segment)
        self.layer2 = TSMResBlock(32, 64, stride=2, n_segment=n_segment)
        self.layer3 = MELiteBeforeTSMResBlock(64, 128, stride=2, n_segment=n_segment, reduction=4)

        # GRU input: AdaptiveAvgPool2d(1,1) collapses spatial dims to 128-d vector per frame
        self.gru = nn.GRU(
            input_size=128,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_dim, num_classes)

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

        x = F.adaptive_avg_pool2d(x, (1, 1))    # (B*T, 128, 1, 1)
        x = x.view(b, t, -1)                    # (B, T, 128)

        # GRU temporal aggregation
        rnn_out, hidden = self.gru(x)           # hidden: (1, B, hidden_dim)
        last_hidden = hidden[-1]                # (B, hidden_dim)

        out = self.fc(last_hidden)              # (B, num_classes)
        return out


# --------------------------
# UltraLightParallelMELiteGRU
# TSM + Parallel Lite ME/TSM (Scheme B) + Standard GRU
# --------------------------

class UltraLightParallelMELiteGRUModel(nn.Module):
    """Ultra-lightweight model: TSM-ResNet with Parallel Lite ME/TSM in layer3 + GRU.

    Architecture:
        Conv1 (3->32, stride=2) -> TSMResBlock [32->32] -> TSMResBlock [32->64]
        -> ParallelMELiteTSMResBlock [64->128] (Parallel Lite ME + TSM -> Conv)
        -> SpatialPool -> Flatten -> GRU -> FC

    Based on UltraLightMELiteGRUModel but uses Scheme B (Parallel ME/TSM),
    where Lite Motion Excitation and Temporal Shift are applied in parallel
    branches and then fused for lightweight motion-aware temporal modeling.

    Reference: ACTION-Net (Wang et al., CVPR 2021) Section 3.3

    Args:
        num_classes: Number of output classes (default 27 for Jester).
        n_segment: Number of temporal segments (sampled frames).
        hidden_dim: GRU hidden state dimension.
    """

    def __init__(self, num_classes=27, n_segment=8, hidden_dim=128):
        super(UltraLightParallelMELiteGRUModel, self).__init__()
        self.n_segment = n_segment
        self.hidden_dim = hidden_dim

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.layer1 = TSMResBlock(32, 32, stride=1, n_segment=n_segment)
        self.layer2 = TSMResBlock(32, 64, stride=2, n_segment=n_segment)
        self.layer3 = ParallelMELiteTSMResBlock(64, 128, stride=2, n_segment=n_segment, reduction=4)

        # GRU input: AdaptiveAvgPool2d(1,1) collapses spatial dims to 128-d vector per frame
        self.gru = nn.GRU(
            input_size=128,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_dim, num_classes)

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

        x = F.adaptive_avg_pool2d(x, (1, 1))    # (B*T, 128, 1, 1)
        x = x.view(b, t, -1)                    # (B, T, 128)

        # GRU temporal aggregation
        rnn_out, hidden = self.gru(x)           # hidden: (1, B, hidden_dim)
        last_hidden = hidden[-1]                # (B, hidden_dim)

        out = self.fc(last_hidden)              # (B, num_classes)
        return out
