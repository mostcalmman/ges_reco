"""
Video gesture recognition model architectures.
All models accept input shape (B, T, 3, H, W) and output (B, num_classes).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet50_Weights, MobileNet_V2_Weights
from torchvision.models import MobileNet_V3_Large_Weights, MobileNet_V3_Small_Weights
from torchvision.models import ShuffleNet_V2_X1_0_Weights, ShuffleNet_V2_X2_0_Weights

from modules import *

modelList = ['ResNet18', 'LightTSM', 'LightTSMGRU', 'LightTMFGRU', 'TMFin1', 'TMFin2', 'TMFin123', 'ab1', 'ab2', 'LightTSMGRU', 'LightResNet',
             'LightTMF2GRU', 'LightTMF25GRU', 'LightTMF26GRU', 'LightTMF3GRU', 'LightTMF4GRU',
             'ResNet50', 'MobileNetV2', 'ShuffleNetV2x10',
             'ResNet50_ACSSTMF3', 'MobileNetV2_ACSSTMF3', 'MobileNetV3Large_ACSSTMF3',
             'MobileNetV3Small_ACSSTMF3', 'ShuffleNetV2x10_ACSSTMF3', 'ShuffleNetV2x20_ACSSTMF3',
             'ResNet50_TSM', 'MobileNetV2_TSM', 'ShuffleNetV2x10_TSM',
             'Light_OnlyTMF3_GRU', 'Light_OnlyACSS_GRU',
             'ResNet50_OnlyTMF3', 'ResNet50_OnlyACSS',
             'ResNet50_TSN', 'MobileNetV2_TSN',
             ]


# --------------------------
# MARK: ResNet18
# --------------------------
class ResNet18(nn.Module):
    """Baseline: pretrained ResNet18 with temporal mean pooling.

    Architecture: ResNet18 (frozen except layer4) -> AvgPool -> FC

    Args:
        num_classes: Number of output classes.
        freeze_backbone: If True, freeze all layers except layer4.
    """

    def __init__(self, num_classes, freeze_backbone=False):
        super(ResNet18, self).__init__()
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
# LightTSM
# --------------------------

class LightTSM(nn.Module):
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
        super(LightTSM, self).__init__()
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
# LightTSMGRU
# TSM + Standard GRU
# --------------------------

class LightTSMGRU(nn.Module):
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
        super(LightTSMGRU, self).__init__()
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
# LightTMFGRU
# MARK: 发表
# --------------------------

class LightTMFGRU(nn.Module):
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
        super(LightTMFGRU, self).__init__()
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
        self.layer3 = TMFResBlock(64, 128, stride=2, n_segment=n_segment, reduction=4)

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
    

class LightTMF2GRU(nn.Module):
    """
    新版TMF: TMF2, 具体看module.py
    """

    def __init__(self, num_classes=27, n_segment=8, hidden_dim=128):
        super(LightTMF2GRU, self).__init__()
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
        self.layer3 = TMF2ResBlock(64, 128, stride=2, n_segment=n_segment, reduction=4)

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


class LightTMF25GRU(nn.Module):
    """
    验证 ACSS 有效性, 实际是负作用
    """

    def __init__(self, num_classes=27, n_segment=8, hidden_dim=128):
        super(LightTMF25GRU, self).__init__()
        self.n_segment = n_segment
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(0.5)

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.layer1 = ACSSResBlock(32, 32, stride=1, n_segment=n_segment)
        self.layer2 = ACSSResBlock(32, 64, stride=2, n_segment=n_segment)
        self.layer3 = TMF2ResBlock(64, 128, stride=2, n_segment=n_segment, reduction=4)

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
    

class LightTMF26GRU(nn.Module):
    """
    验证 TMF3 有效性
    """

    def __init__(self, num_classes=27, n_segment=8, hidden_dim=128, fusion_mode='B'):
        super(LightTMF26GRU, self).__init__()
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
        self.layer3 = TMF3ResBlock(64, 128, stride=2, n_segment=n_segment, reduction=4, fusion_mode=fusion_mode)

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
    
# MARK: publish
class LightTMF3GRU(nn.Module):
    """
    TMF3: ACSS + TMF3 + GRU
    """

    def __init__(self, num_classes=27, n_segment=8, hidden_dim=128, fusion_mode='B'):
        super(LightTMF3GRU, self).__init__()
        self.n_segment = n_segment
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(0.5)

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.layer1 = ACSSResBlock(32, 32, stride=1, n_segment=n_segment)
        self.layer2 = ACSSResBlock(32, 64, stride=2, n_segment=n_segment)
        self.layer3 = TMF3ResBlock(64, 128, stride=2, n_segment=n_segment, reduction=4, fusion_mode=fusion_mode)

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


class LightTMF4GRU(nn.Module):
    """
    TMF3 的消融实验
    """
    def __init__(self, num_classes=27, n_segment=8, hidden_dim=128, fusion_mode='B'):
        super(LightTMF4GRU, self).__init__()
        self.n_segment = n_segment
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(0.5)

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.layer1 = TMF3ResBlock(32, 32, stride=1, n_segment=n_segment, reduction=2, fusion_mode=fusion_mode)
        self.layer2 = TMF3ResBlock(32, 64, stride=2, n_segment=n_segment, reduction=2, fusion_mode=fusion_mode)
        self.layer3 = TMF3ResBlock(64, 128, stride=2, n_segment=n_segment, reduction=4, fusion_mode=fusion_mode)

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


class TMFin1(nn.Module):
    """
    消融实验
    """
    def __init__(self, num_classes=27, n_segment=8, hidden_dim=128):
        super(TMFin1, self).__init__()
        self.n_segment = n_segment
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(0.5)

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.layer1 = TMFResBlock(32, 32, stride=1, n_segment=n_segment, reduction=2)
        self.layer2 = TSMResBlock(32, 64, stride=2, n_segment=n_segment)
        self.layer3 = TSMResBlock(64, 128, stride=2, n_segment=n_segment)

        # GRU input: AdaptiveAvgPool2d(1,1) collapses spatial dims to 128-d vector per frame
        self.gru = nn.GRU(
            input_size=128,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
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
    

class TMFin2(nn.Module):
    """
    消融实验
    """
    def __init__(self, num_classes=27, n_segment=8, hidden_dim=128):
        super(TMFin2, self).__init__()
        self.n_segment = n_segment
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(0.5)

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.layer1 = TSMResBlock(32, 32, stride=1, n_segment=n_segment)
        self.layer2 = TMFResBlock(32, 64, stride=2, n_segment=n_segment, reduction=2)
        self.layer3 = TSMResBlock(64, 128, stride=2, n_segment=n_segment)

        # GRU input: AdaptiveAvgPool2d(1,1) collapses spatial dims to 128-d vector per frame
        self.gru = nn.GRU(
            input_size=128,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
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
    

class TMFin123(nn.Module):
    """
    消融实验
    """
    def __init__(self, num_classes=27, n_segment=8, hidden_dim=128):
        super(TMFin123, self).__init__()
        self.n_segment = n_segment
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(0.5)

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.layer1 = TMFResBlock(32, 32, stride=1, n_segment=n_segment, reduction=2)
        self.layer2 = TMFResBlock(32, 64, stride=2, n_segment=n_segment, reduction=2)
        self.layer3 = TMFResBlock(64, 128, stride=2, n_segment=n_segment, reduction=4)

        # GRU input: AdaptiveAvgPool2d(1,1) collapses spatial dims to 128-d vector per frame
        self.gru = nn.GRU(
            input_size=128,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
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
    

class ab1(nn.Module):
    """
    消融实验, 只留ResNetLight
    """
    def __init__(self, num_classes=27, n_segment=8, hidden_dim=128):
        super(ab1, self).__init__()
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
        self.layer3 = TSMResBlock(64, 128, stride=2, n_segment=n_segment)

        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        b, t, c, h, w = x.size()

        x = x.view(b * t, c, h, w)              # (B*T, 3, H, W)
        x = self.conv1(x)                       # (B*T, 32, H/2, W/2)
        x = self.layer1(x)                      # (B*T, 32, H/2, W/2)
        x = self.layer2(x)                      # (B*T, 64, H/4, W/4)
        x = self.layer3(x)                      # (B*T, 128, H/8, W/8)

        x = F.adaptive_avg_pool2d(x, (1, 1))    # (B*T, 128, 1, 1)
        x = x.view(b, t, -1)                    # (B, T, 128)

        x = x.mean(dim=1)                       # (B, 128)

        x = self.dropout(x)
        out = self.fc(x)                        # (B, num_classes)
        return out


class ab2(nn.Module):
    """
    消融实验, 只留ResNetLight + TMF
    """
    def __init__(self, num_classes=27, n_segment=8, hidden_dim=128):
        super(ab2, self).__init__()
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
        self.layer3 = TMFResBlock(64, 128, stride=2, n_segment=n_segment, reduction=4)

        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        b, t, c, h, w = x.size()

        x = x.view(b * t, c, h, w)              # (B*T, 3, H, W)
        x = self.conv1(x)                       # (B*T, 32, H/2, W/2)
        x = self.layer1(x)                      # (B*T, 32, H/2, W/2)
        x = self.layer2(x)                      # (B*T, 64, H/4, W/4)
        x = self.layer3(x)                      # (B*T, 128, H/8, W/8)

        x = F.adaptive_avg_pool2d(x, (1, 1))    # (B*T, 128, 1, 1)
        x = x.view(b, t, -1)                    # (B, T, 128)

        x = x.mean(dim=1)                       # (B, 128)

        x = self.dropout(x)
        out = self.fc(x)                        # (B, num_classes)
        return out


class LightTSMGRU(nn.Module):
    """
    原 ab3
    """
    def __init__(self, num_classes=27, n_segment=8, hidden_dim=128):
        super(LightTSMGRU, self).__init__()
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
        self.layer3 = TSMResBlock(64, 128, stride=2, n_segment=n_segment)

        # GRU input: AdaptiveAvgPool2d(1,1) collapses spatial dims to 128-d vector per frame
        self.gru = nn.GRU(
            input_size=128,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
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


class LightResNet(nn.Module):
    """
    消融实验, 与 ab3 结构一致, 但去掉 TSM
    """
    def __init__(self, num_classes=27, n_segment=8, hidden_dim=128):
        super(LightResNet, self).__init__()
        self.n_segment = n_segment
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(0.5)

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.layer1 = LightResBlock(32, 32, stride=1, n_segment=n_segment)
        self.layer2 = LightResBlock(32, 64, stride=2, n_segment=n_segment)
        self.layer3 = LightResBlock(64, 128, stride=2, n_segment=n_segment)

        # GRU input: AdaptiveAvgPool2d(1,1) collapses spatial dims to 128-d vector per frame
        self.gru = nn.GRU(
            input_size=128,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
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
# MARK: other Backbone
# --------------------------

class ResNet50(nn.Module):
    """Plain ResNet50 baseline without ACSS/TMF3 blocks.

    Architecture:
        conv1+bn1+relu+maxpool -> layer1 -> layer2 -> layer3 -> layer4
        -> pool -> temporal mean or GRU -> FC
    """

    def __init__(self, num_classes=27, n_segment=8, hidden_dim=128,
                 freeze_backbone=False, use_gru=False):
        super(ResNet50, self).__init__()
        self.n_segment = n_segment
        self.use_gru = use_gru

        backbone = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.stem = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        feat_dim = 2048

        self.dropout = nn.Dropout(0.5)
        if use_gru:
            self.gru = nn.GRU(input_size=feat_dim, hidden_size=hidden_dim,
                              num_layers=1, batch_first=True)
            self.fc = nn.Linear(hidden_dim, num_classes)
        else:
            self.fc = nn.Linear(feat_dim, num_classes)

        if freeze_backbone:
            for param in backbone.parameters():
                param.requires_grad = False

    def forward(self, x):
        b, t, c, h, w = x.size()
        x = x.view(b * t, c, h, w)

        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(b, t, -1)

        if self.use_gru:
            rnn_out, hidden = self.gru(x)
            last_hidden = hidden[-1]
            last_hidden = self.dropout(last_hidden)
            out = self.fc(last_hidden)
        else:
            x = x.mean(dim=1)
            x = self.dropout(x)
            out = self.fc(x)
        return out


class MobileNetV2(nn.Module):
    """Plain MobileNetV2 baseline without any temporal module.

    Architecture:
        MobileNetV2 features -> pool -> temporal mean -> FC
    """

    def __init__(self, num_classes=27, n_segment=8, freeze_backbone=False):
        super(MobileNetV2, self).__init__()
        self.n_segment = n_segment

        backbone = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V2)
        self.features = backbone.features
        feat_dim = _get_stage_out_channels(self.features)

        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(feat_dim, num_classes)

        if freeze_backbone:
            for param in backbone.parameters():
                param.requires_grad = False

    def forward(self, x):
        b, t, c, h, w = x.size()
        x = x.view(b * t, c, h, w)

        x = self.features(x)

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(b, t, -1)

        x = x.mean(dim=1)
        x = self.dropout(x)
        out = self.fc(x)
        return out


class ShuffleNetV2x10(nn.Module):
    """Plain ShuffleNetV2 x1.0 baseline without any temporal module.

    Architecture:
        conv1 -> maxpool -> stage2 -> stage3 -> stage4 -> conv5 -> pool -> temporal mean -> FC
    """

    def __init__(self, num_classes=27, n_segment=8, freeze_backbone=False):
        super(ShuffleNetV2x10, self).__init__()
        self.n_segment = n_segment

        backbone = models.shufflenet_v2_x1_0(weights=ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1)
        self.conv1 = backbone.conv1
        self.maxpool = backbone.maxpool
        self.stage2 = backbone.stage2
        self.stage3 = backbone.stage3
        self.stage4 = backbone.stage4
        self.conv5 = backbone.conv5
        feat_dim = 1024

        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(feat_dim, num_classes)

        if freeze_backbone:
            for param in backbone.parameters():
                param.requires_grad = False

    def forward(self, x):
        b, t, c, h, w = x.size()
        x = x.view(b * t, c, h, w)

        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(b, t, -1)

        x = x.mean(dim=1)
        x = self.dropout(x)
        out = self.fc(x)
        return out


def _split_mobilenet_by_stride(features):
    """Split a MobileNet features Sequential into stages by stride-2 boundaries.

    Scans each sub-module for stride-2 convolutions and splits at those
    boundaries. Returns a list of nn.Sequential stages.
    """
    stages = []
    current = []
    for i, module in enumerate(features):
        is_stride2 = False
        for m in module.modules():
            if isinstance(m, nn.Conv2d) and hasattr(m, 'stride'):
                if m.stride == (2, 2) or m.stride == 2:
                    is_stride2 = True
                    break
        if is_stride2 and len(current) > 0:
            stages.append(nn.Sequential(*current))
            current = [module]
        else:
            current.append(module)
    if current:
        stages.append(nn.Sequential(*current))
    return stages


def _get_stage_out_channels(stage):
    """Get output channel count from the last BatchNorm or Conv in a stage."""
    out_ch = None
    for m in stage.modules():
        if isinstance(m, nn.BatchNorm2d):
            out_ch = m.num_features
        elif isinstance(m, nn.Conv2d):
            out_ch = m.out_channels
    return out_ch


class ResNet50_ACSSTMF3(nn.Module):
    """ResNet50 backbone with ACSS before shallow-deep stages and TMF3 before layer4.

    Architecture:
        conv1+bn1+relu+maxpool → layer1(256) → ACSS → layer2(512) → ACSS
        → layer3(1024) → TMF3 → layer4(2048) → pool → GRU → FC
    """

    def __init__(self, num_classes=27, n_segment=8, hidden_dim=128,
                 freeze_backbone=False, use_gru=False, fusion_mode='B', reduction='auto'):
        super(ResNet50_ACSSTMF3, self).__init__()
        self.n_segment = n_segment
        self.use_gru = use_gru

        backbone = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.stem = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool
        )
        self.layer1 = backbone.layer1   # -> 256ch
        self.layer2 = backbone.layer2   # -> 512ch
        self.layer3 = backbone.layer3   # -> 1024ch
        self.layer4 = backbone.layer4   # -> 2048ch
        feat_dim = 2048
        tmf3_in_dim = 1024

        self.acss1 = ACSS_Module(256, n_segment)
        self.acss2 = ACSS_Module(512, n_segment)

        red = TMF3Module._auto_reduction(tmf3_in_dim) if reduction == 'auto' else int(reduction)
        self.tmf3 = TMF3Module(tmf3_in_dim, n_segment, reduction=red, fusion_mode=fusion_mode)

        self.dropout = nn.Dropout(0.5)
        if use_gru:
            self.gru = nn.GRU(input_size=feat_dim, hidden_size=hidden_dim,
                              num_layers=1, batch_first=True)
            self.fc = nn.Linear(hidden_dim, num_classes)
        else:
            self.fc = nn.Linear(feat_dim, num_classes)

        if freeze_backbone:
            for param in backbone.parameters():
                param.requires_grad = False

    def forward(self, x):
        b, t, c, h, w = x.size()
        x = x.view(b * t, c, h, w)

        x = self.stem(x)
        x = self.layer1(x)
        x = self.acss1(x)
        x = self.layer2(x)
        x = self.acss2(x)
        x = self.layer3(x)
        x = self.tmf3(x)
        x = self.layer4(x)

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(b, t, -1)

        if self.use_gru:
            rnn_out, hidden = self.gru(x)
            last_hidden = hidden[-1]
            last_hidden = self.dropout(last_hidden)
            out = self.fc(last_hidden)
        else:
            x = x.mean(dim=1)
            x = self.dropout(x)
            out = self.fc(x)
        return out


class MobileNetV2_ACSSTMF3(nn.Module):
    """MobileNetV2 backbone with ACSS before non-deepest stages and TMF3 before final.

    Features are split dynamically by stride-2 boundaries.
    """

    def __init__(self, num_classes=27, n_segment=8,
                 freeze_backbone=False, fusion_mode='B', reduction='auto'):
        super(MobileNetV2_ACSSTMF3, self).__init__()
        self.n_segment = n_segment

        backbone = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V2)
        stages = _split_mobilenet_by_stride(backbone.features)
        self.stages = nn.ModuleList(stages)

        if len(stages) < 2:
            raise ValueError("MobileNetV2 split produced fewer than 2 stages.")

        tmf3_in_dim = _get_stage_out_channels(stages[-2])
        feat_dim = _get_stage_out_channels(stages[-1])

        acss_modules = []
        for stage in stages[:-2]:
            out_ch = _get_stage_out_channels(stage)
            acss_modules.append(ACSS_Module(out_ch, n_segment))
        self.acss_modules = nn.ModuleList(acss_modules)

        red = TMF3Module._auto_reduction(tmf3_in_dim) if reduction == 'auto' else int(reduction)
        self.tmf3 = TMF3Module(tmf3_in_dim, n_segment, reduction=red, fusion_mode=fusion_mode)

        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(feat_dim, num_classes)

        if freeze_backbone:
            for param in backbone.parameters():
                param.requires_grad = False

    def forward(self, x):
        b, t, c, h, w = x.size()
        x = x.view(b * t, c, h, w)

        deepest_pre_idx = len(self.stages) - 2
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i < deepest_pre_idx:
                x = self.acss_modules[i](x)
            elif i == deepest_pre_idx:
                x = self.tmf3(x)

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(b, t, -1)

        x = x.mean(dim=1)
        x = self.dropout(x)
        out = self.fc(x)
        return out


class MobileNetV3Large_ACSSTMF3(nn.Module):
    """MobileNetV3-Large backbone with ACSS before non-deepest stages and TMF3 before final."""

    def __init__(self, num_classes=27, n_segment=8,
                 freeze_backbone=False, fusion_mode='B', reduction='auto'):
        super(MobileNetV3Large_ACSSTMF3, self).__init__()
        self.n_segment = n_segment

        backbone = models.mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2)
        stages = _split_mobilenet_by_stride(backbone.features)
        self.stages = nn.ModuleList(stages)

        if len(stages) < 2:
            raise ValueError("MobileNetV3-Large split produced fewer than 2 stages.")

        tmf3_in_dim = _get_stage_out_channels(stages[-2])
        feat_dim = _get_stage_out_channels(stages[-1])

        acss_modules = []
        for stage in stages[:-2]:
            out_ch = _get_stage_out_channels(stage)
            acss_modules.append(ACSS_Module(out_ch, n_segment))
        self.acss_modules = nn.ModuleList(acss_modules)

        red = TMF3Module._auto_reduction(tmf3_in_dim) if reduction == 'auto' else int(reduction)
        self.tmf3 = TMF3Module(tmf3_in_dim, n_segment, reduction=red, fusion_mode=fusion_mode)

        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(feat_dim, num_classes)

        if freeze_backbone:
            for param in backbone.parameters():
                param.requires_grad = False

    def forward(self, x):
        b, t, c, h, w = x.size()
        x = x.view(b * t, c, h, w)

        deepest_pre_idx = len(self.stages) - 2
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i < deepest_pre_idx:
                x = self.acss_modules[i](x)
            elif i == deepest_pre_idx:
                x = self.tmf3(x)

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(b, t, -1)

        x = x.mean(dim=1)
        x = self.dropout(x)
        out = self.fc(x)
        return out


class MobileNetV3Small_ACSSTMF3(nn.Module):
    """MobileNetV3-Small backbone with ACSS before non-deepest stages and TMF3 before final.

    Uses V1 weights (only available variant for MobileNetV3-Small).
    """

    def __init__(self, num_classes=27, n_segment=8,
                 freeze_backbone=False, fusion_mode='B', reduction='auto'):
        super(MobileNetV3Small_ACSSTMF3, self).__init__()
        self.n_segment = n_segment

        backbone = models.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        stages = _split_mobilenet_by_stride(backbone.features)
        self.stages = nn.ModuleList(stages)

        if len(stages) < 2:
            raise ValueError("MobileNetV3-Small split produced fewer than 2 stages.")

        tmf3_in_dim = _get_stage_out_channels(stages[-2])
        feat_dim = _get_stage_out_channels(stages[-1])

        acss_modules = []
        for stage in stages[:-2]:
            out_ch = _get_stage_out_channels(stage)
            acss_modules.append(ACSS_Module(out_ch, n_segment))
        self.acss_modules = nn.ModuleList(acss_modules)

        red = TMF3Module._auto_reduction(tmf3_in_dim) if reduction == 'auto' else int(reduction)
        self.tmf3 = TMF3Module(tmf3_in_dim, n_segment, reduction=red, fusion_mode=fusion_mode)

        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(feat_dim, num_classes)

        if freeze_backbone:
            for param in backbone.parameters():
                param.requires_grad = False

    def forward(self, x):
        b, t, c, h, w = x.size()
        x = x.view(b * t, c, h, w)

        deepest_pre_idx = len(self.stages) - 2
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i < deepest_pre_idx:
                x = self.acss_modules[i](x)
            elif i == deepest_pre_idx:
                x = self.tmf3(x)

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(b, t, -1)

        x = x.mean(dim=1)
        x = self.dropout(x)
        out = self.fc(x)
        return out


class ShuffleNetV2x10_ACSSTMF3(nn.Module):
    """ShuffleNetV2 x1.0 backbone with ACSS before stage3/4 and TMF3 before conv5.

    Channels: conv1(24) → stage2(116) → stage3(232) → stage4(464) → conv5(1024)
    """

    def __init__(self, num_classes=27, n_segment=8,
                 freeze_backbone=False, fusion_mode='B', reduction='auto'):
        super(ShuffleNetV2x10_ACSSTMF3, self).__init__()
        self.n_segment = n_segment

        backbone = models.shufflenet_v2_x1_0(weights=ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1)
        self.conv1 = backbone.conv1       # -> 24ch
        self.maxpool = backbone.maxpool
        self.stage2 = backbone.stage2     # -> 116ch
        self.stage3 = backbone.stage3     # -> 232ch
        self.stage4 = backbone.stage4     # -> 464ch
        self.conv5 = backbone.conv5       # -> 1024ch
        feat_dim = 1024
        tmf3_in_dim = 464

        self.acss2 = ACSS_Module(116, n_segment)
        self.acss3 = ACSS_Module(232, n_segment)

        red = TMF3Module._auto_reduction(tmf3_in_dim) if reduction == 'auto' else int(reduction)
        self.tmf3 = TMF3Module(tmf3_in_dim, n_segment, reduction=red, fusion_mode=fusion_mode)

        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(feat_dim, num_classes)

        if freeze_backbone:
            for param in backbone.parameters():
                param.requires_grad = False

    def forward(self, x):
        b, t, c, h, w = x.size()
        x = x.view(b * t, c, h, w)

        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.acss2(x)
        x = self.stage3(x)
        x = self.acss3(x)
        x = self.stage4(x)
        x = self.tmf3(x)
        x = self.conv5(x)

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(b, t, -1)

        x = x.mean(dim=1)
        x = self.dropout(x)
        out = self.fc(x)
        return out


class ShuffleNetV2x20_ACSSTMF3(nn.Module):
    """ShuffleNetV2 x2.0 backbone with ACSS before stage3/4 and TMF3 before conv5.

    Channels: conv1(24) → stage2(244) → stage3(488) → stage4(976) → conv5(2048)
    """

    def __init__(self, num_classes=27, n_segment=8,
                 freeze_backbone=False, fusion_mode='B', reduction='auto'):
        super(ShuffleNetV2x20_ACSSTMF3, self).__init__()
        self.n_segment = n_segment

        backbone = models.shufflenet_v2_x2_0(weights=ShuffleNet_V2_X2_0_Weights.IMAGENET1K_V1)
        self.conv1 = backbone.conv1       # -> 24ch
        self.maxpool = backbone.maxpool
        self.stage2 = backbone.stage2     # -> 244ch
        self.stage3 = backbone.stage3     # -> 488ch
        self.stage4 = backbone.stage4     # -> 976ch
        self.conv5 = backbone.conv5       # -> 2048ch
        feat_dim = 2048
        tmf3_in_dim = 976

        self.acss2 = ACSS_Module(244, n_segment)
        self.acss3 = ACSS_Module(488, n_segment)

        red = TMF3Module._auto_reduction(tmf3_in_dim) if reduction == 'auto' else int(reduction)
        self.tmf3 = TMF3Module(tmf3_in_dim, n_segment, reduction=red, fusion_mode=fusion_mode)

        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(feat_dim, num_classes)

        if freeze_backbone:
            for param in backbone.parameters():
                param.requires_grad = False

    def forward(self, x):
        b, t, c, h, w = x.size()
        x = x.view(b * t, c, h, w)

        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.acss2(x)
        x = self.stage3(x)
        x = self.acss3(x)
        x = self.stage4(x)
        x = self.tmf3(x)
        x = self.conv5(x)

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(b, t, -1)

        x = x.mean(dim=1)
        x = self.dropout(x)
        out = self.fc(x)
        return out


# --------------------------
# MARK: Ablation (ACSS/TMF3 单独消融)
# --------------------------

class Light_OnlyTMF3_GRU(nn.Module):
    """LightTMF3GRU 消融: 去除浅层ACSS, 保留深层TMF3.

    Architecture:
        Conv1 -> LightResBlock -> LightResBlock -> TMF3ResBlock -> GRU -> FC
    """

    def __init__(self, num_classes=27, n_segment=8, hidden_dim=128, fusion_mode='B'):
        super(Light_OnlyTMF3_GRU, self).__init__()
        self.n_segment = n_segment
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(0.5)

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.layer1 = LightResBlock(32, 32, stride=1, n_segment=n_segment)
        self.layer2 = LightResBlock(32, 64, stride=2, n_segment=n_segment)
        self.layer3 = TMF3ResBlock(64, 128, stride=2, n_segment=n_segment, reduction=4, fusion_mode=fusion_mode)

        self.gru = nn.GRU(
            input_size=128,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        b, t, c, h, w = x.size()

        x = x.view(b * t, c, h, w)
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(b, t, -1)

        rnn_out, hidden = self.gru(x)
        last_hidden = hidden[-1]

        last_hidden = self.dropout(last_hidden)
        out = self.fc(last_hidden)
        return out


class Light_OnlyACSS_GRU(nn.Module):
    """LightTMF3GRU 消融: 去除深层TMF3, 保留浅层ACSS.

    Architecture:
        Conv1 -> ACSSResBlock -> ACSSResBlock -> LightResBlock -> GRU -> FC
    """

    def __init__(self, num_classes=27, n_segment=8, hidden_dim=128):
        super(Light_OnlyACSS_GRU, self).__init__()
        self.n_segment = n_segment
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(0.5)

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.layer1 = ACSSResBlock(32, 32, stride=1, n_segment=n_segment)
        self.layer2 = ACSSResBlock(32, 64, stride=2, n_segment=n_segment)
        self.layer3 = LightResBlock(64, 128, stride=2, n_segment=n_segment)

        self.gru = nn.GRU(
            input_size=128,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        b, t, c, h, w = x.size()

        x = x.view(b * t, c, h, w)
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(b, t, -1)

        rnn_out, hidden = self.gru(x)
        last_hidden = hidden[-1]

        last_hidden = self.dropout(last_hidden)
        out = self.fc(last_hidden)
        return out


class ResNet50_OnlyTMF3(nn.Module):
    """ResNet50_ACSSTMF3 消融: 去除浅层ACSS, 保留深层TMF3.

    Architecture:
        stem -> layer1 -> layer2 -> layer3 -> TMF3 -> layer4 -> pool -> FC
    """

    def __init__(self, num_classes=27, n_segment=8, hidden_dim=128,
                 freeze_backbone=False, use_gru=False, fusion_mode='B', reduction='auto'):
        super(ResNet50_OnlyTMF3, self).__init__()
        self.n_segment = n_segment
        self.use_gru = use_gru

        backbone = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.stem = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool
        )
        self.layer1 = backbone.layer1   # -> 256ch
        self.layer2 = backbone.layer2   # -> 512ch
        self.layer3 = backbone.layer3   # -> 1024ch
        self.layer4 = backbone.layer4   # -> 2048ch
        feat_dim = 2048
        tmf3_in_dim = 1024

        red = TMF3Module._auto_reduction(tmf3_in_dim) if reduction == 'auto' else int(reduction)
        self.tmf3 = TMF3Module(tmf3_in_dim, n_segment, reduction=red, fusion_mode=fusion_mode)

        self.dropout = nn.Dropout(0.5)
        if use_gru:
            self.gru = nn.GRU(input_size=feat_dim, hidden_size=hidden_dim,
                              num_layers=1, batch_first=True)
            self.fc = nn.Linear(hidden_dim, num_classes)
        else:
            self.fc = nn.Linear(feat_dim, num_classes)

        if freeze_backbone:
            for param in backbone.parameters():
                param.requires_grad = False

    def forward(self, x):
        b, t, c, h, w = x.size()
        x = x.view(b * t, c, h, w)

        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.tmf3(x)
        x = self.layer4(x)

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(b, t, -1)

        if self.use_gru:
            rnn_out, hidden = self.gru(x)
            last_hidden = hidden[-1]
            last_hidden = self.dropout(last_hidden)
            out = self.fc(last_hidden)
        else:
            x = x.mean(dim=1)
            x = self.dropout(x)
            out = self.fc(x)
        return out


class ResNet50_OnlyACSS(nn.Module):
    """ResNet50_ACSSTMF3 消融: 去除深层TMF3, 保留浅层ACSS.

    Architecture:
        stem -> layer1 -> ACSS -> layer2 -> ACSS -> layer3 -> layer4 -> pool -> FC
    """

    def __init__(self, num_classes=27, n_segment=8, hidden_dim=128,
                 freeze_backbone=False, use_gru=False):
        super(ResNet50_OnlyACSS, self).__init__()
        self.n_segment = n_segment
        self.use_gru = use_gru

        backbone = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.stem = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool
        )
        self.layer1 = backbone.layer1   # -> 256ch
        self.layer2 = backbone.layer2   # -> 512ch
        self.layer3 = backbone.layer3   # -> 1024ch
        self.layer4 = backbone.layer4   # -> 2048ch
        feat_dim = 2048

        self.acss1 = ACSS_Module(256, n_segment)
        self.acss2 = ACSS_Module(512, n_segment)

        self.dropout = nn.Dropout(0.5)
        if use_gru:
            self.gru = nn.GRU(input_size=feat_dim, hidden_size=hidden_dim,
                              num_layers=1, batch_first=True)
            self.fc = nn.Linear(hidden_dim, num_classes)
        else:
            self.fc = nn.Linear(feat_dim, num_classes)

        if freeze_backbone:
            for param in backbone.parameters():
                param.requires_grad = False

    def forward(self, x):
        b, t, c, h, w = x.size()
        x = x.view(b * t, c, h, w)

        x = self.stem(x)
        x = self.layer1(x)
        x = self.acss1(x)
        x = self.layer2(x)
        x = self.acss2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(b, t, -1)

        if self.use_gru:
            rnn_out, hidden = self.gru(x)
            last_hidden = hidden[-1]
            last_hidden = self.dropout(last_hidden)
            out = self.fc(last_hidden)
        else:
            x = x.mean(dim=1)
            x = self.dropout(x)
            out = self.fc(x)
        return out


# --------------------------
# MARK: TSM-only
# --------------------------

class ResNet50_TSM(nn.Module):
    """ResNet50 with vanilla TSM inserted before each layer.

    Architecture:
        stem -> TSM -> layer1 -> TSM -> layer2 -> TSM -> layer3 -> TSM -> layer4
        -> pool -> temporal mean -> FC
    """

    def __init__(self, num_classes=27, n_segment=8, hidden_dim=128,
                 freeze_backbone=False, use_gru=False):
        super(ResNet50_TSM, self).__init__()
        self.n_segment = n_segment
        self.use_gru = use_gru

        backbone = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.stem = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool
        )
        self.layer1 = backbone.layer1   # -> 256ch
        self.layer2 = backbone.layer2   # -> 512ch
        self.layer3 = backbone.layer3   # -> 1024ch
        self.layer4 = backbone.layer4   # -> 2048ch
        feat_dim = 2048

        self.dropout = nn.Dropout(0.5)
        if use_gru:
            self.gru = nn.GRU(input_size=feat_dim, hidden_size=hidden_dim,
                              num_layers=1, batch_first=True)
            self.fc = nn.Linear(hidden_dim, num_classes)
        else:
            self.fc = nn.Linear(feat_dim, num_classes)

        if freeze_backbone:
            for param in backbone.parameters():
                param.requires_grad = False

    def forward(self, x):
        b, t, c, h, w = x.size()
        x = x.view(b * t, c, h, w)

        x = self.stem(x)
        x = temporal_shift(x, self.n_segment)
        x = self.layer1(x)
        x = temporal_shift(x, self.n_segment)
        x = self.layer2(x)
        x = temporal_shift(x, self.n_segment)
        x = self.layer3(x)
        x = temporal_shift(x, self.n_segment)
        x = self.layer4(x)

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(b, t, -1)

        if self.use_gru:
            rnn_out, hidden = self.gru(x)
            last_hidden = hidden[-1]
            last_hidden = self.dropout(last_hidden)
            out = self.fc(last_hidden)
        else:
            x = x.mean(dim=1)
            x = self.dropout(x)
            out = self.fc(x)
        return out


class MobileNetV2_TSM(nn.Module):
    """MobileNetV2 with vanilla TSM inserted before each stage.

    Features are split by stride-2 boundaries (same as ACSS+TMF3 variant).
    TSM is applied before each stage.
    """

    def __init__(self, num_classes=27, n_segment=8, freeze_backbone=False):
        super(MobileNetV2_TSM, self).__init__()
        self.n_segment = n_segment

        backbone = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V2)
        stages = _split_mobilenet_by_stride(backbone.features)
        self.stages = nn.ModuleList(stages)

        feat_dim = _get_stage_out_channels(stages[-1])

        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(feat_dim, num_classes)

        if freeze_backbone:
            for param in backbone.parameters():
                param.requires_grad = False

    def forward(self, x):
        b, t, c, h, w = x.size()
        x = x.view(b * t, c, h, w)

        for stage in self.stages:
            x = temporal_shift(x, self.n_segment)
            x = stage(x)

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(b, t, -1)

        x = x.mean(dim=1)
        x = self.dropout(x)
        out = self.fc(x)
        return out


class ShuffleNetV2x10_TSM(nn.Module):
    """ShuffleNetV2 x1.0 with vanilla TSM inserted before each stage.

    Architecture:
        conv1 -> maxpool -> TSM -> stage2 -> TSM -> stage3 -> TSM -> stage4 -> TSM -> conv5
        -> pool -> temporal mean -> FC
    """

    def __init__(self, num_classes=27, n_segment=8, freeze_backbone=False):
        super(ShuffleNetV2x10_TSM, self).__init__()
        self.n_segment = n_segment

        backbone = models.shufflenet_v2_x1_0(weights=ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1)
        self.conv1 = backbone.conv1
        self.maxpool = backbone.maxpool
        self.stage2 = backbone.stage2
        self.stage3 = backbone.stage3
        self.stage4 = backbone.stage4
        self.conv5 = backbone.conv5
        feat_dim = 1024

        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(feat_dim, num_classes)

        if freeze_backbone:
            for param in backbone.parameters():
                param.requires_grad = False

    def forward(self, x):
        b, t, c, h, w = x.size()
        x = x.view(b * t, c, h, w)

        x = self.conv1(x)
        x = self.maxpool(x)
        x = temporal_shift(x, self.n_segment)
        x = self.stage2(x)
        x = temporal_shift(x, self.n_segment)
        x = self.stage3(x)
        x = temporal_shift(x, self.n_segment)
        x = self.stage4(x)
        x = temporal_shift(x, self.n_segment)
        x = self.conv5(x)

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(b, t, -1)

        x = x.mean(dim=1)
        x = self.dropout(x)
        out = self.fc(x)
        return out


# --------------------------
# MARK: TSN 标准范式验证
# 之前的实现 TSN 不标准: 标准的是 FC -> Avg, 之前是 Avg -> FC
# 二者在数学上等价, 参数完全一致, FLOPs影响<0.01%, 可忽略. dropout会导致细微影响, 但先忽略
# --------------------------

class ResNet50_TSN(nn.Module):
    """ResNet50 严格遵循 TSN 标准范式: 每帧独立 Dropout+FC, 再 temporal mean consensus.

    Architecture:
        stem -> layer1 -> layer2 -> layer3 -> layer4
        -> pool -> Dropout -> FC (per segment) -> mean consensus
    """

    def __init__(self, num_classes=27, n_segment=8, freeze_backbone=False):
        super(ResNet50_TSN, self).__init__()
        self.n_segment = n_segment

        backbone = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.stem = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        feat_dim = 2048

        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(feat_dim, num_classes)

        if freeze_backbone:
            for param in backbone.parameters():
                param.requires_grad = False

    def forward(self, x):
        b, t, c, h, w = x.size()
        x = x.view(b * t, c, h, w)

        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = F.adaptive_avg_pool2d(x, (1, 1))   # (B*T, 2048, 1, 1)
        x = x.view(b * t, -1)                  # (B*T, 2048)

        x = self.dropout(x)                    # per-segment dropout
        x = self.fc(x)                         # (B*T, num_classes)
        x = x.view(b, t, -1)                   # (B, T, num_classes)
        out = x.mean(dim=1)                    # (B, num_classes) consensus
        return out


class MobileNetV2_TSN(nn.Module):
    """MobileNetV2 严格遵循 TSN 标准范式: 每帧独立 Dropout+FC, 再 temporal mean consensus.

    Architecture:
        MobileNetV2 features -> pool -> Dropout -> FC (per segment) -> mean consensus
    """

    def __init__(self, num_classes=27, n_segment=8, freeze_backbone=False):
        super(MobileNetV2_TSN, self).__init__()
        self.n_segment = n_segment

        backbone = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V2)
        self.features = backbone.features
        feat_dim = _get_stage_out_channels(self.features)

        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(feat_dim, num_classes)

        if freeze_backbone:
            for param in backbone.parameters():
                param.requires_grad = False

    def forward(self, x):
        b, t, c, h, w = x.size()
        x = x.view(b * t, c, h, w)

        x = self.features(x)

        x = F.adaptive_avg_pool2d(x, (1, 1))   # (B*T, 1280, 1, 1)
        x = x.view(b * t, -1)                  # (B*T, 1280)

        x = self.dropout(x)                    # per-segment dropout
        x = self.fc(x)                         # (B*T, num_classes)
        x = x.view(b, t, -1)                   # (B, T, num_classes)
        out = x.mean(dim=1)                    # (B, num_classes) consensus
        return out

