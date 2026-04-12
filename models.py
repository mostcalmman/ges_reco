"""
Video gesture recognition model architectures.
All models accept input shape (B, T, 3, H, W) and output (B, num_classes).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from modules import *

modelList = ['ResNet18', 'LightTSM', 'LightTSMGRU', 'LightTMFGRU', 'TMFin1', 'TMFin2', 'TMFin123', 'ab1', 'ab2', 'ab3',
             'LightTMF2GRU', 'LightTMF25GRU', 'LightTMF26GRU', 'LightTMF3GRU', 'LightTMF4GRU', 'LightTMF4GRU'
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

    def __init__(self, num_classes, freeze_backbone=True):
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
    新版TMF: TMF2.5, 具体看module.py
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
    新版TMF: TMF2.6, 具体看module.py
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
    

class LightTMF3GRU(nn.Module):
    """
    新版TMF: ACSS + TMF3 + GRU
    Architecture: ACSS in shallow stages (layer1, layer2), TMF3 in deep stage (layer3)
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
    新版TMF: ACSS + TMF3 + GRU
    Architecture: ACSS in shallow stages (layer1, layer2), TMF3 in deep stage (layer3)
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


class LightTMF5GRU(nn.Module):
    """
    新版TMF: TMF5 (ME + ACSS soft shift) + GRU
    Architecture: TMF5 in deep stage (layer3), combining TMF2's multi-frame ME 
                  with ACSS's learnable soft temporal shift in parallel fusion.
    """

    def __init__(self, num_classes=27, n_segment=8, hidden_dim=128, fold_div=8):
        super(LightTMF5GRU, self).__init__()
        self.n_segment = n_segment
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(0.5)

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # Shallow stages: standard TSM (or could use ACSSResBlock for consistency)
        self.layer1 = ACSSResBlock(32, 32, stride=1, n_segment=n_segment)
        self.layer2 = ACSSResBlock(32, 64, stride=2, n_segment=n_segment)
        # Deep stage: TMF5 with ME + ACSS soft shift
        self.layer3 = TMF5ResBlock(64, 128, stride=2, n_segment=n_segment, reduction=4, fold_div=fold_div)

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


class ab3(nn.Module):
    """
    消融实验, 只留ResNetLight + GRU
    """
    def __init__(self, num_classes=27, n_segment=8, hidden_dim=128):
        super(ab3, self).__init__()
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

