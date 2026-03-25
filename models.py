"""
Video gesture recognition model architectures.
All models accept input shape (B, T, 3, H, W) and output (B, num_classes).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from modules import *

modelList = ['resnet', 'resnet_gru', 'lightweight_tsm', 'ultralight_convgru',
             'ultralight_convgru_pooled', 'lightweight_tsm_resnet', 'ultralight_convgru_resnet',
             'ultralight_gru', 'ultralight_me_gru', 'ultralight_me_lite_gru',
             'ultralight_me_before_gru', 'ultralight_parallel_me_gru',
             'ultralight_me_lite_before_gru', 'ultralight_parallel_me_lite_gru', 'ParallelMEAdd']


def _load_resnet18_weights_to_tsm(target_model, pretrained_resnet):
    """Load ResNet18 pretrained weights into TSMResBlock layers.
    
    ResNet18 has 2 BasicBlocks per layer. We map the first BasicBlock's weights
    to our single TSMResBlock per layer.
    
    Mapping:
        ResNet18.layerX.0.{conv1,bn1,conv2,bn2,downsample} 
        -> target_model.layerX.{conv1,bn1,conv2,bn2,shortcut}
    
    Args:
        target_model: Model with layer1, layer2, layer3, (layer4) TSMResBlocks
        pretrained_resnet: Pretrained ResNet18 model from torchvision
    """
    source_state = pretrained_resnet.state_dict()
    target_state = target_model.state_dict()
    
    # Map layer indices: ResNet18 layer1 -> our layer1, etc.
    # ResNet18 has 2 blocks per layer (0 and 1), we take block 0
    layer_names = ['layer1', 'layer2', 'layer3']
    if hasattr(target_model, 'layer4'):
        layer_names.append('layer4')
    
    loaded_keys = []
    for layer_name in layer_names:
        if not hasattr(target_model, layer_name):
            continue
            
        # Map conv1, bn1, conv2, bn2
        for block_idx in range(2):  # 0 and 1 for conv/bn layers
            src_prefix = f'{layer_name}.0.conv{block_idx+1}'
            tgt_prefix = f'{layer_name}.conv{block_idx+1}'
            
            # Conv weight
            src_key = f'{src_prefix}.weight'
            tgt_key = f'{tgt_prefix}.weight'
            if src_key in source_state and tgt_key in target_state:
                target_state[tgt_key].copy_(source_state[src_key])
                loaded_keys.append(tgt_key)
            
            # BN: weight, bias, running_mean, running_var
            for bn_param in ['weight', 'bias', 'running_mean', 'running_var', 'num_batches_tracked']:
                src_key = f'{src_prefix.replace("conv", "bn")}.{bn_param}'
                tgt_key = f'{tgt_prefix.replace("conv", "bn")}.{bn_param}'
                if src_key in source_state and tgt_key in target_state:
                    target_state[tgt_key].copy_(source_state[src_key])
                    loaded_keys.append(tgt_key)
        
        # Map shortcut (downsample in ResNet) if exists
        src_downsample_key = f'{layer_name}.0.downsample.0.weight'
        tgt_shortcut_key = f'{layer_name}.shortcut.0.weight'
        if src_downsample_key in source_state and tgt_shortcut_key in target_state:
            target_state[tgt_shortcut_key].copy_(source_state[src_downsample_key])
            loaded_keys.append(tgt_shortcut_key)
            
            # Map BN in shortcut
            for bn_param in ['weight', 'bias', 'running_mean', 'running_var', 'num_batches_tracked']:
                src_key = f'{layer_name}.0.downsample.1.{bn_param}'
                tgt_key = f'{layer_name}.shortcut.1.{bn_param}'
                if src_key in source_state and tgt_key in target_state:
                    target_state[tgt_key].copy_(source_state[src_key])
                    loaded_keys.append(tgt_key)
    
    print(f"Loaded {len(loaded_keys)} parameters from ResNet18 pretrained weights")
    return loaded_keys

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
# MARK: ResnetGRU
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
# MARK: LightTSM
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
# MARK: UltraLightMELiteGRU
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
# MARK: UltraLightMEBeforeGRU
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


# --------------------------
# MARK: UltraLightParallelMEGRU
# TSM + Parallel ME/TSM (Scheme B) + Standard GRU
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

        out = self.fc(last_hidden)              # (B, num_classes)
        return out

# MARK: LSR
class ParallelMEGRUAddModel(nn.Module):
    def __init__(self, num_classes=27, n_segment=8, hidden_dim=128):
        super(ParallelMEGRUAddModel, self).__init__()
        self.n_segment = n_segment
        self.hidden_dim = hidden_dim

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.layer1 = TSMResBlock(32, 32, stride=1, n_segment=n_segment)
        self.layer2 = TSMResBlock(32, 64, stride=2, n_segment=n_segment)
        self.layer3 = ParallelMETSMAddResBlock(64, 128, stride=2, n_segment=n_segment, reduction=4)

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
# MARK: UltraLightMELiteBeforeGRU
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
# MARK: UltraLightParallelMELiteGRU
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

# --------------------------
# MARK: Ultralight_ConvGRU
# TSM + ConvGRU
# --------------------------

class UltraLightConvGRUModel(nn.Module):
    """Ultra-lightweight model: 3-layer TSM-ResNet with early ConvGRU.

    Architecture:
        Conv1 (3->32, stride=2) -> 3x TSMResBlock [32->64->128]
        -> ConvGRU (128->128) -> GlobalAvgPool -> FC

    Removes layer4 entirely, attaching ConvGRU after layer3 at 128 channels
    for maximum parameter reduction. Preserves full spatial resolution from
    layer3 output for ConvGRU to maintain spatial-temporal feature learning.
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

        # 输入尺寸100x176时, layer3输出为12x22 (H/8=12, W/8=22)
        _, c_out, h_out, w_out = x.size()
        x = x.view(b, t, c_out, h_out, w_out)   # (B, T, 128, H/8, W/8)
        x = self.convgru(x)                     # (B, 128, H/8, W/8)

        x = F.adaptive_avg_pool2d(x, (1, 1))    # (B, 128, 1, 1)
        x = x.view(b, -1)                       # (B, 128)
        out = self.fc(x)                        # (B, num_classes)
        return out

# --------------------------
# MARK: Ultralight_ConvGRU_Pooled
# TSM + ConvGRU, convGRU前缩特征图
# --------------------------

class UltraLightConvGRUPooledModel(nn.Module):
    """Ultra-lightweight model with spatial pooling before ConvGRU.

    Architecture:
        Conv1 (3->32, stride=2) -> 3x TSMResBlock [32->64->128]
        -> SpatialPool (7x7) -> ConvGRU (128->128) -> GlobalAvgPool -> FC

    Similar to UltraLightConvGRUModel but adds spatial pooling before ConvGRU
    to reduce computational cost. Trades some spatial precision for efficiency.
    Target: < 2.5M parameters, lower compute than full-resolution version.

    Args:
        num_classes: Number of output classes (default 27 for Jester).
        n_segment: Number of temporal segments (sampled frames).
        pool_size: Spatial pooling size before ConvGRU (default 7).
    """

    def __init__(self, num_classes=27, n_segment=8, pool_size=7):
        super(UltraLightConvGRUPooledModel, self).__init__()
        self.n_segment = n_segment
        self.pool_size = pool_size

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
        x = F.adaptive_avg_pool2d(x, (self.pool_size, self.pool_size))
                                                # (B*T, 128, pool_size, pool_size)

        _, c_out, h_out, w_out = x.size()
        x = x.view(b, t, c_out, h_out, w_out)   # (B, T, 128, pool_size, pool_size)
        x = self.convgru(x)                     # (B, 128, pool_size, pool_size)

        x = F.adaptive_avg_pool2d(x, (1, 1))    # (B, 128, 1, 1)
        x = x.view(b, -1)                       # (B, 128)
        out = self.fc(x)                        # (B, num_classes)
        return out


# --------------------------
# MARK: Pre-TSM
# ResNet18-channel TSM Models (with pretrained weights)
# --------------------------

class LightweightTSMResNetModel(nn.Module):
    """Lightweight TSM model with ResNet18 channel dimensions.

    Architecture:
        Conv1 (3->64, stride=2) -> 4x TSMResBlock [64->128->256->512]
        -> temporal mean pooling -> GlobalAvgPool -> FC

    Uses ResNet18 channel progression (64, 128, 256, 512) to enable loading
    pretrained ImageNet weights from ResNet18 backbone.
    Target: ~4M parameters (higher than LightweightTSMModel but better convergence).

    Args:
        num_classes: Number of output classes (default 27 for Jester).
        n_segment: Number of temporal segments (sampled frames).
        pretrained: If True, load ResNet18 pretrained weights into backbone.
    """

    def __init__(self, num_classes=27, n_segment=8, pretrained=True):
        super(LightweightTSMResNetModel, self).__init__()
        self.n_segment = n_segment

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # ResNet18 channel progression: 64 -> 128 -> 256 -> 512
        self.layer1 = TSMResBlock(64, 64, stride=1, n_segment=n_segment)
        self.layer2 = TSMResBlock(64, 128, stride=2, n_segment=n_segment)
        self.layer3 = TSMResBlock(128, 256, stride=2, n_segment=n_segment)
        self.layer4 = TSMResBlock(256, 512, stride=2, n_segment=n_segment)

        self.fc = nn.Linear(512, num_classes)

        # Load ResNet18 pretrained weights
        if pretrained:
            resnet_pretrained = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            # ResNet18 conv1 is Conv2d(3,64,7,2,3), our conv1 is Sequential with same layer
            self.conv1[0].load_state_dict(resnet_pretrained.conv1.state_dict())
            self.conv1[1].load_state_dict(resnet_pretrained.bn1.state_dict())
            _load_resnet18_weights_to_tsm(self, resnet_pretrained)
            print(f"Loaded ResNet18 pretrained weights into {self.__class__.__name__}")

    def forward(self, x):
        """
        Args:
            x: (B, T, 3, H, W) video frame sequence.
        Returns:
            (B, num_classes) classification logits.
        """
        b, t, c, h, w = x.size()

        x = x.view(b * t, c, h, w)              # (B*T, 3, H, W)
        x = self.conv1(x)                       # (B*T, 64, H/4, W/4)
        x = self.layer1(x)                      # (B*T, 64, H/4, W/4)
        x = self.layer2(x)                      # (B*T, 128, H/8, W/8)
        x = self.layer3(x)                      # (B*T, 256, H/16, W/16)
        x = self.layer4(x)                      # (B*T, 512, H/32, W/32)

        _, c_out, h_out, w_out = x.size()
        x = x.view(b, t, c_out, h_out, w_out)   # (B, T, 512, H/32, W/32)
        x = x.mean(dim=1)                       # (B, 512, H/32, W/32)

        x = F.adaptive_avg_pool2d(x, (1, 1))    # (B, 512, 1, 1)
        x = x.view(b, -1)                       # (B, 512)
        out = self.fc(x)                        # (B, num_classes)
        return out

# --------------------------
# MARK: Pre-UltraLight_ConvGRU_Pooled
# ResNet18-channel TSM Models (with pretrained weights)
# --------------------------

class UltraLightConvGRUResNetModel(nn.Module):
    """Ultra-lightweight ConvGRU model with ResNet18 channel dimensions.

    Architecture:
        Conv1 (3->64, stride=2) -> 3x TSMResBlock [64->128->256]
        -> SpatialPool (7x7) -> ConvGRU (256->256) -> GlobalAvgPool -> FC

    Uses ResNet18 channel progression (64, 128, 256) for first 3 layers
    to enable loading pretrained ImageNet weights. Attaches ConvGRU after
    layer3 for temporal aggregation.
    Target: ~3.5M parameters.

    Args:
        num_classes: Number of output classes (default 27 for Jester).
        n_segment: Number of temporal segments (sampled frames).
        pretrained: If True, load ResNet18 pretrained weights into backbone.
    """

    def __init__(self, num_classes=27, n_segment=8, pretrained=True):
        super(UltraLightConvGRUResNetModel, self).__init__()
        self.n_segment = n_segment

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # ResNet18 channel progression for first 3 layers: 64 -> 128 -> 256
        self.layer1 = TSMResBlock(64, 64, stride=1, n_segment=n_segment)
        self.layer2 = TSMResBlock(64, 128, stride=2, n_segment=n_segment)
        self.layer3 = TSMResBlock(128, 256, stride=2, n_segment=n_segment)
        # No layer4 — ConvGRU attaches directly after layer3

        self.convgru = ConvGRU(input_channels=256, hidden_channels=256)

        self.fc = nn.Linear(256, num_classes)

        # Load ResNet18 pretrained weights (layer1-3)
        if pretrained:
            resnet_pretrained = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            self.conv1[0].load_state_dict(resnet_pretrained.conv1.state_dict())
            self.conv1[1].load_state_dict(resnet_pretrained.bn1.state_dict())
            _load_resnet18_weights_to_tsm(self, resnet_pretrained)
            print(f"Loaded ResNet18 pretrained weights into {self.__class__.__name__}")

    def forward(self, x):
        """
        Args:
            x: (B, T, 3, H, W) video frame sequence.
        Returns:
            (B, num_classes) classification logits.
        """
        b, t, c, h, w = x.size()

        x = x.view(b * t, c, h, w)              # (B*T, 3, H, W)
        x = self.conv1(x)                       # (B*T, 64, H/4, W/4)
        x = self.layer1(x)                      # (B*T, 64, H/4, W/4)
        x = self.layer2(x)                      # (B*T, 128, H/8, W/8)
        x = self.layer3(x)                      # (B*T, 256, H/16, W/16)

        # 降低空间分辨率, 减少 ConvGRU 计算量和显存占用
        x = F.adaptive_avg_pool2d(x, (7, 7))    # (B*T, 256, 7, 7)

        _, c_out, h_out, w_out = x.size()
        x = x.view(b, t, c_out, h_out, w_out)   # (B, T, 256, 7, 7)
        x = self.convgru(x)                     # (B, 256, 7, 7)

        x = F.adaptive_avg_pool2d(x, (1, 1))    # (B, 256, 1, 1)
        x = x.view(b, -1)                       # (B, 256)
        out = self.fc(x)                        # (B, num_classes)
        return out
