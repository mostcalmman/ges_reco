import torch

from models import *

def build_model(model_type, config, pretrained=None, freeze_backbone=None, device=None):
    """根据模型类型创建模型实例。"""
    num_classes = config.get("num_classes", 27)
    num_frames = config.get("num_frames", 16)
    hidden_dim = config.get("hidden_dim", 128)

    if freeze_backbone is None:
        freeze_backbone = bool(config.get("freeze_backbone", False))

    if model_type == "ResNet18":
        model = ResNet18(
            num_classes=num_classes,
            freeze_backbone=freeze_backbone,
        )
    elif model_type == "LightTSM":
        model = LightTSM(
            num_classes=num_classes,
            n_segment=num_frames,
        )
    elif model_type == "LightTSMGRU":
        model = LightTSMGRU(
            num_classes=num_classes,
            n_segment=num_frames,
            hidden_dim=hidden_dim,
        )
    elif model_type == "LightTMFGRU":
        model = LightTMFGRU(
            num_classes=num_classes,
            n_segment=num_frames,
            hidden_dim=hidden_dim,
        )
    elif model_type == "LightTMF2GRU":
        model = LightTMF2GRU(
            num_classes=num_classes,
            n_segment=num_frames,
            hidden_dim=hidden_dim,
        )
    elif model_type == "LightTMF25GRU":
        model = LightTMF25GRU(
            num_classes=num_classes,
            n_segment=num_frames,
            hidden_dim=hidden_dim,
        )
    elif model_type == "LightTMF26GRU":
        model = LightTMF26GRU(
            num_classes=num_classes,
            n_segment=num_frames,
            hidden_dim=hidden_dim,
        )
    elif model_type == "LightTMF3GRU":
        model = LightTMF3GRU(
            num_classes=num_classes,
            n_segment=num_frames,
            hidden_dim=hidden_dim,
        )
    elif model_type == "LightTMF4GRU":
        model = LightTMF4GRU(
            num_classes=num_classes,
            n_segment=num_frames,
            hidden_dim=hidden_dim,
        )
    elif model_type == "TMFin1":
        model = TMFin1(
            num_classes=num_classes,
            n_segment=num_frames,
            hidden_dim=hidden_dim,
        )
    elif model_type == "TMFin2":
        model = TMFin2(
            num_classes=num_classes,
            n_segment=num_frames,
            hidden_dim=hidden_dim,
        )
    elif model_type == "TMFin123":
        model = TMFin123(
            num_classes=num_classes,
            n_segment=num_frames,
            hidden_dim=hidden_dim,
        )
    elif model_type == "ab1":
        model = ab1(
            num_classes=num_classes,
            n_segment=num_frames,
            hidden_dim=hidden_dim,
        )
    elif model_type == "ab2":
        model = ab2(
            num_classes=num_classes,
            n_segment=num_frames,
            hidden_dim=hidden_dim,
        )
    elif model_type == "ab3":
        model = ab3(
            num_classes=num_classes,
            n_segment=num_frames,
            hidden_dim=hidden_dim,
        )
    elif model_type == "LightResNet":
        model = LightResNet(
            num_classes=num_classes,
            n_segment=num_frames,
            hidden_dim=hidden_dim,
        )
    elif model_type == "ResNet50":
        model = ResNet50(
            num_classes=num_classes,
            n_segment=num_frames,
            hidden_dim=hidden_dim,
            freeze_backbone=freeze_backbone,
            use_gru=False,
        )
    elif model_type == "MobileNetV2":
        model = MobileNetV2(
            num_classes=num_classes,
            n_segment=num_frames,
            freeze_backbone=freeze_backbone,
        )
    elif model_type == "ShuffleNetV2x10":
        model = ShuffleNetV2x10(
            num_classes=num_classes,
            n_segment=num_frames,
            freeze_backbone=freeze_backbone,
        )
    elif model_type == "ResNet50_TSM":
        model = ResNet50_TSM(
            num_classes=num_classes,
            n_segment=num_frames,
            hidden_dim=hidden_dim,
            freeze_backbone=freeze_backbone,
            use_gru=False,
        )
    elif model_type == "MobileNetV2_TSM":
        model = MobileNetV2_TSM(
            num_classes=num_classes,
            n_segment=num_frames,
            freeze_backbone=freeze_backbone,
        )
    elif model_type == "ShuffleNetV2x10_TSM":
        model = ShuffleNetV2x10_TSM(
            num_classes=num_classes,
            n_segment=num_frames,
            freeze_backbone=freeze_backbone,
        )
    elif model_type == "ResNet50_ACSSTMF3":
        model = ResNet50_ACSSTMF3(
            num_classes=num_classes,
            n_segment=num_frames,
            hidden_dim=hidden_dim,
            freeze_backbone=freeze_backbone,
            use_gru=False,
            fusion_mode='B',
            reduction='auto',
        )
    elif model_type == "MobileNetV2_ACSSTMF3":
        model = MobileNetV2_ACSSTMF3(
            num_classes=num_classes,
            n_segment=num_frames,
            freeze_backbone=freeze_backbone,
            fusion_mode='B',
            reduction='auto',
        )
    elif model_type == "MobileNetV3Large_ACSSTMF3":
        model = MobileNetV3Large_ACSSTMF3(
            num_classes=num_classes,
            n_segment=num_frames,
            freeze_backbone=freeze_backbone,
            fusion_mode='B',
            reduction='auto',
        )
    elif model_type == "MobileNetV3Small_ACSSTMF3":
        model = MobileNetV3Small_ACSSTMF3(
            num_classes=num_classes,
            n_segment=num_frames,
            freeze_backbone=freeze_backbone,
            fusion_mode='B',
            reduction='auto',
        )
    elif model_type == "ShuffleNetV2x10_ACSSTMF3":
        model = ShuffleNetV2x10_ACSSTMF3(
            num_classes=num_classes,
            n_segment=num_frames,
            freeze_backbone=freeze_backbone,
            fusion_mode='B',
            reduction='auto',
        )
    elif model_type == "ShuffleNetV2x20_ACSSTMF3":
        model = ShuffleNetV2x20_ACSSTMF3(
            num_classes=num_classes,
            n_segment=num_frames,
            freeze_backbone=freeze_backbone,
            fusion_mode='B',
            reduction='auto',
        )
    elif model_type == "Light_OnlyTMF3_GRU":
        model = Light_OnlyTMF3_GRU(
            num_classes=num_classes,
            n_segment=num_frames,
            hidden_dim=hidden_dim,
        )
    elif model_type == "Light_OnlyACSS_GRU":
        model = Light_OnlyACSS_GRU(
            num_classes=num_classes,
            n_segment=num_frames,
            hidden_dim=hidden_dim,
        )
    elif model_type == "ResNet50_OnlyTMF3":
        model = ResNet50_OnlyTMF3(
            num_classes=num_classes,
            n_segment=num_frames,
            hidden_dim=hidden_dim,
            freeze_backbone=freeze_backbone,
            use_gru=False,
            fusion_mode='B',
            reduction='auto',
        )
    elif model_type == "ResNet50_OnlyACSS":
        model = ResNet50_OnlyACSS(
            num_classes=num_classes,
            n_segment=num_frames,
            hidden_dim=hidden_dim,
            freeze_backbone=freeze_backbone,
            use_gru=False,
        )
    elif model_type == "ResNet50_TSN":
        model = ResNet50_TSN(
            num_classes=num_classes,
            n_segment=num_frames,
            freeze_backbone=freeze_backbone,
        )
    elif model_type == "MobileNetV2_TSN":
        model = MobileNetV2_TSN(
            num_classes=num_classes,
            n_segment=num_frames,
            freeze_backbone=freeze_backbone,
        )
    else:
        raise ValueError(f"未知的模型类型: {model_type}")

    if device is not None:
        model = model.to(device)

    return model


def load_model_weights(model, model_weight_path, device, strict=True, eval_mode=True):
    """加载模型权重，兼容纯 state_dict 与 checkpoint 字典。"""
    state = torch.load(model_weight_path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]

    model.load_state_dict(state, strict=strict)
    model = model.to(device)
    if eval_mode:
        model.eval()
    return model


def build_and_load_model(
    model_type,
    config,
    device,
    model_weight_path,
    pretrained=None,
    freeze_backbone=None,
    strict=True,
    eval_mode=True,
):
    """创建模型并加载权重。"""
    model = build_model(
        model_type=model_type,
        config=config,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
        device=device,
    )
    return load_model_weights(
        model=model,
        model_weight_path=model_weight_path,
        device=device,
        strict=strict,
        eval_mode=eval_mode,
    )
