import torch

from models import *

def build_model(model_type, config, pretrained=None, freeze_backbone=None, device=None):
    """根据模型类型创建模型实例。"""
    num_classes = config.get("num_classes", 27)
    num_frames = config.get("num_frames", 16)
    hidden_dim = config.get("hidden_dim", 128)

    if freeze_backbone is None:
        freeze_backbone = True

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
    elif model_type == "TMF1":
        model = TMF1(
            num_classes=num_classes,
            n_segment=num_frames,
            hidden_dim=hidden_dim,
        )
    elif model_type == "TMF2":
        model = TMF2(
            num_classes=num_classes,
            n_segment=num_frames,
            hidden_dim=hidden_dim,
        )
    elif model_type == "TMF123":
        model = TMF123(
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
