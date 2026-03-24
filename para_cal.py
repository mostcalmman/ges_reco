import os
import argparse
import torch

from models import modelList, ResNetVideoModel, ResNetGRUVideoModel, LightweightTSMModel, UltraLightConvGRUModel, LightweightTSMResNetModel, UltraLightConvGRUResNetModel, UltraLightGRUModel, UltraLightMEGRUModel, UltraLightMELiteGRUModel, UltraLightMEBeforeGRUModel, UltraLightParallelMEGRUModel, UltraLightMELiteBeforeGRUModel, UltraLightParallelMELiteGRUModel
from utils import get_config, get_platform_name

# 尝试导入 thop，如果未安装则给出提示
try:
    from thop import profile
    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False
    profile = None


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Calculate Model Parameters")
    parser.add_argument("--model_type", type=str, choices=modelList, required=True, help="使用的模型结构名称")
    return parser.parse_args()


def create_model(model_type, config):
    """
    根据模型类型创建模型实例
    
    Args:
        model_type: 模型类型字符串
        config: 配置字典
        
    Returns:
        nn.Module: 初始化后的模型
    """
    if model_type == 'resnet_gru':
        model = ResNetGRUVideoModel(
            num_classes=config.get("num_classes", 27), 
            hidden_dim=config.get("hidden_dim", 256),
            freeze_backbone=True
        )
    elif model_type == 'resnet':
        model = ResNetVideoModel(
            num_classes=config.get("num_classes", 27),
            freeze_backbone=True
        )
    elif model_type == 'lightweight_tsm':
        model = LightweightTSMModel(
            num_classes=config.get("num_classes", 27),
            n_segment=config.get("num_frames", 37)
        )
    elif model_type == 'ultralight_convgru':
        model = UltraLightConvGRUModel(
            num_classes=config.get("num_classes", 27),
            n_segment=config.get("num_frames", 37)
        )
    elif model_type == 'lightweight_tsm_resnet':
        model = LightweightTSMResNetModel(
            num_classes=config.get("num_classes", 27),
            n_segment=config.get("num_frames", 37),
            pretrained=True
        )
    elif model_type == 'ultralight_convgru_resnet':
        model = UltraLightConvGRUResNetModel(
            num_classes=config.get("num_classes", 27),
            n_segment=config.get("num_frames", 37),
            pretrained=True
        )
    elif model_type == 'ultralight_gru':
        model = UltraLightGRUModel(
            num_classes=config.get("num_classes", 27),
            n_segment=config.get("num_frames", 37),
            hidden_dim=config.get("hidden_dim", 128)
        )
    elif model_type == 'ultralight_me_gru':
        model = UltraLightMEGRUModel(
            num_classes=config.get("num_classes", 27),
            n_segment=config.get("num_frames", 37),
            hidden_dim=config.get("hidden_dim", 128)
        )
    elif model_type == 'ultralight_me_lite_gru':
        model = UltraLightMELiteGRUModel(
            num_classes=config.get("num_classes", 27),
            n_segment=config.get("num_frames", 37),
            hidden_dim=config.get("hidden_dim", 128)
        )
    elif model_type == 'ultralight_me_before_gru':
        model = UltraLightMEBeforeGRUModel(
            num_classes=config.get("num_classes", 27),
            n_segment=config.get("num_frames", 37),
            hidden_dim=config.get("hidden_dim", 128)
        )
    elif model_type == 'ultralight_parallel_me_gru':
        model = UltraLightParallelMEGRUModel(
            num_classes=config.get("num_classes", 27),
            n_segment=config.get("num_frames", 37),
            hidden_dim=config.get("hidden_dim", 128)
        )
    elif model_type == 'ultralight_me_lite_before_gru':
        model = UltraLightMELiteBeforeGRUModel(
            num_classes=config.get("num_classes", 27),
            n_segment=config.get("num_frames", 37),
            hidden_dim=config.get("hidden_dim", 128)
        )
    elif model_type == 'ultralight_parallel_me_lite_gru':
        model = UltraLightParallelMELiteGRUModel(
            num_classes=config.get("num_classes", 27),
            n_segment=config.get("num_frames", 37),
            hidden_dim=config.get("hidden_dim", 128)
        )
    else:
        raise ValueError(f"未知的模型类型: {model_type}")
    
    return model


def calculate_parameters(model):
    """
    计算模型的参数量统计
    
    Args:
        model: PyTorch 模型实例
        
    Returns:
        tuple: (total_params, trainable_params, frozen_params)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    return total_params, trainable_params, frozen_params


def calculate_flops(model, config):
    """
    计算模型的 FLOPs (浮点运算次数)
    
    Args:
        model: PyTorch 模型实例
        config: 配置字典，包含 num_frames 和 img_size
        
    Returns:
        tuple: (flops, params) - FLOPs 和参数量 (来自 thop.profile)
    """
    # 从配置读取输入尺寸
    num_frames = config.get("num_frames", 37)
    img_size = config.get("img_size", (100, 176))
    height, width = img_size
    
    # 构造 dummy input: (B=1, T, C=3, H, W)
    dummy_input = torch.randn(1, num_frames, 3, height, width)
    
    # 计算 FLOPs 和参数量
    flops, params = profile(
        model,
        inputs=(dummy_input,),
        verbose=False
    )
    
    return flops, params


def main():
    args = parse_args()
    
    # 加载配置
    config = get_config()
    
    # 打印平台信息
    platform_name = get_platform_name()
    print(f"\nPlatform: {platform_name}")
    print(f"=== 模型 [{args.model_type}] 统计信息 ===")
    
    # 创建模型
    model = create_model(args.model_type, config)
    
    # 计算参数量
    total_params, trainable_params, frozen_params = calculate_parameters(model)
    
    print(f"\n[参数量统计]")
    print(f"  总模型参数量 (Total Parameters): {total_params:,}")
    print(f"  可训练参数量 (Trainable Parameters): {trainable_params:,}")
    print(f"  冻结参数量 (Frozen Parameters): {frozen_params:,}")
    
    # 计算 FLOPs
    if THOP_AVAILABLE:
        flops, _ = calculate_flops(model, config)
        num_frames = config.get("num_frames", 37)
        img_size = config.get("img_size", (100, 176))
        
        print(f"\n[计算量统计]")
        print(f"  输入尺寸: (B=1, T={num_frames}, C=3, H={img_size[0]}, W={img_size[1]})")
        print(f"  FLOPs: {flops/1e9:.3f} G ({flops:,.0f})")
        print(f"  FLOPs per frame: {flops/num_frames/1e6:.3f} M")
    else:
        print(f"\n[计算量统计]")
        print("  提示: 安装 thop 库以计算 FLOPs")
        print("        pip install thop")


if __name__ == "__main__":
    main()

# python para_cal.py --model_type resnet
