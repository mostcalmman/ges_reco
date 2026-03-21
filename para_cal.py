import os
import argparse
import torch

from models import modelList, ResNetVideoModel, ResNetGRUVideoModel, LightweightTSMModel, UltraLightConvGRUModel, LightweightTSMResNetModel, UltraLightConvGRUResNetModel
from dataset import CONFIG, CONFIG_Linux, IS_WINDOWS, IS_LINUX, get_config

def parse_args():
    parser = argparse.ArgumentParser(description="Calculate Model Parameters")
    parser.add_argument("--model_type", type=str, choices=modelList, required=True, help="使用的模型结构名称")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 根据平台获取相应配置
    config = get_config()
    
    # 打印平台信息
    platform_name = "Windows" if IS_WINDOWS else ("Linux" if IS_LINUX else "Unknown")
    print(f"\nPlatform: {platform_name}")
    print(f"=== 模型 [{args.model_type}] 参数量统计 ===")
    
    if args.model_type == 'resnet_gru':
        model = ResNetGRUVideoModel(
            num_classes=config.get("num_classes", 27), 
            hidden_dim=config.get("hidden_dim", 256),
            freeze_backbone=True
        )
    elif args.model_type == 'resnet':
        model = ResNetVideoModel(
            num_classes=config.get("num_classes", 27),
            freeze_backbone=True
        )
    elif args.model_type == 'lightweight_tsm':
        model = LightweightTSMModel(
            num_classes=config.get("num_classes", 27),
            n_segment=config.get("num_frames", 37)
        )
    elif args.model_type == 'ultralight_convgru':
        model = UltraLightConvGRUModel(
            num_classes=config.get("num_classes", 27),
            n_segment=config.get("num_frames", 37)
        )
    elif args.model_type == 'lightweight_tsm_resnet':
        model = LightweightTSMResNetModel(
            num_classes=config.get("num_classes", 27),
            n_segment=config.get("num_frames", 37),
            pretrained=True
        )
    elif args.model_type == 'ultralight_convgru_resnet':
        model = UltraLightConvGRUResNetModel(
            num_classes=config.get("num_classes", 27),
            n_segment=config.get("num_frames", 37),
            pretrained=True
        )
    else:
        print(f"❌ 未知的模型类型: {args.model_type}")
        return

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    print(f"总模型参数量 (Total Parameters): {total_params:,}")
    print(f"可训练参数量 (Trainable Parameters): {trainable_params:,}")
    print(f"冻结参数量 (Frozen Parameters): {frozen_params:,}")

if __name__ == "__main__":
    main()

# python para_cal.py --model_type resnet