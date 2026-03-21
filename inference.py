import os
import argparse
import pandas as pd
import torch
import sys
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset import CONFIG, JesterDataset, val_transform
from models import ResNetVideoModel, ResNetGRUVideoModel, LightweightTSMModel, UltraLightConvGRUModel, LightweightTSMResNetModel, UltraLightConvGRUResNetModel

def parse_args():
    parser = argparse.ArgumentParser(description="Gesture Recognition Inference")
    parser.add_argument("--model_type", type=str, choices=['resnet', 'resnet_gru', 'lightweight_tsm', 'ultralight_convgru', 'lightweight_tsm_resnet', 'ultralight_convgru_resnet'], default='resnet', help="使用的模型结构")
    parser.add_argument("--csv_path", type=str, default="", help="要推理的 CSV 文件路径(数据集推理)")
    parser.add_argument("--root_dir", type=str, default="dataset/Test", help="要推理的视频图片根目录")
    parser.add_argument("--video_path", type=str, default="", help="单个视频文件夹路径(单视频推理)")
    parser.add_argument("--model_weight", type=str, required=True, help="模型权重文件路径")
    parser.add_argument("--output_csv", type=str, default="checkpoint/inference_results.csv", help="推理结果输出的 CSV 文件路径")
    parser.add_argument("--batch_size", type=int, default=16, help="推理时的 Batch Size")
    return parser.parse_args()

def infer_single_video(args, model, device):
    print(f"正在对单个视频进行推理: {args.video_path}")
    
    frames = []
    if os.path.exists(args.video_path):
        frame_files = sorted([f for f in os.listdir(args.video_path) if f.endswith('.jpg')])
        total_frames = len(frame_files)
        
        if total_frames <= CONFIG["num_frames"]:
            indices = np.linspace(1, total_frames, total_frames, dtype=int)
            padding = np.ones(CONFIG["num_frames"] - total_frames, dtype=int) * total_frames
            indices = np.concatenate((indices, padding))
        else:
            indices = np.linspace(1, total_frames, CONFIG["num_frames"], dtype=int)
            
        for i in indices:
            frame_name = f"{i:05d}.jpg"
            img_path = os.path.join(args.video_path, frame_name)
            try:
                img = Image.open(img_path).convert('RGB')
            except FileNotFoundError:
                img = Image.new('RGB', (CONFIG["img_size"][1], CONFIG["img_size"][0]), color=0)
            img = val_transform(img)
            frames.append(img)
            
    if not frames:
        print("未找到视频帧，请检查路径。")
        return
        
    inputs = torch.stack(frames).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        
    pred_label = predicted.item()
    print(f"✅ 单视频推理完成! 预测标签 ID: {pred_label}")
    
    results_df = pd.DataFrame({"video_path": [args.video_path], "predicted_label_id": [pred_label]})
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    results_df.to_csv(args.output_csv, index=False)
    print(f"✅ 推理结果已保存至: {args.output_csv}")

def infer_dataset(args, model, device):
    if not os.path.exists(args.csv_path):
        print(f"❌ 找不到 CSV 文件: {args.csv_path}")
        return
        
    print(f"读取数据集: {args.csv_path}")
    test_dataset = JesterDataset(
        csv_file=args.csv_path,
        root_dir=args.root_dir,
        num_frames=CONFIG["num_frames"],
        transform=val_transform,
        is_test=False 
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    predictions = []
    video_ids = []
    
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    has_labels = False
    
    criterion = torch.nn.CrossEntropyLoss()
    
    print(f"开始对 {len(test_dataset)} 个样本进行推理...")
    with torch.no_grad():
        for i, (inputs, labels, v_ids) in enumerate(test_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            if labels[0].item() != -1:
                has_labels = True
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()
            
            predictions.extend(predicted.cpu().numpy())
            video_ids.extend(v_ids)
            
            if (i + 1) % 10 == 0 or (i + 1) == len(test_loader):
                print(f"推理进度: [{i+1}/{len(test_loader)}]")

    print("-" * 30)
    if has_labels and test_total > 0:
        final_loss = test_loss / len(test_loader)
        final_acc = 100. * test_correct / test_total
        print(f"🎯 数据集包含真实标签，评估结果如下:")
        print(f"   Loss: {final_loss:.4f}")
        print(f"   Accuracy: {final_acc:.2f}%")
    else:
        print("⚠️ 推理的数据集没有提供真实标签，已跳过 Loss 和 Acc 计算。")

    results_df = pd.DataFrame({
        "video_id": video_ids,
        "predicted_label_id": predictions
    })
    
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    results_df.to_csv(args.output_csv, index=False)
    print(f"✅ 推理完成！预测结果已保存至: {args.output_csv}")

def run_inference():
    args = parse_args()
    device = CONFIG["device"]
    
    print(f"正在加载模型并准备推理...")
    print(f"使用设备: {device}")

    if not os.path.exists(args.model_weight):
        print(f"❌ 找不到模型权重文件: {args.model_weight}")
        return

    if args.model_type == 'resnet_gru':
        model = ResNetGRUVideoModel(num_classes=CONFIG["num_classes"], hidden_dim=CONFIG["hidden_dim"])
    elif args.model_type == 'lightweight_tsm':
        model = LightweightTSMModel(
            num_classes=CONFIG.get("num_classes", 27),
            n_segment=CONFIG.get("num_frames", 37)
        )
    elif args.model_type == 'ultralight_convgru':
        model = UltraLightConvGRUModel(
            num_classes=CONFIG.get("num_classes", 27),
            n_segment=CONFIG.get("num_frames", 37)
        )
    elif args.model_type == 'lightweight_tsm_resnet':
        model = LightweightTSMResNetModel(
            num_classes=CONFIG.get("num_classes", 27),
            n_segment=CONFIG.get("num_frames", 37),
            pretrained=False  # 推理时不加载预训练权重
        )
    elif args.model_type == 'ultralight_convgru_resnet':
        model = UltraLightConvGRUResNetModel(
            num_classes=CONFIG.get("num_classes", 27),
            n_segment=CONFIG.get("num_frames", 37),
            pretrained=False  # 推理时不加载预训练权重
        )
    else:
        model = ResNetVideoModel(num_classes=CONFIG["num_classes"])
        
    model.load_state_dict(torch.load(args.model_weight, map_location=device))
    model.to(device)
    model.eval()

    if args.video_path:
        infer_single_video(args, model, device)
    elif args.csv_path:
        infer_dataset(args, model, device)
    else:
        print("❌ 请提供 --csv_path (用于数据集) 或 --video_path (用于单视频)")

if __name__ == "__main__":
    run_inference()

# 执行单视频推断:
# python inference.py --video_path "dataset/Test/100010" --model_type "resnet" --model_weight "checkpoint/lightweight_gesture_model.pth"

# 执行整个测试集推断（自动结算 Loss / Acc 等）:
# python inference.py --csv_path "dataset/Test.csv" --root_dir "dataset/Test" --model_type "resnet" --model_weight "checkpoint/lightweight_gesture_model.pth"