import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import os
import argparse
import pandas as pd
import sys

# 动态添加当前目录到系统路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from baseline import LightweightVideoModel, CONFIG, JesterDataset

def parse_args():
    parser = argparse.ArgumentParser(description="Gesture Recognition Inference")
    parser.add_argument("--csv_path", type=str, default="dataset/Test.csv", help="要推理的 CSV 文件路径")
    parser.add_argument("--root_dir", type=str, default="dataset/Test", help="要推理的视频图片根目录")
    parser.add_argument("--model_weight", type=str, default="checkpoint/lightweight_gesture_model.pth", help="模型权重文件路径")
    parser.add_argument("--output_csv", type=str, default="checkpoint/inference_results.csv", help="推理结果输出的 CSV 文件路径")
    parser.add_argument("--batch_size", type=int, default=16, help="推理时的 Batch Size")
    return parser.parse_args()

def run_inference():
    args = parse_args()
    device = CONFIG["device"]
    
    print(f"正在加载模型并准备推理...")
    print(f"使用设备: {device}")

    # 1. 检查并加载模型
    if not os.path.exists(args.model_weight):
        print(f"❌ 找不到模型权重文件: {args.model_weight}")
        print("请先运行 python baseline.py 训练模型！")
        return

    # 这里需要确保推断的类别数与训练时一致，通常从训练集的配置或模型本身推断
    model = LightweightVideoModel(num_classes=CONFIG["num_classes"], hidden_dim=CONFIG["hidden_dim"])
    model.load_state_dict(torch.load(args.model_weight, map_location=device))
    model.to(device)
    model.eval()

    # 2. 图像预处理 (必须和训练时验证集的处理一致)
    val_transform = transforms.Compose([
        transforms.Resize(CONFIG["img_size"]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 3. 准备 Dataset 和 DataLoader
    if not os.path.exists(args.csv_path):
        print(f"❌ 找不到 CSV 文件: {args.csv_path}")
        return
        
    print(f"读取数据集: {args.csv_path}")
    test_dataset = JesterDataset(
        csv_file=args.csv_path,
        root_dir=args.root_dir,
        num_frames=CONFIG["num_frames"],
        transform=val_transform,
        is_test=False # 设置为 False 以便 JesterDataset 尝试读取标签
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # 4. 开始推理
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
            
            # 检查这批数据是否有真实标签 (不是 -1)
            if labels[0].item() != -1:
                has_labels = True
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()
            
            predictions.extend(predicted.cpu().numpy())
            video_ids.extend(v_ids)
            
            # 简单打印进度
            if (i + 1) % 10 == 0 or (i + 1) == len(test_loader):
                print(f"推理进度: [{i+1}/{len(test_loader)}]")

    # 5. 输出结果和指标
    print("-" * 30)
    if has_labels and test_total > 0:
        final_loss = test_loss / len(test_loader)
        final_acc = 100. * test_correct / test_total
        print(f"🎯 数据集包含真实标签，评估结果如下:")
        print(f"   Loss: {final_loss:.4f}")
        print(f"   Accuracy: {final_acc:.2f}%")
    else:
        print("⚠️ 推理的数据集没有提供正确结果的标签，已跳过 Loss 和 Acc 计算。")

    # 6. 保存预测结果到 CSV
    results_df = pd.DataFrame({
        "video_id": video_ids,
        "predicted_label_id": predictions
    })
    
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    results_df.to_csv(args.output_csv, index=False)
    print(f"✅ 推理完成！预测结果已保存至: {args.output_csv}")

if __name__ == "__main__":
    run_inference()