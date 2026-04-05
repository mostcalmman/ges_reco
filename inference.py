import os
import argparse
import pandas as pd
import torch
import sys
import numpy as np
import time
from datetime import datetime
from PIL import Image
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import get_config, get_platform_name, build_model, load_model_weights
from dataset import (
    DEFAULT_NUM_FRAMES,
    JesterDataset,
    SAMPLING_UNIFORM,
    get_val_transform,
    sample_frame_indices,
)
from models import modelList


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Gesture Recognition Inference")
    parser.add_argument("--model_type", type=str, choices=modelList, default='resnet', help="使用的模型结构")
    parser.add_argument("--csv_path", type=str, default="dataset/Test.csv", help="要推理的 CSV 文件路径(数据集推理)")
    parser.add_argument("--root_dir", type=str, default="dataset/Test", help="要推理的视频图片根目录")
    parser.add_argument("--video_path", type=str, default="", help="单个视频文件夹路径(单视频推理)")
    parser.add_argument("--model_weight", type=str, required=True, help="模型权重文件路径")
    parser.add_argument("--output", type=str, default="checkpoint/inference_results", help="推理输出路径")
    parser.add_argument("--batch_size", type=int, default=None, help="推理时的 Batch Size")
    return parser.parse_args()


def ensure_parent_dir(file_path):
    """确保目标文件的父目录存在。"""
    parent_dir = os.path.dirname(file_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)


def synchronize_if_cuda(device):
    """在 CUDA 上进行计时前后同步，避免异步导致计时偏差。"""
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.synchronize()


def resolve_output_paths(output_arg, is_single_video):
    """根据 --output 参数解析结果文件路径。"""
    output_arg = output_arg.strip() if output_arg else "checkpoint/inference_results"
    output_lower = output_arg.lower()

    if is_single_video:
        if output_lower.endswith(".txt"):
            return output_arg, None
        return os.path.join(output_arg, "results.txt"), None

    if output_lower.endswith(".csv"):
        csv_path = output_arg
        results_txt_path = os.path.join(os.path.dirname(output_arg) or ".", "results.txt")
    elif output_lower.endswith(".txt"):
        results_txt_path = output_arg
        csv_path = os.path.join(os.path.dirname(output_arg) or ".", "predictions.csv")
    else:
        results_txt_path = os.path.join(output_arg, "results.txt")
        csv_path = os.path.join(output_arg, "predictions.csv")

    return results_txt_path, csv_path


def load_model(model_type, config, device, model_weight_path):
    """
    加载指定类型的模型
    
    Args:
        model_type: 模型类型
        config: 配置字典
        device: 计算设备
        model_weight_path: 模型权重文件路径
        
    Returns:
        model: 加载好权重的模型实例
    """
    model = build_model(
        model_type=model_type,
        config=config,
        pretrained=False,  # 推理阶段不触发额外预训练权重下载
        device=device,
    )
    return load_model_weights(model, model_weight_path, device)


def infer_single_video(args, model, device, config):
    """对单个视频进行推理"""
    print(f"正在对单个视频进行推理: {args.video_path}")
    
    frames = []
    if os.path.exists(args.video_path):
        frame_files = sorted([f for f in os.listdir(args.video_path) if f.endswith('.jpg')])
        total_frames = len(frame_files)
        num_frames = DEFAULT_NUM_FRAMES
        img_size = tuple(config.get("img_size", (100, 176)))

        indices = sample_frame_indices(
            total_frames=total_frames,
            num_frames=num_frames,
            sampling_mode=SAMPLING_UNIFORM,
        )
        
        # 创建 transform
        val_transform = get_val_transform(
            img_size=img_size,
            normalize_mean=config.get("normalize_mean"),
            normalize_std=config.get("normalize_std")
        )
            
        for i in indices:
            frame_name = f"{i:05d}.jpg"
            img_path = os.path.join(args.video_path, frame_name)
            try:
                img = Image.open(img_path).convert('RGB')
            except FileNotFoundError:
                img = Image.new('RGB', (img_size[1], img_size[0]), color=0)
            img = val_transform(img)
            frames.append(img)
            
    if not frames:
        print("未找到视频帧，请检查路径。")
        return
        
    inputs = torch.stack(frames).unsqueeze(0).to(device)
    
    synchronize_if_cuda(device)
    start_time = time.perf_counter()
    with torch.no_grad():
        outputs = model(inputs)
        _, predicted = outputs.max(1)
    synchronize_if_cuda(device)
    inference_time_ms = (time.perf_counter() - start_time) * 1000.0
        
    pred_label = predicted.item()
    print(f"✅ 单视频推理完成! 预测标签 ID: {pred_label}")

    results_txt_path, _ = resolve_output_paths(args.output, is_single_video=True)
    ensure_parent_dir(results_txt_path)

    with open(results_txt_path, "w", encoding="utf-8") as f:
        f.write(f"inference_time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"video_path: {args.video_path}\n")
        f.write(f"predicted_label_id: {pred_label}\n")
        f.write(f"inference_time_ms: {inference_time_ms:.4f}\n")

    print(f"✅ 推理结果已保存至: {results_txt_path}")


def infer_dataset(args, model, device, config):
    """对整个数据集进行推理"""
    if not os.path.exists(args.csv_path):
        print(f"❌ 找不到 CSV 文件: {args.csv_path}")
        return
        
    print(f"读取数据集: {args.csv_path}")
    
    # 创建 transform
    img_size = tuple(config.get("img_size", (100, 176)))
    val_transform = get_val_transform(
        img_size=img_size,
        normalize_mean=config.get("normalize_mean"),
        normalize_std=config.get("normalize_std")
    )
    
    csv_df = pd.read_csv(args.csv_path)
    has_label_column = "label_id" in csv_df.columns
    has_label_values = has_label_column and csv_df["label_id"].notna().any()

    test_dataset = JesterDataset(
        csv_file=args.csv_path,
        root_dir=args.root_dir,
        num_frames=DEFAULT_NUM_FRAMES,
        transform=val_transform,
        is_test=False,
        sampling_mode=SAMPLING_UNIFORM,
    )
    
    # 使用 config 中的 batch_size（如果未指定）
    batch_size = args.batch_size if args.batch_size is not None else config["batch_size"]
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=config["num_workers"], 
        pin_memory=config["pin_memory"]
    )

    predictions = []
    video_ids = []
    per_clip_inference_time_ms = []

    top1_correct = 0
    top5_correct = 0
    metric_total = 0
    
    print(f"开始对 {len(test_dataset)} 个样本进行推理...")
    with torch.no_grad():
        for i, (inputs, labels, v_ids) in enumerate(test_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            synchronize_if_cuda(device)
            start_time = time.perf_counter()
            outputs = model(inputs)
            synchronize_if_cuda(device)
            batch_time_ms = (time.perf_counter() - start_time) * 1000.0
            batch_size_current = inputs.size(0)
            avg_clip_time_ms = batch_time_ms / max(batch_size_current, 1)

            _, predicted = outputs.max(1)

            valid_mask = labels != -1
            if valid_mask.any():
                valid_labels = labels[valid_mask]
                valid_outputs = outputs[valid_mask]
                valid_predicted = predicted[valid_mask]

                top1_correct += valid_predicted.eq(valid_labels).sum().item()

                topk = min(5, valid_outputs.size(1))
                topk_indices = valid_outputs.topk(k=topk, dim=1).indices
                top5_correct += topk_indices.eq(valid_labels.unsqueeze(1)).any(dim=1).sum().item()
                metric_total += valid_labels.size(0)

            predictions.extend(predicted.cpu().numpy())
            video_ids.extend(v_ids)
            per_clip_inference_time_ms.extend([avg_clip_time_ms] * batch_size_current)
            
            if (i + 1) % 10 == 0 or (i + 1) == len(test_loader):
                print(f"推理进度: [{i+1}/{len(test_loader)}]")

    print("-" * 30)
    if metric_total > 0:
        top1_acc = 100.0 * top1_correct / metric_total
        top5_acc = 100.0 * top5_correct / metric_total
        print("🎯 数据集包含真实标签，评估结果如下:")
        print(f"   Top-1 Accuracy: {top1_acc:.2f}%")
        print(f"   Top-5 Accuracy: {top5_acc:.2f}%")
    else:
        print("⚠️ 推理的数据集没有提供真实标签，已跳过 Top-1 / Top-5 计算。")

    results_df = pd.DataFrame({
        "video_id": video_ids,
        "predicted_label_id": predictions,
        "inference_time_ms": per_clip_inference_time_ms,
    })

    results_txt_path, results_csv_path = resolve_output_paths(args.output, is_single_video=False)
    ensure_parent_dir(results_csv_path)
    results_df.to_csv(results_csv_path, index=False)

    avg_inference_time_ms = float(np.mean(per_clip_inference_time_ms)) if per_clip_inference_time_ms else 0.0
    summary_lines = [
        f"inference_time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"dataset_csv: {args.csv_path}",
        f"total_clips: {len(test_dataset)}",
        f"average_inference_time_ms: {avg_inference_time_ms:.4f}",
        f"has_ground_truth: {str(has_label_values and metric_total > 0).lower()}",
    ]
    if metric_total > 0:
        top1_acc = 100.0 * top1_correct / metric_total
        top5_acc = 100.0 * top5_correct / metric_total
        summary_lines.append(f"top1_accuracy: {top1_acc:.4f}%")
        summary_lines.append(f"top5_accuracy: {top5_acc:.4f}%")
    elif has_label_column and not has_label_values:
        summary_lines.append("note: CSV 包含 label_id 列，但标签为空，无法计算准确率")

    ensure_parent_dir(results_txt_path)
    with open(results_txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines) + "\n")

    print(f"✅ 推理完成！预测结果 CSV 已保存至: {results_csv_path}")
    print(f"✅ 统计结果已保存至: {results_txt_path}")


def run_inference():
    """主推理流程"""
    args = parse_args()
    
    # 加载配置
    config = get_config()
    config["num_frames"] = DEFAULT_NUM_FRAMES
    device = config["device"]
    
    # 打印平台信息
    platform_name = get_platform_name()
    print(f"Platform: {platform_name}")
    print(f"正在加载模型并准备推理...")
    print(f"使用设备: {device}")

    if not os.path.exists(args.model_weight):
        print(f"❌ 找不到模型权重文件: {args.model_weight}")
        return

    # 加载模型
    model = load_model(args.model_type, config, device, args.model_weight)
    model.eval()

    # 执行推理
    if args.video_path:
        infer_single_video(args, model, device, config)
    elif args.csv_path:
        infer_dataset(args, model, device, config)
    else:
        print("❌ 请提供 --csv_path (用于数据集) 或 --video_path (用于单视频)")


if __name__ == "__main__":
    run_inference()

# 执行单视频推断:
# python inference.py --video_path "dataset/Test/100010" --model_type "resnet" --model_weight "checkpoint/lightweight_gesture_model.pth" --output "checkpoint/single_run"

# 执行整个测试集推断（自动计算 Top-1 / Top-5）:
# python inference.py --csv_path dataset/Test.csv --root_dir dataset/Test --model_type ultralight_parallel_me_gru --model_weight checkpoint/final_2/model_ultralight_parallel_me_gru.pth --output results/final_2/test
# python inference.py --csv_path dataset/Train.csv --root_dir dataset/Train --model_type ultralight_parallel_me_gru --model_weight checkpoint/final_2/model_ultralight_parallel_me_gru.pth --output results/final_2/train
# python inference.py --csv_path dataset/Validation.csv --root_dir dataset/Validation --model_type ultralight_parallel_me_gru --model_weight checkpoint/final_2/model_ultralight_parallel_me_gru.pth --output results/final_2/val
