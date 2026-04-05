import argparse
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader

from config_loader import get_config, get_platform_name
from dataset import (
    JesterDataset,
    SAMPLING_UNIFORM,
    get_val_transform,
    sample_frame_indices,
)
from models import MODEL_TYPE, build_model, load_model_weights


def parse_args():
    parser = argparse.ArgumentParser(
        description="Standalone inference for UltraLightParallelMEGRUModel"
    )
    parser.add_argument("--config", type=str, default=None, help="Path to split config json")
    parser.add_argument("--model_weight", type=str, required=True, help="Model weight path")
    parser.add_argument("--data_dir", type=str, default=None, help="Dataset root path")
    parser.add_argument("--csv_path", type=str, default=None, help="CSV path for dataset inference")
    parser.add_argument("--root_dir", type=str, default=None, help="Frames root dir for dataset inference")
    parser.add_argument("--video_path", type=str, default="", help="Single video folder path")
    parser.add_argument("--output", type=str, default="../results/split_ultralight_parallel_me_gru")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_frames", type=int, default=None)
    parser.add_argument("--hidden_dim", type=int, default=None)
    return parser.parse_args()


def _abs_if_needed(path_value):
    if path_value is None:
        return None
    return path_value if os.path.isabs(path_value) else os.path.abspath(path_value)


def apply_overrides(config, args):
    if args.data_dir is not None:
        config["data_dir"] = _abs_if_needed(args.data_dir)
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.num_frames is not None:
        config["num_frames"] = args.num_frames
    if args.hidden_dim is not None:
        config["hidden_dim"] = args.hidden_dim


def ensure_parent_dir(file_path):
    parent_dir = os.path.dirname(file_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)


def synchronize_if_cuda(device):
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.synchronize()


def resolve_output_paths(output_arg, is_single_video):
    output_arg = output_arg.strip() if output_arg else "../results/split_ultralight_parallel_me_gru"
    output_lower = output_arg.lower()

    if is_single_video:
        if output_lower.endswith(".txt"):
            return output_arg, None
        return os.path.join(output_arg, "results.txt"), None

    if output_lower.endswith(".csv"):
        csv_path = output_arg
        txt_path = os.path.join(os.path.dirname(output_arg) or ".", "results.txt")
    elif output_lower.endswith(".txt"):
        txt_path = output_arg
        csv_path = os.path.join(os.path.dirname(output_arg) or ".", "predictions.csv")
    else:
        txt_path = os.path.join(output_arg, "results.txt")
        csv_path = os.path.join(output_arg, "predictions.csv")

    return txt_path, csv_path


def infer_single_video(args, model, device, config):
    video_path = _abs_if_needed(args.video_path)
    if not video_path or not os.path.exists(video_path):
        raise FileNotFoundError(f"Single video path not found: {args.video_path}")

    print(f"Running single-video inference: {video_path}")

    frame_files = sorted([name for name in os.listdir(video_path) if name.endswith(".jpg")])
    total_frames = len(frame_files)

    indices = sample_frame_indices(
        total_frames=total_frames,
        num_frames=config["num_frames"],
        sampling_mode=SAMPLING_UNIFORM,
    )

    img_size = tuple(config.get("img_size", (100, 176)))
    val_transform = get_val_transform(
        img_size=img_size,
        normalize_mean=config.get("normalize_mean"),
        normalize_std=config.get("normalize_std"),
    )

    frames = []
    for idx in indices:
        frame_name = f"{idx:05d}.jpg"
        img_path = os.path.join(video_path, frame_name)
        try:
            img = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            img = Image.new("RGB", (img_size[1], img_size[0]), color=0)
        frames.append(val_transform(img))

    inputs = torch.stack(frames).unsqueeze(0).to(device)

    synchronize_if_cuda(device)
    start_time = time.perf_counter()
    with torch.no_grad():
        outputs = model(inputs)
        probs = torch.softmax(outputs, dim=1)
        predicted = outputs.argmax(dim=1)
        topk = min(5, outputs.size(1))
        topk_scores, topk_indices = probs.topk(k=topk, dim=1)
    synchronize_if_cuda(device)
    inference_time_ms = (time.perf_counter() - start_time) * 1000.0

    pred_label = int(predicted.item())
    topk_pairs = [
        (int(label), float(score))
        for label, score in zip(topk_indices[0].tolist(), topk_scores[0].tolist())
    ]

    print(f"Predicted label id: {pred_label}")

    results_txt_path, _ = resolve_output_paths(args.output, is_single_video=True)
    results_txt_path = _abs_if_needed(results_txt_path)
    ensure_parent_dir(results_txt_path)

    with open(results_txt_path, "w", encoding="utf-8") as f:
        f.write(f"inference_time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"model_type: {MODEL_TYPE}\n")
        f.write(f"video_path: {video_path}\n")
        f.write(f"predicted_label_id: {pred_label}\n")
        f.write(f"inference_time_ms: {inference_time_ms:.4f}\n")
        f.write("top5:\n")
        for label, score in topk_pairs:
            f.write(f"  label_id={label}, score={score:.6f}\n")

    print(f"Single-video result saved: {results_txt_path}")


def infer_dataset(args, model, device, config):
    if args.csv_path is None:
        csv_path = os.path.join(config["data_dir"], "Test.csv")
    else:
        csv_path = _abs_if_needed(args.csv_path)

    if args.root_dir is None:
        root_dir = os.path.join(config["data_dir"], "Test")
    else:
        root_dir = _abs_if_needed(args.root_dir)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV path not found: {csv_path}")
    if not os.path.exists(root_dir):
        raise FileNotFoundError(f"Root dir not found: {root_dir}")

    print(f"Running dataset inference with CSV: {csv_path}")

    img_size = tuple(config.get("img_size", (100, 176)))
    val_transform = get_val_transform(
        img_size=img_size,
        normalize_mean=config.get("normalize_mean"),
        normalize_std=config.get("normalize_std"),
    )

    csv_df = pd.read_csv(csv_path)
    has_label_column = "label_id" in csv_df.columns
    has_label_values = has_label_column and csv_df["label_id"].notna().any()

    dataset = JesterDataset(
        csv_file=csv_path,
        root_dir=root_dir,
        num_frames=config["num_frames"],
        transform=val_transform,
        is_test=False,
        sampling_mode=SAMPLING_UNIFORM,
        img_size=img_size,
    )

    loader_kwargs = {
        "batch_size": config["batch_size"],
        "num_workers": config.get("num_workers", 0),
        "pin_memory": config.get("pin_memory", True),
        "shuffle": False,
    }
    if loader_kwargs["num_workers"] > 0:
        loader_kwargs["prefetch_factor"] = config.get("prefetch_factor", 2)

    loader = DataLoader(dataset, **loader_kwargs)

    predictions = []
    video_ids = []
    per_clip_time_ms = []

    top1_correct = 0
    top5_correct = 0
    metric_total = 0

    with torch.no_grad():
        for step, (inputs, labels, vid_ids) in enumerate(loader, start=1):
            inputs = inputs.to(device)
            labels = labels.to(device)

            synchronize_if_cuda(device)
            start_time = time.perf_counter()
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            synchronize_if_cuda(device)

            batch_time_ms = (time.perf_counter() - start_time) * 1000.0
            avg_clip_time_ms = batch_time_ms / max(inputs.size(0), 1)

            predicted = outputs.argmax(dim=1)

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

            predictions.extend(predicted.cpu().numpy().tolist())
            video_ids.extend(list(vid_ids))
            per_clip_time_ms.extend([avg_clip_time_ms] * inputs.size(0))

            if step % 10 == 0 or step == len(loader):
                print(f"Inference progress: [{step}/{len(loader)}]")

    results_df = pd.DataFrame(
        {
            "video_id": video_ids,
            "predicted_label_id": predictions,
            "inference_time_ms": per_clip_time_ms,
        }
    )

    results_txt_path, results_csv_path = resolve_output_paths(args.output, is_single_video=False)
    results_txt_path = _abs_if_needed(results_txt_path)
    results_csv_path = _abs_if_needed(results_csv_path)

    ensure_parent_dir(results_csv_path)
    results_df.to_csv(results_csv_path, index=False)

    avg_inference_time_ms = float(np.mean(per_clip_time_ms)) if per_clip_time_ms else 0.0
    summary_lines = [
        f"inference_time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"model_type: {MODEL_TYPE}",
        f"dataset_csv: {csv_path}",
        f"dataset_root: {root_dir}",
        f"total_clips: {len(dataset)}",
        f"average_inference_time_ms: {avg_inference_time_ms:.4f}",
        f"has_ground_truth: {str(has_label_values and metric_total > 0).lower()}",
    ]

    if metric_total > 0:
        top1_acc = 100.0 * top1_correct / metric_total
        top5_acc = 100.0 * top5_correct / metric_total
        summary_lines.append(f"top1_accuracy: {top1_acc:.4f}%")
        summary_lines.append(f"top5_accuracy: {top5_acc:.4f}%")
    elif has_label_column and not has_label_values:
        summary_lines.append("note: label_id exists but is empty, skipped accuracy")

    ensure_parent_dir(results_txt_path)
    with open(results_txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines) + "\n")

    print(f"Prediction CSV saved: {results_csv_path}")
    print(f"Summary saved: {results_txt_path}")


def run_inference():
    args = parse_args()
    config = get_config(args.config)
    apply_overrides(config, args)

    print(f"Platform: {get_platform_name()}")
    print(f"Device: {config['device']}")

    model_weight_path = _abs_if_needed(args.model_weight)
    if not os.path.exists(model_weight_path):
        raise FileNotFoundError(f"Model weight not found: {model_weight_path}")

    model = build_model(config, device=config["device"])
    model = load_model_weights(model, model_weight_path, config["device"], strict=True, eval_mode=True)

    if args.video_path:
        infer_single_video(args, model, config["device"], config)
    else:
        infer_dataset(args, model, config["device"], config)


if __name__ == "__main__":
    run_inference()
