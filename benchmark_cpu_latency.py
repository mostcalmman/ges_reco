import argparse
import glob
import json
import os
import platform
import time

import numpy as np
import torch

from models import modelList
from utils.config_loader import get_config
from utils.model_loader import build_and_load_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark pure forward latency of a trained model on CPU."
    )
    parser.add_argument("--model_weight", type=str, default="", help="训练好的模型权重路径")
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="模型类型；传入后优先使用该值，否则回退到配置中的 model_type",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="配置文件路径；不传则优先在权重同目录下查找 config_*.json，再回退到根目录 config.json",
    )
    parser.add_argument("--warmup_iters", type=int, default=100, help="warm-up 轮数")
    parser.add_argument("--benchmark_iters", type=int, default=1000, help="正式统计轮数")
    parser.add_argument("--batch_size", type=int, default=1, help="benchmark batch size")
    parser.add_argument("--num_frames", type=int, default=None, help="覆盖配置中的帧数")
    parser.add_argument("--img_height", type=int, default=None, help="覆盖配置中的输入高度")
    parser.add_argument("--img_width", type=int, default=None, help="覆盖配置中的输入宽度")
    parser.add_argument("--hidden_dim", type=int, default=None, help="覆盖配置中的 hidden_dim")
    parser.add_argument("--num_classes", type=int, default=None, help="覆盖配置中的类别数")
    parser.add_argument("--num_threads", type=int, default=None, help="限制 CPU 线程数，便于稳定复现")
    parser.add_argument("--output", type=str, default="", help="可选，结果保存到文本文件")
    parser.add_argument("--list_models", action="store_true", help="列出 model_loader 支持的模型类型并退出")
    return parser.parse_args()


def resolve_abs_path(path_value):
    if not path_value:
        return path_value
    return path_value if os.path.isabs(path_value) else os.path.abspath(path_value)


def merge_platform_config(config_data):
    merged_config = config_data.get("common", {}).copy()

    system = platform.system()
    if system == "Linux":
        platform_config = config_data.get("linux", {})
    elif system == "Windows":
        platform_config = config_data.get("windows", {})
    else:
        platform_config = config_data.get("windows", {})

    merged_config.update(platform_config)

    dataset_name = merged_config.get("dataset")
    if dataset_name and "datasets" in config_data:
        dataset_config = config_data["datasets"].get(dataset_name, {})
        merged_config.update(dataset_config)

    return merged_config


def load_json_config(config_path):
    with open(config_path, "r", encoding="utf-8") as file:
        config_data = json.load(file)
    return merge_platform_config(config_data)


def detect_config_path(args):
    if args.config:
        config_path = resolve_abs_path(args.config)
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        return config_path

    model_weight_path = resolve_abs_path(args.model_weight)
    checkpoint_dir = os.path.dirname(model_weight_path)
    candidate_paths = sorted(glob.glob(os.path.join(checkpoint_dir, "config*.json")))
    if candidate_paths:
        return candidate_paths[-1]

    return None


def load_benchmark_config(args):
    config_path = detect_config_path(args)

    if config_path is None:
        config = get_config()
    else:
        config = load_json_config(config_path)

    config["device"] = "cpu"

    if args.num_frames is not None:
        config["num_frames"] = args.num_frames
    if args.hidden_dim is not None:
        config["hidden_dim"] = args.hidden_dim
    if args.num_classes is not None:
        config["num_classes"] = args.num_classes

    img_size = tuple(config.get("img_size", (100, 176)))
    if len(img_size) != 2:
        raise ValueError(f"img_size 配置非法: {img_size}")

    height, width = int(img_size[0]), int(img_size[1])
    if args.img_height is not None:
        height = args.img_height
    if args.img_width is not None:
        width = args.img_width
    config["img_size"] = (height, width)

    return config, config_path


def canonicalize_model_type(model_type):
    if not model_type:
        return None

    current_aliases = {name.lower(): name for name in modelList}
    return current_aliases.get(model_type.lower(), model_type)


def resolve_model_type(args, config):
    model_type = args.model_type or config.get("model_type")
    if not model_type:
        raise ValueError("无法确定 model_type，请通过 --model_type 显式传入")
    model_type = canonicalize_model_type(model_type)
    if model_type not in modelList:
        raise ValueError(
            "model_loader 不支持该 model_type: "
            + f"{model_type}；可用类型: {', '.join(modelList)}"
        )
    return model_type


def build_and_load_cpu_model(model_type, config, model_weight_path):
    model = build_and_load_model(
        model_type=model_type,
        config=config,
        device="cpu",
        model_weight_path=model_weight_path,
        pretrained=False,
    )
    return model


def validate_args(args):
    if args.list_models:
        return

    if not args.model_weight:
        raise ValueError("未传入 --model_weight")
    if args.warmup_iters < 0:
        raise ValueError("warmup_iters 必须 >= 0")
    if args.benchmark_iters <= 0:
        raise ValueError("benchmark_iters 必须 > 0")
    if args.batch_size <= 0:
        raise ValueError("batch_size 必须 > 0")
    if args.num_threads is not None and args.num_threads <= 0:
        raise ValueError("num_threads 必须 > 0")


def benchmark_forward_latency(model, inputs, warmup_iters, benchmark_iters):
    with torch.inference_mode():
        for _ in range(warmup_iters):
            _ = model(inputs)

    latency_ms = []
    with torch.inference_mode():
        for _ in range(benchmark_iters):
            start_time_ns = time.perf_counter_ns()
            _ = model(inputs)
            end_time_ns = time.perf_counter_ns()
            latency_ms.append((end_time_ns - start_time_ns) / 1_000_000.0)

    return np.asarray(latency_ms, dtype=np.float64)


def maybe_write_output(output_path, lines):
    if not output_path:
        return

    output_path = resolve_abs_path(output_path)
    parent_dir = os.path.dirname(output_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        file.write("\n".join(lines) + "\n")


def main():
    args = parse_args()

    if args.list_models:
        print("model_loader 支持的模型类型:")
        for name in modelList:
            print(name)
        return

    validate_args(args)

    model_weight_path = resolve_abs_path(args.model_weight)
    if not os.path.exists(model_weight_path):
        raise FileNotFoundError(f"模型权重不存在: {model_weight_path}")

    if args.num_threads is not None:
        torch.set_num_threads(args.num_threads)

    config, config_path = load_benchmark_config(args)
    model_type = resolve_model_type(args, config)
    config["model_type"] = model_type
    model = build_and_load_cpu_model(model_type, config, model_weight_path)
    model.eval()

    num_frames = int(config.get("num_frames", 16))
    height, width = config["img_size"]
    inputs = torch.randn(args.batch_size, num_frames, 3, height, width, device="cpu")

    latency_ms = benchmark_forward_latency(
        model=model,
        inputs=inputs,
        warmup_iters=args.warmup_iters,
        benchmark_iters=args.benchmark_iters,
    )

    mean_ms = float(latency_ms.mean())
    std_ms = float(latency_ms.std(ddof=0))
    p50_ms = float(np.percentile(latency_ms, 50))
    p95_ms = float(np.percentile(latency_ms, 95))

    summary_lines = [
        f"device: cpu",
        f"model_type: {model_type}",
        f"model_source: utils.model_loader",
        f"model_weight: {model_weight_path}",
        f"config: {config_path if config_path else 'config.json (root default)'}",
        f"input_shape: {(args.batch_size, num_frames, 3, height, width)}",
        f"warmup_iters: {args.warmup_iters}",
        f"benchmark_iters: {args.benchmark_iters}",
        f"num_threads: {torch.get_num_threads()}",
        "timing_scope: model(inputs) only; excludes data loading, preprocessing, and input preparation",
        f"latency(ms) = {mean_ms:.1f} ± {std_ms:.1f}",
        f"p50(ms) = {p50_ms:.1f}",
        f"p95(ms) = {p95_ms:.1f}",
    ]

    print("开始 CPU 前向 latency benchmark...")
    for line in summary_lines:
        print(line)

    maybe_write_output(args.output, summary_lines)


if __name__ == "__main__":
    main()