import argparse
import os
from datetime import datetime

SPLITS = ("train", "val", "test")


def parse_args():
    parser = argparse.ArgumentParser(description="汇总 train/val/test 推理结果")
    parser.add_argument(
        "--dir",
        type=str,
        required=True,
        help="包含 train/val/test 子目录的结果根目录",
    )
    return parser.parse_args()


def parse_percent(value):
    value = value.strip()
    if value.endswith("%"):
        value = value[:-1].strip()
    return float(value)


def parse_bool(value):
    return value.strip().lower() == "true"


def parse_results_txt(file_path):
    result = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or ":" not in line:
                continue
            key, value = line.split(":", 1)
            result[key.strip()] = value.strip()

    required = ["total_clips", "average_inference_time_ms"]
    for key in required:
        if key not in result:
            raise ValueError(f"文件缺少字段 {key}: {file_path}")

    parsed = {
        "dataset_csv": result.get("dataset_csv", ""),
        "total_clips": int(result["total_clips"]),
        "average_inference_time_ms": float(result["average_inference_time_ms"]),
        "has_ground_truth": parse_bool(result.get("has_ground_truth", "false")),
    }

    if "top1_accuracy" in result:
        parsed["top1_accuracy"] = parse_percent(result["top1_accuracy"])
    if "top5_accuracy" in result:
        parsed["top5_accuracy"] = parse_percent(result["top5_accuracy"])

    return parsed


def weighted_average(pairs):
    total_weight = sum(weight for _, weight in pairs)
    if total_weight == 0:
        return None
    weighted_sum = sum(value * weight for value, weight in pairs)
    return weighted_sum / total_weight


def main():
    args = parse_args()
    base_dir = args.dir

    split_results = {}
    for split in SPLITS:
        file_path = os.path.join(base_dir, split, "results.txt")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"缺少结果文件: {file_path}")
        split_results[split] = parse_results_txt(file_path)

    total_samples = sum(item["total_clips"] for item in split_results.values())

    time_pairs = [
        (item["average_inference_time_ms"], item["total_clips"])
        for item in split_results.values()
    ]
    weighted_time = weighted_average(time_pairs)

    top1_pairs = []
    top5_pairs = []
    for item in split_results.values():
        if item["has_ground_truth"] and "top1_accuracy" in item and "top5_accuracy" in item:
            top1_pairs.append((item["top1_accuracy"], item["total_clips"]))
            top5_pairs.append((item["top5_accuracy"], item["total_clips"]))

    weighted_top1 = weighted_average(top1_pairs)
    weighted_top5 = weighted_average(top5_pairs)
    metric_samples = sum(weight for _, weight in top1_pairs)

    lines = [
        f"sum_time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"base_dir: {base_dir}",
        f"total_samples: {total_samples}",
        f"weighted_average_inference_time_ms: {weighted_time:.4f}" if weighted_time is not None else "weighted_average_inference_time_ms: N/A",
    ]

    if weighted_top1 is not None and weighted_top5 is not None:
        lines.append(f"metric_samples: {metric_samples}")
        lines.append(f"weighted_top1_accuracy: {weighted_top1:.4f}%")
        lines.append(f"weighted_top5_accuracy: {weighted_top5:.4f}%")
    else:
        lines.append("metric_samples: 0")
        lines.append("weighted_top1_accuracy: N/A")
        lines.append("weighted_top5_accuracy: N/A")

    for split in SPLITS:
        item = split_results[split]
        lines.append(
            f"{split}_samples: {item['total_clips']}, {split}_avg_inference_time_ms: {item['average_inference_time_ms']:.4f}"
        )

    sum_path = os.path.join(base_dir, "sum.txt")
    with open(sum_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"汇总完成: {sum_path}")


if __name__ == "__main__":
    main()
