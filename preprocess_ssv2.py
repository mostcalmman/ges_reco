"""
SSv2 数据集预处理脚本
将 .webm 视频提取为 Jester 格式的 JPG 帧序列 + CSV 标签文件，
使现有训练管线无需任何改动即可用于 SSv2。

用法:
    python preprocess_ssv2.py --video_dir dataset/SSv2/videos --workers 8
"""

import os
import json
import csv
import argparse
import shutil
import bisect
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2


DONE_MARKER = "_EXTRACT_DONE"


def parse_args():
    parser = argparse.ArgumentParser(description="SSv2 预处理：webm → JPG 帧 + Jester 格式 CSV")
    parser.add_argument("--video_dir", type=str, required=True,
                        help="存放 .webm 视频文件的目录")
    parser.add_argument("--output_dir", type=str, default="dataset/SSv2",
                        help="输出目录（默认 dataset/SSv2）")
    parser.add_argument("--labels_dir", type=str, default=None,
                        help="标签目录（默认 {output_dir}/labels）")
    parser.add_argument("--workers", type=int, default=16,
                        help="并行提取的进程数")
    parser.add_argument("--jpg_quality", type=int, default=100,
                        help="JPEG 保存质量 (1-100)")
    parser.add_argument("--splits", type=str, nargs="+",
                        default=["Train", "Validation", "Test"],
                        help="要处理的数据集划分 (默认全部)")
    parser.add_argument("--uniform_16", action=argparse.BooleanOptionalAction, default=True,
                        help="按整个视频时间均匀抽16帧（默认）；--no-uniform_16 提取全部帧")
    return parser.parse_args()


# ============================================================
# 标签解析
# ============================================================

def load_template_to_id(labels_path):
    """加载 labels.json，返回 template_text → label_id 映射"""
    with open(labels_path, "r", encoding="utf-8") as f:
        labels = json.load(f)
    return {k: int(v) for k, v in labels.items()}


def normalize_template(template):
    """[something] → something，用于匹配 labels.json 的 key"""
    return template.replace("[", "").replace("]", "")


def load_split_json(json_path, template_to_id):
    """解析 split json，返回 [(video_id, label_id), ...]"""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = []
    unmapped = 0
    for item in data:
        vid = str(item["id"])
        if "template" in item:
            normalized = normalize_template(item["template"])
            label_id = template_to_id.get(normalized, -1)
            if label_id == -1:
                unmapped += 1
        else:
            label_id = -1
        records.append((vid, label_id))

    if unmapped:
        print(f"  ⚠️ {unmapped} 条记录未能映射到类别")
    return records


def load_test_answers(csv_path, template_to_id):
    """解析 test-answers.csv (分号分隔)，返回 vid → label_id 映射"""
    answer_map = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(";")
            if len(parts) == 2:
                vid, label_text = parts
                answer_map[vid] = template_to_id.get(label_text, -1)
    return answer_map


# ============================================================
# 帧提取
# ============================================================

def extract_frames(args_tuple):
    """Uniformly sample 16 timestamps across the video, or extract all frames."""
    video_path, output_folder, jpg_quality, uniform_16 = args_tuple
    marker_path = os.path.join(output_folder, DONE_MARKER)
    mode = 'uniform_time_16' if uniform_16 else 'all_frames'
    if os.path.exists(marker_path):
        try:
            with open(marker_path, encoding='utf-8') as f:
                marker = json.load(f)
            count = marker.get('count', 0) if isinstance(marker, dict) else 0
            if (isinstance(marker, dict) and marker.get('mode') == mode and count > 0
                    and all(os.path.isfile(os.path.join(output_folder, f"{i:05d}.jpg"))
                            for i in range(1, count + 1))):
                return count, True
        except (ValueError, OSError):
            pass

    # Decode first to obtain actual timestamps, avoiding unreliable frame-count metadata.
    cap = cv2.VideoCapture(video_path)
    timestamps = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC))
    cap.release()
    if not timestamps:
        return 0, False
    selected = None
    if uniform_16:
        # Some backends expose no timestamps: uniform indices are the CFR fallback.
        if not all(timestamps[i] > timestamps[i - 1] for i in range(1, len(timestamps))):
            timestamps = list(range(len(timestamps)))
        selected = []
        for i in range(16):
            target = timestamps[0] + (timestamps[-1] - timestamps[0]) * i / 15
            j = min(bisect.bisect_left(timestamps, target), len(timestamps) - 1)
            if j > 0 and target - timestamps[j - 1] < timestamps[j] - target:
                j -= 1
            selected.append(j)

    # Old markers/modes must not leave stale frames in the output.
    if os.path.isdir(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    source_idx = 0
    written = 0
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            copies = 1 if selected is None else selected.count(source_idx)
            for _ in range(copies):
                written += 1
                path = os.path.join(output_folder, f"{written:05d}.jpg")
                if not cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, jpg_quality]):
                    raise OSError(f"Cannot write frame: {path}")
            source_idx += 1
            if selected is not None and written == 16:
                break
    finally:
        cap.release()
    expected = 16 if uniform_16 else len(timestamps)
    if written != expected:
        shutil.rmtree(output_folder)
        return 0, False
    with open(marker_path, 'w', encoding='utf-8') as f:
        json.dump({'mode': mode, 'count': written}, f)
    return written, False


# ============================================================
# Split 处理 & CSV 生成
# ============================================================

def process_split(split_name, records, video_dir, output_dir, workers, jpg_quality, uniform_16=True):
    """处理一个 split，返回 {vid: frame_count}"""
    split_dir = os.path.join(output_dir, split_name)
    os.makedirs(split_dir, exist_ok=True)

    tasks = []
    for vid, _ in records:
        video_path = os.path.join(video_dir, f"{vid}.webm")
        output_folder = os.path.join(split_dir, vid)
        tasks.append((video_path, output_folder, jpg_quality, uniform_16))

    print(f"\n{'=' * 60}")
    print(f"处理 {split_name}: {len(tasks)} 个视频 → {split_dir}")
    print(f"{'=' * 60}")

    results = {}
    skipped = 0
    failed = 0

    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_to_vid = {}
        for (vid, _), task in zip(records, tasks):
            future = executor.submit(extract_frames, task)
            future_to_vid[future] = vid

        done_count = 0
        total = len(future_to_vid)
        for future in as_completed(future_to_vid):
            vid = future_to_vid[future]
            done_count += 1
            try:
                frame_count, was_skipped = future.result()
                if frame_count > 0:
                    results[vid] = frame_count
                    if was_skipped:
                        skipped += 1
                else:
                    failed += 1
            except Exception as e:
                print(f"  ✗ {vid}: {e}")
                failed += 1

            if done_count % 2000 == 0 or done_count == total:
                print(f"  进度: {done_count}/{total}  成功: {len(results)}  跳过: {skipped}  失败: {failed}")

    return results


def write_csv(csv_path, records, frame_counts):
    """写入 Jester 兼容的 CSV (video_id, frames, label_id)"""
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["video_id", "frames", "label_id"])
        written = 0
        for vid, label_id in records:
            if vid in frame_counts:
                writer.writerow([vid, frame_counts[vid], label_id])
                written += 1
    print(f"  ✓ CSV 已保存: {csv_path} ({written} 条记录)")


# ============================================================
# 主流程
# ============================================================

SPLIT_JSON_MAP = {
    "Train": "train.json",
    "Validation": "validation.json",
    "Test": "test.json",
}


def main():
    args = parse_args()
    labels_dir = args.labels_dir or os.path.join(args.output_dir, "labels")

    # 1. 加载类别映射
    labels_path = os.path.join(labels_dir, "labels.json")
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"labels.json 不存在: {labels_path}")
    template_to_id = load_template_to_id(labels_path)
    print(f"已加载 {len(template_to_id)} 个类别")

    # 2. 可选加载 test 答案
    test_answers_path = os.path.join(labels_dir, "test-answers.csv")
    test_answer_map = None
    if os.path.exists(test_answers_path):
        test_answer_map = load_test_answers(test_answers_path, template_to_id)
        print(f"已加载 {len(test_answer_map)} 条 test 答案")

    # 3. 逐 split 处理
    for split_name in args.splits:
        json_file = SPLIT_JSON_MAP.get(split_name)
        if json_file is None:
            print(f"⚠️ 未知 split: {split_name}，跳过")
            continue

        json_path = os.path.join(labels_dir, json_file)
        if not os.path.exists(json_path):
            print(f"⚠️ 跳过 {split_name}: {json_path} 不存在")
            continue

        records = load_split_json(json_path, template_to_id)

        # 为 Test 补充答案标签
        if split_name == "Test" and test_answer_map is not None:
            records = [
                (vid, test_answer_map.get(vid, lid))
                for vid, lid in records
            ]

        frame_counts = process_split(
            split_name, records, args.video_dir, args.output_dir,
            args.workers, args.jpg_quality, args.uniform_16,
        )

        csv_path = os.path.join(args.output_dir, f"{split_name}.csv")
        write_csv(csv_path, records, frame_counts)

    print(f"\n{'=' * 60}")
    print("预处理完成！")
    print(f"输出目录: {args.output_dir}")
    print(f"切换训练: config.json 中 \"dataset\" 改为 \"ssv2\" 即可")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
