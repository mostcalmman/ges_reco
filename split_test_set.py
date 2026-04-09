import argparse
import os
import shutil

import pandas as pd
from tqdm import tqdm


def _sample_from_split(data_dir, split_name, sample_size, seed):
    csv_path = os.path.join(data_dir, f"{split_name}.csv")
    split_dir = os.path.join(data_dir, split_name)

    print(f"Reading {split_name} csv: {csv_path}")
    df = pd.read_csv(csv_path)

    if len(df) < sample_size:
        raise ValueError(
            f"{split_name} set only has {len(df)} samples, cannot extract {sample_size}."
        )

    sampled_df = df.sample(n=sample_size, random_state=seed).copy()
    sampled_df["source_split"] = split_name
    sampled_df["source_video_id"] = sampled_df["video_id"]

    return sampled_df, split_dir


def build_final_set(data_dir="dataset", sample_size=5000, seed=42):
    """从 Train/Test/Validation 各抽样 sample_size，复制到 Final，不改动原数据。"""
    final_split_name = "Final"
    final_dir = os.path.join(data_dir, final_split_name)
    final_csv_path = os.path.join(data_dir, f"{final_split_name}.csv")

    os.makedirs(final_dir, exist_ok=True)

    sampled_train, train_dir = _sample_from_split(data_dir, "Train", sample_size, seed)
    sampled_test, test_dir = _sample_from_split(data_dir, "Test", sample_size, seed)
    sampled_val, val_dir = _sample_from_split(data_dir, "Validation", sample_size, seed)

    final_df = pd.concat([sampled_train, sampled_test, sampled_val], ignore_index=True)

    print("\nSampling finished:")
    print(f"  - Train sampled: {len(sampled_train)}")
    print(f"  - Test sampled: {len(sampled_test)}")
    print(f"  - Validation sampled: {len(sampled_val)}")
    print(f"  - Final total: {len(final_df)}")

    copied_count = 0
    missing_count = 0

    records = final_df.to_dict("records")
    for row in tqdm(records, desc="Copying folders to Final"):
        split_name = str(row["source_split"])
        video_id = str(row["video_id"])

        if split_name == "Train":
            src_root = train_dir
        elif split_name == "Test":
            src_root = test_dir
        else:
            src_root = val_dir

        src_path = os.path.join(src_root, video_id)

        # 使用前缀避免不同 split 下同名 video_id 的潜在冲突
        final_video_id = f"{split_name}_{video_id}"
        dst_path = os.path.join(final_dir, final_video_id)

        if os.path.exists(src_path):
            if os.path.exists(dst_path):
                shutil.rmtree(dst_path)
            shutil.copytree(src_path, dst_path)
            copied_count += 1
            row["video_id"] = final_video_id
        else:
            missing_count += 1

    print(f"\nCopy completed! Success: {copied_count}, Missing: {missing_count}")

    # records 里 video_id 可能已被改为 final_video_id，重新构建 DataFrame
    out_df = pd.DataFrame(records)
    out_df.to_csv(final_csv_path, index=False)

    print("Final set generated successfully!")
    print(f"  - Final dir: {final_dir}")
    print(f"  - Final csv: {final_csv_path}")
    print("  - Original Train/Test/Validation data are NOT removed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="从 Train/Test/Validation 各抽取样本生成 1:1:1 的 Final 集（仅复制，不删除原数据）"
    )
    parser.add_argument("--data_dir", type=str, default="dataset", help="数据集根目录")
    parser.add_argument("--sample_size", type=int, default=5000, help="每个 split 抽取样本数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    build_final_set(args.data_dir, args.sample_size, args.seed)
