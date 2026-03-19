import os
import pandas as pd
import shutil
import argparse
from tqdm import tqdm

def split_test_set(data_dir="dataset", sample_size=5000):
    train_csv_path = os.path.join(data_dir, "Train.csv")
    test_csv_path = os.path.join(data_dir, "Test.csv")
    train_dir = os.path.join(data_dir, "Train")
    test_dir = os.path.join(data_dir, "Test")

    print(f"Reading train csv: {train_csv_path}")
    df = pd.read_csv(train_csv_path)
    
    if len(df) <= sample_size:
        print(f"Error: Train set only has {len(df)} samples, cannot extract {sample_size}!")
        return

    # 随机抽取样本
    print(f"Sampling {sample_size} from {len(df)} records for the new test set...")
    test_df = df.sample(n=sample_size, random_state=42)
    # 从原训练集中剔除抽取的样本
    train_df = df.drop(test_df.index)

    print(f"Sampling finished: ")
    print(f"   - Original Train count: {len(df)}")
    print(f"   - New Train count: {len(train_df)}")
    print(f"   - New Test count: {len(test_df)}")

    # 创建 Test 目录
    os.makedirs(test_dir, exist_ok=True)

    print(f"Moving {sample_size} folders from Train to Test...")
    moved_count = 0
    missing_count = 0
    
    # 转换为字典列表以提高遍历速度
    test_records = test_df.to_dict('records')
    
    for row in tqdm(test_records, desc="Moving folders"):
        video_id = str(row['video_id'])
        src_path = os.path.join(train_dir, video_id)
        dst_path = os.path.join(test_dir, video_id)
        
        if os.path.exists(src_path):
            shutil.move(src_path, dst_path)
            moved_count += 1
        else:
            missing_count += 1

    print(f"\nMove completed! Success: {moved_count}, Missing: {missing_count}")

    print("Saving updated CSV files...")
    # 保存新的 Test.csv 和更新后的 Train.csv
    test_df.to_csv(test_csv_path, index=False)
    train_df.to_csv(train_csv_path, index=False)
    print(f"CSV files saved!\n- New Test: {test_csv_path}\n- New Train: {train_csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从训练集中随机抽取生成测试集")
    parser.add_argument("--data_dir", type=str, default="dataset", help="数据集根目录")
    parser.add_argument("--sample_size", type=int, default=5000, help="抽取的测试集样本数")
    args = parser.parse_args()
    
    split_test_set(args.data_dir, args.sample_size)
