import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

# --------------------------
# 1. 默认配置参数 (将被命令行参数覆盖)
# --------------------------
CONFIG = {
    "data_dir": "dataset",
    "checkpoint_dir": "checkpoint",
    "batch_size": 48,
    "num_workers": 6,
    "pin_memory": True,
    "prefetch_factor": 2,
    "num_frames": 37,
    "img_size": (100, 176),
    "num_classes": 27,
    "hidden_dim": 256,
    "num_epochs": 20,
    "learning_rate": 1e-3,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

# --------------------------
# 2. 数据集加载器 (Dataset)
# --------------------------
class JesterDataset(Dataset):
    def __init__(self, csv_file, root_dir, num_frames=37, transform=None, is_test=False):
        self.data_info = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.num_frames = num_frames
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.data_info)

    def _sample_indices(self, total_frames):
        # 处理视频长度和模型要求不一致的情况, 返回一个长度为 num_frames 的索引列表
        if total_frames <= self.num_frames:
            indices = np.linspace(1, total_frames, total_frames, dtype=int)
            padding = np.ones(self.num_frames - total_frames, dtype=int) * total_frames
            indices = np.concatenate((indices, padding))
        else:
            indices = np.linspace(1, total_frames, self.num_frames, dtype=int)
        return indices

    def __getitem__(self, idx):
        row = self.data_info.iloc[idx]
        # 兼容 Train.csv (video_id) 和 Test.csv (id)
        video_id = str(row.get('video_id', row.get('id'))) 
        total_frames = int(row['frames'])
        
        # 如果没有标签，填充 -1
        if pd.isna(row.get('label_id')):
            label = -1
        else:
            label = int(row['label_id'])

        video_path = os.path.join(self.root_dir, video_id)
        frame_indices = self._sample_indices(total_frames)

        frames = []
        for i in frame_indices:
            frame_name = f"{i:05d}.jpg"
            img_path = os.path.join(video_path, frame_name)
            try:
                img = Image.open(img_path).convert('RGB')
            except FileNotFoundError:
                # 使用元组来创建黑色备用图像
                img = Image.new('RGB', (CONFIG["img_size"][1], CONFIG["img_size"][0]), color=0)
                
            if self.transform:
                img = self.transform(img)
            frames.append(img)

        frames_tensor = torch.stack(frames) # (37, 3, H, W)
        return frames_tensor, label, video_id

# 数据预处理
train_transform = transforms.Compose([
    transforms.Resize(CONFIG["img_size"]),
    transforms.ColorJitter(brightness=0.2, contrast=0.2), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize(CONFIG["img_size"]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
