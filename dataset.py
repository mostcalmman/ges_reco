import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from utils import get_config


# --------------------------
# 数据集加载器 (Dataset)
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
        if total_frames <= 0:
            return np.ones(self.num_frames, dtype=int)

        if total_frames <= self.num_frames:
            indices = np.linspace(1, total_frames, total_frames, dtype=int)
            padding = np.ones(self.num_frames - total_frames, dtype=int) * total_frames
            indices = np.concatenate((indices, padding))
            return indices

        boundaries = np.linspace(0, total_frames, self.num_frames + 1, dtype=int)
        sampled = []
        for i in range(self.num_frames):
            start = boundaries[i]
            end = boundaries[i + 1]
            if end <= start:
                chosen = start
            else:
                chosen = np.random.randint(start, end)
            sampled.append(chosen + 1)  # 转成 1-based 帧编号

        return np.array(sampled, dtype=int)

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
                # 使用元组来创建黑色备用图像 (height, width)
                config = get_config()
                img_size = config.get("img_size", (100, 176))
                img = Image.new('RGB', (img_size[1], img_size[0]), color=0)
                
            if self.transform:
                img = self.transform(img)
            frames.append(img)

        frames_tensor = torch.stack(frames) # (37, 3, H, W)
        return frames_tensor, label, video_id


def get_train_transform(img_size=(100, 176), normalize_mean=None, normalize_std=None):
    """
    获取训练集数据预处理变换
    
    Args:
        img_size: 目标图像尺寸 (height, width)
        normalize_mean: 归一化均值
        normalize_std: 归一化标准差
        
    Returns:
        transforms.Compose: 训练集变换组合
    """
    if normalize_mean is None:
        normalize_mean = [0.485, 0.456, 0.406]
    if normalize_std is None:
        normalize_std = [0.229, 0.224, 0.225]

    h = 100
    return transforms.Compose([
        transforms.Resize(h),
        transforms.CenterCrop(h),
        transforms.ColorJitter(brightness=0.2, contrast=0.2), 
        transforms.ToTensor(),
        transforms.Normalize(mean=normalize_mean, std=normalize_std)
    ])


def get_val_transform(img_size=(100, 176), normalize_mean=None, normalize_std=None):
    """
    获取验证/测试集数据预处理变换
    
    Args:
        img_size: 目标图像尺寸 (height, width)
        normalize_mean: 归一化均值
        normalize_std: 归一化标准差
        
    Returns:
        transforms.Compose: 验证集变换组合
    """
    if normalize_mean is None:
        normalize_mean = [0.485, 0.456, 0.406]
    if normalize_std is None:
        normalize_std = [0.229, 0.224, 0.225]
        
    h = 100
    return transforms.Compose([
        transforms.Resize(h),
        transforms.CenterCrop(h),
        transforms.ToTensor(),
        transforms.Normalize(mean=normalize_mean, std=normalize_std)
    ])


# # 为了保持向后兼容，保留默认的 transforms
# _default_config = get_config()
# train_transform = get_train_transform(
#     img_size=_default_config.get("img_size", (100, 176)),
#     normalize_mean=_default_config.get("normalize_mean"),
#     normalize_std=_default_config.get("normalize_std")
# )
# val_transform = get_val_transform(
#     img_size=_default_config.get("img_size", (100, 176)),
#     normalize_mean=_default_config.get("normalize_mean"),
#     normalize_std=_default_config.get("normalize_std")
# )
