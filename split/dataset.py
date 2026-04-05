import os

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset


DEFAULT_NUM_FRAMES = 16
SAMPLING_RANDOM = "random"
SAMPLING_UNIFORM = "uniform"


def sample_frame_indices(total_frames, num_frames=DEFAULT_NUM_FRAMES, sampling_mode=SAMPLING_UNIFORM):
    if total_frames <= 0:
        return np.ones(num_frames, dtype=int)

    if total_frames <= num_frames:
        indices = np.linspace(1, total_frames, total_frames, dtype=int)
        padding = np.ones(num_frames - total_frames, dtype=int) * total_frames
        return np.concatenate((indices, padding))

    if sampling_mode == SAMPLING_RANDOM:
        boundaries = np.linspace(0, total_frames, num_frames + 1, dtype=int)
        sampled = []
        for i in range(num_frames):
            start = boundaries[i]
            end = boundaries[i + 1]
            if end <= start:
                chosen = start
            else:
                chosen = np.random.randint(start, end)
            sampled.append(chosen + 1)
        return np.array(sampled, dtype=int)

    if sampling_mode == SAMPLING_UNIFORM:
        return np.linspace(1, total_frames, num_frames, dtype=int)

    raise ValueError(f"Unsupported sampling mode: {sampling_mode}")


class JesterDataset(Dataset):
    def __init__(
        self,
        csv_file,
        root_dir,
        num_frames=DEFAULT_NUM_FRAMES,
        transform=None,
        is_test=False,
        sampling_mode=None,
        img_size=(100, 176),
    ):
        self.data_info = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.num_frames = num_frames
        self.transform = transform
        self.is_test = is_test
        self.img_size = img_size

        if sampling_mode is None:
            self.sampling_mode = SAMPLING_UNIFORM if is_test else SAMPLING_RANDOM
        else:
            self.sampling_mode = sampling_mode

    def __len__(self):
        return len(self.data_info)

    def _sample_indices(self, total_frames):
        return sample_frame_indices(
            total_frames=total_frames,
            num_frames=self.num_frames,
            sampling_mode=self.sampling_mode,
        )

    def __getitem__(self, idx):
        row = self.data_info.iloc[idx]
        video_id = str(row.get("video_id", row.get("id")))
        total_frames = int(row["frames"])

        if pd.isna(row.get("label_id")):
            label = -1
        else:
            label = int(row["label_id"])

        video_path = os.path.join(self.root_dir, video_id)
        frame_indices = self._sample_indices(total_frames)

        frames = []
        for i in frame_indices:
            frame_name = f"{i:05d}.jpg"
            img_path = os.path.join(video_path, frame_name)
            try:
                img = Image.open(img_path).convert("RGB")
            except FileNotFoundError:
                img = Image.new("RGB", (self.img_size[1], self.img_size[0]), color=0)

            if self.transform:
                img = self.transform(img)
            frames.append(img)

        frames_tensor = torch.stack(frames)
        return frames_tensor, label, video_id


def get_train_transform(img_size=(100, 176), normalize_mean=None, normalize_std=None):
    if normalize_mean is None:
        normalize_mean = [0.485, 0.456, 0.406]
    if normalize_std is None:
        normalize_std = [0.229, 0.224, 0.225]

    h = int(img_size[0])
    return transforms.Compose(
        [
            transforms.Resize(h),
            transforms.CenterCrop(h),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=normalize_mean, std=normalize_std),
        ]
    )


def get_val_transform(img_size=(100, 176), normalize_mean=None, normalize_std=None):
    if normalize_mean is None:
        normalize_mean = [0.485, 0.456, 0.406]
    if normalize_std is None:
        normalize_std = [0.229, 0.224, 0.225]

    h = int(img_size[0])
    return transforms.Compose(
        [
            transforms.Resize(h),
            transforms.CenterCrop(h),
            transforms.ToTensor(),
            transforms.Normalize(mean=normalize_mean, std=normalize_std),
        ]
    )
