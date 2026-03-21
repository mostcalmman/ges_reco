# AGENTS.md

Guidelines for AI agents working in this gesture recognition repository.

## Project Overview

PyTorch-based video gesture recognition system using ResNet18 variants (with optional GRU) on the Jester dataset.

## Commands

### Training
```bash
# Train ResNet model (default)
python train.py --model_type resnet --epochs 20 --batch_size 48

# Train ResNet+GRU model
python train.py --model_type resnet_gru --epochs 20 --batch_size 48
```

### Inference
```bash
# Single video inference
python inference.py --video_path "dataset/Test/100010" --model_type resnet --model_weight "checkpoint/model_resnet.pth"

# Dataset inference
python inference.py --csv_path "dataset/Test.csv" --root_dir "dataset/Test" --model_type resnet --model_weight "checkpoint/model_resnet.pth"
```

### Utilities
```bash
# Calculate model parameters
python para_cal.py --model_type resnet

# Split dataset (create test set from train)
python split_test_set.py --data_dir dataset --sample_size 5000
```

## Code Style Guidelines

### Imports
- Order: standard library → third-party → local modules
- Example:
```python
import os
import argparse
import pandas as pd
import torch
from dataset import CONFIG, JesterDataset
```

### Configuration
- Global config lives in `dataset.py` as `CONFIG` dict
- Override via command-line arguments using `argparse`
- Key configs: `data_dir`, `checkpoint_dir`, `batch_size`, `num_frames`, `device`

### Naming Conventions
- Classes: `PascalCase` (e.g., `ResNetVideoModel`, `JesterDataset`)
- Functions: `snake_case` (e.g., `parse_args`, `train_model`)
- Constants: `UPPER_CASE` in CONFIG dict
- Private methods: `_leading_underscore` (e.g., `_sample_indices`)

### Type Annotations
- Not currently used; optional for new code
- If adding, use Python 3.9+ syntax: `list[str]`, `dict[str, int]`

### Comments
- Chinese comments acceptable (项目现有中文注释)
- Use `# MARK:` for section headers
- Use `# --------------------------` for visual separation

### Error Handling
- Use try/except for file operations with fallback defaults
- Example: missing frames fallback to black image

### Model Patterns
- Inherit from `nn.Module`
- Implement `__init__` and `forward`
- Use `freeze_backbone=True` to freeze ResNet layers (unfreeze layer4)

## Project Structure

```
.
├── dataset.py          # Dataset loader, transforms, CONFIG
├── models.py           # ResNetVideoModel, ResNetGRUVideoModel
├── train.py            # Training loop with early stopping
├── inference.py        # Single video & dataset inference
├── para_cal.py         # Model parameter counter
├── split_test_set.py   # Dataset splitting utility
├── checkpoint/         # Model weights & results (gitignored)
└── dataset/            # Data directory (gitignored)
```

## Dependencies

Core requirements (inferred from imports):
- `torch` + `torchvision`
- `pandas`
- `numpy`
- `Pillow` (PIL)
- `matplotlib`
- `tqdm`

No formal requirements.txt exists; install manually as needed.

## Device Handling

Always use `CONFIG["device"]` (auto-detects CUDA/CPU):
```python
model.to(CONFIG["device"])
inputs = inputs.to(CONFIG["device"])
```

## Data Format

- Input: Video frames as `.jpg` files in folders
- CSV format: `video_id`, `frames`, `label_id` (Train/Val) or `id`, `frames` (Test)
- Frame naming: `{frame_num:05d}.jpg` (e.g., `00001.jpg`)
