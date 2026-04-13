# PROJECT KNOWLEDGE BASE

**Project**: Video Gesture Recognition (Jester Dataset)  
**Stack**: PyTorch + ResNet18 variants + optional GRU  
**Last Updated**: 2026-04-13

---

## OVERVIEW
PyTorch-based video gesture recognition system using ResNet18 variants (with optional GRU) on the Jester dataset. Supports multiple model architectures including LightTSM, LightTMF series with GRU variants.

## STRUCTURE
```
.
├── train.py              # Main training entry point
├── inference.py          # Single video & dataset inference
├── models.py             # Core model architectures (15+ variants)
├── dataset.py            # Dataset loader, transforms, CONFIG
├── modules.py            # Shared model components
├── para_cal.py           # Model parameter counter
├── split_test_set.py     # Dataset splitting utility
├── config.json           # Platform-aware configuration
├── utils/                # Config/model loading utilities
│   ├── config_loader.py  # Platform-specific config loading
│   └── model_loader.py   # Model instantiation
├── split/                # Alternative implementation
├── temporal-shift-module/# TSM (external reference)
├── ActionNet/            # ActionNet reference implementation
├── checkpoint/           # Model weights (gitignored)
└── dataset/              # Jester data (gitignored)
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Add new model | `models.py` | Inherit from `nn.Module`, follow existing pattern |
| Modify data loading | `dataset.py` | Check `JesterDataset` class |
| Change hyperparams | `config.json` | Platform-specific (windows/linux) |
| Add utility | `utils/` | Import via `from utils import ...` |
| Train model | `train.py` | Use `--model_type`, `--epochs` flags |
| Run inference | `inference.py` | Supports single video or CSV batch |

## CODE MAP
| Symbol | Type | Location | Role |
|--------|------|----------|------|
| `CONFIG` | Dict | `dataset.py:12` | Global configuration |
| `JesterDataset` | Class | `dataset.py:49` | Data loader |
| `LightTSMGRU` | Class | `models.py:123` | TSM + GRU model |
| `LightTMF3GRU` | Class | `models.py:444` | TMF fusion model |
| `train_model` | Function | `train.py:559` | Main training loop |
| `infer_single_video` | Function | `inference.py:96` | Single video inference |
| `get_config` | Function | `utils/config_loader.py:11` | Load config.json |
| `build_model` | Function | `utils/model_loader.py:5` | Model factory |

## CONVENTIONS

### Import Order
```python
import os
import argparse
import pandas as pd
import torch
from dataset import CONFIG, JesterDataset
```

### Configuration Pattern
```python
config = get_config()  # From config.json
model.to(config["device"])  # Always use config device
```

### Model Definition Pattern
```python
class LightXXXGRU(nn.Module):
    def __init__(self, num_classes, n_segment, hidden_dim):
        # Standard layers: conv1, layer1-3, GRU, fc
        # Freeze backbone: for param in resnet.parameters(): param.requires_grad = False
        
    def forward(self, x):
        # B, T, C, H, W -> reshape -> CNN -> GRU -> fc
```

### Comments
- Chinese comments acceptable (项目现有中文注释)
- Use `# MARK:` for section headers
- Use `# --------------------------` for visual separation

## ANTI-PATTERNS
| Pattern | Why Forbidden | Location |
|---------|--------------|----------|
| `.cuda()` hardcoded | Not portable | Avoid - use `.to(device)` |
| `.data` access | Deprecated | Avoid - use `.detach()` |
| `global` variables | Breaks encapsulation | temporal-shift-module/ only |
| Direct dict access | No defaults | Use `.get(key, default)` |

## COMMANDS
```bash
# Training
python train.py --model_type LightTMF3GRU --epochs 50 --batch_size 48

# Inference (single video)
python inference.py --video_path "dataset/Test/100010" --model_type LightTMF3GRU --model_weight "checkpoint/model.pth"

# Inference (dataset)
python inference.py --csv_path "dataset/Test.csv" --root_dir "dataset/Test" --model_type LightTMF3GRU

# Calculate parameters
python para_cal.py --model_type LightTMF3GRU

# Split dataset
python split_test_set.py --data_dir dataset --sample_size 5000
```

## DEVICE HANDLING
Always use `CONFIG["device"]` (auto-detects CUDA/CPU):
```python
model.to(CONFIG["device"])
inputs = inputs.to(CONFIG["device"])
```

## DATA FORMAT
- Input: Video frames as `.jpg` files in folders
- CSV: `video_id`, `frames`, `label_id` (Train/Val) or `id`, `frames` (Test)
- Frame naming: `{frame_num:05d}.jpg` (e.g., `00001.jpg`)
- Num frames: 16 (default), configurable in config.json
