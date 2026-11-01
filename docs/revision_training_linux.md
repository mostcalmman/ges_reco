# Linux 服务器补实验步骤

## 1. 环境准备

在仓库根目录执行；替换下面的 `/srv/...` 路径。推荐独立 Python 3.11 环境。PyTorch/torchvision 必须使用与服务器驱动及 GPU 匹配的同一套官方构建；不要复制 Windows 的 CPU wheel。

```bash
cd /srv/gesture
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# 先按 https://pytorch.org/get-started/locally/ 的 Linux/Pip/Python/CUDA 选项安装 torch、torchvision。
# 不在这里固定 CUDA wheel 版本，因为服务器驱动/GPU 未指定。
python -m pip install -r requirements-training.txt
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__); print('CUDA:', torch.cuda.is_available()); assert torch.cuda.is_available()"
export PYTHONUTF8=1
export PYTHONUNBUFFERED=1
export TORCH_HOME=/srv/cache/torch
mkdir -p "$TORCH_HOME"
```

GPU依赖安装命令以官方选择器为准。本地验证环境为 torch 2.11.0 / torchvision 0.26.0 CPU；未验证 Linux CUDA/NCCL。

## 2. 数据与预训练权重

```bash
export JESTER=/srv/data/jester
export SSV2=/srv/data/ssv2
export OUT=/srv/experiments/gesture_revision
mkdir -p "$OUT/logs"
```

每个数据根目录必须包含大小写精确的 `Train.csv`、`Validation.csv`、`Train/<video_id>/00001.jpg`、`Validation/<video_id>/00001.jpg`。CSV 有表头 `video_id,frames,label_id`，label_id 为0起始整数：Jester 0–26，SSv2 0–173；frames 为真实连续帧数。Linux 路径区分大小写。

SSv2 原始 webm 如未转换，可使用仓库脚本（已转换则跳过）：

```bash
python -m pip install opencv-python-headless
# /srv/data/ssv2_labels 内须有 labels.json、train.json、validation.json
python preprocess_ssv2.py --video_dir /srv/data/ssv2_videos --labels_dir /srv/data/ssv2_labels --output_dir "$SSV2" --workers 8 --splits Train Validation
```

该脚本读取现有文件，不下载 SSv2。已有加载器对缺失帧使用黑图回退，所以训练前确认转换完成、CSV和帧目录对应，不能只以无异常启动判断数据正确。

预下载 Heavy/Mobile ImageNet 权重，避免多进程同时首次下载：

```bash
python -c "from torchvision.models import resnet50, mobilenet_v2, ResNet50_Weights, MobileNet_V2_Weights; resnet50(weights=ResNet50_Weights.IMAGENET1K_V2); mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V2)"
```

离线服务器可在联网机器运行相同命令，将 `$TORCH_HOME/hub/checkpoints/` 中下载的文件复制到服务器同一缓存子目录。当前模型自动加载这些预训练权重；Micro 无需下载。

## 3. 首次短训练

```bash
python smoke_revision.py
python train.py --dataset jester --data_dir "$JESTER" --model_type LightTMF3GRU_ShortOnly --img_size 100 --num_frames 16 --hidden_dim 256 --epochs 1 --batch_size 2 --num_workers 0 --max_samples 4 --optimizer sgd --learning_rate 0.01 --scheduler multistep --save_every 1 --skip_test --checkpoint_dir "$OUT/smoke_micro"
python train.py --dataset ssv2 --data_dir "$SSV2" --model_type ResNet50_ACSSTMF3 --img_size 224 --num_frames 16 --epochs 1 --batch_size 1 --num_workers 0 --max_samples 2 --optimizer sgd --learning_rate 0.01 --scheduler multistep --save_every 1 --skip_test --checkpoint_dir "$OUT/smoke_heavy"
```

`--max_samples` 只截取训练和验证各自前 N 条，用于短训练，不用于正式实验。`--skip_test` 跳过周期保存及结束后的 Test 推理，不跳过每轮 Validation。

## 4. 正式实验：单卡或多卡启动

下面14条实验命令共用启动数组，选择一种设置；不要同时执行两个设置块。所有命令明确覆盖数据集、分辨率、帧数、hidden、优化器、学习率及训练轮数。

单卡：

```bash
export CUDA_VISIBLE_DEVICES=0
LAUNCH=(python)
BATCH=16
set -euo pipefail
```

四卡（单机 Linux CUDA/NCCL）：

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
LAUNCH=(torchrun --standalone --nproc_per_node=4)
BATCH=4
set -euo pipefail
```

这里有效 batch 均为16；`--batch_size` 是每GPU数值。16只是统一起点，并非论文历史 batch 的断言。按实际显存调整时，整组对照保持相同有效 batch，不因模型或分辨率单独改变。代码不自动缩放学习率。DDP 使用每卡独立 BatchNorm，不保证与单卡数值完全相同；同组使用一致的GPU数量。

多卡首次可先运行：

```bash
torchrun --standalone --nproc_per_node=4 train.py --dataset jester --data_dir "$JESTER" --model_type LightTMF3GRU_NoSpatial --img_size 100 --num_frames 16 --hidden_dim 256 --epochs 1 --batch_size 1 --num_workers 0 --max_samples 8 --save_every 1 --skip_test --checkpoint_dir "$OUT/smoke_ddp"
```

### A. Jester Micro 基线与四个消融

```bash
"${LAUNCH[@]}" train.py --config config.json --dataset jester --data_dir "$JESTER" --model_type LightTMF3GRU --img_size 100 --num_frames 16 --hidden_dim 256 --epochs 50 --batch_size "$BATCH" --num_workers 4 --optimizer sgd --learning_rate 0.01 --weight_decay 0.0005 --scheduler multistep --seed 42 --save_every 10 --skip_test --checkpoint_dir "$OUT/jester_100_LightTMF3GRU" 2>&1 | tee -a "$OUT/logs/jester_100_LightTMF3GRU.log"

"${LAUNCH[@]}" train.py --config config.json --dataset jester --data_dir "$JESTER" --model_type LightTMF3GRU_SwapPlacement --img_size 100 --num_frames 16 --hidden_dim 256 --epochs 50 --batch_size "$BATCH" --num_workers 4 --optimizer sgd --learning_rate 0.01 --weight_decay 0.0005 --scheduler multistep --seed 42 --save_every 10 --skip_test --checkpoint_dir "$OUT/jester_100_LightTMF3GRU_SwapPlacement" 2>&1 | tee -a "$OUT/logs/jester_100_LightTMF3GRU_SwapPlacement.log"

"${LAUNCH[@]}" train.py --config config.json --dataset jester --data_dir "$JESTER" --model_type LightTMF3GRU_ShortOnly --img_size 100 --num_frames 16 --hidden_dim 256 --epochs 50 --batch_size "$BATCH" --num_workers 4 --optimizer sgd --learning_rate 0.01 --weight_decay 0.0005 --scheduler multistep --seed 42 --save_every 10 --skip_test --checkpoint_dir "$OUT/jester_100_LightTMF3GRU_ShortOnly" 2>&1 | tee -a "$OUT/logs/jester_100_LightTMF3GRU_ShortOnly.log"

"${LAUNCH[@]}" train.py --config config.json --dataset jester --data_dir "$JESTER" --model_type LightTMF3GRU_NoSpatial --img_size 100 --num_frames 16 --hidden_dim 256 --epochs 50 --batch_size "$BATCH" --num_workers 4 --optimizer sgd --learning_rate 0.01 --weight_decay 0.0005 --scheduler multistep --seed 42 --save_every 10 --skip_test --checkpoint_dir "$OUT/jester_100_LightTMF3GRU_NoSpatial" 2>&1 | tee -a "$OUT/logs/jester_100_LightTMF3GRU_NoSpatial.log"

"${LAUNCH[@]}" train.py --config config.json --dataset jester --data_dir "$JESTER" --model_type LightTMF3_NoGRU --img_size 100 --num_frames 16 --hidden_dim 256 --epochs 50 --batch_size "$BATCH" --num_workers 4 --optimizer sgd --learning_rate 0.01 --weight_decay 0.0005 --scheduler multistep --seed 42 --save_every 10 --skip_test --checkpoint_dir "$OUT/jester_100_LightTMF3_NoGRU" 2>&1 | tee -a "$OUT/logs/jester_100_LightTMF3_NoGRU.log"
```

### B. SSv2 ResNet-50 同设置比较与模块消融

```bash
"${LAUNCH[@]}" train.py --config config.json --dataset ssv2 --data_dir "$SSV2" --model_type ResNet50_TSN --img_size 224 --num_frames 16 --hidden_dim 256 --epochs 50 --batch_size "$BATCH" --num_workers 4 --optimizer sgd --learning_rate 0.01 --weight_decay 0.0005 --scheduler multistep --seed 42 --save_every 10 --skip_test --checkpoint_dir "$OUT/ssv2_224_ResNet50_TSN" 2>&1 | tee -a "$OUT/logs/ssv2_224_ResNet50_TSN.log"

"${LAUNCH[@]}" train.py --config config.json --dataset ssv2 --data_dir "$SSV2" --model_type ResNet50_TSM --img_size 224 --num_frames 16 --hidden_dim 256 --epochs 50 --batch_size "$BATCH" --num_workers 4 --optimizer sgd --learning_rate 0.01 --weight_decay 0.0005 --scheduler multistep --seed 42 --save_every 10 --skip_test --checkpoint_dir "$OUT/ssv2_224_ResNet50_TSM" 2>&1 | tee -a "$OUT/logs/ssv2_224_ResNet50_TSM.log"

"${LAUNCH[@]}" train.py --config config.json --dataset ssv2 --data_dir "$SSV2" --model_type ResNet50_OnlyACSS --img_size 224 --num_frames 16 --hidden_dim 256 --epochs 50 --batch_size "$BATCH" --num_workers 4 --optimizer sgd --learning_rate 0.01 --weight_decay 0.0005 --scheduler multistep --seed 42 --save_every 10 --skip_test --checkpoint_dir "$OUT/ssv2_224_ResNet50_OnlyACSS" 2>&1 | tee -a "$OUT/logs/ssv2_224_ResNet50_OnlyACSS.log"

"${LAUNCH[@]}" train.py --config config.json --dataset ssv2 --data_dir "$SSV2" --model_type ResNet50_OnlyTMF3 --img_size 224 --num_frames 16 --hidden_dim 256 --epochs 50 --batch_size "$BATCH" --num_workers 4 --optimizer sgd --learning_rate 0.01 --weight_decay 0.0005 --scheduler multistep --seed 42 --save_every 10 --skip_test --checkpoint_dir "$OUT/ssv2_224_ResNet50_OnlyTMF3" 2>&1 | tee -a "$OUT/logs/ssv2_224_ResNet50_OnlyTMF3.log"

"${LAUNCH[@]}" train.py --config config.json --dataset ssv2 --data_dir "$SSV2" --model_type ResNet50_ACSSTMF3 --img_size 224 --num_frames 16 --hidden_dim 256 --epochs 50 --batch_size "$BATCH" --num_workers 4 --optimizer sgd --learning_rate 0.01 --weight_decay 0.0005 --scheduler multistep --seed 42 --save_every 10 --skip_test --checkpoint_dir "$OUT/ssv2_224_ResNet50_ACSSTMF3" 2>&1 | tee -a "$OUT/logs/ssv2_224_ResNet50_ACSSTMF3.log"
```

### C. SSv2 Micro/Mobile 分辨率对照

```bash
"${LAUNCH[@]}" train.py --config config.json --dataset ssv2 --data_dir "$SSV2" --model_type LightTMF3GRU --img_size 112 --num_frames 16 --hidden_dim 256 --epochs 50 --batch_size "$BATCH" --num_workers 4 --optimizer sgd --learning_rate 0.01 --weight_decay 0.0005 --scheduler multistep --seed 42 --save_every 10 --skip_test --checkpoint_dir "$OUT/ssv2_112_LightTMF3GRU" 2>&1 | tee -a "$OUT/logs/ssv2_112_LightTMF3GRU.log"

"${LAUNCH[@]}" train.py --config config.json --dataset ssv2 --data_dir "$SSV2" --model_type LightTMF3GRU --img_size 224 --num_frames 16 --hidden_dim 256 --epochs 50 --batch_size "$BATCH" --num_workers 4 --optimizer sgd --learning_rate 0.01 --weight_decay 0.0005 --scheduler multistep --seed 42 --save_every 10 --skip_test --checkpoint_dir "$OUT/ssv2_224_LightTMF3GRU" 2>&1 | tee -a "$OUT/logs/ssv2_224_LightTMF3GRU.log"

"${LAUNCH[@]}" train.py --config config.json --dataset ssv2 --data_dir "$SSV2" --model_type MobileNetV2_ACSSTMF3 --img_size 112 --num_frames 16 --hidden_dim 256 --epochs 50 --batch_size "$BATCH" --num_workers 4 --optimizer sgd --learning_rate 0.01 --weight_decay 0.0005 --scheduler multistep --seed 42 --save_every 10 --skip_test --checkpoint_dir "$OUT/ssv2_112_MobileNetV2_ACSSTMF3" 2>&1 | tee -a "$OUT/logs/ssv2_112_MobileNetV2_ACSSTMF3.log"

"${LAUNCH[@]}" train.py --config config.json --dataset ssv2 --data_dir "$SSV2" --model_type MobileNetV2_ACSSTMF3 --img_size 224 --num_frames 16 --hidden_dim 256 --epochs 50 --batch_size "$BATCH" --num_workers 4 --optimizer sgd --learning_rate 0.01 --weight_decay 0.0005 --scheduler multistep --seed 42 --save_every 10 --skip_test --checkpoint_dir "$OUT/ssv2_224_MobileNetV2_ACSSTMF3" 2>&1 | tee -a "$OUT/logs/ssv2_224_MobileNetV2_ACSSTMF3.log"
```

## 5. 输出与恢复训练

每个实验有独立目录，包含 `resolved_config.json`（CLI覆盖后的最终配置）、原配置备份、训练历史CSV、曲线图、最终模型纯state_dict及 `result.txt`。终端日志位于 `$OUT/logs/`。

完整恢复状态在 `intermediate_checkpoints/epoch_N/model_<model_type>_epoch_N.pth`，包括模型、优化器、调度器和历史。最终 `model_<model_type>.pth` 只有权重，不能传给训练 `--resume`。恢复时保持模型、类别、hidden、优化器、数据设置一致，epochs 表示总目标轮数，不是额外轮数。

例如恢复 Micro 短程消融第10轮，继续到50轮（使用第4节配置的 LAUNCH/BATCH）：

```bash
RESUME="$OUT/jester_100_LightTMF3GRU_ShortOnly/intermediate_checkpoints/epoch_10/model_LightTMF3GRU_ShortOnly_epoch_10.pth"
test -f "$RESUME"
"${LAUNCH[@]}" train.py --config config.json --dataset jester --data_dir "$JESTER" --model_type LightTMF3GRU_ShortOnly --img_size 100 --num_frames 16 --hidden_dim 256 --epochs 50 --batch_size "$BATCH" --num_workers 4 --optimizer sgd --learning_rate 0.01 --weight_decay 0.0005 --scheduler multistep --seed 42 --save_every 10 --skip_test --checkpoint_dir "$OUT/jester_100_LightTMF3GRU_ShortOnly" --resume "$RESUME" 2>&1 | tee -a "$OUT/logs/jester_100_LightTMF3GRU_ShortOnly.log"
```

训练参数优先级：common → 当前平台配置 → 选定dataset profile → CLI覆盖。不要修改 config.json 来回切换实验。当前 config 的 early_stopping=false、freeze_backbone=false 应保持；训练使用现有 label_smoothing=0.1、SGD momentum=0.9、多步衰减30/40/45乘0.1，以及既有训练随机采样、验证均匀采样和图像增强。seed固定随机数起点，不承诺跨平台或断点续训逐位一致。

PyTorch 启动方式参考：https://github.com/pytorch/pytorch/blob/main/docs/source/distributed/elastic/train_script.md
