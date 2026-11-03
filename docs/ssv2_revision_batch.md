# SSv2 第二轮批量训练（AdamW）

从仓库根目录运行，数据放在 dataset/SSv2/，保持既有 Train.csv、Validation.csv 和帧目录格式。
同步 config.json、batch_train.py 及之前实现的模型/训练代码到服务器。

```bash
python batch_train.py --preset ssv2_revision --run_root checkpoint/ssv2_revision_adamw --dry_run
python batch_train.py --preset ssv2_revision --run_root checkpoint/ssv2_revision_adamw --stop_on_error
```

后台启动时用下面这条替代正式运行命令：

```bash
nohup python -u batch_train.py --preset ssv2_revision --run_root checkpoint/ssv2_revision_adamw --stop_on_error > ssv2_batch.log 2>&1 &
tail -f ssv2_batch.log
```

## 配置与实验队列

统一参数读取 config.json：dataset=ssv2、optimizer=adamw、learning_rate=0.005、weight_decay=0.0005、hidden_dim=256、16帧；Linux为50轮、batch_size=64、num_workers=16。其余数值沿用原配置，未因切换AdamW自动调整学习率。
已移除 scheduler 和 cosine_eta_min 配置项。train.py 缺省调度为 cosine，eta_min=1e-5，删除配置项不会关闭调度。

实验清单也保存在 config.json 的 batch_runs.ssv2_revision。name 是独立实验标识，用于输出目录和完成状态；model_type 选择模型，img_size 选择该组分辨率。脚本只传模型、输出目录、分辨率，训练超参数使用公共配置。112与224不能由一个全局分辨率同时表达，因此按实验传入。

| 序号 | 模型 | 分辨率 | 审稿要求 |
|---|---|---|---|
| 1 | ResNet50_TSN | 224 | R2-1 无模块基线 |
| 2 | ResNet50_OnlyACSS | 224 | R2-1 仅 ACSS |
| 3 | ResNet50_OnlyTMF3 | 224 | R2-1 仅 DTMF |
| 4 | ResNet50_ACSSTMF3 | 224 | R2-1 完整模型 / R1-5 Heavy 同设置对照 |
| 5 | LightTMF3GRU | 112 | R2-2 Micro 低分辨率 / R1-2 原布局 / R1-5 Micro 同设置对照 |
| 6 | LightTMF3GRU | 224 | R2-2 Micro 高分辨率 |
| 7 | MobileNetV2_ACSSTMF3 | 112 | R2-2 Mobile 低分辨率 / R1-5 Mobile 同设置对照 |
| 8 | MobileNetV2_ACSSTMF3 | 224 | R2-2 Mobile 高分辨率 |
| 9 | LightTMF3GRU_SwapPlacement | 112 | R1-2 Micro 交换布局 |
| 10 | LightTSMGRU | 112 | R1-5 Micro TSM |
| 11 | MobileNetV2_TSM | 112 | R1-5 Mobile TSM |
| 12 | ResNet50_TSM | 224 | R1-5 Heavy TSM |

完整目录映射记录在 `run_root/logs/checkpoint_reviews.log`，每组训练日志也记录本组对应的意见。

R2-1沿用已确认方案，在ResNet-50上做四条件消融，不扩展到另外两个骨干。重复完整模型已去重；同组比较都使用本轮AdamW训练结果，不混用上轮Jester/SGD结果。

已有架构覆盖全部实验，无需新增模型类。Heavy/Mobile加载已有ImageNet V2预训练权重，服务器需能下载或已有缓存。Micro随机初始化。

## 运行与结果

同一时间顺序运行一个模型（单GPU）。若显存不足，在 config.json 的 linux.batch_size 调小，整组统一使用；64是沿用值，未在服务器验证容量。
输出位于 run_root/实验name，日志位于 run_root/logs，完成状态在 batch_state.json。不同分辨率独立计数，互不覆盖。重新启动相同命令跳过完成项目；中断项目默认从头运行，不自动恢复。
默认沿用Train/Validation及周期Test评估流程；不准备Test时会跳过缺失的Test.csv。若明确跳过Test，在命令末尾加 -- --skip_test。
更改配置后需使用新的run_root，避免跳过旧结果。旧jester_revision预设仍在，但现在配置数据集是ssv2，要重新跑Jester需先改回配置。

本地验证：12条命令dry-run与训练参数解析通过；未加载SSv2数据，未启动训练，Linux CUDA运行需在服务器执行。
