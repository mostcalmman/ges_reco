# 补实验模型说明

## 基准和接口

所有模型输入 `(B,T,3,H,W)`，输出未归一化 logits `(B,num_classes)`，直接用于 CrossEntropyLoss。
正式实验 T=16；Jester 27 类，SSv2 174 类。Micro 使用随机初始化，3×3 stride=2 首层、32/64/128 三阶段，每阶段一个残差块，空间全局平均池化后接单层 GRU(hidden=256)、dropout(0.5)、分类层。

| 论文变体 | model_type | 时间聚合 |
|---|---|---|
| Micro | LightTMF3GRU | GRU 最后隐状态 |
| Heavy | ResNet50_ACSSTMF3 | 时间平均池化，无 GRU |
| Mobile | MobileNetV2_ACSSTMF3 | 时间平均池化，无 GRU |

ACSS 使用 `ACSS_Module/ACSSResBlock` 的静态可学习 `(C,3)` 权重，不使用 ACSS_1。DTMF 即代码 TMF3，完整模型使用 fusion_mode=B：`x + spatial * channel * x + spatial * MGDS(x)`。

## 四个新增模型

| model_type | stage1 / stage2 / stage3 | 分类前聚合 | 用途 |
|---|---|---|---|
| LightTMF3GRU_SwapPlacement | DTMF / ACSS / ACSS | GRU | 模块位置交换 |
| LightTMF3GRU_ShortOnly | ACSS / ACSS / 短程DTMF | GRU | 差分跨度消融 |
| LightTMF3GRU_NoSpatial | ACSS / ACSS / 无空间门控DTMF | GRU | 空间注意力消融 |
| LightTMF3_NoGRU | ACSS / ACSS / DTMF | 时间平均 | GRU消融 |

交换位置只改变第一、第三阶段的模块类型，阶段输出通道和 stride 保持 32/1、64/2、128/2；前移的 DTMF 输入通道为32、reduction=4，成本会随位置变化。

ShortOnly 将完整压缩宽度全部分配给短程差分 `conv(U_t)-U_(t-1)`，首帧置零；取消长程差分 `conv(U_(t+1))-U_(t-1)`。并非仅将原长程通道置零，压缩宽度与参数量保持不变。

NoSpatial 不创建空间卷积及 sigmoid，用全1空间权重替代，保留通道注意力、MGDS和残差。NoGRU 保留完整 CNN，逐帧空间池化后沿时间平均，再经 dropout(0.5) 与 Linear(128,num_classes)；没有 GRU 参数。

新增 GRU 模型类默认 hidden_dim=256；现有 LightTMF3GRU 类默认仍为128以保持兼容。模型工厂读取配置 hidden_dim，因此所有实验命令显式传256。新增模型默认 n_segment=8，但正式命令传16；NoGRU 忽略配置中的 hidden_dim。

## DTMF 可选接口

TMF3Module 和 TMF3ResBlock 均支持 `motion_mode='dual'`（默认）或 `'short_only'`，以及 `use_spatial_attention=True`（默认）或 False。默认路径保留原有参数名称和行为。消融模型的参数形状或键集合可能改变，不应把完整模型 checkpoint 当作消融训练的 resume。

## 现有 SSv2 模型

ResNet50_TSN、ResNet50_TSM、ResNet50_OnlyACSS、ResNet50_OnlyTMF3、ResNet50_ACSSTMF3 均已接入工厂。后四种时间模块布局沿用项目实现：TSM 在四个 stage 前移位；ACSS 在 layer2/layer3 前，DTMF 在 layer4 前。TSM 是仓库现有 stage 级复现，不是另行引入的逐残差块官方实现。

Heavy 使用 ImageNet ResNet50 V2 权重；Mobile 使用 ImageNet MobileNetV2 V2 权重。两者默认不冻结骨干。高分辨率不改变模型类，通过 --img_size 指定。

## 最小验证

运行 `python smoke_revision.py`：四个模型各检查27/174类前向、loss与反向，解析训练参数；代表性模型检查16帧224输入。无需数据和预训练下载。本次在 Windows、PyTorch 2.11.0+cpu / torchvision 0.26.0+cpu 通过；Linux CUDA/DDP 需在服务器按训练说明做短训练确认。

另用模拟帧和CSV完成一次本地训练/验证及checkpoint保存（CPU，1条样本），未进行完整数据集训练。
