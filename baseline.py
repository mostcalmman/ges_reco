import os
import argparse
import pandas as pd
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt

# --------------------------
# 1. 默认配置参数 (将被命令行参数覆盖)
# --------------------------
CONFIG = {
    "data_dir": "dataset",
    "checkpoint_dir": "checkpoint",
    "batch_size": 48,
    "num_workers": 4,
    "pin_memory": True,
    "prefetch_factor": 2,
    "num_frames": 37,
    "img_size": (100, 176),
    "num_classes": 27,
    "hidden_dim": 256,
    "num_epochs": 10,
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
        # 如果视频帧数不足 num_frames，则使用最后一帧进行填充, 假设需要5帧, 输入3帧, 则返回 [1, 2, 3, 3, 3]
        # 如果视频帧数超过 num_frames，则均匀采样 num_frames 帧, 假设需要5帧, 输入10帧, 则返回 [1, 3, 5, 7, 9]
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
                print(f"⚠️ 警告: 未找到帧图像 {img_path}，使用黑色图像替代")
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

# --------------------------
# 3. 模型定义 (2D CNN + GRU)
# --------------------------
class LightweightVideoModel(nn.Module):
    def __init__(self, num_classes, hidden_dim, freeze_backbone=True):
        super(LightweightVideoModel, self).__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        if freeze_backbone:
            for param in resnet.parameters():
                param.requires_grad = False
            for param in resnet.layer4.parameters():
                param.requires_grad = True

        self.cnn = nn.Sequential(*list(resnet.children())[:-1])
        cnn_out_dim = 512 
        
        self.rnn = nn.GRU(
            input_size=cnn_out_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )
        
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        b, t, c, h, w = x.size()
        x = x.view(b * t, c, h, w)
        cnn_features = self.cnn(x) 
        cnn_features = cnn_features.view(b, t, -1) 
        rnn_out, hidden = self.rnn(cnn_features) 
        last_hidden = hidden[-1] 
        out = self.fc(last_hidden)
        return out

# --------------------------
# 4. 训练和推理流程
# --------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Lightweight Gesture Recognition Training")
    parser.add_argument("--data_dir", type=str, default=CONFIG["data_dir"], help="数据集所在的目录")
    parser.add_argument("--checkpoint_dir", type=str, default=CONFIG["checkpoint_dir"], help="模型和预测结果保存的目录")
    parser.add_argument("--batch_size", type=int, default=CONFIG["batch_size"])
    parser.add_argument("--epochs", type=int, default=CONFIG["num_epochs"])
    return parser.parse_args()

def train_model():
    args = parse_args()
    
    # 更新配置
    CONFIG["data_dir"] = args.data_dir
    CONFIG["checkpoint_dir"] = args.checkpoint_dir
    CONFIG["batch_size"] = args.batch_size
    CONFIG["num_epochs"] = args.epochs
    
    os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)
    
    print(f"Using device: {CONFIG['device']}")
    print(f"Data Directory: {CONFIG['data_dir']}")
    print(f"Checkpoint Directory: {CONFIG['checkpoint_dir']}")
    
    train_csv = os.path.join(CONFIG["data_dir"], "Train.csv")
    df = pd.read_csv(train_csv)

    # 准备 Dataloader
    train_dataset = JesterDataset(
        csv_file=train_csv,
        root_dir=os.path.join(CONFIG["data_dir"], "Train"),
        num_frames=CONFIG["num_frames"],
        transform=train_transform
    )
    
    val_dataset = JesterDataset(
        csv_file=os.path.join(CONFIG["data_dir"], "Validation.csv"),
        root_dir=os.path.join(CONFIG["data_dir"], "Validation"),
        num_frames=CONFIG["num_frames"],
        transform=val_transform
    )

    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=CONFIG["num_workers"], pin_memory=CONFIG["pin_memory"], prefetch_factor=CONFIG["prefetch_factor"])
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"], pin_memory=CONFIG["pin_memory"], prefetch_factor=CONFIG["prefetch_factor"])

    # 初始化模型
    model = LightweightVideoModel(
        num_classes=CONFIG["num_classes"], 
        hidden_dim=CONFIG["hidden_dim"],
        freeze_backbone=True
    ).to(CONFIG["device"])

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=CONFIG["learning_rate"])

    # 用于记录历史数据
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    best_val_acc = 0.0

    # 训练循环
    for epoch in range(CONFIG["num_epochs"]):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for i, (inputs, labels, _) in enumerate(train_loader):
            inputs, labels = inputs.to(CONFIG["device"]), labels.to(CONFIG["device"])
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            # 每 10 个 batch 输出一次训练进度
            if (i + 1) % 10 == 0 or (i + 1) == len(train_loader):
                print(f"Epoch [{epoch+1}/{CONFIG['num_epochs']}], "
                      f"Step [{i+1}/{len(train_loader)}], "
                      f"Loss: {loss.item():.4f}, "
                      f"Acc: {100. * train_correct / train_total:.2f}%")
            
        train_epoch_loss = train_loss / len(train_loader)
        train_epoch_acc = 100. * train_correct / train_total

        # 验证循环
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels, _ in val_loader:
                inputs, labels = inputs.to(CONFIG["device"]), labels.to(CONFIG["device"])
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
        val_epoch_loss = val_loss / len(val_loader)
        val_epoch_acc = 100. * val_correct / val_total
                
        print(f"Epoch [{epoch+1}/{CONFIG['num_epochs']}] | "
              f"Train Loss: {train_epoch_loss:.4f} Acc: {train_epoch_acc:.2f}% | "
              f"Val Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.2f}%")
              
        # 记录数据
        train_losses.append(train_epoch_loss)
        train_accs.append(train_epoch_acc)
        val_losses.append(val_epoch_loss)
        val_accs.append(val_epoch_acc)
        
        # 提前停止策略 (Early Stopping)
        # 在 epoch 超过总 epoch 数的 70% 后生效
        if epoch >= int(0.7 * CONFIG["num_epochs"]):
            if val_epoch_acc < best_val_acc:
                print(f"⚠️ 触发提前停止：当前验证集准确率({val_epoch_acc:.2f}%) 低于历史最佳({best_val_acc:.2f}%)")
                break
                
        # 更新最佳验证集准确率
        if val_epoch_acc > best_val_acc:
            best_val_acc = val_epoch_acc
    
    # ==========================
    # 保存训练记录和绘制图像
    # ==========================
    history_df = pd.DataFrame({
        "epoch": range(1, len(train_losses) + 1),
        "train_loss": train_losses,
        "train_acc": train_accs,
        "val_loss": val_losses,
        "val_acc": val_accs
    })
    history_csv_path = os.path.join(CONFIG["checkpoint_dir"], "training_history.csv")
    history_df.to_csv(history_csv_path, index=False)
    print(f"📊 训练历史数据已保存至: {history_csv_path}")

    # 绘制曲线
    plt.figure(figsize=(12, 5))
    
    # Loss 曲线
    plt.subplot(1, 2, 1)
    plt.plot(history_df["epoch"], history_df["train_loss"], label='Train Loss', marker='o')
    plt.plot(history_df["epoch"], history_df["val_loss"], label='Val Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    # Accuracy 曲线
    plt.subplot(1, 2, 2)
    plt.plot(history_df["epoch"], history_df["train_acc"], label='Train Acc', marker='o')
    plt.plot(history_df["epoch"], history_df["val_acc"], label='Val Acc', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy Curve')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    plt.tight_layout()
    plot_path = os.path.join(CONFIG["checkpoint_dir"], "training_curves.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"📈 训练曲线图像已保存至: {plot_path}")

    # 保存模型
    model_save_path = os.path.join(CONFIG["checkpoint_dir"], "lightweight_gesture_model.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"Training finished! Model saved to {model_save_path}")
    
    # ==========================
    # 5. 测试集推理并保存 CSV
    # ==========================
    print("\n--- 开始对测试集进行推理 ---")
    test_csv_path = os.path.join(CONFIG["data_dir"], "Test.csv")
    if os.path.exists(test_csv_path):
        test_dataset = JesterDataset(
            csv_file=test_csv_path,
            root_dir=os.path.join(CONFIG["data_dir"], "Test"),
            num_frames=CONFIG["num_frames"],
            transform=val_transform,
            is_test=True
        )
        test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"])
        
        predictions = []
        video_ids = []
        
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        has_labels = False
        
        model.eval()
        with torch.no_grad():
            for inputs, labels, v_ids in test_loader:
                inputs = inputs.to(CONFIG["device"])
                labels = labels.to(CONFIG["device"])
                
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                
                # 如果有真实的标签，计算Loss和Acc (Test.csv 现在有真实标签了)
                if labels[0].item() != -1:
                    has_labels = True
                    loss = criterion(outputs, labels)
                    test_loss += loss.item()
                    test_total += labels.size(0)
                    test_correct += predicted.eq(labels).sum().item()
                
                predictions.extend(predicted.cpu().numpy())
                video_ids.extend(v_ids)
                
        if has_labels and test_total > 0:
            final_test_loss = test_loss / len(test_loader)
            final_test_acc = 100. * test_correct / test_total
            print(f"🎯 测试集评估结果 -> Loss: {final_test_loss:.4f}, Acc: {final_test_acc:.2f}%")
        else:
            print("⚠️ 测试集没有提供真实标签，已跳过 Loss 和 Acc 计算。")
                
        # 保存预测结果到 CSV
        results_df = pd.DataFrame({
            "video_id": video_ids,
            "predicted_label_id": predictions
        })
        
        output_csv_path = os.path.join(CONFIG["checkpoint_dir"], "test_predictions.csv")
        results_df.to_csv(output_csv_path, index=False)
        print(f"✅ 测试集预测完成！结果已保存至: {output_csv_path}")
    else:
        print(f"⚠️ 未找到测试集配置表: {test_csv_path}，跳过测试集推理。")

if __name__ == "__main__":
    train_model()