import os
import argparse
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from dataset import CONFIG, JesterDataset, train_transform, val_transform
from models import ResNetVideoModel, ResNetGRUVideoModel

# --------------------------
# 训练和推理流程
# --------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Gesture Recognition Training")
    parser.add_argument("--model_type", type=str, choices=['resnet', 'resnet_gru'], default='resnet', help="使用的模型结构")
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
    print(f"Model Type: {args.model_type}")
    print(f"Data Directory: {CONFIG['data_dir']}")
    print(f"Checkpoint Directory: {CONFIG['checkpoint_dir']}")
    
    train_csv = os.path.join(CONFIG["data_dir"], "Train.csv")

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

    # 根据参数选择并初始化模型
    if args.model_type == 'resnet_gru':
        model = ResNetGRUVideoModel(
            num_classes=CONFIG["num_classes"], 
            hidden_dim=CONFIG["hidden_dim"],
            freeze_backbone=True
        ).to(CONFIG["device"])
    else:
        model = ResNetVideoModel(
            num_classes=CONFIG["num_classes"],
            freeze_backbone=True
        ).to(CONFIG["device"])

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=CONFIG["learning_rate"])

    # 用于记录历史数据
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    best_val_acc = 0.0

    print(f"\n--- 开始训练 ({args.model_type}) ---")
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
        if epoch + 1 >= int(0.7 * CONFIG["num_epochs"]):
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
    history_csv_path = os.path.join(CONFIG["checkpoint_dir"], f"training_history_{args.model_type}.csv")
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
    plot_path = os.path.join(CONFIG["checkpoint_dir"], f"training_curves_{args.model_type}.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"📈 训练曲线图像已保存至: {plot_path}")

    # 保存模型
    model_save_name = f"model_{args.model_type}.pth"
    model_save_path = os.path.join(CONFIG["checkpoint_dir"], model_save_name)
    torch.save(model.state_dict(), model_save_path)
    print(f"Training finished! Model saved to {model_save_path}")
    
    # ==========================
    # 测试集一次性验证 (可选)
    # ==========================
    print("\n--- 开始对测试集进行初步评估 ---")
    test_csv_path = os.path.join(CONFIG["data_dir"], "Test.csv")
    if os.path.exists(test_csv_path):
        test_dataset = JesterDataset(
            csv_file=test_csv_path,
            root_dir=os.path.join(CONFIG["data_dir"], "Test"),
            num_frames=CONFIG["num_frames"],
            transform=val_transform,
            is_test=False # 设置为 False 以便验证是否有标签
        )
        test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"])
        
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        has_labels = False
        
        model.eval()
        with torch.no_grad():
            for inputs, labels, _ in test_loader:
                inputs = inputs.to(CONFIG["device"])
                labels = labels.to(CONFIG["device"])
                
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                
                if labels[0].item() != -1:
                    has_labels = True
                    loss = criterion(outputs, labels)
                    test_loss += loss.item()
                    test_total += labels.size(0)
                    test_correct += predicted.eq(labels).sum().item()
                
        if has_labels and test_total > 0:
            final_test_loss = test_loss / len(test_loader)
            final_test_acc = 100. * test_correct / test_total
            print(f"🎯 最终测试集评估结果 -> Loss: {final_test_loss:.4f}, Acc: {final_test_acc:.2f}%")
        else:
            final_test_loss, final_test_acc = None, None
            print("⚠️ 测试集没有提供真实标签，不支持自动评估。若需输出预测文件，请使用 inference.py。")
    else:
        final_test_loss, final_test_acc = None, None
        print(f"⚠️ 未找到测试集配置表: {test_csv_path}，跳过此步验证。")

    # ==========================
    # 生成 result.txt
    # ==========================
    result_txt_path = os.path.join(CONFIG["checkpoint_dir"], "result.txt")
    total_params = sum(p.numel() for p in model.parameters())
    
    with open(result_txt_path, "w", encoding="utf-8") as f:
        f.write(f"Model Type: {args.model_type}\n")
        f.write(f"Total Parameters: {total_params:,}\n")
        if train_losses:
            f.write(f"Last Epoch Train Loss: {train_losses[-1]:.4f}\n")
            f.write(f"Last Epoch Train Acc: {train_accs[-1]:.2f}%\n")
            f.write(f"Last Epoch Val Loss: {val_losses[-1]:.4f}\n")
            f.write(f"Last Epoch Val Acc: {val_accs[-1]:.2f}%\n")
        if final_test_loss is not None:
            f.write(f"Final Test Loss: {final_test_loss:.4f}\n")
            f.write(f"Final Test Acc: {final_test_acc:.2f}%\n")
        else:
            f.write("Final Test Loss: N/A\n")
            f.write("Final Test Acc: N/A\n")
            
    print(f"📝 结果已保存至: {result_txt_path}")

if __name__ == "__main__":
    train_model()

# python train.py --model_type resnet