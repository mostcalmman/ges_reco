import os
import platform
import argparse
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from dataset import CONFIG, CONFIG_Linux, IS_WINDOWS, IS_LINUX, get_config, JesterDataset, train_transform, val_transform
from models import modelList, ResNetVideoModel, ResNetGRUVideoModel, LightweightTSMModel, UltraLightConvGRUModel, LightweightTSMResNetModel, UltraLightConvGRUResNetModel, UltraLightConvGRUPooledModel, UltraLightGRUModel

# --------------------------
# 训练和推理流程
# --------------------------
def parse_args():
    config = get_config()
    parser = argparse.ArgumentParser(description="Gesture Recognition Training")
    parser.add_argument("--model_type", type=str, choices=modelList, default='resnet', help="使用的模型结构")
    parser.add_argument("--data_dir", type=str, default=config["data_dir"], help="数据集所在的目录")
    parser.add_argument("--checkpoint_dir", type=str, default=config["checkpoint_dir"], help="模型和预测结果保存的目录")
    parser.add_argument("--batch_size", type=int, default=config["batch_size"])
    parser.add_argument("--epochs", type=int, default=config["num_epochs"])
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training from")
    parser.add_argument("--save_every", type=int, default=0, help="Save checkpoint every N epochs (0 to disable)")
    return parser.parse_args()

def train_model():
    args = parse_args()
    config = get_config()
    
    # 更新配置
    config["data_dir"] = args.data_dir
    config["checkpoint_dir"] = args.checkpoint_dir
    config["batch_size"] = args.batch_size
    config["num_epochs"] = args.epochs
    
    os.makedirs(config["checkpoint_dir"], exist_ok=True)
    
    # 打印平台信息
    platform_name = "Windows" if IS_WINDOWS else ("Linux" if IS_LINUX else platform.system())
    print(f"Platform: {platform_name}")
    print(f"Using device: {config['device']}")
    print(f"Batchsize: {config['batch_size']}")
    print(f"num_workers: {config['num_workers']}, pin_memory: {config['pin_memory']}")
    print(f"Model Type: {args.model_type}")
    print(f"Data Directory: {config['data_dir']}")
    print(f"Checkpoint Directory: {config['checkpoint_dir']}")
    
    train_csv = os.path.join(config["data_dir"], "Train.csv")

    # 准备 Dataloader
    train_dataset = JesterDataset(
        csv_file=train_csv,
        root_dir=os.path.join(config["data_dir"], "Train"),
        num_frames=config["num_frames"],
        transform=train_transform
    )
    
    val_dataset = JesterDataset(
        csv_file=os.path.join(config["data_dir"], "Validation.csv"),
        root_dir=os.path.join(config["data_dir"], "Validation"),
        num_frames=config["num_frames"],
        transform=val_transform
    )

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=config["num_workers"], pin_memory=config["pin_memory"], prefetch_factor=config["prefetch_factor"])
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"], pin_memory=config["pin_memory"], prefetch_factor=config["prefetch_factor"])

    # 根据参数选择并初始化模型
    if args.model_type == 'resnet_gru':
        model = ResNetGRUVideoModel(
            num_classes=config["num_classes"], 
            hidden_dim=config["hidden_dim"],
            freeze_backbone=True
        ).to(config["device"])
    elif args.model_type == 'lightweight_tsm':
        model = LightweightTSMModel(
            num_classes=config["num_classes"],
            n_segment=config["num_frames"]
        ).to(config["device"])
    elif args.model_type == 'ultralight_convgru':
        model = UltraLightConvGRUModel(
            num_classes=config["num_classes"],
            n_segment=config["num_frames"]
        ).to(config["device"])
    elif args.model_type == 'ultralight_convgru_pooled':
        model = UltraLightConvGRUPooledModel(
            num_classes=config["num_classes"],
            n_segment=config["num_frames"]
        ).to(config["device"])
    elif args.model_type == 'lightweight_tsm_resnet':
        model = LightweightTSMResNetModel(
            num_classes=config["num_classes"],
            n_segment=config["num_frames"],
            pretrained=True
        ).to(config["device"])
    elif args.model_type == 'ultralight_convgru_resnet':
        model = UltraLightConvGRUResNetModel(
            num_classes=config["num_classes"],
            n_segment=config["num_frames"],
            pretrained=True
        ).to(config["device"])
    elif args.model_type == 'ultralight_gru':
        model = UltraLightGRUModel(
            num_classes=config["num_classes"],
            n_segment=config["num_frames"],
            hidden_dim=config["hidden_dim"]
        ).to(config["device"])
    else:
        model = ResNetVideoModel(
            num_classes=config["num_classes"],
            freeze_backbone=True
        ).to(config["device"])

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config["learning_rate"])

    # 用于记录历史数据
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    best_val_acc = 0.0
    start_epoch = 0

    # 加载检查点（如果指定了 --resume）
    if args.resume is not None:
        if os.path.exists(args.resume):
            print(f"正在从检查点恢复: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=config["device"])
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            train_losses = checkpoint.get('train_losses', [])
            train_accs = checkpoint.get('train_accs', [])
            val_losses = checkpoint.get('val_losses', [])
            val_accs = checkpoint.get('val_accs', [])
            start_epoch = checkpoint.get('epoch', 0) + 1
            best_val_acc = checkpoint.get('best_val_acc', 0.0)
            print(f"✓ 已从检查点恢复训练: 起始 epoch {start_epoch}")
        else:
            print(f"⚠️ 检查点文件不存在: {args.resume}，从头开始训练")

    print(f"\n--- 开始训练 ({args.model_type}) ---")
    # 训练循环
    for epoch in range(start_epoch, config["num_epochs"]):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for i, (inputs, labels, _) in enumerate(train_loader):
            inputs, labels = inputs.to(config["device"]), labels.to(config["device"])
            
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
                print(f"Epoch [{epoch+1}/{config['num_epochs']}], "
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
                inputs, labels = inputs.to(config["device"]), labels.to(config["device"])
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
        val_epoch_loss = val_loss / len(val_loader)
        val_epoch_acc = 100. * val_correct / val_total
                
        print(f"Epoch [{epoch+1}/{config['num_epochs']}] | "
              f"Train Loss: {train_epoch_loss:.4f} Acc: {train_epoch_acc:.2f}% | "
              f"Val Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.2f}%")
              
        # 记录数据
        train_losses.append(train_epoch_loss)
        train_accs.append(train_epoch_acc)
        val_losses.append(val_epoch_loss)
        val_accs.append(val_epoch_acc)
        
        # 提前停止策略 (Early Stopping)
        # 在 epoch 超过总 epoch 数的 70% 后生效
        if epoch + 1 >= int(0.7 * config["num_epochs"]):
            if val_epoch_acc < best_val_acc:
                print(f"⚠️ 触发提前停止：当前验证集准确率({val_epoch_acc:.2f}%) 低于历史最佳({best_val_acc:.2f}%)")
                break
                
        # 更新最佳验证集准确率
        if val_epoch_acc > best_val_acc:
            best_val_acc = val_epoch_acc
        
        # ==========================
        # 定期保存检查点
        # ==========================
        if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
            print(f"\n--- 保存周期检查点 (Epoch {epoch+1}) ---")
            
            # 创建子目录
            checkpoint_subdir = os.path.join(
                config["checkpoint_dir"], 
                "intermediate_checkpoints", 
                f"epoch_{epoch+1}"
            )
            os.makedirs(checkpoint_subdir, exist_ok=True)
            
            # 保存模型检查点
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses': train_losses,
                'train_accs': train_accs,
                'val_losses': val_losses,
                'val_accs': val_accs,
                'best_val_acc': best_val_acc,
                'model_type': args.model_type
            }
            checkpoint_path = os.path.join(checkpoint_subdir, f"model_{args.model_type}_epoch_{epoch+1}.pth")
            torch.save(checkpoint, checkpoint_path)
            print(f"✓ 检查点已保存: {checkpoint_path}")
            
            # 保存训练历史 CSV
            history_df = pd.DataFrame({
                "epoch": range(1, len(train_losses) + 1),
                "train_loss": train_losses,
                "train_acc": train_accs,
                "val_loss": val_losses,
                "val_acc": val_accs
            })
            history_csv_path = os.path.join(checkpoint_subdir, f"training_history_epoch_{epoch+1}.csv")
            history_df.to_csv(history_csv_path, index=False)
            print(f"✓ 训练历史已保存: {history_csv_path}")
            
            # 绘制并保存训练曲线
            plt.figure(figsize=(12, 5))
            
            plt.subplot(1, 2, 1)
            plt.plot(history_df["epoch"], history_df["train_loss"], label='Train Loss', marker='o')
            plt.plot(history_df["epoch"], history_df["val_loss"], label='Val Loss', marker='o')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title(f'Loss Curve (Epoch {epoch+1})')
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.legend()
            
            plt.subplot(1, 2, 2)
            plt.plot(history_df["epoch"], history_df["train_acc"], label='Train Acc', marker='o')
            plt.plot(history_df["epoch"], history_df["val_acc"], label='Val Acc', marker='o')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy (%)')
            plt.title(f'Accuracy Curve (Epoch {epoch+1})')
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.legend()
            
            plt.tight_layout()
            plot_path = os.path.join(checkpoint_subdir, f"training_curves_epoch_{epoch+1}.png")
            plt.savefig(plot_path, dpi=150)
            plt.close()
            print(f"✓ 训练曲线已保存: {plot_path}")
            
            # 运行测试集评估
            print(f"--- 评估测试集 (Epoch {epoch+1}) ---")
            test_csv_path = os.path.join(config["data_dir"], "Test.csv")
            test_result_loss, test_result_acc = None, None
            
            if os.path.exists(test_csv_path):
                test_dataset = JesterDataset(
                    csv_file=test_csv_path,
                    root_dir=os.path.join(config["data_dir"], "Test"),
                    num_frames=config["num_frames"],
                    transform=val_transform,
                    is_test=False
                )
                test_loader = DataLoader(
                    test_dataset, 
                    batch_size=config["batch_size"], 
                    shuffle=False, 
                    num_workers=config["num_workers"]
                )
                
                test_loss = 0.0
                test_correct = 0
                test_total = 0
                has_labels = False
                
                model.eval()
                with torch.no_grad():
                    for inputs, labels, _ in test_loader:
                        inputs = inputs.to(config["device"])
                        labels = labels.to(config["device"])
                        
                        outputs = model(inputs)
                        _, predicted = outputs.max(1)
                        
                        if labels[0].item() != -1:
                            has_labels = True
                            loss = criterion(outputs, labels)
                            test_loss += loss.item()
                            test_total += labels.size(0)
                            test_correct += predicted.eq(labels).sum().item()
                
                if has_labels and test_total > 0:
                    test_result_loss = test_loss / len(test_loader)
                    test_result_acc = 100. * test_correct / test_total
                    print(f"✓ 测试集评估 -> Loss: {test_result_loss:.4f}, Acc: {test_result_acc:.2f}%")
                else:
                    print("⚠️ 测试集没有提供真实标签")
            else:
                print(f"⚠️ 未找到测试集配置表: {test_csv_path}")
            
            # 保存 result.txt 到检查点目录
            result_txt_path = os.path.join(checkpoint_subdir, "result.txt")
            total_params = sum(p.numel() for p in model.parameters())
            
            with open(result_txt_path, "w", encoding="utf-8") as f:
                f.write(f"Model Type: {args.model_type}\n")
                f.write(f"Checkpoint Epoch: {epoch + 1}\n")
                f.write(f"Total Parameters: {total_params:,}\n")
                f.write(f"Last Epoch Train Loss: {train_losses[-1]:.4f}\n")
                f.write(f"Last Epoch Train Acc: {train_accs[-1]:.2f}%\n")
                f.write(f"Last Epoch Val Loss: {val_losses[-1]:.4f}\n")
                f.write(f"Last Epoch Val Acc: {val_accs[-1]:.2f}%\n")
                f.write(f"Best Val Acc So Far: {best_val_acc:.2f}%\n")
                if test_result_loss is not None:
                    f.write(f"Test Loss: {test_result_loss:.4f}\n")
                    f.write(f"Test Acc: {test_result_acc:.2f}%\n")
                else:
                    f.write("Test Loss: N/A\n")
                    f.write("Test Acc: N/A\n")
            
            print(f"✓ 结果已保存: {result_txt_path}")
            print(f"--- 周期检查点保存完成 ---\n")
    
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
    history_csv_path = os.path.join(config["checkpoint_dir"], f"training_history_{args.model_type}.csv")
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
    plot_path = os.path.join(config["checkpoint_dir"], f"training_curves_{args.model_type}.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"📈 训练曲线图像已保存至: {plot_path}")

    # 保存模型
    model_save_name = f"model_{args.model_type}.pth"
    model_save_path = os.path.join(config["checkpoint_dir"], model_save_name)
    torch.save(model.state_dict(), model_save_path)
    print(f"Training finished! Model saved to {model_save_path}")
    
    # ==========================
    # 测试集一次性验证 (可选)
    # ==========================
    print("\n--- 开始对测试集进行初步评估 ---")
    test_csv_path = os.path.join(config["data_dir"], "Test.csv")
    if os.path.exists(test_csv_path):
        test_dataset = JesterDataset(
            csv_file=test_csv_path,
            root_dir=os.path.join(config["data_dir"], "Test"),
            num_frames=config["num_frames"],
            transform=val_transform,
            is_test=False # 设置为 False 以便验证是否有标签
        )
        test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"])
        
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        has_labels = False
        
        model.eval()
        with torch.no_grad():
            for inputs, labels, _ in test_loader:
                inputs = inputs.to(config["device"])
                labels = labels.to(config["device"])
                
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
    result_txt_path = os.path.join(config["checkpoint_dir"], "result.txt")
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

# python train.py --model_type ultralight_convgru --checkpoint_dir ./checkpoint/ultraLight