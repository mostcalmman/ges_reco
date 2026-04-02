import os
import argparse
import shutil
from datetime import datetime
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from utils import get_config, get_platform_name, is_windows, is_linux, build_model
from dataset import JesterDataset, get_train_transform, get_val_transform
from models import modelList

LR_MILESTONES = [10, 20, 25]
LR_GAMMA = 0.1
OPTIMIZER_CHOICES = ['sgd', 'adam', 'adamw']


# ============================================================
# 配置与初始化函数
# ============================================================

def parse_args():
    """解析命令行参数"""
    config = get_config()
    model_type = config.get("model_type", "resnet")
    default_checkpoint_dir = os.path.join(config["checkpoint_dir"], model_type)
    parser = argparse.ArgumentParser(description="Gesture Recognition Training")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=default_checkpoint_dir,
        help="模型和预测结果保存目录，默认: config.checkpoint_dir/{model_type}",
    )
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training from")
    return parser.parse_args()


def backup_config_to_checkpoint(config_path, checkpoint_dir):
    """将本次训练配置文件备份到 checkpoint 目录"""
    if not os.path.exists(config_path):
        print(f"⚠️ 配置文件不存在，跳过备份: {config_path}")
        return None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_filename = f"config_{timestamp}.json"
    backup_path = os.path.join(checkpoint_dir, backup_filename)
    shutil.copy2(config_path, backup_path)
    print(f"✓ 配置文件已备份: {backup_path}")
    return backup_path


def setup_training():
    """
    初始化训练环境：解析参数、更新配置、创建目录、打印信息
    
    Returns:
        tuple: (args, config)
    """
    args = parse_args()
    config = get_config()

    model_type = config.get("model_type", "resnet")
    if model_type not in modelList:
        raise ValueError(
            f"config.json 中的 model_type 无效: {model_type}，"
            f"可选值: {modelList}"
        )
    optimizer_name = config.get("optimizer", "adamw")
    if optimizer_name not in OPTIMIZER_CHOICES:
        raise ValueError(
            f"config.json 中的 optimizer 无效: {optimizer_name}，"
            f"可选值: {OPTIMIZER_CHOICES}"
        )

    config["optimizer"] = optimizer_name
    config["save_every"] = int(config.get("save_every", 10))
    config["early_stopping"] = bool(config.get("early_stopping", False))
    args.model_type = model_type
    
    # 更新配置
    config["checkpoint_dir"] = args.checkpoint_dir
    
    # 创建检查点目录
    os.makedirs(config["checkpoint_dir"], exist_ok=True)

    # 在训练开始前备份本次配置文件
    backup_config_to_checkpoint("config.json", config["checkpoint_dir"])
    
    # 打印训练配置信息
    print_training_info(args, config)
    
    return args, config


def print_training_info(args, config):
    """打印训练配置信息"""
    platform_name = get_platform_name()
    print(f"Platform: {platform_name}")
    print(f"Using device: {config['device']}")
    print(f"Batchsize: {config['batch_size']}")
    print(f"num_workers: {config['num_workers']}, pin_memory: {config['pin_memory']}")
    print(f"Model Type: {args.model_type}")
    print(f"Optimizer: {config['optimizer'].upper()}")
    print(f"Data Directory: {config['data_dir']}")
    print(f"Checkpoint Directory: {config['checkpoint_dir']}")


# ============================================================
# 数据加载函数
# ============================================================

def create_dataloaders(config):
    """
    创建训练和验证数据加载器
    
    Args:
        config: 配置字典
        
    Returns:
        tuple: (train_loader, val_loader)
    """
    # 根据配置创建 transforms
    img_size = tuple(config.get("img_size", (100, 176)))
    normalize_mean = config.get("normalize_mean")
    normalize_std = config.get("normalize_std")
    
    train_transform = get_train_transform(img_size, normalize_mean, normalize_std)
    val_transform = get_val_transform(img_size, normalize_mean, normalize_std)
    
    # 创建数据集
    train_dataset = JesterDataset(
        csv_file=os.path.join(config["data_dir"], "Train.csv"),
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

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config["batch_size"], 
        shuffle=True, 
        num_workers=config["num_workers"], 
        pin_memory=config["pin_memory"], 
        prefetch_factor=config["prefetch_factor"]
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config["batch_size"], 
        shuffle=False, 
        num_workers=config["num_workers"], 
        pin_memory=config["pin_memory"], 
        prefetch_factor=config["prefetch_factor"]
    )
    
    return train_loader, val_loader


def build_optimizer(optimizer_name, model, base_lr):
    """根据优化器名称创建优化器"""
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())

    if optimizer_name == 'adam':
        return torch.optim.Adam(trainable_params, lr=base_lr, weight_decay=5e-4)
    if optimizer_name == 'adamw':
        return torch.optim.AdamW(trainable_params, lr=base_lr, weight_decay=5e-4)

    return torch.optim.SGD(
        trainable_params,
        lr=base_lr,
        momentum=0.9,
        weight_decay=5e-4
    )


def infer_lr_by_epoch(base_lr, start_epoch, milestones, gamma):
    """根据即将开始的 epoch 自动推导当前阶段学习率"""
    decay_count = sum(1 for milestone in milestones if start_epoch >= milestone)
    return base_lr * (gamma ** decay_count)


def load_checkpoint_if_needed(args, model, optimizer, device, base_lr, milestones, gamma):
    """
    如果需要，从检查点恢复训练状态
    
    Args:
        args: 命令行参数
        model: 模型实例
        optimizer: 优化器
        device: 计算设备
        
    Returns:
        tuple: (train_losses, train_accs, val_losses, val_accs, best_val_acc, start_epoch, scheduler_state_dict)
    """
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    best_val_acc = 0.0
    start_epoch = 0
    scheduler_state_dict = None
    
    if args.resume is not None:
        if os.path.exists(args.resume):
            print(f"正在从检查点恢复: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            checkpoint_optimizer_type = checkpoint.get('optimizer_type')
            if checkpoint_optimizer_type and checkpoint_optimizer_type != optimizer.__class__.__name__:
                print(f"⚠️ 检查点优化器类型为 {checkpoint_optimizer_type}，当前为 {optimizer.__class__.__name__}，跳过优化器状态恢复")
            elif 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print("✓ 已恢复优化器状态")
            else:
                print("⚠️ 检查点中没有 optimizer_state_dict，使用当前优化器默认状态")
            
            start_epoch = checkpoint.get('epoch', 0) + 1
            scheduler_state_dict = checkpoint.get('scheduler_state_dict')

            auto_lr = infer_lr_by_epoch(base_lr, start_epoch, milestones, gamma)
            for param_group in optimizer.param_groups:
                param_group['lr'] = auto_lr
            print(
                f"✓ 根据 checkpoint epoch={checkpoint.get('epoch', 0)} 自动设置学习率为: {auto_lr:.6f} "
                f"(milestones={milestones}, gamma={gamma})"
            )
            
            train_losses = checkpoint.get('train_losses', [])
            train_accs = checkpoint.get('train_accs', [])
            val_losses = checkpoint.get('val_losses', [])
            val_accs = checkpoint.get('val_accs', [])
            best_val_acc = checkpoint.get('best_val_acc', 0.0)
            print(f"✓ 已从检查点恢复训练: 起始 epoch {start_epoch}")
        else:
            print(f"⚠️ 检查点文件不存在: {args.resume}，从头开始训练")
    
    return train_losses, train_accs, val_losses, val_accs, best_val_acc, start_epoch, scheduler_state_dict


# ============================================================
# 训练与验证函数
# ============================================================

def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, num_epochs):
    """
    执行一个训练 epoch
    
    Returns:
        tuple: (avg_loss, accuracy)
    """
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    
    for i, (inputs, labels, _) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        
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
            print(f"Epoch [{epoch+1}/{num_epochs}], "
                  f"Step [{i+1}/{len(train_loader)}], "
                  f"Loss: {loss.item():.4f}, "
                  f"Acc: {100. * train_correct / train_total:.2f}%")
    
    avg_loss = train_loss / len(train_loader)
    accuracy = 100. * train_correct / train_total
    return avg_loss, accuracy


def validate(model, val_loader, criterion, device):
    """
    在验证集上评估模型
    
    Returns:
        tuple: (avg_loss, accuracy)
    """
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for inputs, labels, _ in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()
    
    avg_loss = val_loss / len(val_loader)
    accuracy = 100. * val_correct / val_total
    return avg_loss, accuracy


def log_epoch_results(epoch, num_epochs, train_loss, train_acc, val_loss, val_acc):
    """打印 epoch 训练结果"""
    print(f"Epoch [{epoch+1}/{num_epochs}] | "
          f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
          f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")


def should_early_stop(epoch, num_epochs, val_acc, best_val_acc, early_stopping_enabled):
    """
    检查是否触发提前停止
    
    在 epoch 超过总 epoch 数的 70% 后生效
    """
    if not early_stopping_enabled:
        return False
    
    if epoch + 1 >= int(0.7 * num_epochs):
        if val_acc < best_val_acc:
            print(f"⚠️ 触发提前停止：当前验证集准确率({val_acc:.2f}%) 低于历史最佳({best_val_acc:.2f}%)")
            return True
    return False


# ============================================================
# 保存与绘图函数
# ============================================================

def save_intermediate_checkpoint(args, config, model, optimizer, scheduler, history, epoch, best_val_acc):
    """
    保存中间检查点（模型、历史记录、曲线图、测试结果）
    
    Args:
        args: 命令行参数
        config: 配置字典
        model: 模型实例
        optimizer: 优化器
        scheduler: 学习率调度器
        history: 训练历史字典
        epoch: 当前 epoch（0-based）
        best_val_acc: 最佳验证准确率
    """
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
        'optimizer_type': optimizer.__class__.__name__,
        'scheduler_state_dict': scheduler.state_dict(),
        'train_losses': history['train_losses'],
        'train_accs': history['train_accs'],
        'val_losses': history['val_losses'],
        'val_accs': history['val_accs'],
        'best_val_acc': best_val_acc,
        'model_type': args.model_type
    }
    checkpoint_path = os.path.join(checkpoint_subdir, f"model_{args.model_type}_epoch_{epoch+1}.pth")
    torch.save(checkpoint, checkpoint_path)
    print(f"✓ 检查点已保存: {checkpoint_path}")
    
    # 保存训练历史 CSV
    save_history_csv(history, os.path.join(checkpoint_subdir, f"training_history_epoch_{epoch+1}.csv"))
    
    # 绘制并保存训练曲线
    plot_training_curves(history, os.path.join(checkpoint_subdir, f"training_curves_epoch_{epoch+1}.png"))
    
    # 运行测试集评估并保存结果
    test_loss, test_acc = evaluate_test_set(config, model)
    save_result_txt(
        args.model_type, model, history, best_val_acc,
        os.path.join(checkpoint_subdir, "result.txt"),
        test_loss=test_loss, test_acc=test_acc
    )
    
    print(f"--- 周期检查点保存完成 ---\n")


def save_history_csv(history, filepath):
    """保存训练历史到 CSV"""
    history_df = pd.DataFrame({
        "epoch": range(1, len(history['train_losses']) + 1),
        "train_loss": history['train_losses'],
        "train_acc": history['train_accs'],
        "val_loss": history['val_losses'],
        "val_acc": history['val_accs']
    })
    history_df.to_csv(filepath, index=False)
    print(f"✓ 训练历史已保存: {filepath}")


def plot_training_curves(history, filepath):
    """绘制并保存训练曲线"""
    epochs = range(1, len(history['train_losses']) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # Loss 曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_losses'], label='Train Loss', marker='o')
    plt.plot(epochs, history['val_losses'], label='Val Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    # Accuracy 曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_accs'], label='Train Acc', marker='o')
    plt.plot(epochs, history['val_accs'], label='Val Acc', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy Curve')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close()
    print(f"✓ 训练曲线已保存: {filepath}")


def save_result_txt(model_type, model, history, best_val_acc, filepath, test_loss=None, test_acc=None):
    """保存训练结果到文本文件"""
    total_params = sum(p.numel() for p in model.parameters())
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"Model Type: {model_type}\n")
        f.write(f"Total Parameters: {total_params:,}\n")
        
        if history['train_losses']:
            f.write(f"Last Epoch Train Loss: {history['train_losses'][-1]:.4f}\n")
            f.write(f"Last Epoch Train Acc: {history['train_accs'][-1]:.2f}%\n")
            f.write(f"Last Epoch Val Loss: {history['val_losses'][-1]:.4f}\n")
            f.write(f"Last Epoch Val Acc: {history['val_accs'][-1]:.2f}%\n")
        
        f.write(f"Best Val Acc: {best_val_acc:.2f}%\n")
        
        if test_loss is not None:
            f.write(f"Test Loss: {test_loss:.4f}\n")
            f.write(f"Test Acc: {test_acc:.2f}%\n")
        else:
            f.write("Test Loss: N/A\n")
            f.write("Test Acc: N/A\n")
    
    print(f"✓ 结果已保存: {filepath}")


# ============================================================
# 测试集评估函数
# ============================================================

def evaluate_test_set(config, model):
    """
    在测试集上评估模型
    
    Returns:
        tuple: (test_loss, test_acc) 如果测试集没有标签则返回 (None, None)
    """
    test_csv_path = os.path.join(config["data_dir"], "Test.csv")
    
    if not os.path.exists(test_csv_path):
        print(f"⚠️ 未找到测试集配置表: {test_csv_path}")
        return None, None
    
    # 创建测试集 transform
    img_size = tuple(config.get("img_size", (100, 176)))
    normalize_mean = config.get("normalize_mean")
    normalize_std = config.get("normalize_std")
    val_transform = get_val_transform(img_size, normalize_mean, normalize_std)
    
    test_dataset = JesterDataset(
        csv_file=test_csv_path,
        root_dir=os.path.join(config["data_dir"], "Test"),
        num_frames=config["num_frames"],
        transform=val_transform,
        is_test=False  # 设置为 False 以便验证是否有标签
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config["batch_size"], 
        shuffle=False, 
        num_workers=config["num_workers"]
    )
    
    criterion = nn.CrossEntropyLoss()
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
        test_loss = test_loss / len(test_loader)
        test_acc = 100. * test_correct / test_total
        print(f"✓ 测试集评估 -> Loss: {test_loss:.4f}, Acc: {test_acc:.2f}%")
        return test_loss, test_acc
    else:
        print("⚠️ 测试集没有提供真实标签")
        return None, None


# ============================================================
# 主训练函数
# ============================================================

def train_model():
    """主训练流程"""
    # ---------- 1. 初始化 ----------
    args, config = setup_training()
    train_loader, val_loader = create_dataloaders(config)
    model = build_model(args.model_type, config, device=config["device"])
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1) # change: add label smoothing
    base_lr = config["learning_rate"] if config["learning_rate"] != 0 else 0.01
    optimizer = build_optimizer(config["optimizer"], model, base_lr)
    
    # 初始化历史记录
    history = {
        'train_losses': [], 'train_accs': [],
        'val_losses': [], 'val_accs': []
    }
    best_val_acc = 0.0
    
    # 加载检查点（如果需要）
    (history['train_losses'], history['train_accs'], 
     history['val_losses'], history['val_accs'], 
     best_val_acc, start_epoch, scheduler_state_dict) = load_checkpoint_if_needed(
        args, model, optimizer, config["device"], base_lr, LR_MILESTONES, LR_GAMMA
    )

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=LR_MILESTONES,
        gamma=LR_GAMMA
    )

    if scheduler_state_dict is not None:
        scheduler.load_state_dict(scheduler_state_dict)
        print(f"✓ 已恢复学习率调度器状态: last_epoch={scheduler.last_epoch}")
    elif start_epoch > 0:
        scheduler.last_epoch = start_epoch - 1
        scheduler._last_lr = [group['lr'] for group in optimizer.param_groups]
        print(f"✓ 已按起始 epoch 对齐调度器进度: last_epoch={scheduler.last_epoch}")
    
    print(f"\n--- 开始训练 ({args.model_type}) ---")
    optimizer_name = optimizer.__class__.__name__
    if config["optimizer"] in ('adam', 'adamw'):
        print(f"优化器: {optimizer_name} | 初始学习率: {optimizer.param_groups[0]['lr']:.6f} | weight_decay: 5e-4")
    else:
        print(f"优化器: {optimizer_name} | 初始学习率: {optimizer.param_groups[0]['lr']:.6f} | momentum: 0.9 | weight_decay: 5e-4")
    print(f"学习率策略: MultiStepLR, milestones={LR_MILESTONES}, gamma={LR_GAMMA}")
    
    # ---------- 2. 训练循环 ----------
    for epoch in range(start_epoch, config["num_epochs"]):
        # 训练一个 epoch
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, 
            config["device"], epoch, config["num_epochs"]
        )
        
        # 验证
        val_loss, val_acc = validate(model, val_loader, criterion, config["device"])
        
        # 记录并打印结果
        log_epoch_results(epoch, config["num_epochs"], train_loss, train_acc, val_loss, val_acc)
        history['train_losses'].append(train_loss)
        history['train_accs'].append(train_acc)
        history['val_losses'].append(val_loss)
        history['val_accs'].append(val_acc)
        
        # 更新最佳准确率
        if val_acc > best_val_acc:
            best_val_acc = val_acc
        
        # 提前停止检查
        if should_early_stop(epoch, config["num_epochs"], val_acc, best_val_acc, config["early_stopping"]):
            break
        
        # 定期保存检查点
        if config["save_every"] > 0 and (epoch + 1) % config["save_every"] == 0:
            save_intermediate_checkpoint(
                args, config, model, optimizer, scheduler, history, 
                epoch, best_val_acc
            )

        # 学习率调度（按 epoch 结束后更新）
        scheduler.step()
    
    # ---------- 3. 训练结束保存 ----------
    print("\n--- 训练完成，保存最终结果 ---")
    
    # 保存训练历史
    final_history_path = os.path.join(config["checkpoint_dir"], f"training_history_{args.model_type}.csv")
    save_history_csv(history, final_history_path)
    print(f"📊 训练历史数据已保存至: {final_history_path}")
    
    # 保存最终曲线图
    final_plot_path = os.path.join(config["checkpoint_dir"], f"training_curves_{args.model_type}.png")
    plot_training_curves(history, final_plot_path)
    print(f"📈 训练曲线图像已保存至: {final_plot_path}")
    
    # 保存最终模型
    model_save_path = os.path.join(config["checkpoint_dir"], f"model_{args.model_type}.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"✓ 模型已保存至: {model_save_path}")
    
    # 最终测试集评估
    print("\n--- 开始对测试集进行初步评估 ---")
    test_loss, test_acc = evaluate_test_set(config, model)
    
    # 保存最终结果文件
    result_txt_path = os.path.join(config["checkpoint_dir"], "result.txt")
    save_result_txt(args.model_type, model, history, best_val_acc, result_txt_path, test_loss, test_acc)
    print(f"📝 结果已保存至: {result_txt_path}")


if __name__ == "__main__":
    train_model()

# python train.py --checkpoint_dir ./checkpoint/ultraLight
