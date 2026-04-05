import argparse
import csv
import os
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config_loader import get_config, get_platform_name
from dataset import (
    JesterDataset,
    SAMPLING_RANDOM,
    SAMPLING_UNIFORM,
    get_train_transform,
    get_val_transform,
)
from models import MODEL_TYPE, build_model


OPTIMIZER_CHOICES = ["sgd", "adam", "adamw"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Standalone training for UltraLightParallelMEGRUModel"
    )
    parser.add_argument("--config", type=str, default=None, help="Path to split config json")
    parser.add_argument("--data_dir", type=str, default=None, help="Dataset root path")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Output checkpoint directory")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path for resuming")

    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size")
    parser.add_argument("--num_frames", type=int, default=None, help="Frames per clip")
    parser.add_argument("--hidden_dim", type=int, default=None, help="GRU hidden dim")
    parser.add_argument("--learning_rate", type=float, default=None, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=None, help="Weight decay")
    parser.add_argument("--optimizer", type=str, default=None, choices=OPTIMIZER_CHOICES)
    parser.add_argument("--num_workers", type=int, default=None, help="DataLoader workers")
    parser.add_argument("--save_every", type=int, default=None, help="Save interval by epoch")
    return parser.parse_args()


def _abs_if_needed(path_value):
    if path_value is None:
        return None
    return path_value if os.path.isabs(path_value) else os.path.abspath(path_value)


def apply_overrides(config, args):
    path_overrides = {
        "data_dir": _abs_if_needed(args.data_dir),
        "checkpoint_dir": _abs_if_needed(args.checkpoint_dir),
    }
    for key, value in path_overrides.items():
        if value is not None:
            config[key] = value

    scalar_overrides = {
        "num_epochs": args.epochs,
        "batch_size": args.batch_size,
        "num_frames": args.num_frames,
        "hidden_dim": args.hidden_dim,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "optimizer": args.optimizer,
        "num_workers": args.num_workers,
        "save_every": args.save_every,
    }
    for key, value in scalar_overrides.items():
        if value is not None:
            config[key] = value

    config["model_type"] = MODEL_TYPE
    config["optimizer"] = str(config.get("optimizer", "adamw")).lower()
    if config["optimizer"] not in OPTIMIZER_CHOICES:
        raise ValueError(
            f"Unsupported optimizer: {config['optimizer']}. Choose from {OPTIMIZER_CHOICES}."
        )



def print_training_info(config):
    print(f"Platform: {get_platform_name()}")
    print(f"Device: {config['device']}")
    print(f"Model Type: {config['model_type']}")
    print(f"Data Dir: {config['data_dir']}")
    print(f"Checkpoint Dir: {config['checkpoint_dir']}")
    print(f"Batch Size: {config['batch_size']}")
    print(f"Num Workers: {config['num_workers']}")
    print(f"Num Frames: {config['num_frames']}")
    print(f"Hidden Dim: {config['hidden_dim']}")
    print(f"Optimizer: {config['optimizer']}")
    print(f"Learning Rate: {config['learning_rate']}")
    print(f"Weight Decay: {config['weight_decay']}")


def create_dataloaders(config):
    data_dir = config["data_dir"]
    train_csv = os.path.join(data_dir, "Train.csv")
    val_csv = os.path.join(data_dir, "Validation.csv")
    train_root = os.path.join(data_dir, "Train")
    val_root = os.path.join(data_dir, "Validation")

    img_size = tuple(config.get("img_size", (100, 176)))
    normalize_mean = config.get("normalize_mean")
    normalize_std = config.get("normalize_std")

    train_transform = get_train_transform(img_size, normalize_mean, normalize_std)
    val_transform = get_val_transform(img_size, normalize_mean, normalize_std)

    train_dataset = JesterDataset(
        csv_file=train_csv,
        root_dir=train_root,
        num_frames=config["num_frames"],
        transform=train_transform,
        sampling_mode=SAMPLING_RANDOM,
        img_size=img_size,
    )
    val_dataset = JesterDataset(
        csv_file=val_csv,
        root_dir=val_root,
        num_frames=config["num_frames"],
        transform=val_transform,
        sampling_mode=SAMPLING_UNIFORM,
        img_size=img_size,
    )

    loader_kwargs = {
        "batch_size": config["batch_size"],
        "num_workers": config["num_workers"],
        "pin_memory": config.get("pin_memory", True),
    }
    if config["num_workers"] > 0:
        loader_kwargs["prefetch_factor"] = config.get("prefetch_factor", 2)

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    return train_loader, val_loader


def build_optimizer(config, model):
    lr = float(config["learning_rate"])
    wd = float(config["weight_decay"])
    params = filter(lambda p: p.requires_grad, model.parameters())

    optimizer_name = config["optimizer"]
    if optimizer_name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=wd)
    if optimizer_name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=wd)


def train_one_epoch(model, loader, criterion, optimizer, device, epoch, total_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for step, (inputs, labels, _) in enumerate(loader, start=1):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        predicted = outputs.argmax(dim=1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        if step % 10 == 0 or step == len(loader):
            acc = 100.0 * correct / max(total, 1)
            print(
                f"Epoch [{epoch + 1}/{total_epochs}] Step [{step}/{len(loader)}] "
                f"Loss: {loss.item():.4f} Acc: {acc:.2f}%"
            )

    avg_loss = running_loss / max(len(loader), 1)
    avg_acc = 100.0 * correct / max(total, 1)
    return avg_loss, avg_acc


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels, _ in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            predicted = outputs.argmax(dim=1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    avg_loss = running_loss / max(len(loader), 1)
    avg_acc = 100.0 * correct / max(total, 1)
    return avg_loss, avg_acc


def save_history_csv(history, csv_path):
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])
        for idx in range(len(history["train_losses"])):
            writer.writerow(
                [
                    idx + 1,
                    history["train_losses"][idx],
                    history["train_accs"][idx],
                    history["val_losses"][idx],
                    history["val_accs"][idx],
                ]
            )


def save_checkpoint(path, epoch, model, optimizer, scheduler, history, best_val_acc):
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "optimizer_type": optimizer.__class__.__name__,
        "scheduler_state_dict": scheduler.state_dict(),
        "train_losses": history["train_losses"],
        "train_accs": history["train_accs"],
        "val_losses": history["val_losses"],
        "val_accs": history["val_accs"],
        "best_val_acc": best_val_acc,
        "model_type": MODEL_TYPE,
    }
    torch.save(state, path)


def backup_resolved_config(config, checkpoint_dir):
    os.makedirs(checkpoint_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(checkpoint_dir, f"split_config_{timestamp}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        import json

        json.dump(config, f, ensure_ascii=False, indent=2)
    return output_path


def train_model():
    args = parse_args()
    config = get_config(args.config)
    apply_overrides(config, args)

    os.makedirs(config["checkpoint_dir"], exist_ok=True)
    backup_path = backup_resolved_config(config, config["checkpoint_dir"])

    print_training_info(config)
    print(f"Resolved config saved to: {backup_path}")

    train_loader, val_loader = create_dataloaders(config)

    model = build_model(config, device=config["device"])
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = build_optimizer(config, model)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["num_epochs"], eta_min=1e-5
    )

    history = {
        "train_losses": [],
        "train_accs": [],
        "val_losses": [],
        "val_accs": [],
    }
    best_val_acc = 0.0
    start_epoch = 0

    if args.resume is not None:
        resume_path = _abs_if_needed(args.resume)
        if not os.path.exists(resume_path):
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")

        checkpoint = torch.load(resume_path, map_location=config["device"])
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        history["train_losses"] = checkpoint.get("train_losses", [])
        history["train_accs"] = checkpoint.get("train_accs", [])
        history["val_losses"] = checkpoint.get("val_losses", [])
        history["val_accs"] = checkpoint.get("val_accs", [])
        best_val_acc = checkpoint.get("best_val_acc", 0.0)
        start_epoch = checkpoint.get("epoch", -1) + 1
        print(f"Resumed from {resume_path}, start epoch = {start_epoch}")

    for epoch in range(start_epoch, config["num_epochs"]):
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            config["device"],
            epoch,
            config["num_epochs"],
        )
        val_loss, val_acc = validate(model, val_loader, criterion, config["device"])

        history["train_losses"].append(train_loss)
        history["train_accs"].append(train_acc)
        history["val_losses"].append(val_loss)
        history["val_accs"].append(val_acc)

        print(
            f"Epoch [{epoch + 1}/{config['num_epochs']}] "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%"
        )

        best_val_acc = max(best_val_acc, val_acc)

        if config.get("save_every", 0) > 0 and (epoch + 1) % config["save_every"] == 0:
            intermediate_dir = os.path.join(
                config["checkpoint_dir"], "intermediate_checkpoints", f"epoch_{epoch + 1}"
            )
            os.makedirs(intermediate_dir, exist_ok=True)
            ckpt_path = os.path.join(
                intermediate_dir, f"model_{MODEL_TYPE}_epoch_{epoch + 1}.pth"
            )
            save_checkpoint(
                ckpt_path, epoch, model, optimizer, scheduler, history, best_val_acc
            )
            print(f"Intermediate checkpoint saved: {ckpt_path}")

        scheduler.step()

    final_model_path = os.path.join(config["checkpoint_dir"], f"model_{MODEL_TYPE}.pth")
    torch.save(model.state_dict(), final_model_path)

    final_history_path = os.path.join(
        config["checkpoint_dir"], f"training_history_{MODEL_TYPE}.csv"
    )
    save_history_csv(history, final_history_path)

    final_ckpt_path = os.path.join(config["checkpoint_dir"], f"checkpoint_{MODEL_TYPE}.pth")
    save_checkpoint(
        final_ckpt_path,
        config["num_epochs"] - 1,
        model,
        optimizer,
        scheduler,
        history,
        best_val_acc,
    )

    print("Training complete.")
    print(f"Final model saved to: {final_model_path}")
    print(f"Final checkpoint saved to: {final_ckpt_path}")
    print(f"Training history saved to: {final_history_path}")


if __name__ == "__main__":
    train_model()
