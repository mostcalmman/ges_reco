import os
import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.models.video as video_models
from PIL import Image

# ==========================================
# 1. 自定义数据集类 (Dataset)
# ==========================================
class GestureDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, is_test=False):
        """
        :param csv_file: CSV文件路径
        :param root_dir: 包含图片的文件夹路径 (如 dataset/Train)
        :param transform: 图像预处理
        :param is_test: 是否为测试集 (测试集没有label_id)
        """
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        # 获取视频对应的文件夹名 (video_id)
        video_id = str(self.data_frame.iloc[idx]['video_id'])
        video_dir = os.path.join(self.root_dir, video_id)
        
        # 读取文件夹下所有的图片并排序，确保时间顺序正确
        image_files = sorted([f for f in os.listdir(video_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
        
        frames = []
        # 提取这37帧
        for img_name in image_files:
            img_path = os.path.join(video_dir, img_name)
            # 使用 RGB 模式打开，确保通道数为3
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            frames.append(image)
            
        # 将 list of tensors (37个 [C, H, W]) 堆叠成一个 tensor: [T, C, H, W]
        # T=37, C=3
        video_tensor = torch.stack(frames)
        
        # PyTorch 的 3D CNN 要求输入维度是 [C, T, H, W]，因此需要调整维度顺序
        video_tensor = video_tensor.permute(1, 0, 2, 3)
        
        if self.is_test:
            return video_tensor, video_id
        else:
            # 训练和验证集有 label_id
            label = int(self.data_frame.iloc[idx]['label_id'])
            return video_tensor, label

# ==========================================
# 2. 训练与验证逻辑
# ==========================================
def main(args):
    # 路径
    DATASET_DIR = args.dataset
    TRAIN_CSV = os.path.join(DATASET_DIR, "Train.csv")
    VAL_CSV = os.path.join(DATASET_DIR, "Validation.csv")
    TEST_CSV = os.path.join(DATASET_DIR, "Test.csv")
    
    TRAIN_DIR = os.path.join(DATASET_DIR, "Train")
    VAL_DIR = os.path.join(DATASET_DIR, "Validation")
    TEST_DIR = os.path.join(DATASET_DIR, "Test")

    # 超参数
    BATCH_SIZE = 8
    EPOCHS = 10
    LEARNING_RATE = 0.001
    NUM_CLASSES = 27

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # 定义图像预处理 (Resizing 和 归一化)
    transform = transforms.Compose([
        transforms.Resize((112, 112)),  # R3D模型标准输入分辨率
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989])
    ])

    # 实例化 Dataset 和 DataLoader
    train_dataset = GestureDataset(TRAIN_CSV, TRAIN_DIR, transform=transform)
    val_dataset = GestureDataset(VAL_CSV, VAL_DIR, transform=transform)
    test_dataset = GestureDataset(TEST_CSV, TEST_DIR, transform=transform, is_test=True)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # 初始化 3D ResNet18 模型
    model = video_models.r3d_18(weights=video_models.R3D_18_Weights.DEFAULT)
    # 替换全连接层，以适应我们的分类数量
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
    model = model.to(device)

    # 损失函数与优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # ==========================================
    # 3. 训练和验证循环
    # ==========================================
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        # 训练阶段
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
        train_acc = 100 * correct_train / total_train
        
        # 验证阶段
        model.eval()
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
                
        val_acc = 100 * correct_val / total_val
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

    print("Training completed. Starting test prediction...")

    # ==========================================
    # 4. 测试集预测 (因为测试集没有Label，只输出预测结果)
    # ==========================================
    model.eval()
    test_results = []
    
    with torch.no_grad():
        for inputs, video_ids in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            for vid, pred in zip(video_ids, predicted):
                test_results.append({'video_id': vid, 'predicted_label_id': pred.item()})
                
    # 将预测结果保存为 CSV
    results_df = pd.DataFrame(test_results)
    results_df.to_csv("test_predictions.csv", index=False)
    print("Test prediction completed. Results saved in 'test_predictions.csv'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3D ResNet18 Gesture Recognition")
    parser.add_argument("--dataset", default="dataset", help="训练集路径")
    args = parser.parse_args()
    main(args)