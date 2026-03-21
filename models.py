import torch
import torch.nn as nn
import torchvision.models as models

# --------------------------
# MARK: 基础模型 (ResNet18 -> 平均池化 -> 全连接层)
# --------------------------
class ResNetVideoModel(nn.Module):
    def __init__(self, num_classes, freeze_backbone=True):
        super(ResNetVideoModel, self).__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        if freeze_backbone:
            for param in resnet.parameters():
                param.requires_grad = False
            for param in resnet.layer4.parameters():
                param.requires_grad = True

        self.cnn = nn.Sequential(*list(resnet.children())[:-1])
        cnn_out_dim = 512 
        
        self.fc = nn.Linear(cnn_out_dim, num_classes)

    def forward(self, x):
        b, t, c, h, w = x.size()
        x = x.view(b * t, c, h, w)
        cnn_features = self.cnn(x) 
        cnn_features = cnn_features.view(b, t, -1) 
        # 取时间维度的平均值作为视频的整体特征
        features = cnn_features.mean(dim=1)
        out = self.fc(features)
        return out

# --------------------------
# MARK: 时序模型 (ResNet18 -> GRU -> 全连接层)
# --------------------------
class ResNetGRUVideoModel(nn.Module):
    def __init__(self, num_classes, hidden_dim, freeze_backbone=True):
        super(ResNetGRUVideoModel, self).__init__()
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