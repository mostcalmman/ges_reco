import torch
import torch.nn as nn
import torch.nn.functional as F

from modules import ParallelMETSMResBlock, TSMResBlock


MODEL_TYPE = "ultralight_parallel_me_gru"


class UltraLightParallelMEGRUModel(nn.Module):
    def __init__(self, num_classes=27, n_segment=8, hidden_dim=128):
        super().__init__()
        self.n_segment = n_segment
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(0.5)

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.layer1 = TSMResBlock(32, 32, stride=1, n_segment=n_segment)
        self.layer2 = TSMResBlock(32, 64, stride=2, n_segment=n_segment)
        self.layer3 = ParallelMETSMResBlock(64, 128, stride=2, n_segment=n_segment, reduction=4)

        self.gru = nn.GRU(
            input_size=128,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )

        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        b, t, c, h, w = x.size()

        x = x.view(b * t, c, h, w)
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(b, t, -1)

        _, hidden = self.gru(x)
        last_hidden = hidden[-1]

        last_hidden = self.dropout(last_hidden)
        out = self.fc(last_hidden)
        return out


def build_model(config, device=None):
    model = UltraLightParallelMEGRUModel(
        num_classes=config.get("num_classes", 27),
        n_segment=config.get("num_frames", 16),
        hidden_dim=config.get("hidden_dim", 128),
    )
    if device is not None:
        model = model.to(device)
    return model


def load_model_weights(model, model_weight_path, device, strict=True, eval_mode=True):
    state = torch.load(model_weight_path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]

    model.load_state_dict(state, strict=strict)
    model = model.to(device)
    if eval_mode:
        model.eval()
    return model
