"""Minimal CPU/CUDA smoke check; no dataset or pretrained download required."""
import torch
from utils import build_model
from train import parse_args


def main():
    torch.set_num_threads(2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    names = ['LightTMF3GRU_SwapPlacement', 'LightTMF3GRU_ShortOnly',
             'LightTMF3GRU_NoSpatial', 'LightTMF3_NoGRU']
    for name in names:
        for classes in (27, 174):
            config = dict(num_classes=classes, num_frames=4, hidden_dim=256, device=str(device))
            model = build_model(name, config, device=device)
            logits = model(torch.randn(1, 4, 3, 32, 32, device=device))
            assert logits.shape == (1, classes)
            loss = torch.nn.functional.cross_entropy(logits, torch.zeros(1, dtype=torch.long, device=device))
            loss.backward()
            assert torch.isfinite(loss)
        args = parse_args(['--model_type', name, '--dataset', 'ssv2', '--img_size', '224',
                           '--epochs', '1', '--num_workers', '0', '--hidden_dim', '256'])
        assert args.model_type == name
        print(f'PASS {name}: forward/backward, 27/174 classes, CLI')
    model.eval()
    with torch.no_grad():
        assert model(torch.randn(1, 16, 3, 224, 224, device=device)).shape == (1, 174)
    print(f'PASS 224x224 / 16 frames on {device}')


if __name__ == '__main__':
    main()
