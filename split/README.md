# split: UltraLightParallelMEGRUModel standalone

This directory is a minimal standalone migration for training and inference of UltraLightParallelMEGRUModel.

## Scope
- Only includes code required by UltraLightParallelMEGRUModel.
- No imports from repository-level modules outside split/.
- Dataset files are reused in ../dataset by default and are not migrated.

## Files
- config.json: standalone training/inference config
- config_loader.py: config merge and path resolution
- modules.py: temporal_shift, TSMResBlock, ParallelMETSMResBlock
- models.py: UltraLightParallelMEGRUModel + model build/load helpers
- dataset.py: JesterDataset + frame sampling + transforms
- train.py: standalone training entrypoint
- inference.py: standalone inference entrypoint

## Train
From repository root:

```bash
python split/train.py
```

Optional overrides:

```bash
python split/train.py \
  --config split/config.json \
  --data_dir dataset \
  --checkpoint_dir checkpoint/split_run \
  --epochs 20 \
  --batch_size 64
```

## Inference
Dataset inference:

```bash
python split/inference.py \
  --model_weight checkpoint/final_2/model_ultralight_parallel_me_gru.pth \
  --csv_path dataset/Test.csv \
  --root_dir dataset/Test \
  --output results/split/test
```

Single-video inference:

```bash
python split/inference.py \
  --model_weight checkpoint/final_2/model_ultralight_parallel_me_gru.pth \
  --video_path dataset/Test/100010 \
  --output results/split/single
```
