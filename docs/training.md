# Training Guide

## Quick Start

```bash
# 1. Prepare dataset
python scripts/prepare_dataset.py --source /path/to/ArASL

# 2. Start training
python training/train.py --config configs/train_config.yaml

# 3. Resume from checkpoint
python training/train.py --config configs/train_config.yaml --resume checkpoints/latest.pt
```

## Configuration

Edit `configs/train_config.yaml` to change hyperparameters. Key settings:

```yaml
training:
  learning_rate: 5.0e-5
  batch_size: 16
  num_epochs: 3
  weight_decay: 0.01

template:
  use_template_constraint: true   # Critical for accuracy

confidence:
  threshold: 0.85
```

## Ablation Studies

Reproduce the paper's ablation experiments:

```bash
python scripts/run_experiments.py --ablation
```

This runs three conditions:
1. BLIP without template constraint (baseline ~22%)
2. BLIP + template constraint (~92%)
3. Full model with confidence calibration (~94% effective)

## Hardware Requirements

| Mode | GPU Memory | RAM | Time (3 epochs) |
|---|---|---|---|
| Training (batch=16) | 16 GB | 32 GB | ~2 hours |
| Training (batch=8) | 10 GB | 16 GB | ~3 hours |
| Inference only | 8 GB | 8 GB | 47 ms/image |

## Checkpoints

Checkpoints are saved to `checkpoints/` after each epoch. The `latest.pt` symlink always points to the most recent checkpoint.

To convert a checkpoint to a Hugging Face model:

```python
from transformers import BlipForConditionalGeneration, BlipProcessor
import torch

ckpt = torch.load("checkpoints/epoch_3.pt")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
model.load_state_dict(ckpt["model"])
model.save_pretrained("models/arsl-blip")
```
