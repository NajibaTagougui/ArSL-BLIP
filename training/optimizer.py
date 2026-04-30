"""
Optimizer builder for ArSL-BLIP training.

Authors: Najiba Tagougui, Ansar Hani, Monji Kherallah
Year: 2026
"""

import torch
from torch.optim import AdamW


def build_optimizer(model, config: dict) -> torch.optim.Optimizer:
    """Build AdamW optimizer with weight-decay skip for bias/norm params."""
    no_decay = {"bias", "LayerNorm.weight", "layer_norm.weight"}

    params = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": config["training"]["weight_decay"],
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    opt_cfg = config.get("optimizer", {})
    optimizer = AdamW(
        params,
        lr=config["training"]["learning_rate"],
        betas=tuple(opt_cfg.get("betas", [0.9, 0.999])),
        eps=float(opt_cfg.get("eps", 1e-8)),
    )
    return optimizer
