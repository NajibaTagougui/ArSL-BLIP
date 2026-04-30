"""
Learning Rate Scheduler for ArSL-BLIP training.

Authors: Najiba Tagougui, Ansar Hani, Monji Kherallah
Year: 2026
"""

import torch
from transformers import get_cosine_schedule_with_warmup


def build_scheduler(optimizer, config: dict, steps_per_epoch: int):
    """Build cosine LR scheduler with linear warmup."""
    num_epochs = config["training"]["num_epochs"]
    warmup_steps = config["training"].get("warmup_steps", 100)
    total_steps = steps_per_epoch * num_epochs

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    return scheduler
