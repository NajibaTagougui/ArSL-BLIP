"""
Fine-tuning Script for ArSL-BLIP

Fine-tunes Salesforce/blip-image-captioning-base on the ArASL dataset
using template-constrained generation.

Authors: Najiba Tagougui, Ansar Hani, Monji Kherallah
Year: 2026

Usage:
    python training/train.py --config configs/train_config.yaml
    python training/train.py --config configs/train_config.yaml --resume checkpoints/latest.pt
"""

import os
import argparse
import yaml
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
from pathlib import Path
from tqdm import tqdm

from training.optimizer import build_optimizer
from training.scheduler import build_scheduler


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

TEMPLATE = "arabic sign language letter"


class ArASLDataset(Dataset):
    """ArASL captioning dataset for BLIP fine-tuning."""

    def __init__(self, data_dir: str, processor: BlipProcessor, split: str = "train"):
        self.processor = processor
        self.samples = []

        data_path = Path(data_dir) / split
        for class_dir in sorted(data_path.iterdir()):
            if not class_dir.is_dir():
                continue
            label = class_dir.name.lower()
            caption = f"{TEMPLATE} {label}"
            for img_file in class_dir.glob("*"):
                if img_file.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    self.samples.append((str(img_file), caption))

        print(f"[{split}] {len(self.samples)} samples across "
              f"{len(set(s[1] for s in self.samples))} classes")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, caption = self.samples[idx]
        image = Image.open(img_path).convert("RGB").resize((224, 224))
        encoding = self.processor(
            images=image,
            text=caption,
            return_tensors="pt",
            padding="max_length",
            max_length=50,
            truncation=True,
        )
        return {k: v.squeeze(0) for k, v in encoding.items()}


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(config: dict, resume: str = None) -> None:
    device = torch.device(config["hardware"]["device"] if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    # Load model
    model_name = config["model"]["name"]
    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name).to(device)

    # Datasets
    train_ds = ArASLDataset(config["data"]["dataset_path"], processor, "train")
    val_ds = ArASLDataset(config["data"]["dataset_path"], processor, "val")

    train_loader = DataLoader(
        train_ds,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["data"]["num_workers"],
        pin_memory=config["data"]["pin_memory"],
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["data"]["num_workers"],
    )

    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config, len(train_loader))

    start_epoch = 0
    if resume:
        ckpt = torch.load(resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from epoch {start_epoch}")

    output_dir = Path(config["logging"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(start_epoch, config["training"]["num_epochs"]):
        model.train()
        total_loss = 0.0

        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            batch = {k: v.to(device) for k, v in batch.items()}

            labels = batch["input_ids"].clone()
            labels[labels == processor.tokenizer.pad_token_id] = -100

            outputs = model(
                pixel_values=batch["pixel_values"],
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=labels,
            )
            loss = outputs.loss

            loss.backward()

            if (step + 1) % config["training"]["gradient_accumulation_steps"] == 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config["training"]["gradient_clip"]
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            total_loss += loss.item()

            if (step + 1) % config["logging"]["log_every"] == 0:
                avg = total_loss / (step + 1)
                print(f"  step {step+1} | loss {avg:.4f} | lr {scheduler.get_last_lr()[0]:.2e}")

        # Validation
        val_loss = _validate(model, val_loader, processor, device)
        print(f"Epoch {epoch+1} | train_loss={total_loss/len(train_loader):.4f} | val_loss={val_loss:.4f}")

        # Save checkpoint
        ckpt_path = output_dir / f"epoch_{epoch+1}.pt"
        torch.save({"epoch": epoch, "model": model.state_dict(),
                    "optimizer": optimizer.state_dict()}, ckpt_path)
        # Symlink latest
        latest = output_dir / "latest.pt"
        if latest.exists():
            latest.unlink()
        latest.symlink_to(ckpt_path.name)
        print(f"Saved checkpoint → {ckpt_path}")

    # Save final model
    save_path = output_dir / "arsl-blip-final"
    model.save_pretrained(save_path)
    processor.save_pretrained(save_path)
    print(f"Final model saved to {save_path}")


@torch.no_grad()
def _validate(model, loader, processor, device) -> float:
    model.eval()
    total_loss = 0.0
    for batch in tqdm(loader, desc="Validating", leave=False):
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100
        outputs = model(
            pixel_values=batch["pixel_values"],
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=labels,
        )
        total_loss += outputs.loss.item()
    return total_loss / len(loader)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fine-tune BLIP for ArSL")
    parser.add_argument("--config", type=str, default="configs/train_config.yaml")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    train(config, args.resume)


if __name__ == "__main__":
    main()
