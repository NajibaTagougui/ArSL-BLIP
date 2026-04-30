"""
Train/Validation Dataset Splitting for ArASL

Authors: Najiba Tagougui, Ansar Hani, Monji Kherallah
Year: 2026
"""

import os
import shutil
import argparse
import random
from pathlib import Path
from tqdm import tqdm


def split_dataset(
    data_dir: str,
    output_dir: str,
    train_ratio: float = 0.85,
    val_ratio: float = 0.15,
    seed: int = 42,
    images_per_class: int = 200,
) -> None:
    """Split preprocessed dataset into train/val sets."""
    random.seed(seed)
    data = Path(data_dir)
    output = Path(output_dir)

    train_dir = output / "train"
    val_dir = output / "val"

    class_dirs = [d for d in data.iterdir() if d.is_dir()]
    print(f"Splitting {len(class_dirs)} classes (train={train_ratio}, val={val_ratio})")

    for class_dir in tqdm(class_dirs, desc="Splitting"):
        class_name = class_dir.name
        images = list(class_dir.glob("*"))
        images = [f for f in images if f.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        random.shuffle(images)

        # Sample up to images_per_class
        images = images[:images_per_class]

        n_train = int(len(images) * train_ratio)
        train_imgs = images[:n_train]
        val_imgs = images[n_train:]

        for subset, imgs in [("train", train_imgs), ("val", val_imgs)]:
            dest = output / subset / class_name
            dest.mkdir(parents=True, exist_ok=True)
            for img in imgs:
                shutil.copy2(img, dest / img.name)

    print(f"\nDataset split saved to {output_dir}")
    _print_stats(output)


def _print_stats(output_dir: Path) -> None:
    for split in ["train", "val"]:
        split_dir = output_dir / split
        if not split_dir.exists():
            continue
        total = sum(len(list(c.glob("*"))) for c in split_dir.iterdir() if c.is_dir())
        classes = len(list(split_dir.iterdir()))
        print(f"  {split}: {total} images across {classes} classes")


def main():
    parser = argparse.ArgumentParser(description="Split ArASL dataset into train/val")
    parser.add_argument("--data", type=str, required=True, help="Preprocessed data directory")
    parser.add_argument("--output", type=str, default="data/splits", help="Output directory")
    parser.add_argument("--train", type=float, default=0.85, help="Training split ratio")
    parser.add_argument("--val", type=float, default=0.15, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--per_class", type=int, default=200, help="Max images per class")
    args = parser.parse_args()

    split_dataset(args.data, args.output, args.train, args.val, args.seed, args.per_class)


if __name__ == "__main__":
    main()
