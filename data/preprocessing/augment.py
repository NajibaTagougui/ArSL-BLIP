"""
Data Augmentation Utilities for ArASL

Authors: Najiba Tagougui, Ansar Hani, Monji Kherallah
Year: 2026
"""

import os
import random
from pathlib import Path
from PIL import Image, ImageEnhance, ImageOps
import numpy as np
from tqdm import tqdm


def augment_image(image: Image.Image) -> list[Image.Image]:
    """Apply augmentations and return a list of augmented variants."""
    augmented = []

    # Horizontal flip
    augmented.append(ImageOps.mirror(image))

    # Rotation
    for angle in [-10, 10, 15, -15]:
        augmented.append(image.rotate(angle, fillcolor=(255, 255, 255)))

    # Brightness
    for factor in [0.8, 1.2]:
        enhancer = ImageEnhance.Brightness(image)
        augmented.append(enhancer.enhance(factor))

    # Contrast
    for factor in [0.9, 1.1]:
        enhancer = ImageEnhance.Contrast(image)
        augmented.append(enhancer.enhance(factor))

    return augmented


def augment_dataset(data_dir: str, output_dir: str) -> None:
    """Augment the full dataset and save to output_dir."""
    data = Path(data_dir)
    output = Path(output_dir)

    class_dirs = [d for d in data.iterdir() if d.is_dir()]
    print(f"Augmenting {len(class_dirs)} classes...")

    for class_dir in tqdm(class_dirs, desc="Augmenting"):
        class_name = class_dir.name
        out_class = output / class_name
        out_class.mkdir(parents=True, exist_ok=True)

        for img_path in class_dir.glob("*"):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            img = Image.open(img_path).convert("RGB")
            # Save original
            img.save(out_class / img_path.name)
            # Save augmented
            for i, aug_img in enumerate(augment_image(img)):
                stem = img_path.stem
                aug_img.save(out_class / f"{stem}_aug{i}{img_path.suffix}")

    print(f"Augmented dataset saved to {output_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Augment ArASL dataset")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--output", type=str, default="data/augmented")
    args = parser.parse_args()
    augment_dataset(args.data, args.output)
