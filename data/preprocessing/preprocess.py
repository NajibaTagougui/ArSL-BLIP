"""
Image Preprocessing Pipeline for ArASL Dataset

Authors: Najiba Tagougui, Ansar Hani, Monji Kherallah
Year: 2026
"""

import os
import argparse
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm


IMAGE_SIZE = (224, 224)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def preprocess_image(image_path: str, output_path: str, size: tuple = IMAGE_SIZE) -> bool:
    """Preprocess a single image: resize, convert to RGB, save."""
    try:
        img = Image.open(image_path)
        # Convert grayscale to RGB if needed
        if img.mode != "RGB":
            img = img.convert("RGB")
        img = img.resize(size, Image.LANCZOS)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        img.save(output_path)
        return True
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return False


def preprocess_dataset(source_dir: str, output_dir: str) -> None:
    """Preprocess the full ArASL dataset directory."""
    source = Path(source_dir)
    output = Path(output_dir)

    class_dirs = [d for d in source.iterdir() if d.is_dir()]
    print(f"Found {len(class_dirs)} class directories.")

    total, success = 0, 0
    for class_dir in tqdm(class_dirs, desc="Classes"):
        class_name = class_dir.name
        out_class_dir = output / class_name
        out_class_dir.mkdir(parents=True, exist_ok=True)

        for img_file in class_dir.glob("*"):
            if img_file.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                out_path = out_class_dir / img_file.name
                if preprocess_image(str(img_file), str(out_path)):
                    success += 1
                total += 1

    print(f"\nPreprocessed {success}/{total} images → {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess ArASL dataset")
    parser.add_argument("--source", type=str, required=True, help="Path to raw dataset")
    parser.add_argument("--output", type=str, default="data/processed", help="Output directory")
    parser.add_argument("--size", type=int, nargs=2, default=[224, 224], help="Target image size")
    args = parser.parse_args()

    preprocess_dataset(args.source, args.output)


if __name__ == "__main__":
    main()
