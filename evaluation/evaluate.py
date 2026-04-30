"""
Evaluation Script for ArSL-BLIP

Computes overall accuracy, per-class accuracy, effective accuracy,
and saves a detailed report.

Authors: Najiba Tagougui, Ansar Hani, Monji Kherallah
Year: 2026

Usage:
    python evaluation/evaluate.py --checkpoint models/arsl-blip --data data/splits/val
"""

import argparse
import json
from pathlib import Path
from PIL import Image
from tqdm import tqdm

from models.blip_arabic import ArabicSignRecognizer
from evaluation.metrics import compute_metrics


def evaluate(model_path: str, data_dir: str, threshold: float = 0.85) -> dict:
    recognizer = ArabicSignRecognizer(model_path=model_path, confidence_threshold=threshold)

    data = Path(data_dir)
    y_true, y_pred, accepted_mask = [], [], []

    for class_dir in sorted(data.iterdir()):
        if not class_dir.is_dir():
            continue
        label = class_dir.name.lower()
        for img_path in tqdm(list(class_dir.glob("*")), desc=label, leave=False):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            image = Image.open(img_path).convert("RGB")
            result = recognizer.predict(image)

            y_true.append(label)
            y_pred.append(result["letter"] or "")
            accepted_mask.append(result["accepted"])

    metrics = compute_metrics(y_true, y_pred, accepted_mask)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate ArSL-BLIP")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data", type=str, default="data/splits/val")
    parser.add_argument("--threshold", type=float, default=0.85)
    parser.add_argument("--output", type=str, default="results/evaluation.json")
    args = parser.parse_args()

    metrics = evaluate(args.checkpoint, args.data, args.threshold)

    print("\n=== Evaluation Results ===")
    print(f"  Overall Accuracy : {metrics['overall_accuracy']:.4f}")
    print(f"  Effective Accuracy (τ={args.threshold}): {metrics['effective_accuracy']:.4f}")
    print(f"  Coverage         : {metrics['coverage']:.4f}")
    print(f"  Macro F1         : {metrics['macro_f1']:.4f}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nFull report saved → {args.output}")


if __name__ == "__main__":
    main()
