"""
Inference Script for ArSL-BLIP

Authors: Najiba Tagougui, Ansar Hani, Monji Kherallah
Year: 2026

Usage:
    python inference/predict.py --image path/to/sign.jpg
    python inference/predict.py --folder ./test_images/ --output results.json
    python inference/predict.py --interactive
"""

import argparse
import json
from pathlib import Path
from PIL import Image
from tqdm import tqdm

from models.blip_arabic import ArabicSignRecognizer


def predict_single(recognizer: ArabicSignRecognizer, image_path: str) -> dict:
    image = Image.open(image_path).convert("RGB")
    result = recognizer.predict(image)
    return result


def predict_folder(recognizer: ArabicSignRecognizer, folder: str) -> list:
    folder = Path(folder)
    extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    image_paths = [p for p in folder.rglob("*") if p.suffix.lower() in extensions]

    results = []
    for path in tqdm(image_paths, desc="Processing images"):
        result = predict_single(recognizer, str(path))
        result["file"] = str(path)
        results.append(result)
    return results


def interactive_mode(recognizer: ArabicSignRecognizer) -> None:
    print("ArSL-BLIP Interactive Mode — type 'quit' to exit")
    while True:
        path = input("\nEnter image path: ").strip()
        if path.lower() in {"quit", "exit", "q"}:
            break
        if not Path(path).exists():
            print(f"  File not found: {path}")
            continue
        result = predict_single(recognizer, path)
        print(f"  Letter    : {result['letter']}")
        print(f"  Arabic    : {result['arabic_char']}")
        print(f"  Confidence: {result['confidence']:.2%}")
        print(f"  Status    : {'ACCEPTED' if result['accepted'] else 'REJECTED (low confidence)'}")


def main():
    parser = argparse.ArgumentParser(description="ArSL-BLIP inference")
    parser.add_argument("--checkpoint", type=str, default="models/arsl-blip")
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--folder", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--threshold", type=float, default=0.85)
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args()

    recognizer = ArabicSignRecognizer(
        model_path=args.checkpoint,
        confidence_threshold=args.threshold,
    )

    if args.interactive:
        interactive_mode(recognizer)

    elif args.image:
        result = predict_single(recognizer, args.image)
        print(f"Letter    : {result['letter']}")
        print(f"Arabic    : {result['arabic_char']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"Status    : {'ACCEPTED' if result['accepted'] else 'REJECTED'}")

    elif args.folder:
        results = predict_folder(recognizer, args.folder)
        if args.output:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"Results saved to {args.output}")
        else:
            for r in results:
                print(f"{r['file']}: {r['letter']} ({r['confidence']:.2%})")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
