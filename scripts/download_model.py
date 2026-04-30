"""
Download fine-tuned ArSL-BLIP model weights.

Authors: Najiba Tagougui, Ansar Hani, Monji Kherallah
Year: 2026

Usage:
    python scripts/download_model.py
    python scripts/download_model.py --output models/my-arsl-blip
"""

import argparse
from pathlib import Path


HF_MODEL_ID = "NajibaTagougui/arsl-blip"  # Will be published upon acceptance


def download_from_huggingface(model_id: str, output_dir: str) -> None:
    try:
        from transformers import BlipProcessor, BlipForConditionalGeneration
    except ImportError:
        raise RuntimeError("Install transformers: pip install transformers")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    print(f"Downloading '{model_id}' from Hugging Face ...")
    processor = BlipProcessor.from_pretrained(model_id)
    model = BlipForConditionalGeneration.from_pretrained(model_id)

    processor.save_pretrained(output)
    model.save_pretrained(output)
    print(f"Model saved to {output}")


def main():
    parser = argparse.ArgumentParser(description="Download ArSL-BLIP model")
    parser.add_argument("--model", type=str, default=HF_MODEL_ID)
    parser.add_argument("--output", type=str, default="models/arsl-blip")
    args = parser.parse_args()

    print("NOTE: The Hugging Face model will be published upon paper acceptance.")
    print(f"Attempting to download from: {args.model}")
    download_from_huggingface(args.model, args.output)


if __name__ == "__main__":
    main()
