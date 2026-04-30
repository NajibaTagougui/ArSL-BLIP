"""
End-to-End Dataset Preparation Script for ArASL.

Runs: preprocess -> augment (optional) -> split

Authors: Najiba Tagougui, Ansar Hani, Monji Kherallah
Year: 2026

Usage:
    python scripts/prepare_dataset.py --source /path/to/ArASL
    python scripts/prepare_dataset.py --source /path/to/ArASL --augment
"""

import argparse
import subprocess
import sys


def run(cmd: list) -> None:
    print(f"\n>>> {' '.join(cmd)}")
    result = subprocess.run(cmd, check=True)
    if result.returncode != 0:
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(description="Prepare ArASL dataset")
    parser.add_argument("--source", type=str, required=True, help="Raw ArASL dataset folder")
    parser.add_argument("--output", type=str, default="data/processed")
    parser.add_argument("--splits", type=str, default="data/splits")
    parser.add_argument("--augment", action="store_true", help="Apply augmentation")
    parser.add_argument("--per_class", type=int, default=200)
    args = parser.parse_args()

    # Step 1: Preprocess
    run([sys.executable, "data/preprocessing/preprocess.py",
         "--source", args.source, "--output", args.output])

    # Step 2: Augment (optional)
    if args.augment:
        run([sys.executable, "data/preprocessing/augment.py",
             "--data", args.output, "--output", args.output + "_augmented"])
        prepped = args.output + "_augmented"
    else:
        prepped = args.output

    # Step 3: Split
    run([sys.executable, "data/preprocessing/dataset_split.py",
         "--data", prepped, "--output", args.splits,
         "--per_class", str(args.per_class)])

    print(f"\nDataset ready at: {args.splits}")


if __name__ == "__main__":
    main()
