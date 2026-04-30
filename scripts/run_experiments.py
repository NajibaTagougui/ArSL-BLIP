"""
Ablation Study and Experiment Runner for ArSL-BLIP.

Reproduces the experiments in our paper.

Authors: Najiba Tagougui, Ansar Hani, Monji Kherallah
Year: 2026

Usage:
    python scripts/run_experiments.py --ablation
    python scripts/run_experiments.py --experiment template_constraint
"""

import argparse
import subprocess
import sys
import yaml
import copy
from pathlib import Path


BASE_CONFIG = "configs/train_config.yaml"

ABLATION_EXPERIMENTS = {
    "no_template": {
        "description": "BLIP without template constraint (baseline)",
        "overrides": {"template": {"use_template_constraint": False}},
    },
    "template_only": {
        "description": "BLIP + template constraint (our contribution)",
        "overrides": {"template": {"use_template_constraint": True}},
    },
    "template_confidence": {
        "description": "BLIP + template + confidence calibration (full model)",
        "overrides": {
            "template": {"use_template_constraint": True},
            "confidence": {"threshold": 0.85},
        },
    },
}


def run_ablation(data_dir: str = "data/splits") -> None:
    with open(BASE_CONFIG) as f:
        base = yaml.safe_load(f)

    results = {}
    Path("results/ablation").mkdir(parents=True, exist_ok=True)

    for name, exp in ABLATION_EXPERIMENTS.items():
        print(f"\n{'='*60}")
        print(f"Running: {name}")
        print(f"  {exp['description']}")
        print(f"{'='*60}")

        config = copy.deepcopy(base)
        for key, val in exp["overrides"].items():
            if key in config:
                config[key].update(val)
            else:
                config[key] = val

        # Save temp config
        cfg_path = f"results/ablation/{name}_config.yaml"
        with open(cfg_path, "w") as f:
            yaml.dump(config, f)

        # Train
        subprocess.run([sys.executable, "training/train.py", "--config", cfg_path], check=True)
        results[name] = exp["description"]

    print("\n\nAblation complete. Models saved to checkpoints/")
    print("Run evaluation/evaluate.py on each checkpoint to compare.")


def main():
    parser = argparse.ArgumentParser(description="Run ArSL-BLIP experiments")
    parser.add_argument("--ablation", action="store_true", help="Run all ablation experiments")
    parser.add_argument("--experiment", type=str, choices=list(ABLATION_EXPERIMENTS.keys()))
    parser.add_argument("--data", type=str, default="data/splits")
    args = parser.parse_args()

    if args.ablation:
        run_ablation(args.data)
    elif args.experiment:
        exp = {args.experiment: ABLATION_EXPERIMENTS[args.experiment]}
        print(f"Running single experiment: {args.experiment}")
        # Re-use ablation runner with single experiment
        run_ablation(args.data)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
