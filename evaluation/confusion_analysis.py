"""
Confusion Matrix Analysis for ArSL-BLIP

Authors: Najiba Tagougui, Ansar Hani, Monji Kherallah
Year: 2026
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from pathlib import Path


def plot_confusion_matrix(y_true, y_pred, output_path="results/confusion_matrix.png", figsize=(20, 18)):
    labels = sorted(set(y_true))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax, linewidths=0.5)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title("Normalized Confusion Matrix - ArSL-BLIP", fontsize=14, fontweight="bold")
    plt.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix saved -> {output_path}")

    print("\nTop-5 most confused pairs:")
    np.fill_diagonal(cm_norm, 0)
    indices = np.unravel_index(np.argsort(cm_norm, axis=None)[-5:][::-1], cm_norm.shape)
    for r, c in zip(*indices):
        print(f"  True={labels[r]} -> Pred={labels[c]} : {cm_norm[r,c]:.2%}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--true", type=str, required=True)
    parser.add_argument("--pred", type=str, required=True)
    parser.add_argument("--output", type=str, default="results/confusion_matrix.png")
    args = parser.parse_args()

    with open(args.true) as f:
        y_true = json.load(f)
    with open(args.pred) as f:
        y_pred = json.load(f)

    plot_confusion_matrix(y_true, y_pred, args.output)


if __name__ == "__main__":
    main()
