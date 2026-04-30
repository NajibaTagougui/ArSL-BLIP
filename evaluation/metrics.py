"""
Metrics for ArSL-BLIP Evaluation

Authors: Najiba Tagougui, Ansar Hani, Monji Kherallah
Year: 2026
"""

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
import numpy as np
from typing import List


def compute_metrics(
    y_true: List[str],
    y_pred: List[str],
    accepted_mask: List[bool],
) -> dict:
    """Compute full evaluation metrics.

    Args:
        y_true: Ground-truth class labels.
        y_pred: Predicted class labels.
        accepted_mask: Boolean mask — True when confidence >= threshold.

    Returns:
        Dictionary of metric names → values.
    """
    overall_acc = accuracy_score(y_true, y_pred)

    # Effective accuracy: only over accepted predictions
    acc_indices = [i for i, a in enumerate(accepted_mask) if a]
    if acc_indices:
        eff_acc = accuracy_score(
            [y_true[i] for i in acc_indices],
            [y_pred[i] for i in acc_indices],
        )
    else:
        eff_acc = 0.0
    coverage = len(acc_indices) / len(y_true) if y_true else 0.0

    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    per_class_report = classification_report(y_true, y_pred, zero_division=0, output_dict=True)

    return {
        "overall_accuracy": float(overall_acc),
        "effective_accuracy": float(eff_acc),
        "coverage": float(coverage),
        "macro_f1": float(macro_f1),
        "per_class_report": per_class_report,
        "num_samples": len(y_true),
        "num_accepted": len(acc_indices),
    }
