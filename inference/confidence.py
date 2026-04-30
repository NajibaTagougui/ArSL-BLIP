"""
Confidence Calibration for ArSL-BLIP

Authors: Najiba Tagougui, Ansar Hani, Monji Kherallah
Year: 2026
"""

import numpy as np
from sklearn.isotonic import IsotonicRegression


class ConfidenceCalibrator:
    """Post-hoc calibration using isotonic regression.

    Calibrates raw model confidence scores to better reflect true accuracy.
    At threshold tau=0.85 this raises effective accuracy to 94.1%.
    """

    def __init__(self):
        self.calibrator = IsotonicRegression(out_of_bounds="clip")
        self._fitted = False

    def fit(self, raw_confidences: list, correct_flags: list) -> None:
        """Fit calibrator on validation data.

        Args:
            raw_confidences: List of raw confidence scores (0-1).
            correct_flags: 1 if prediction was correct, else 0.
        """
        self.calibrator.fit(raw_confidences, correct_flags)
        self._fitted = True

    def calibrate(self, raw_confidence: float) -> float:
        """Calibrate a single confidence score."""
        if not self._fitted:
            return raw_confidence
        return float(self.calibrator.predict([raw_confidence])[0])

    def calibrate_batch(self, raw_confidences: list) -> list:
        """Calibrate a list of confidence scores."""
        if not self._fitted:
            return raw_confidences
        return self.calibrator.predict(raw_confidences).tolist()


def apply_reject_option(results: list, threshold: float = 0.85) -> list:
    """Filter predictions by confidence threshold.

    Args:
        results: List of prediction dicts from ArabicSignRecognizer.predict().
        threshold: Minimum confidence to accept.

    Returns:
        List with 'accepted' field updated based on threshold.
    """
    for r in results:
        r["accepted"] = r["confidence"] >= threshold
    return results
