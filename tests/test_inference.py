"""
Unit tests for inference utilities

Authors: Najiba Tagougui, Ansar Hani, Monji Kherallah
Year: 2026

Run: pytest tests/test_inference.py
"""

import pytest
from inference.confidence import apply_reject_option, ConfidenceCalibrator


def test_reject_option_accept():
    results = [{"confidence": 0.92, "accepted": False}]
    out = apply_reject_option(results, threshold=0.85)
    assert out[0]["accepted"] is True


def test_reject_option_reject():
    results = [{"confidence": 0.70, "accepted": True}]
    out = apply_reject_option(results, threshold=0.85)
    assert out[0]["accepted"] is False


def test_reject_option_boundary():
    results = [{"confidence": 0.85, "accepted": False}]
    out = apply_reject_option(results, threshold=0.85)
    assert out[0]["accepted"] is True


def test_calibrator_unfitted():
    cal = ConfidenceCalibrator()
    score = cal.calibrate(0.9)
    assert score == 0.9  # Returns raw if not fitted


def test_calibrator_fitted():
    cal = ConfidenceCalibrator()
    cal.fit([0.6, 0.7, 0.8, 0.9], [0, 0, 1, 1])
    calibrated = cal.calibrate(0.75)
    assert 0.0 <= calibrated <= 1.0


def test_calibrator_batch():
    cal = ConfidenceCalibrator()
    cal.fit([0.5, 0.8, 0.9], [0, 1, 1])
    results = cal.calibrate_batch([0.6, 0.85])
    assert len(results) == 2
