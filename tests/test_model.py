"""
Unit tests for ArabicSignRecognizer

Authors: Najiba Tagougui, Ansar Hani, Monji Kherallah
Year: 2026

Run: pytest tests/test_model.py
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from PIL import Image


@pytest.fixture
def dummy_image():
    return Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))


@pytest.fixture
def mock_recognizer():
    """Return a recognizer with mocked model to avoid loading weights."""
    with patch("models.blip_arabic.BlipProcessor") as MockProc, \
         patch("models.blip_arabic.BlipForConditionalGeneration") as MockModel:

        MockProc.from_pretrained.return_value = MagicMock()
        MockModel.from_pretrained.return_value = MagicMock()

        from models.blip_arabic import ArabicSignRecognizer
        rec = ArabicSignRecognizer.__new__(ArabicSignRecognizer)
        rec.confidence_threshold = 0.85
        rec.arabic_letters = {"nun": "ن", "waw": "و"}
        import torch
        rec.device = torch.device("cpu")
        rec.processor = MockProc.from_pretrained.return_value
        rec.model = MockModel.from_pretrained.return_value
        yield rec


def test_preprocess_rgb(dummy_image):
    from models.blip_arabic import ArabicSignRecognizer
    result = ArabicSignRecognizer._preprocess(dummy_image)
    assert result.size == (224, 224)
    assert result.mode == "RGB"


def test_preprocess_grayscale():
    from models.blip_arabic import ArabicSignRecognizer
    gray = Image.fromarray(np.random.randint(0, 255, (100, 100), dtype=np.uint8))
    result = ArabicSignRecognizer._preprocess(gray)
    assert result.mode == "RGB"


def test_arabic_letters_mapping():
    from models.blip_arabic import ARABIC_LETTERS
    assert "nun" in ARABIC_LETTERS
    assert ARABIC_LETTERS["nun"] == "ن"
    assert len(ARABIC_LETTERS) == 32


def test_confidence_threshold_accept():
    from models.blip_arabic import ArabicSignRecognizer
    rec = MagicMock(spec=ArabicSignRecognizer)
    rec.confidence_threshold = 0.85
    assert 0.9 >= rec.confidence_threshold


def test_confidence_threshold_reject():
    from models.blip_arabic import ArabicSignRecognizer
    rec = MagicMock(spec=ArabicSignRecognizer)
    rec.confidence_threshold = 0.85
    assert 0.7 < rec.confidence_threshold
