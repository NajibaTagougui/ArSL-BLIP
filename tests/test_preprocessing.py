"""
Unit tests for preprocessing pipeline

Authors: Najiba Tagougui, Ansar Hani, Monji Kherallah
Year: 2026

Run: pytest tests/test_preprocessing.py
"""

import pytest
import tempfile
import os
import numpy as np
from PIL import Image
from pathlib import Path

from data.preprocessing.preprocess import preprocess_image
from data.preprocessing.augment import augment_image


@pytest.fixture
def sample_image_path(tmp_path):
    img = Image.fromarray(np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8))
    path = tmp_path / "test.jpg"
    img.save(path)
    return str(path)


def test_preprocess_image_size(sample_image_path, tmp_path):
    out = str(tmp_path / "out.jpg")
    result = preprocess_image(sample_image_path, out)
    assert result is True
    processed = Image.open(out)
    assert processed.size == (224, 224)


def test_preprocess_image_rgb(tmp_path):
    gray = Image.fromarray(np.random.randint(0, 255, (100, 100), dtype=np.uint8))
    src = str(tmp_path / "gray.png")
    out = str(tmp_path / "rgb.png")
    gray.save(src)
    preprocess_image(src, out)
    result = Image.open(out)
    assert result.mode == "RGB"


def test_augment_image_returns_list():
    img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    augmented = augment_image(img)
    assert isinstance(augmented, list)
    assert len(augmented) > 0
    for a in augmented:
        assert isinstance(a, Image.Image)


def test_preprocess_invalid_path(tmp_path):
    result = preprocess_image("/nonexistent/path.jpg", str(tmp_path / "out.jpg"))
    assert result is False
