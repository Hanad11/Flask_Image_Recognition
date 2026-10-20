"""Tests for image preprocessing and prediction in model.py."""
from io import BytesIO
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image

import model


def make_image(size=(50, 30), mode="RGB"):
    """Return an in-memory PNG image of the given size and colour mode."""
    buffer = BytesIO()
    Image.new(mode, size).save(buffer, format="PNG")
    buffer.seek(0)
    return buffer


def test_preprocess_output_shape():
    """Preprocessing returns a batch of one 224x224 RGB image."""
    result = model.preprocess_img(make_image())
    assert result.shape == (1, 224, 224, 3)


def test_preprocess_values_are_normalised():
    """Pixel values are scaled into the range 0 to 1."""
    result = model.preprocess_img(make_image())
    assert result.min() >= 0.0
    assert result.max() <= 1.0


@pytest.mark.parametrize("size", [(10, 10), (500, 300)])
def test_preprocess_resizes_any_input_size(size):
    """Very small and large images are both resized to 224x224."""
    result = model.preprocess_img(make_image(size=size))
    assert result.shape == (1, 224, 224, 3)


def test_preprocess_rejects_non_image_data():
    """Bytes that are not an image raise an OSError."""
    with pytest.raises(OSError):
        model.preprocess_img(BytesIO(b"not an image"))


def test_preprocess_rejects_grayscale_image():
    """A one-channel image cannot be reshaped to three channels."""
    with pytest.raises(ValueError):
        model.preprocess_img(make_image(mode="L"))


def test_predict_result_returns_highest_scoring_class(monkeypatch):
    """predict_result returns the index of the highest model score."""
    fake_model = MagicMock()
    fake_model.predict.return_value = np.array([[0.1, 0.7, 0.2]])
    monkeypatch.setattr(model, "model", fake_model)
    assert model.predict_result(np.zeros((1, 224, 224, 3))) == 1
