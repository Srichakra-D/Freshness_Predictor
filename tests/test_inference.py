from pathlib import Path

import pytest
import torch
import torch.nn as nn
from PIL import Image

from inference import (
    CLASS_NAMES,
    FRESHNESS_LABELS,
    ModelVGG16,
    load_model,
    predict,
    preprocess_image,
)


class FixedOutputModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))

    def forward(self, image):
        batch_size = image.shape[0]
        fruit_logits = torch.zeros(batch_size, len(CLASS_NAMES))
        freshness_logits = torch.zeros(batch_size, len(FRESHNESS_LABELS))
        fruit_logits[:, 7] = 5.0
        freshness_logits[:, 1] = 4.0
        return fruit_logits, freshness_logits


def test_preprocess_image_converts_to_rgb_tensor():
    image = Image.new("L", (40, 80), color=128)

    tensor = preprocess_image(image)

    assert tensor.shape == (1, 3, 224, 224)
    assert tensor.dtype == torch.float32
    assert torch.all((tensor >= 0) & (tensor <= 1))


def test_preprocess_image_rejects_non_image():
    with pytest.raises(TypeError, match="PIL Image"):
        preprocess_image("not-an-image")


def test_model_output_dimensions():
    model = ModelVGG16().eval()

    with torch.inference_mode():
        fruit_logits, freshness_logits = model(torch.zeros(1, 3, 224, 224))

    assert fruit_logits.shape == (1, len(CLASS_NAMES))
    assert freshness_logits.shape == (1, len(FRESHNESS_LABELS))


def test_predict_maps_logits_to_labels_and_confidence():
    result = predict(
        FixedOutputModel(),
        Image.new("RGB", (224, 224), color="green"),
        torch.device("cpu"),
    )

    assert result.fruit == "potato"
    assert result.freshness == "Spoiled"
    assert result.fruit_confidence > 0.9
    assert result.freshness_confidence > 0.9
    assert not result.is_low_confidence


def test_load_model_rejects_missing_checkpoint(tmp_path):
    missing_checkpoint = tmp_path / "missing.pth"

    with pytest.raises(FileNotFoundError, match="checkpoint not found"):
        load_model(missing_checkpoint, torch.device("cpu"))


def test_repository_checkpoint_is_compatible():
    checkpoint = Path(__file__).parents[1] / "model.pth"

    model = load_model(checkpoint, torch.device("cpu"))

    assert not model.training
    assert model.block2[-1].out_features == len(CLASS_NAMES)
    assert model.block3[-1].out_features == len(FRESHNESS_LABELS)
