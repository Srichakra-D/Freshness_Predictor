from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


CLASS_NAMES = (
    "apples",
    "banana",
    "bittergroud",
    "capsicum",
    "cucumber",
    "okra",
    "oranges",
    "potato",
    "tomato",
)
FRESHNESS_LABELS = ("Fresh", "Spoiled")
LOW_CONFIDENCE_THRESHOLD = 0.55

INFERENCE_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]),
    ]
)


@dataclass(frozen=True)
class Prediction:
    fruit: str
    freshness: str
    fruit_confidence: float
    freshness_confidence: float

    @property
    def is_low_confidence(self) -> bool:
        return min(self.fruit_confidence, self.freshness_confidence) < LOW_CONFIDENCE_THRESHOLD


class ModelVGG16(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = models.vgg16(weights=None)
        self.base.classifier = nn.Sequential()
        self.block1 = nn.Sequential(
            nn.Linear(512 * 7 * 7, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
        )
        self.block2 = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, len(CLASS_NAMES)),
        )
        self.block3 = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, len(FRESHNESS_LABELS)),
        )

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = torch.flatten(self.base.features(image), 1)
        shared_features = self.block1(features)
        return self.block2(shared_features), self.block3(shared_features)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_path: str | Path, device: torch.device | None = None) -> ModelVGG16:
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {path}")

    target_device = device or get_device()
    model = ModelVGG16().to(target_device)
    try:
        state_dict = torch.load(path, map_location=target_device, weights_only=True)
    except TypeError:
        state_dict = torch.load(path, map_location=target_device)

    model.load_state_dict(state_dict)
    model.eval()
    return model


def preprocess_image(image: Image.Image) -> torch.Tensor:
    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL Image")
    return INFERENCE_TRANSFORM(image.convert("RGB")).unsqueeze(0)


def predict(
    model: nn.Module,
    image: Image.Image,
    device: torch.device | None = None,
) -> Prediction:
    target_device = device or next(model.parameters()).device
    image_tensor = preprocess_image(image).to(target_device)

    with torch.inference_mode():
        fruit_logits, freshness_logits = model(image_tensor)
        fruit_probabilities = torch.softmax(fruit_logits, dim=1)
        freshness_probabilities = torch.softmax(freshness_logits, dim=1)

    fruit_confidence, fruit_index = fruit_probabilities.max(dim=1)
    freshness_confidence, freshness_index = freshness_probabilities.max(dim=1)
    return Prediction(
        fruit=CLASS_NAMES[fruit_index.item()],
        freshness=FRESHNESS_LABELS[freshness_index.item()],
        fruit_confidence=fruit_confidence.item(),
        freshness_confidence=freshness_confidence.item(),
    )
