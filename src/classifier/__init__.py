"""신발/의류 zero-shot 분류 모듈."""
from .model import ZeroShotClassifier, Prediction, MODELS, pick_device  # noqa: F401
from . import taxonomy  # noqa: F401

__all__ = ["ZeroShotClassifier", "Prediction", "MODELS", "pick_device", "taxonomy"]
