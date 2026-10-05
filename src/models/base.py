"""Shared interface for interactive segmentation model adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from src.prompts import PromptRecord


@dataclass(frozen=True)
class ModelPrediction:
    mask: np.ndarray
    score: float
    seconds: float


class InteractiveSegmenter(Protocol):
    name: str
    device: str

    def set_image(self, image_rgb: np.ndarray) -> float:
        """Encode one RGB image and return set_image time in seconds."""

    def predict(self, prompt: PromptRecord) -> ModelPrediction:
        """Predict one mask and time only the prompt-specific predict call."""
