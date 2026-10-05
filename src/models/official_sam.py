"""Adapters backed by the official SAM and MobileSAM repositories."""

from __future__ import annotations

import importlib
from pathlib import Path
import sys
import time
from types import ModuleType

import numpy as np
import torch

from src.models.base import ModelPrediction
from src.prompts import PromptRecord


class OfficialSamAdapter:
    """Common predictor wrapper that enforces the team's single-mask contract."""

    def __init__(
        self,
        *,
        name: str,
        package_name: str,
        model_type: str,
        source_path: str | Path,
        checkpoint_path: str | Path,
        device: str = "auto",
    ) -> None:
        self.name = name
        self.source_path = Path(source_path).resolve()
        self.checkpoint_path = Path(checkpoint_path).resolve()
        if not self.source_path.is_dir():
            raise FileNotFoundError(f"Official source directory not found: {self.source_path}")
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        self.device = self._resolve_device(device)
        package = self._import_package(package_name)
        model = package.sam_model_registry[model_type](checkpoint=str(self.checkpoint_path))
        model.to(device=self.device)
        model.eval()
        self.predictor = package.SamPredictor(model)
        self._image_shape: tuple[int, int] | None = None

    @staticmethod
    def _resolve_device(requested: str) -> str:
        if requested == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
        if requested not in {"cpu", "cuda"}:
            raise ValueError(f"Unsupported device: {requested}")
        return requested

    def _import_package(self, package_name: str) -> ModuleType:
        source = str(self.source_path)
        if source not in sys.path:
            sys.path.insert(0, source)
        return importlib.import_module(package_name)

    def set_image(self, image_rgb: np.ndarray) -> float:
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3 or image_rgb.dtype != np.uint8:
            raise ValueError(
                f"Expected RGB uint8 HxWx3 input, got shape={image_rgb.shape}, "
                f"dtype={image_rgb.dtype}"
            )
        started = time.perf_counter()
        with torch.inference_mode():
            self.predictor.set_image(image_rgb)
        elapsed = time.perf_counter() - started
        self._image_shape = image_rgb.shape[:2]
        return elapsed

    def predict(self, prompt: PromptRecord) -> ModelPrediction:
        if self._image_shape is None:
            raise RuntimeError("set_image must be called before predict")

        point_coords = None
        point_labels = None
        box = None
        if prompt.prompt_type == "point":
            if prompt.point_xy is None or prompt.point_label is None:
                raise ValueError(f"Point prompt {prompt.prompt_id} has no coordinates/label")
            point_coords = np.asarray([prompt.point_xy], dtype=np.float32)
            point_labels = np.asarray([prompt.point_label], dtype=np.int32)
        elif prompt.prompt_type == "box":
            if prompt.box_xyxy is None:
                raise ValueError(f"Box prompt {prompt.prompt_id} has no box")
            box = np.asarray(prompt.box_xyxy, dtype=np.float32)
        else:
            raise ValueError(f"Unsupported prompt_type={prompt.prompt_type!r}")

        started = time.perf_counter()
        with torch.inference_mode():
            masks, scores, _ = self.predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=box,
                multimask_output=False,
            )
        elapsed = time.perf_counter() - started

        if masks.shape[0] != 1:
            raise RuntimeError(
                f"{self.name} returned {masks.shape[0]} masks for one prompt; expected 1"
            )
        mask = np.asarray(masks[0], dtype=bool)
        if mask.shape != self._image_shape:
            raise RuntimeError(
                f"{self.name} returned mask shape {mask.shape}; expected {self._image_shape}"
            )
        return ModelPrediction(mask=mask, score=float(scores[0]), seconds=elapsed)
