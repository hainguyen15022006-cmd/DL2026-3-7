"""Construct model adapters by the stable names used in experiment logs."""

from __future__ import annotations

from pathlib import Path

from src.models.base import InteractiveSegmenter


MODEL_NAMES = ("mobile_sam", "sam_vit_b")


def create_model(
    name: str, root: str | Path, model_config: dict, device: str
) -> InteractiveSegmenter:
    if name == "sam_vit_b":
        from src.models.sam_vit_b import create_model as create_sam_vit_b

        return create_sam_vit_b(root, model_config, device)
    if name == "mobile_sam":
        from src.models.mobile_sam import create_model as create_mobile_sam

        return create_mobile_sam(root, model_config, device)
    raise ValueError(f"Unknown model {name!r}; expected one of {MODEL_NAMES}")
