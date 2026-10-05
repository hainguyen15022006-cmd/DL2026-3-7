"""SAM ViT-B main-model adapter."""

from __future__ import annotations

from pathlib import Path

from src.models.official_sam import OfficialSamAdapter


def create_model(
    root: str | Path,
    config: dict,
    device: str = "auto",
) -> OfficialSamAdapter:
    root = Path(root).resolve()
    return OfficialSamAdapter(
        name="sam_vit_b",
        package_name="segment_anything",
        model_type=config["model_type"],
        source_path=root / config["source_path"],
        checkpoint_path=root / config["checkpoint_path"],
        device=device,
    )
