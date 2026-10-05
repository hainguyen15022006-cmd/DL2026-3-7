"""Load and validate the fixed COCO evaluation handoff used by experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


MANIFEST_FIELDS = frozenset(
    {
        "image_id",
        "annotation_id",
        "file_name",
        "width",
        "height",
        "category_id",
        "bbox_xywh",
        "area",
    }
)


class DataContractError(ValueError):
    """Raised when the data handoff violates the agreed experiment contract."""


@dataclass(frozen=True)
class EvalInstance:
    image_id: int
    annotation_id: int
    file_name: str
    width: int
    height: int
    category_id: int
    bbox_xywh: tuple[float, float, float, float]
    area: float


@dataclass(frozen=True)
class ValidationReport:
    manifest_path: str
    manifest_rows: int
    checked_rows: int
    unique_images: int
    unique_annotations: int
    category_count: int
    status: str = "ok"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _positive_int(value: Any, field: str, row_number: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise DataContractError(
            f"Row {row_number}: {field} must be a positive integer, got {value!r}"
        )
    return value


def _finite_positive_number(value: Any, field: str, row_number: int) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DataContractError(
            f"Row {row_number}: {field} must be numeric, got {value!r}"
        )
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise DataContractError(
            f"Row {row_number}: {field} must be finite and positive, got {value!r}"
        )
    return result


def _parse_row(row: Any, row_number: int) -> EvalInstance:
    if not isinstance(row, dict):
        raise DataContractError(f"Row {row_number}: expected an object, got {type(row).__name__}")

    fields = frozenset(row)
    missing = sorted(MANIFEST_FIELDS - fields)
    extra = sorted(fields - MANIFEST_FIELDS)
    if missing or extra:
        raise DataContractError(
            f"Row {row_number}: manifest fields do not match the contract; "
            f"missing={missing}, extra={extra}"
        )

    file_name = row["file_name"]
    if (
        not isinstance(file_name, str)
        or not file_name
        or Path(file_name).name != file_name
        or Path(file_name).suffix.lower() not in {".jpg", ".jpeg"}
    ):
        raise DataContractError(
            f"Row {row_number}: file_name must be a JPEG basename, got {file_name!r}"
        )

    width = _positive_int(row["width"], "width", row_number)
    height = _positive_int(row["height"], "height", row_number)

    bbox = row["bbox_xywh"]
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise DataContractError(
            f"Row {row_number}: bbox_xywh must be a four-value list, got {bbox!r}"
        )
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for value in bbox
    ):
        raise DataContractError(f"Row {row_number}: bbox_xywh contains a non-finite value")

    x, y, box_width, box_height = (float(value) for value in bbox)
    if x < 0 or y < 0 or box_width <= 0 or box_height <= 0:
        raise DataContractError(f"Row {row_number}: bbox_xywh has invalid coordinates {bbox!r}")
    tolerance = 1e-6
    if x + box_width > width + tolerance or y + box_height > height + tolerance:
        raise DataContractError(
            f"Row {row_number}: bbox_xywh {bbox!r} exceeds image size {(width, height)}"
        )

    return EvalInstance(
        image_id=_positive_int(row["image_id"], "image_id", row_number),
        annotation_id=_positive_int(row["annotation_id"], "annotation_id", row_number),
        file_name=file_name,
        width=width,
        height=height,
        category_id=_positive_int(row["category_id"], "category_id", row_number),
        bbox_xywh=(x, y, box_width, box_height),
        area=_finite_positive_number(row["area"], "area", row_number),
    )


def load_manifest(
    root: str | Path,
    manifest_path: str | Path = "configs/eval_manifest.json",
    expected_count: int | None = 50,
) -> tuple[Path, list[EvalInstance]]:
    """Read the manifest and validate its schema and unique one-instance-per-image IDs."""
    root = Path(root).resolve()
    manifest = Path(manifest_path)
    if not manifest.is_absolute():
        manifest = root / manifest
    manifest = manifest.resolve()
    if not manifest.is_file():
        raise DataContractError(f"Manifest not found: {manifest}")

    try:
        raw = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DataContractError(f"Cannot read manifest {manifest}: {error}") from error

    if not isinstance(raw, list):
        raise DataContractError("Manifest root must be a JSON list")
    if expected_count is not None and len(raw) != expected_count:
        raise DataContractError(
            f"Manifest must contain {expected_count} rows, found {len(raw)}"
        )

    records = [_parse_row(row, index) for index, row in enumerate(raw, start=1)]
    image_ids = [record.image_id for record in records]
    annotation_ids = [record.annotation_id for record in records]
    if len(set(image_ids)) != len(image_ids):
        raise DataContractError("Manifest violates one-instance-per-image: duplicate image_id")
    if len(set(annotation_ids)) != len(annotation_ids):
        raise DataContractError("Manifest contains duplicate annotation_id")
    return manifest, records


def _validate_files(root: Path, record: EvalInstance, row_number: int) -> None:
    image_path = root / "data" / "coco" / "val2017" / record.file_name
    mask_path = root / "data" / "coco" / "gt_masks" / f"{record.annotation_id}.png"

    if not image_path.is_file():
        raise DataContractError(f"Row {row_number}: image not found: {image_path}")
    if not mask_path.is_file():
        raise DataContractError(f"Row {row_number}: GT mask not found: {mask_path}")

    try:
        with Image.open(image_path) as image:
            image.load()
            if image.size != (record.width, record.height):
                raise DataContractError(
                    f"Row {row_number}: image size {image.size} does not match "
                    f"manifest {(record.width, record.height)}"
                )
            rgb = np.asarray(image.convert("RGB"))
    except DataContractError:
        raise
    except (OSError, ValueError) as error:
        raise DataContractError(f"Row {row_number}: cannot decode image: {error}") from error
    if rgb.shape != (record.height, record.width, 3) or rgb.dtype != np.uint8:
        raise DataContractError(
            f"Row {row_number}: RGB image has invalid array contract "
            f"shape={rgb.shape}, dtype={rgb.dtype}"
        )

    try:
        with Image.open(mask_path) as mask_image:
            mask_image.load()
            if mask_image.mode != "L":
                raise DataContractError(
                    f"Row {row_number}: GT mask must be 8-bit grayscale, got mode={mask_image.mode}"
                )
            mask = np.asarray(mask_image)
    except DataContractError:
        raise
    except (OSError, ValueError) as error:
        raise DataContractError(f"Row {row_number}: cannot decode GT mask: {error}") from error

    if mask.shape != (record.height, record.width):
        raise DataContractError(
            f"Row {row_number}: mask shape {mask.shape} does not match "
            f"manifest {(record.height, record.width)}"
        )
    values = set(np.unique(mask).tolist())
    if not values.issubset({0, 255}) or 255 not in values:
        raise DataContractError(
            f"Row {row_number}: GT mask must be non-empty and binary 0/255, got {sorted(values)}"
        )


def validate_dataset(
    root: str | Path,
    manifest_path: str | Path = "configs/eval_manifest.json",
    expected_count: int | None = 50,
    limit: int | None = None,
) -> ValidationReport:
    """Validate manifest plus image/mask files and return a machine-readable report."""
    root = Path(root).resolve()
    manifest, records = load_manifest(root, manifest_path, expected_count)

    if limit is not None and (limit <= 0 or limit > len(records)):
        raise DataContractError(
            f"limit must be between 1 and {len(records)}, got {limit}"
        )
    checked = records if limit is None else records[:limit]
    for row_number, record in enumerate(checked, start=1):
        _validate_files(root, record, row_number)

    return ValidationReport(
        manifest_path=str(manifest),
        manifest_rows=len(records),
        checked_rows=len(checked),
        unique_images=len({record.image_id for record in records}),
        unique_annotations=len({record.annotation_id for record in records}),
        category_count=len({record.category_id for record in records}),
    )
