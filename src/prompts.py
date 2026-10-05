"""Deterministic point and box prompts for the fixed COCO evaluation set."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
from typing import Any, Iterable

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt

from src.data_contract import EvalInstance


PROMPT_TYPES = ("point", "box")
NOISE_LEVELS = (0.1, 0.2)
TRIALS = (1, 2, 3)
DIRECTIONS = ((-1, -1), (-1, 1), (1, -1), (1, 1))


class PromptContractError(ValueError):
    """Raised when generated or loaded prompts violate the locked protocol."""


@dataclass(frozen=True)
class PromptRecord:
    prompt_id: str
    image_id: int
    annotation_id: int
    prompt_type: str
    noise_level: float
    trial: int
    seed: int
    point_xy: tuple[int, int] | None
    point_label: int | None
    box_xyxy: tuple[int, int, int, int] | None
    sx: int
    sy: int
    dx: int
    dy: int
    point_inside_gt: bool | None
    box_iou_gt: float | None

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["point_xy"] = list(self.point_xy) if self.point_xy is not None else None
        row["box_xyxy"] = list(self.box_xyxy) if self.box_xyxy is not None else None
        return row


def load_gt_mask(root: str | Path, annotation_id: int) -> np.ndarray:
    path = Path(root) / "data" / "coco" / "gt_masks" / f"{annotation_id}.png"
    if not path.is_file():
        raise PromptContractError(f"GT mask not found: {path}")
    with Image.open(path) as image:
        mask = np.asarray(image) > 0
    if mask.ndim != 2 or not mask.any():
        raise PromptContractError(
            f"GT mask for annotation_id={annotation_id} must be a non-empty 2D mask"
        )
    return mask


def deepest_positive_point(mask: np.ndarray) -> tuple[int, int]:
    """Return the first row-major pixel with maximum Euclidean distance to background."""
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2 or not mask.any():
        raise PromptContractError("A clean point requires a non-empty 2D GT mask")
    distances = distance_transform_edt(mask)
    y, x = np.unravel_index(int(np.argmax(distances)), distances.shape)
    return int(x), int(y)


def tight_box(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Return the inclusive tight box [x_min, y_min, x_max, y_max] of a mask."""
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2 or not mask.any():
        raise PromptContractError("A clean box requires a non-empty 2D GT mask")
    y_coordinates, x_coordinates = np.nonzero(mask)
    return (
        int(x_coordinates.min()),
        int(y_coordinates.min()),
        int(x_coordinates.max()),
        int(y_coordinates.max()),
    )


def pixel_box_iou(
    first: tuple[int, int, int, int], second: tuple[int, int, int, int]
) -> float:
    """Compute IoU for inclusive pixel-coordinate boxes."""
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection_width = max(0, right - left + 1)
    intersection_height = max(0, bottom - top + 1)
    intersection = intersection_width * intersection_height

    first_area = max(0, first[2] - first[0] + 1) * max(0, first[3] - first[1] + 1)
    second_area = max(0, second[2] - second[0] + 1) * max(0, second[3] - second[1] + 1)
    union = first_area + second_area - intersection
    return float(intersection / union) if union else 0.0


def _clip_point(x: int, y: int, width: int, height: int) -> tuple[int, int]:
    return min(max(x, 0), width - 1), min(max(y, 0), height - 1)


def _clip_box(
    box: tuple[int, int, int, int], width: int, height: int
) -> tuple[int, int, int, int]:
    x_min, y_min = _clip_point(box[0], box[1], width, height)
    x_max, y_max = _clip_point(box[2], box[3], width, height)
    if x_min > x_max or y_min > y_max:
        raise PromptContractError(f"Clipped box is invalid: {(x_min, y_min, x_max, y_max)}")
    return x_min, y_min, x_max, y_max


def _prompt_id(
    annotation_id: int, prompt_type: str, noise_level: float, trial: int
) -> str:
    noise_token = f"{int(round(noise_level * 100)):02d}"
    return f"ann{annotation_id}_{prompt_type}_n{noise_token}_t{trial}"


def _clean_records(
    instance: EvalInstance,
    mask: np.ndarray,
    point: tuple[int, int],
    box: tuple[int, int, int, int],
    seed: int,
) -> list[PromptRecord]:
    return [
        PromptRecord(
            prompt_id=_prompt_id(instance.annotation_id, "point", 0.0, 0),
            image_id=instance.image_id,
            annotation_id=instance.annotation_id,
            prompt_type="point",
            noise_level=0.0,
            trial=0,
            seed=seed,
            point_xy=point,
            point_label=1,
            box_xyxy=None,
            sx=0,
            sy=0,
            dx=0,
            dy=0,
            point_inside_gt=bool(mask[point[1], point[0]]),
            box_iou_gt=None,
        ),
        PromptRecord(
            prompt_id=_prompt_id(instance.annotation_id, "box", 0.0, 0),
            image_id=instance.image_id,
            annotation_id=instance.annotation_id,
            prompt_type="box",
            noise_level=0.0,
            trial=0,
            seed=seed,
            point_xy=None,
            point_label=None,
            box_xyxy=box,
            sx=0,
            sy=0,
            dx=0,
            dy=0,
            point_inside_gt=None,
            box_iou_gt=1.0,
        ),
    ]


def generate_prompts_for_instance(
    instance: EvalInstance,
    mask: np.ndarray,
    directions: Iterable[tuple[int, int]],
    seed: int = 2026,
) -> list[PromptRecord]:
    """Generate two clean prompts and twelve noisy prompts for one instance."""
    if mask.shape != (instance.height, instance.width):
        raise PromptContractError(
            f"annotation_id={instance.annotation_id}: mask shape {mask.shape} does not match "
            f"{(instance.height, instance.width)}"
        )
    directions = tuple(directions)
    if len(directions) != 3 or len(set(directions)) != 3:
        raise PromptContractError("Each instance requires three unique sign pairs")
    if any(direction not in DIRECTIONS for direction in directions):
        raise PromptContractError(f"Unsupported direction in {directions!r}")

    clean_point = deepest_positive_point(mask)
    clean_box = tight_box(mask)
    records = _clean_records(instance, mask, clean_point, clean_box, seed)
    gt_width = float(instance.bbox_xywh[2])
    gt_height = float(instance.bbox_xywh[3])

    for noise_level in NOISE_LEVELS:
        for trial, (sx, sy) in enumerate(directions, start=1):
            dx = round(sx * noise_level * gt_width)
            dy = round(sy * noise_level * gt_height)

            noisy_point = _clip_point(
                clean_point[0] + dx,
                clean_point[1] + dy,
                instance.width,
                instance.height,
            )
            records.append(
                PromptRecord(
                    prompt_id=_prompt_id(
                        instance.annotation_id, "point", noise_level, trial
                    ),
                    image_id=instance.image_id,
                    annotation_id=instance.annotation_id,
                    prompt_type="point",
                    noise_level=noise_level,
                    trial=trial,
                    seed=seed,
                    point_xy=noisy_point,
                    point_label=1,
                    box_xyxy=None,
                    sx=sx,
                    sy=sy,
                    dx=dx,
                    dy=dy,
                    point_inside_gt=bool(mask[noisy_point[1], noisy_point[0]]),
                    box_iou_gt=None,
                )
            )

            shifted_box = tuple(
                value + offset
                for value, offset in zip(clean_box, (dx, dy, dx, dy), strict=True)
            )
            noisy_box = _clip_box(shifted_box, instance.width, instance.height)
            records.append(
                PromptRecord(
                    prompt_id=_prompt_id(
                        instance.annotation_id, "box", noise_level, trial
                    ),
                    image_id=instance.image_id,
                    annotation_id=instance.annotation_id,
                    prompt_type="box",
                    noise_level=noise_level,
                    trial=trial,
                    seed=seed,
                    point_xy=None,
                    point_label=None,
                    box_xyxy=noisy_box,
                    sx=sx,
                    sy=sy,
                    dx=dx,
                    dy=dy,
                    point_inside_gt=None,
                    box_iou_gt=pixel_box_iou(noisy_box, clean_box),
                )
            )
    return records


def generate_prompt_set(
    root: str | Path, instances: Iterable[EvalInstance], seed: int = 2026
) -> list[PromptRecord]:
    """Generate the complete prompt set with one seeded RNG over manifest order."""
    root = Path(root)
    rng = random.Random(seed)
    records: list[PromptRecord] = []
    for instance in instances:
        directions = rng.sample(DIRECTIONS, k=3)
        mask = load_gt_mask(root, instance.annotation_id)
        records.extend(
            generate_prompts_for_instance(instance, mask, directions, seed=seed)
        )
    expected_instances = len(instances) if isinstance(instances, list) else None
    validate_prompt_matrix(records, expected_instances=expected_instances)
    return records


def validate_prompt_matrix(
    records: Iterable[PromptRecord], expected_instances: int | None = 50
) -> None:
    records = list(records)
    prompt_ids = [record.prompt_id for record in records]
    if len(set(prompt_ids)) != len(prompt_ids):
        raise PromptContractError("Duplicate prompt_id values found")

    annotation_ids = sorted({record.annotation_id for record in records})
    if expected_instances is not None and len(annotation_ids) != expected_instances:
        raise PromptContractError(
            f"Expected {expected_instances} instances, found {len(annotation_ids)}"
        )

    expected_conditions = {
        (prompt_type, 0.0, 0)
        for prompt_type in PROMPT_TYPES
    } | {
        (prompt_type, noise_level, trial)
        for prompt_type in PROMPT_TYPES
        for noise_level in NOISE_LEVELS
        for trial in TRIALS
    }
    for annotation_id in annotation_ids:
        subset = [record for record in records if record.annotation_id == annotation_id]
        conditions = {
            (record.prompt_type, record.noise_level, record.trial) for record in subset
        }
        if conditions != expected_conditions:
            missing = sorted(expected_conditions - conditions)
            extra = sorted(conditions - expected_conditions)
            raise PromptContractError(
                f"annotation_id={annotation_id}: incomplete prompt matrix; "
                f"missing={missing}, extra={extra}"
            )


CSV_FIELDS = (
    "prompt_id",
    "image_id",
    "annotation_id",
    "prompt_type",
    "noise_level",
    "trial",
    "seed",
    "point_x",
    "point_y",
    "point_label",
    "box_x_min",
    "box_y_min",
    "box_x_max",
    "box_y_max",
    "sx",
    "sy",
    "dx",
    "dy",
    "point_inside_gt",
    "box_iou_gt",
)


def _csv_row(record: PromptRecord) -> dict[str, Any]:
    point = record.point_xy or (None, None)
    box = record.box_xyxy or (None, None, None, None)
    return {
        "prompt_id": record.prompt_id,
        "image_id": record.image_id,
        "annotation_id": record.annotation_id,
        "prompt_type": record.prompt_type,
        "noise_level": record.noise_level,
        "trial": record.trial,
        "seed": record.seed,
        "point_x": point[0],
        "point_y": point[1],
        "point_label": record.point_label,
        "box_x_min": box[0],
        "box_y_min": box[1],
        "box_x_max": box[2],
        "box_y_max": box[3],
        "sx": record.sx,
        "sy": record.sy,
        "dx": record.dx,
        "dy": record.dy,
        "point_inside_gt": record.point_inside_gt,
        "box_iou_gt": record.box_iou_gt,
    }


def write_prompt_files(
    records: Iterable[PromptRecord], json_path: str | Path, csv_path: str | Path
) -> None:
    records = list(records)
    validate_prompt_matrix(records, expected_instances=len({r.annotation_id for r in records}))
    json_path = Path(json_path)
    csv_path = Path(csv_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    temporary_json = json_path.with_suffix(json_path.suffix + ".tmp")
    temporary_json.write_text(
        json.dumps([record.to_dict() for record in records], indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_json.replace(json_path)

    temporary_csv = csv_path.with_suffix(csv_path.suffix + ".tmp")
    with temporary_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(_csv_row(record) for record in records)
    temporary_csv.replace(csv_path)


def load_prompt_records(path: str | Path) -> list[PromptRecord]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise PromptContractError("Prompt JSON root must be a list")
    records: list[PromptRecord] = []
    for index, row in enumerate(raw, start=1):
        try:
            point = tuple(row["point_xy"]) if row["point_xy"] is not None else None
            box = tuple(row["box_xyxy"]) if row["box_xyxy"] is not None else None
            records.append(
                PromptRecord(
                    **{
                        **row,
                        "point_xy": point,
                        "box_xyxy": box,
                    }
                )
            )
        except (KeyError, TypeError, ValueError) as error:
            raise PromptContractError(f"Prompt row {index} is invalid: {error}") from error
    validate_prompt_matrix(records, expected_instances=len({r.annotation_id for r in records}))
    return records
