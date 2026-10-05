"""Input validation for one-image, user-supplied SAM prompts."""

from __future__ import annotations

from src.prompts import PromptRecord


def make_demo_prompt(
    width: int,
    height: int,
    *,
    point: tuple[int, int] | None = None,
    box: tuple[int, int, int, int] | None = None,
) -> PromptRecord:
    if (point is None) == (box is None):
        raise ValueError("Provide exactly one of --point or --box")
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive")
    if point is not None:
        x, y = point
        if not (0 <= x < width and 0 <= y < height):
            raise ValueError(f"Point {(x, y)} lies outside image {(width, height)}")
        return PromptRecord(
            prompt_id="demo_point", image_id=0, annotation_id=0,
            prompt_type="point", noise_level=0.0, trial=0, seed=2026,
            point_xy=point, point_label=1, box_xyxy=None,
            sx=0, sy=0, dx=0, dy=0, point_inside_gt=None, box_iou_gt=None,
        )

    assert box is not None
    x_min, y_min, x_max, y_max = box
    if not (0 <= x_min <= x_max < width and 0 <= y_min <= y_max < height):
        raise ValueError(f"Box {box} lies outside image {(width, height)}")
    return PromptRecord(
        prompt_id="demo_box", image_id=0, annotation_id=0,
        prompt_type="box", noise_level=0.0, trial=0, seed=2026,
        point_xy=None, point_label=None, box_xyxy=box,
        sx=0, sy=0, dx=0, dy=0, point_inside_gt=None, box_iou_gt=None,
    )
