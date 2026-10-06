"""Metric helpers for binary instance-segmentation masks."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Iterable, Sequence


@dataclass(frozen=True)
class BinaryCounts:
    """Pixel counts used by IoU and Dice."""

    true_positive: int
    false_positive: int
    false_negative: int
    true_negative: int

    @property
    def union(self) -> int:
        return self.true_positive + self.false_positive + self.false_negative


def _shape(mask: object) -> tuple[int, ...] | None:
    value = getattr(mask, "shape", None)
    if value is None:
        return None
    return tuple(int(item) for item in value)


def _flat_boolean_values(mask: object) -> list[bool]:
    """Convert a NumPy-like array or nested sequence to flat Boolean values."""
    if hasattr(mask, "ravel"):
        return [bool(value) for value in mask.ravel()]

    if not isinstance(mask, Sequence) or isinstance(mask, (str, bytes)):
        raise TypeError("A mask must be a two-dimensional array or nested sequence")

    values: list[bool] = []
    for row in mask:
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes)):
            raise TypeError("A mask must be a two-dimensional array or nested sequence")
        values.extend(bool(value) for value in row)
    return values


def _sequence_shape(mask: object) -> tuple[int, int]:
    if not isinstance(mask, Sequence) or isinstance(mask, (str, bytes)):
        raise TypeError("A mask must be a two-dimensional array or nested sequence")
    rows = len(mask)
    widths = []
    for row in mask:
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes)):
            raise TypeError("A mask must be a two-dimensional array or nested sequence")
        widths.append(len(row))
    if not widths:
        return (0, 0)
    if len(set(widths)) != 1:
        raise ValueError("Mask rows must all have the same length")
    return (rows, widths[0])


def binary_counts(prediction: object, ground_truth: object) -> BinaryCounts:
    """Count TP, FP, FN and TN after treating non-zero pixels as foreground."""
    prediction_shape = _shape(prediction) or _sequence_shape(prediction)
    ground_truth_shape = _shape(ground_truth) or _sequence_shape(ground_truth)
    if len(prediction_shape) != 2 or len(ground_truth_shape) != 2:
        raise ValueError("Prediction and ground-truth masks must be two-dimensional")
    if prediction_shape != ground_truth_shape:
        raise ValueError(
            "Prediction and ground-truth masks must have the same shape, "
            f"got {prediction_shape} and {ground_truth_shape}"
        )

    predicted = _flat_boolean_values(prediction)
    expected = _flat_boolean_values(ground_truth)
    tp = fp = fn = tn = 0
    for predicted_pixel, expected_pixel in zip(predicted, expected):
        if predicted_pixel and expected_pixel:
            tp += 1
        elif predicted_pixel:
            fp += 1
        elif expected_pixel:
            fn += 1
        else:
            tn += 1
    return BinaryCounts(tp, fp, fn, tn)


def binary_iou(
    prediction: object, ground_truth: object, *, empty_value: float = 1.0
) -> float:
    """Return TP / (TP + FP + FN).

    If both masks are empty, their union is empty and ``empty_value`` is returned.
    The project uses 1.0 because two empty masks agree exactly. Empty ground-truth
    masks should normally be rejected during data preparation.
    """
    counts = binary_counts(prediction, ground_truth)
    if counts.union == 0:
        return float(empty_value)
    return counts.true_positive / counts.union


def binary_dice(
    prediction: object, ground_truth: object, *, empty_value: float = 1.0
) -> float:
    """Return 2TP / (2TP + FP + FN)."""
    counts = binary_counts(prediction, ground_truth)
    denominator = 2 * counts.true_positive + counts.false_positive + counts.false_negative
    if denominator == 0:
        return float(empty_value)
    return 2 * counts.true_positive / denominator


def validate_iou(value: object) -> float:
    """Parse one IoU value and require the closed interval [0, 1]."""
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"IoU is not numeric: {value!r}") from error
    if not isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise ValueError(f"IoU must be finite and in [0, 1], got {value!r}")
    return parsed


def mean(values: Iterable[float]) -> float | None:
    """Return the arithmetic mean, or None for an empty input."""
    items = list(values)
    return sum(items) / len(items) if items else None


def median(values: Iterable[float]) -> float | None:
    """Return the median, or None for an empty input."""
    items = sorted(values)
    if not items:
        return None
    middle = len(items) // 2
    if len(items) % 2:
        return items[middle]
    return (items[middle - 1] + items[middle]) / 2
