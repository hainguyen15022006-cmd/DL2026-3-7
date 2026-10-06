"""Recompute per-run IoU and Dice from saved prediction and GT masks."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.metrics import binary_counts, binary_dice, binary_iou, validate_iou


EXTRA_FIELDS = (
    "recorded_iou",
    "iou_delta",
    "dice",
    "true_positive",
    "false_positive",
    "false_negative",
    "true_negative",
    "metric_status",
    "metric_error",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="results/raw_predictions.csv")
    parser.add_argument("--output", default="results/per_run_metrics.csv")
    parser.add_argument("--gt-mask-dir", default="data/coco/gt_masks")
    parser.add_argument(
        "--root",
        default=".",
        help="Base directory for relative mask_path values (default: current directory)",
    )
    parser.add_argument(
        "--iou-tolerance",
        type=float,
        default=1e-9,
        help="Maximum difference allowed between recorded and recomputed IoU",
    )
    return parser.parse_args()


def load_mask(path: Path) -> list[list[bool]]:
    with Image.open(path) as image:
        gray = image.convert("L")
        width, height = gray.size
        if hasattr(gray, "get_flattened_data"):
            pixels = list(gray.get_flattened_data())
        else:
            pixels = list(gray.getdata())
    return [
        [pixels[y * width + x] != 0 for x in range(width)]
        for y in range(height)
    ]


def evaluate_row(
    row: dict[str, str], root: Path, gt_mask_dir: Path, tolerance: float
) -> dict[str, object]:
    output: dict[str, object] = dict(row)
    output["recorded_iou"] = row.get("iou", "")
    for field in EXTRA_FIELDS[1:]:
        output[field] = ""

    if row.get("status", "").strip().lower() != "ok":
        output["metric_status"] = "skipped"
        output["metric_error"] = "Source run status is not ok"
        return output

    try:
        annotation_id = int(row["annotation_id"])
        prediction_path = Path(row["mask_path"])
        if not prediction_path.is_absolute():
            prediction_path = root / prediction_path
        gt_path = gt_mask_dir / f"{annotation_id}.png"
        prediction = load_mask(prediction_path)
        ground_truth = load_mask(gt_path)
        counts = binary_counts(prediction, ground_truth)
        recomputed_iou = binary_iou(prediction, ground_truth)
        recorded_iou = validate_iou(row["iou"])
        delta = recomputed_iou - recorded_iou

        output.update(
            {
                "iou": recomputed_iou,
                "iou_delta": delta,
                "dice": binary_dice(prediction, ground_truth),
                "true_positive": counts.true_positive,
                "false_positive": counts.false_positive,
                "false_negative": counts.false_negative,
                "true_negative": counts.true_negative,
                "metric_status": "ok" if abs(delta) <= tolerance else "mismatch",
                "metric_error": "" if abs(delta) <= tolerance else "Recorded IoU differs from mask IoU",
            }
        )
    except (KeyError, TypeError, ValueError, OSError) as error:
        output["metric_status"] = "error"
        output["metric_error"] = str(error)
    return output


def main() -> int:
    args = parse_args()
    root = Path(args.root).resolve()
    input_path = Path(args.input)
    output_path = Path(args.output)
    gt_mask_dir = Path(args.gt_mask_dir)
    if not gt_mask_dir.is_absolute():
        gt_mask_dir = root / gt_mask_dir

    with input_path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError("Input CSV has no header")
        required = {"annotation_id", "iou", "status", "mask_path"}
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(f"Input CSV is missing columns: {sorted(missing)}")
        rows = [evaluate_row(row, root, gt_mask_dir, args.iou_tolerance) for row in reader]
        fieldnames = list(reader.fieldnames)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames + list(EXTRA_FIELDS))
        writer.writeheader()
        writer.writerows(rows)

    counts: dict[str, int] = {}
    for row in rows:
        status = str(row["metric_status"])
        counts[status] = counts.get(status, 0) + 1
    print(f"Wrote {len(rows)} rows to {output_path}")
    print("Metric status:", counts)
    return 1 if counts.get("error", 0) or counts.get("mismatch", 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
