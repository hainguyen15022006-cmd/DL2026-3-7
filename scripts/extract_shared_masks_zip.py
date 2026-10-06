"""Validate and install only the 800 masks from Sơn's shared-run bundle.

The archive includes copies of CSV, manifest and COCO data, but this command
never extracts or overwrites those files. Install the processed COCO data first.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path, PurePosixPath
import zipfile

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_PREFIXES = ("son_visualization_bundle/", "")
MODEL_COUNTS = {"sam_vit_b": 700, "mobile_sam": 100}


def csv_rows(data: bytes) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(data.decode("utf-8-sig"))))


def check_member_paths(names: list[str]) -> None:
    if len(names) != len(set(names)):
        raise ValueError("ZIP contains duplicate member names")
    for name in names:
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or "\\" in name:
            raise ValueError(f"Unsafe ZIP member path: {name}")


def load_expected_rows(root: Path) -> list[dict[str, str]]:
    rows = csv_rows((root / "results/raw_predictions.csv").read_bytes())
    if len(rows) != 800 or any(row["status"] != "ok" for row in rows):
        raise ValueError("The tracked shared-run CSV must have 800 successful rows")
    paths = [row["mask_path"] for row in rows]
    if len(set(paths)) != 800:
        raise ValueError("The shared-run CSV has duplicate mask paths")
    counts = {model: 0 for model in MODEL_COUNTS}
    for row in rows:
        model = row["model"]
        path = row["mask_path"]
        expected_prefix = f"results/masks/{model}/"
        if model not in counts or not path.startswith(expected_prefix) or not path.endswith(".png"):
            raise ValueError(f"Unexpected model or mask path in shared CSV: {path}")
        counts[model] += 1
    if counts != MODEL_COUNTS:
        raise ValueError(f"Unexpected per-model mask counts: {counts}")
    return rows


def archive_prefix(names: set[str], expected_paths: set[str]) -> str:
    for prefix in ARCHIVE_PREFIXES:
        found = {name[len(prefix):] for name in names if name.startswith(prefix + "results/masks/")}
        if found == expected_paths:
            return prefix
    raise ValueError("ZIP mask paths do not match the 800 paths in the tracked shared-run CSV")


def compare_metadata(archive: zipfile.ZipFile, prefix: str, root: Path) -> None:
    for rel in ("results/raw_predictions.csv", "results/prompts.csv"):
        stored = csv_rows(archive.read(prefix + rel))
        tracked = csv_rows((root / rel).read_bytes())
        if stored != tracked:
            raise ValueError(f"ZIP {rel} differs from the tracked repository version")
    rel = "configs/eval_manifest.json"
    stored_manifest = json.loads(archive.read(prefix + rel))
    tracked_manifest = json.loads((root / rel).read_text(encoding="utf-8"))
    if stored_manifest != tracked_manifest:
        raise ValueError("ZIP manifest differs from the fixed tracked manifest")


def binary_mask(data: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(data)) as image:
        gray = np.asarray(image.convert("L"))
    if not np.isin(gray, (0, 255)).all():
        raise ValueError("A prediction or GT mask is not a binary 0/255 PNG")
    return gray > 0


def verify_iou(archive: zipfile.ZipFile, prefix: str, row: dict[str, str], root: Path) -> bytes:
    contents = archive.read(prefix + row["mask_path"])
    prediction = binary_mask(contents)
    gt_path = root / "data/coco/gt_masks" / f"{row['annotation_id']}.png"
    if not gt_path.is_file():
        raise FileNotFoundError(f"Install the processed COCO data first; missing {gt_path}")
    gt = binary_mask(gt_path.read_bytes())
    if prediction.shape != gt.shape:
        raise ValueError(f"Prediction/GT dimensions differ for {row['run_id']}")
    union = np.logical_or(prediction, gt).sum()
    measured = float(np.logical_and(prediction, gt).sum() / union) if union else 1.0
    if abs(measured - float(row["iou"])) > 1e-9:
        raise ValueError(
            f"Mask IoU differs from the tracked shared-run CSV for {row['run_id']}: "
            f"measured={measured:.9f}, recorded={float(row['iou']):.9f}"
        )
    return contents


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zip_file", type=Path)
    parser.add_argument("--check-only", action="store_true", help="Validate without installing")
    args = parser.parse_args()

    rows = load_expected_rows(ROOT)
    expected_paths = {row["mask_path"] for row in rows}
    with zipfile.ZipFile(args.zip_file) as archive:
        names = archive.namelist()
        check_member_paths(names)
        prefix = archive_prefix(set(names), expected_paths)
        corrupt = archive.testzip()
        if corrupt is not None:
            raise ValueError(f"ZIP CRC validation failed: {corrupt}")
        compare_metadata(archive, prefix, ROOT)

        # Validate every IoU and every existing destination before writing any file.
        missing = []
        already_present = 0
        for row in rows:
            contents = verify_iou(archive, prefix, row, ROOT)
            destination = (ROOT / row["mask_path"]).resolve()
            if not destination.is_relative_to(ROOT.resolve()):
                raise ValueError(f"Unsafe destination: {destination}")
            if destination.exists():
                if destination.read_bytes() != contents:
                    raise ValueError(f"Existing mask differs; will not overwrite: {destination}")
                already_present += 1
            else:
                missing.append((destination, contents))

        if not args.check_only:
            for destination, contents in missing:
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_suffix(destination.suffix + ".tmp")
                temporary.write_bytes(contents)
                temporary.replace(destination)

    print(json.dumps({
        "status": "ok",
        "validated_masks": len(rows),
        "already_present": already_present,
        "installed": 0 if args.check_only else len(missing),
        "check_only": args.check_only,
        "note": "All 800 mask IoUs match the tracked CSV. Only prediction PNGs are installed.",
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
