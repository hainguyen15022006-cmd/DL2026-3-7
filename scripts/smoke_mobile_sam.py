"""Smoke-test MobileSAM: verify clean-prompt predictions and produce overlays."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_contract import load_manifest  # noqa: E402
from src.experiment import binary_iou, read_results  # noqa: E402
from src.models.factory import create_model  # noqa: E402
from src.prompts import PromptRecord, load_gt_mask, load_prompt_records  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers (testable without a model)
# ---------------------------------------------------------------------------

SMOKE_CSV_FIELDS = (
    "model",
    "image_id",
    "annotation_id",
    "prompt_type",
    "noise_level",
    "trial",
    "iou",
    "seconds",
    "status",
    "seed",
    "prompt_id",
    "encode_seconds",
    "predicted_score",
    "error",
    "committed_iou",
    "abs_diff",
)


@dataclass
class SmokeRow:
    """One result row for the smoke CSV."""

    model: str
    image_id: int
    annotation_id: int
    prompt_type: str
    noise_level: float
    trial: int
    iou: str
    seconds: str
    status: str
    seed: int
    prompt_id: str
    encode_seconds: str
    predicted_score: str
    error: str
    committed_iou: str
    abs_diff: str

    def as_dict(self) -> dict[str, Any]:
        return {field: getattr(self, field) for field in SMOKE_CSV_FIELDS}


def compute_abs_diff(smoke_iou: float | None, committed_iou: float | None) -> float | None:
    """Return |smoke_iou - committed_iou| or None if either is missing."""
    if smoke_iou is None or committed_iou is None:
        return None
    return abs(smoke_iou - committed_iou)


def lookup_committed_iou(
    committed: dict[str, dict[str, str]], model: str, prompt_id: str
) -> float | None:
    """Read the IoU for a given run from the committed raw_predictions.csv."""
    run_id = f"{model}:{prompt_id}"
    row = committed.get(run_id)
    if row is None:
        return None
    raw = row.get("iou", "")
    if raw == "":
        return None
    try:
        return float(raw)
    except (ValueError, TypeError):
        return None


def gt_contour_coords(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return boundary pixel coordinates of a boolean mask for drawing."""
    padded = np.pad(mask.astype(np.uint8), 1, mode="constant", constant_values=0)
    interior = (
        padded[:-2, 1:-1] & padded[2:, 1:-1] & padded[1:-1, :-2] & padded[1:-1, 2:]
    )
    boundary = mask.astype(np.uint8) - interior
    ys, xs = np.nonzero(boundary)
    return list(zip(xs.tolist(), ys.tolist()))


def compose_overlay(
    image_rgb: np.ndarray,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    prompt: PromptRecord,
    iou: float | None,
) -> Image.Image:
    """Create a Pillow overlay image with GT contour, prompt and prediction."""
    overlay = Image.fromarray(image_rgb, mode="RGB").copy()
    draw = ImageDraw.Draw(overlay)

    # GT contour in green
    for x, y in gt_contour_coords(gt_mask):
        draw.point((x, y), fill=(0, 255, 0))

    # Predicted mask in semi-transparent blue
    pred_rgba = np.zeros((*pred_mask.shape, 4), dtype=np.uint8)
    pred_rgba[pred_mask, 0] = 0
    pred_rgba[pred_mask, 1] = 120
    pred_rgba[pred_mask, 2] = 255
    pred_rgba[pred_mask, 3] = 90
    pred_layer = Image.fromarray(pred_rgba, mode="RGBA")
    overlay.paste(pred_layer, mask=pred_layer.split()[3])

    # Draw prompt
    if prompt.prompt_type == "point" and prompt.point_xy is not None:
        x, y = prompt.point_xy
        r = 5
        draw.ellipse([x - r, y - r, x + r, y + r], fill=(255, 0, 0), outline=(255, 255, 255))
    elif prompt.prompt_type == "box" and prompt.box_xyxy is not None:
        x0, y0, x1, y1 = prompt.box_xyxy
        draw.rectangle([x0, y0, x1, y1], outline=(255, 255, 0), width=2)

    # Caption
    iou_text = f"{iou:.4f}" if iou is not None else "N/A"
    caption = f"{prompt.prompt_type} | IoU={iou_text}"
    try:
        font = ImageFont.load_default(size=14)
    except TypeError:
        font = ImageFont.load_default()
    draw.text((5, 5), caption, fill=(255, 255, 255), font=font)

    return overlay


def write_smoke_csv(path: Path, rows: list[SmokeRow]) -> None:
    """Write smoke results to CSV with explicit UTF-8 encoding."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SMOKE_CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.as_dict())


def smoke_is_complete(rows: list[SmokeRow], instance_count: int) -> bool:
    """Require one successful clean point and box result per chosen instance."""
    if instance_count < 1 or len(rows) != 2 * instance_count:
        return False
    groups: dict[int, set[str]] = {}
    for row in rows:
        if row.status != "ok" or row.noise_level != 0 or row.trial != 0:
            return False
        types = groups.setdefault(row.annotation_id, set())
        if row.prompt_type in types or row.prompt_type not in {"point", "box"}:
            return False
        types.add(row.prompt_type)
    return len(groups) == instance_count and all(
        types == {"point", "box"} for types in groups.values()
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _load_config(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=10, help="First N manifest rows")
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/experiment.json")
    )
    parser.add_argument(
        "--out", type=Path, default=Path("results/smoke/mobile_sam")
    )
    args = parser.parse_args()
    if not 1 <= args.n <= 50:
        parser.error("--n must be between 1 and 50")

    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    config = _load_config(config_path)

    # Validate prerequisites
    weights_path = ROOT / config["models"]["mobile_sam"]["checkpoint_path"]
    if not weights_path.is_file():
        print(f"ERROR: MobileSAM weights not found at {weights_path}", file=sys.stderr)
        print("Download mobile_sam.pt and place it in weights/", file=sys.stderr)
        return 1

    source_path = ROOT / config["models"]["mobile_sam"]["source_path"]
    if not source_path.is_dir():
        print(f"ERROR: MobileSAM source not found at {source_path}", file=sys.stderr)
        print("Clone https://github.com/ChaoningZhang/MobileSAM next to this repo", file=sys.stderr)
        return 1

    data_dir = ROOT / "data" / "coco"
    if not data_dir.is_dir():
        print(f"ERROR: COCO data not found at {data_dir}", file=sys.stderr)
        return 1

    # Load manifest and prompts
    _, instances = load_manifest(ROOT, config["manifest_path"], expected_count=50)
    instances = instances[: args.n]

    prompt_path = ROOT / config["prompt_json_path"]
    all_prompts = load_prompt_records(prompt_path)
    selected_annotations = {inst.annotation_id for inst in instances}
    clean_prompts = [
        p
        for p in all_prompts
        if p.annotation_id in selected_annotations and p.noise_level == 0
    ]

    # Load committed results for comparison
    committed_path = ROOT / config["raw_results_path"]
    committed = read_results(committed_path) if committed_path.is_file() else {}

    # Create model
    print(f"Loading MobileSAM model (device={args.device})...")
    model = create_model("mobile_sam", ROOT, config["models"]["mobile_sam"], args.device)
    print("Model loaded.")

    out_dir = args.out if args.out.is_absolute() else ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    smoke_rows: list[SmokeRow] = []
    instance_by_image = {inst.image_id: inst for inst in instances}
    prompts_by_image: dict[int, list[PromptRecord]] = {}
    for p in clean_prompts:
        prompts_by_image.setdefault(p.image_id, []).append(p)

    for idx, inst in enumerate(instances, start=1):
        image_path = ROOT / "data" / "coco" / "val2017" / inst.file_name
        prompts_for_image = prompts_by_image.get(inst.image_id, [])
        print(f"[{idx}/{len(instances)}] image_id={inst.image_id}, {len(prompts_for_image)} clean prompts")

        try:
            with Image.open(image_path) as img:
                image_rgb = np.asarray(img.convert("RGB"))
            encode_seconds = model.set_image(image_rgb)
            gt_mask = load_gt_mask(ROOT, inst.annotation_id)
        except Exception as exc:
            msg = f"instance preparation failed: {type(exc).__name__}: {exc}"
            print(f"  ERROR: {msg}", file=sys.stderr)
            for prompt in prompts_for_image:
                c_iou = lookup_committed_iou(committed, "mobile_sam", prompt.prompt_id)
                smoke_rows.append(
                    SmokeRow(
                        model="mobile_sam",
                        image_id=inst.image_id,
                        annotation_id=inst.annotation_id,
                        prompt_type=prompt.prompt_type,
                        noise_level=0,
                        trial=0,
                        iou="",
                        seconds="",
                        status="error",
                        seed=2026,
                        prompt_id=prompt.prompt_id,
                        encode_seconds="",
                        predicted_score="",
                        error=msg,
                        committed_iou=f"{c_iou}" if c_iou is not None else "",
                        abs_diff="",
                    )
                )
            continue

        for prompt in prompts_for_image:
            try:
                prediction = model.predict(prompt)
                iou = binary_iou(prediction.mask, gt_mask)

                c_iou = lookup_committed_iou(committed, "mobile_sam", prompt.prompt_id)
                diff = compute_abs_diff(iou, c_iou)

                # Save overlay
                overlay = compose_overlay(image_rgb, gt_mask, prediction.mask, prompt, iou)
                overlay_name = f"{prompt.prompt_id}.png"
                overlay.save(out_dir / overlay_name)

                smoke_rows.append(
                    SmokeRow(
                        model="mobile_sam",
                        image_id=inst.image_id,
                        annotation_id=inst.annotation_id,
                        prompt_type=prompt.prompt_type,
                        noise_level=0,
                        trial=0,
                        iou=f"{iou}",
                        seconds=f"{prediction.seconds}",
                        status="ok",
                        seed=2026,
                        prompt_id=prompt.prompt_id,
                        encode_seconds=f"{encode_seconds}",
                        predicted_score=f"{prediction.score}",
                        error="",
                        committed_iou=f"{c_iou}" if c_iou is not None else "",
                        abs_diff=f"{diff}" if diff is not None else "",
                    )
                )
                print(f"  {prompt.prompt_type}: IoU={iou:.4f}, committed={c_iou}, diff={diff}")

            except Exception as exc:
                msg = f"predict failed: {type(exc).__name__}: {exc}"
                print(f"  ERROR ({prompt.prompt_type}): {msg}", file=sys.stderr)
                c_iou = lookup_committed_iou(committed, "mobile_sam", prompt.prompt_id)
                smoke_rows.append(
                    SmokeRow(
                        model="mobile_sam",
                        image_id=inst.image_id,
                        annotation_id=inst.annotation_id,
                        prompt_type=prompt.prompt_type,
                        noise_level=0,
                        trial=0,
                        iou="",
                        seconds="",
                        status="error",
                        seed=2026,
                        prompt_id=prompt.prompt_id,
                        encode_seconds=f"{encode_seconds}",
                        predicted_score="",
                        error=msg,
                        committed_iou=f"{c_iou}" if c_iou is not None else "",
                        abs_diff="",
                    )
                )

    # Write CSV
    csv_path = out_dir / "smoke_results.csv"
    write_smoke_csv(csv_path, smoke_rows)
    print(f"\nSaved {len(smoke_rows)} rows to {csv_path}")

    # Summary
    print("\n=== SMOKE TEST SUMMARY ===")
    for ptype in ("point", "box"):
        subset = [r for r in smoke_rows if r.prompt_type == ptype and r.status == "ok"]
        if not subset:
            print(f"  {ptype}: no successful rows")
            continue
        ious = [float(r.iou) for r in subset]
        n_with_diff = sum(
            1 for r in subset if r.abs_diff and float(r.abs_diff) > 0.01
        )
        print(
            f"  {ptype}: n={len(subset)}, mean_IoU={np.mean(ious):.4f}, "
            f"abs_diff>0.01: {n_with_diff}/{len(subset)}"
        )

    if not smoke_is_complete(smoke_rows, len(instances)):
        print("ERROR: smoke test is incomplete or contains failed predictions", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
