"""Segment one local image with a user-selected point or box; no GT is needed."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.demo import make_demo_prompt  # noqa: E402
from src.models.factory import MODEL_NAMES, create_model  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--model", choices=MODEL_NAMES, default="sam_vit_b")
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.json"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--point", nargs=2, type=int, metavar=("X", "Y"))
    selection.add_argument(
        "--box", nargs=4, type=int, metavar=("X_MIN", "Y_MIN", "X_MAX", "Y_MAX")
    )
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if args.model not in config["models"]:
        parser.error(f"{args.model} is not configured in {config_path}")
    image_path = args.image if args.image.is_absolute() else ROOT / args.image
    with Image.open(image_path) as source:
        rgb = np.asarray(source.convert("RGB"))
    height, width = rgb.shape[:2]
    try:
        prompt = make_demo_prompt(
            width, height,
            point=tuple(args.point) if args.point else None,
            box=tuple(args.box) if args.box else None,
        )
    except ValueError as error:
        parser.error(str(error))

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output_dir = args.output_dir or Path("results/demo") / f"{args.model}_{stamp}"
    output_dir = output_dir if output_dir.is_absolute() else ROOT / output_dir
    output_files = [output_dir / name for name in ("mask.png", "overlay.png", "result.json")]
    if any(path.exists() for path in output_files):
        parser.error(f"Output files already exist in {output_dir}; choose a new --output-dir")

    model = create_model(args.model, ROOT, config["models"][args.model], args.device)
    encode_seconds = model.set_image(rgb)
    prediction = model.predict(prompt)
    output_dir.mkdir(parents=True, exist_ok=True)

    mask = np.asarray(prediction.mask, dtype=bool)
    Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(output_files[0])
    overlay = rgb.copy()
    overlay[mask] = (0.55 * overlay[mask] + 0.45 * np.array([30, 220, 80])).astype(np.uint8)
    preview = Image.fromarray(overlay)
    drawing = ImageDraw.Draw(preview)
    if prompt.point_xy is not None:
        x, y = prompt.point_xy
        drawing.ellipse((x - 5, y - 5, x + 5, y + 5), fill="yellow", outline="black")
    else:
        drawing.rectangle(prompt.box_xyxy, outline="yellow", width=3)
    preview.save(output_files[1])
    details = {
        "image": str(image_path.resolve()),
        "model": args.model,
        "prompt_type": prompt.prompt_type,
        "point_xy": prompt.point_xy,
        "box_xyxy": prompt.box_xyxy,
        "predicted_score": prediction.score,
        "encode_seconds": encode_seconds,
        "predict_seconds": prediction.seconds,
        "mask": str(output_files[0]),
        "overlay": str(output_files[1]),
        "iou": None,
        "note": "No GT mask was provided; IoU cannot be calculated for this demo.",
    }
    output_files[2].write_text(json.dumps(details, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(details, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
