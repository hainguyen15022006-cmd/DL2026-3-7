"""Draw qualitative overlays for the 4 examples in results/selected_examples.csv.

Run from the repo root:  python scripts/make_overlays.py
Outputs: results/examples/example_*.png and results/examples/figure_examples.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

OUT = Path("results/examples")

sel = pd.read_csv("results/selected_examples.csv")
prompts = pd.read_csv("results/prompts.csv").set_index("prompt_id")
raw = pd.read_csv("results/raw_predictions.csv")
with open("configs/eval_manifest.json", encoding="utf-8") as manifest_file:
    manifest = {r["annotation_id"]: r for r in json.load(manifest_file)}


def load_mask(path):
    return np.array(Image.open(path).convert("L")) > 127


def clean_prediction(row):
    """Find the shared-run clean reference for one selected SAM prediction."""
    clean_id = f"ann{int(row['annotation_id'])}_{row['prompt_type']}_n00_t0"
    matches = raw[(raw.prompt_id == clean_id) & (raw.model == "sam_vit_b")]
    if len(matches) != 1:
        raise SystemExit(f"Expected exactly one shared-run clean prediction: {clean_id}")
    return clean_id, matches.iloc[0]


def preflight():
    """Validate every input before replacing any saved qualitative figure."""
    required = {}
    for _, row in sel.iterrows():
        if row["model"] != "sam_vit_b" or row["status"] != "ok":
            raise SystemExit(f"Unsupported selected result: {row['run_id']}")
        matches = raw[(raw.run_id == row["run_id"]) & (raw.model == "sam_vit_b")]
        if len(matches) != 1:
            raise SystemExit(f"Selected result is not unique in the shared run: {row['run_id']}")
        source = matches.iloc[0]
        if abs(float(source["iou"]) - float(row["iou"])) > 1e-9:
            raise SystemExit(f"Selected IoU differs from shared-run CSV: {row['run_id']}")
        required[str(source["run_id"])] = source
        _, clean_row = clean_prediction(row)
        required[str(clean_row["run_id"])] = clean_row

    missing = []
    for row in required.values():
        ann = int(row["annotation_id"])
        if ann not in manifest:
            raise SystemExit(f"Annotation {ann} is absent from the fixed manifest")
        paths = (
            Path(row["mask_path"]),
            Path("data/coco/gt_masks") / f"{ann}.png",
            Path("data/coco/val2017") / manifest[ann]["file_name"],
        )
        missing.extend(str(path) for path in paths if not path.is_file())
    if missing:
        preview = "\n  ".join(sorted(set(missing))[:12])
        raise SystemExit(
            "Cannot regenerate overlays: required files are missing. The prediction "
            "masks must come from the shared 800-row run, not Dương's separate "
            f"700-row run. Missing:\n  {preview}"
        )

    for row in required.values():
        ann = int(row["annotation_id"])
        pred = load_mask(row["mask_path"])
        gt = load_mask(Path("data/coco/gt_masks") / f"{ann}.png")
        if pred.shape != gt.shape:
            raise SystemExit(f"Prediction/GT shape mismatch: {row['run_id']}")
        union = np.logical_or(pred, gt).sum()
        calculated = float(np.logical_and(pred, gt).sum() / union) if union else 1.0
        if abs(calculated - float(row["iou"])) > 1e-9:
            raise SystemExit(
                f"Mask does not reproduce the shared-run IoU for {row['run_id']}: "
                f"mask={calculated:.9f}, CSV={float(row['iou']):.9f}. "
                "Do not substitute masks from another run."
            )


def overlay(ax, img, mask, color, alpha=0.5):
    ax.imshow(img)
    layer = np.zeros((*mask.shape, 4))
    layer[mask] = [*color, alpha]
    ax.imshow(layer)
    ax.contour(mask.astype(float), levels=[0.5], colors=[color], linewidths=1.2)


def draw_prompt(ax, p, color, ls="-"):
    if p["prompt_type"] == "box":
        x0, y0, x1, y1 = p["box_x_min"], p["box_y_min"], p["box_x_max"], p["box_y_max"]
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                   edgecolor=color, linewidth=2, linestyle=ls))
    else:
        ax.plot(p["point_x"], p["point_y"], marker="*", markersize=16,
                color=color, markeredgecolor="black")


def render(row, path_png=None):
    ann = int(row["annotation_id"])
    man = manifest[ann]
    img = np.array(Image.open(f"data/coco/val2017/{man['file_name']}").convert("RGB"))
    gt = load_mask(f"data/coco/gt_masks/{ann}.png")
    pred = load_mask(row["mask_path"])
    p = prompts.loc[row["prompt_id"]]

    clean_id, clean_row = clean_prediction(row)
    clean_p = prompts.loc[clean_id]
    clean_pred = load_mask(clean_row["mask_path"])

    fig, axs = plt.subplots(1, 3, figsize=(13, 4.6))
    # 1) ground truth + prompt
    overlay(axs[0], img, gt, (0.1, 0.8, 0.2))
    if row["noise_level"] > 0:
        draw_prompt(axs[0], clean_p, "white", "--")
    draw_prompt(axs[0], p, "yellow")
    axs[0].set_title("Ground truth (green)\n+ prompt (yellow"
                     + ("; white dashed = clean)" if row["noise_level"] > 0 else ")"), fontsize=9)
    # 2) prediction for this prompt
    overlay(axs[1], img, pred, (0.9, 0.1, 0.1))
    draw_prompt(axs[1], p, "yellow")
    axs[1].set_title(f"SAM ViT-B prediction (red)\nIoU = {row['iou']:.3f}, "
                     f"score = {row['predicted_score']:.2f}", fontsize=9)
    # 3) clean-prompt prediction for reference
    overlay(axs[2], img, clean_pred, (0.2, 0.4, 1.0))
    draw_prompt(axs[2], clean_p, "yellow")
    axs[2].set_title(f"Clean-prompt prediction (blue)\nIoU = {clean_row['iou']:.3f}", fontsize=9)
    for a in axs:
        a.axis("off")
    shift = f"{int(row['noise_level'] * 100)}% shift, trial {int(row['trial'])}" \
        if row["noise_level"] > 0 else "clean prompt"
    fig.suptitle(f"[{row['group']}] annotation {ann} | image {int(row['image_id'])} | "
                 f"{row['prompt_type']} | {shift} | box_iou_gt = {p['box_iou_gt']:.3f}",
                 fontsize=10)
    fig.tight_layout()
    if path_png:
        fig.savefig(path_png, dpi=150)
    return fig


def main():
    preflight()
    OUT.mkdir(parents=True, exist_ok=True)
    files = []
    for i, (_, row) in enumerate(sel.iterrows(), 1):
        path = OUT / f"example_{i}_{row['group']}_ann{int(row['annotation_id'])}.png"
        fig = render(row, path)
        plt.close(fig)
        files.append(path)
        print("saved example", i, row["prompt_id"], "IoU", round(row["iou"], 4))

    # Stack only the images produced in this invocation, not stale files.
    imgs = [Image.open(path) for path in files]
    width = max(img.width for img in imgs)
    canvas = Image.new("RGB", (width, sum(img.height for img in imgs)), "white")
    y = 0
    for img in imgs:
        canvas.paste(img, (0, y))
        y += img.height
        img.close()
    canvas.save(OUT / "figure_examples.png")
    print("saved", OUT / "figure_examples.png")


if __name__ == "__main__":
    main()
