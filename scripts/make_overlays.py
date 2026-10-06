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
OUT.mkdir(parents=True, exist_ok=True)

sel = pd.read_csv("results/selected_examples.csv")
prompts = pd.read_csv("results/prompts.csv").set_index("prompt_id")
raw = pd.read_csv("results/raw_predictions.csv")
manifest = {r["annotation_id"]: r for r in json.load(open("configs/eval_manifest.json"))}


def load_mask(path):
    return np.array(Image.open(path).convert("L")) > 127


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

    clean_id = f"ann{ann}_{row['prompt_type']}_n00_t0"
    clean_p = prompts.loc[clean_id]
    clean_row = raw[(raw.prompt_id == clean_id) & (raw.model == "sam_vit_b")].iloc[0]
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


for i, (_, row) in enumerate(sel.iterrows(), 1):
    fig = render(row, OUT / f"example_{i}_{row['group']}_ann{int(row['annotation_id'])}.png")
    plt.close(fig)
    print("saved example", i, row["prompt_id"], "IoU", round(row["iou"], 4))

# Combined figure (stack the 4 images)
files = sorted(OUT.glob("example_*.png"))
imgs = [Image.open(f) for f in files]
W = max(i.width for i in imgs)
canvas = Image.new("RGB", (W, sum(i.height for i in imgs)), "white")
y = 0
for im in imgs:
    canvas.paste(im, (0, y))
    y += im.height
canvas.save(OUT / "figure_examples.png")
print("saved", OUT / "figure_examples.png")
