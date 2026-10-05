#!/usr/bin/env python
"""Run one adapter (MobileSAM or SAM ViT-B) on the first N instances in the group COCO manifest.

Both models use the same images, canonical prompts (deepest point in GT and tight GT box), and IoU calculation,
so their results can be compared directly. Prompt noise is not included here; it is handled by the Setup 3 runner.

  python scripts/smoke_manifest.py --adapter chuc_mobile_sam.src.models.mobile_sam:MobileSamAdapter ^
      --checkpoint weights/mobile_sam.pt --manifest configs/eval_manifest.csv ^
      --image-dir data/val2017 --ann-file data/annotations/instances_val2017.json --n 10 --out results/smoke_coco_mobile

  # Use pre-exported GT mask PNG files (names follow --gt-pattern, default: {annotation_id}.png):
  python scripts/smoke_manifest.py ... --gt-dir data/gt_masks

The manifest (CSV or JSON) must contain at least these columns: image_id, annotation_id, file_name. Other columns are ignored.
The adapter must provide __init__(checkpoint, device), set_image(img_rgb), and predict(point=... | box=...), returning
an object with .mask (bool HxW), .score, and .decode_seconds attributes. The adapter must also expose .name.

Output in --out:
  results.csv  : model, image_id, annotation_id, prompt_type, noise_level, trial, iou, seconds, status, seed
                 (+ score, encode_seconds, error). seconds is decode time; encode time is recorded on the point row per image.
  prompts.csv  : prompt coordinates (stored separately per group convention) + point_inside_gt, box_iou_gt
  overlay_*.png: image + mask + GT + prompt for the first instances
An instance failure does not stop the run: its rows have status=error with a diagnostic message.
"""
import argparse
import csv
import importlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from common import deepest_point, inside, iou, panel, save_png, tight_box   # noqa: E402

RESULT_FIELDS = ["model", "image_id", "annotation_id", "prompt_type", "noise_level", "trial", "iou", "seconds",
                 "status", "seed", "score", "encode_seconds", "error"]
PROMPT_FIELDS = ["image_id", "annotation_id", "prompt_type", "x", "y", "x_min", "y_min",
                 "x_max", "y_max", "point_inside_gt", "box_iou_gt"]
REQUIRED = {"image_id", "annotation_id", "file_name"}


def load_manifest(path):
    if path.lower().endswith(".json"):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        rows = data if isinstance(data, list) else next((data[k] for k in ("instances", "items", "manifest", "data")
                                                         if k in data), None)
        if rows is None:
            sys.exit("JSON manifest must be a list of rows or contain one of these keys: instances/items/manifest/data.")
    else:
        with open(path, newline="", encoding="utf-8-sig") as f:        # utf-8-sig handles the BOM used by Excel-exported CSV files
            rows = list(csv.DictReader(f))
    if not rows:
        sys.exit("Manifest is empty.")
    missing = REQUIRED - set(rows[0])
    if missing:
        sys.exit(f"Manifest is missing columns {sorted(missing)}; found {sorted(rows[0])}.")
    return rows


class GtFromCoco:
    """Decode GT directly from instances_val2017.json with pycocotools (requires annotation_id)."""

    def __init__(self, ann_file):
        from pycocotools.coco import COCO
        self.coco = COCO(ann_file)

    def mask(self, row):
        aid = int(row["annotation_id"])
        if aid not in self.coco.anns:
            raise KeyError(f"annotation_id {aid} is not present in the annotation file")
        return self.coco.annToMask(self.coco.anns[aid]).astype(bool)


class GtFromDir:
    def __init__(self, directory, pattern):
        self.dir, self.pattern = directory, pattern

    def mask(self, row):
        p = os.path.join(self.dir, self.pattern.format(**row))
        if p.lower().endswith(".npy"):
            return np.load(p).astype(bool)
        from PIL import Image
        return np.array(Image.open(p).convert("L")) > 0


def load_adapter_class(spec):
    mod, _, cls = spec.partition(":")
    if not cls:
        sys.exit("--adapter must use the module:Class format, for example src.models.mobile_sam:MobileSamAdapter")
    return getattr(importlib.import_module(mod), cls)


def write_csv(path, fields, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fields)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", required=True, help="module:Class, e.g. chuc_mobile_sam.src.models.mobile_sam:MobileSamAdapter")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--image-dir", required=True)
    ap.add_argument("--ann-file", help="instances_val2017.json (decode GT with pycocotools)")
    ap.add_argument("--gt-dir", help="Directory containing pre-exported GT masks")
    ap.add_argument("--gt-pattern", default="{annotation_id}.png")
    ap.add_argument("--n", type=int, default=10, help="Number of first manifest instances to process (smoke test: 10)")
    ap.add_argument("--seed", type=int, default=2026, help="Written to CSV only (this script has no random component)")
    ap.add_argument("--max-overlays", type=int, default=10)
    ap.add_argument("--out", default="results/smoke_coco")
    a = ap.parse_args()
    if bool(a.ann_file) == bool(a.gt_dir):
        sys.exit("Choose exactly one GT source: --ann-file OR --gt-dir.")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    from PIL import Image
    rows = load_manifest(a.manifest)[:a.n]
    ids = [r["image_id"] for r in rows]
    if len(set(ids)) != len(ids):
        print("WARNING: manifest contains duplicate image_id values (group convention: one instance per image).")
    gtp = GtFromCoco(a.ann_file) if a.ann_file else GtFromDir(a.gt_dir, a.gt_pattern)
    model = load_adapter_class(a.adapter)(a.checkpoint, a.device)
    os.makedirs(a.out, exist_ok=True)

    results, prompts, n_overlay = [], [], 0
    for row in rows:
        base = {"model": getattr(model, "name", a.adapter), "image_id": row["image_id"],
                "annotation_id": row["annotation_id"], "noise_level": 0, "trial": 0, "seed": a.seed}
        try:
            img = np.array(Image.open(os.path.join(a.image_dir, row["file_name"])).convert("RGB"))
            gt = gtp.mask(row)
            if gt.shape != img.shape[:2]:
                raise ValueError(f"GT shape {gt.shape} does not match image shape {img.shape[:2]}")
            if not gt.any():
                raise ValueError("GT is empty")
            pt, bx = deepest_point(gt), tight_box(gt)
            enc = model.set_image(img)
        except Exception as e:
            msg = f"error: {type(e).__name__}: {e}".replace("\n", " ")[:300]
            for ptype in ("point", "box"):
                results.append({**base, "prompt_type": ptype, "iou": "", "seconds": "", "status": "error",
                                "score": "", "encode_seconds": "", "error": msg})
            print(f"[ERROR] annotation_id={row['annotation_id']}: {msg}")
            continue

        panels = []
        for ptype, kw in (("point", {"point": pt}), ("box", {"box": bx})):
            try:
                r = model.predict(**kw)
                assert r.mask.dtype == bool and r.mask.shape == gt.shape, "Mask must be bool with shape (H, W), matching GT"
                v = iou(r.mask, gt)
                results.append({**base, "prompt_type": ptype, "iou": round(v, 6), "seconds": round(r.decode_seconds, 6),
                                "status": "ok", "score": round(float(r.score), 6),
                                "encode_seconds": round(enc, 6) if ptype == "point" else 0.0, "error": ""})
                prompts.append({"image_id": row["image_id"],
                                "annotation_id": row["annotation_id"], "prompt_type": ptype,
                                "x": pt[0] if ptype == "point" else "", "y": pt[1] if ptype == "point" else "",
                                "x_min": bx[0] if ptype == "box" else "", "y_min": bx[1] if ptype == "box" else "",
                                "x_max": bx[2] if ptype == "box" else "", "y_max": bx[3] if ptype == "box" else "",
                                "point_inside_gt": inside(gt, pt) if ptype == "point" else "",
                                "box_iou_gt": 1.0 if ptype == "box" else ""})
                try:
                    panels.append(panel(img, r.mask, gt, pt if ptype == "point" else None,
                                        bx if ptype == "box" else None, f"{ptype}  IoU={v:.3f}"))
                except Exception as overlay_error:
                    print(f"[WARNING] Could not compose {ptype} overlay for annotation_id={row['annotation_id']}: {overlay_error}")
            except Exception as e:
                msg = f"error: {type(e).__name__}: {e}".replace("\n", " ")[:300]
                results.append({**base, "prompt_type": ptype, "iou": "", "seconds": "", "status": "error",
                                "score": "", "encode_seconds": "", "error": msg})
                print(f"[ERROR] annotation_id={row['annotation_id']} prompt_type={ptype}: {msg}")

        if panels and n_overlay < a.max_overlays:
            try:
                save_png(os.path.join(a.out, f"overlay_{row['annotation_id']}.png"), np.concatenate(panels, axis=1))
                n_overlay += 1
            except Exception as overlay_error:
                print(f"[WARNING] Could not save overlay for annotation_id={row['annotation_id']}: {overlay_error}")

    write_csv(os.path.join(a.out, "results.csv"), RESULT_FIELDS, results)
    write_csv(os.path.join(a.out, "prompts.csv"), PROMPT_FIELDS, prompts)

    ok = [r for r in results if r["status"] == "ok"]
    print(f"\nmodel={base['model']} | {len(rows)} instances | {len(ok)} successful rows, {len(results) - len(ok)} error rows")
    for ptype in ("point", "box"):
        v = np.array([r["iou"] for r in ok if r["prompt_type"] == ptype], float)
        s = np.array([r["seconds"] for r in ok if r["prompt_type"] == ptype], float)
        if len(v):
            print(f"  {ptype:5s}: n={len(v)}  mean IoU={v.mean():.3f}  median={np.median(v):.3f}  "
                  f"min={v.min():.3f}  decode={s.mean() * 1000:.0f} ms")
    off = [p for p in prompts if p["prompt_type"] == "point" and p["point_inside_gt"] is False]
    if off:
        print(f"  WARNING: {len(off)} canonical points fall outside GT (check GT/manifest).")
    print(f"-> {a.out}/ (results.csv, prompts.csv, overlay_*.png)")


if __name__ == "__main__":
    main()
