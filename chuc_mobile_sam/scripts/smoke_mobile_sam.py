#!/usr/bin/env python
"""Smoke-test MobileSAM with one point and one box; verify I/O, report timings, and save an overlay.

  # Default: use a synthetic image with GT (IoU can be computed); no dataset is required
  python scripts/smoke_mobile_sam.py --checkpoint weights/mobile_sam.pt

  # Use a real group image (for example, one COCO image) and custom prompts in original-image pixels;
  # pass --gt to compute IoU
  python scripts/smoke_mobile_sam.py --checkpoint weights/mobile_sam.pt --image img.jpg \
         --point 320 240 --box 100 80 500 400 [--gt gt_mask.png]

To run on multiple instances from the group COCO manifest, use scripts/smoke_manifest.py.
Output: <out>/overlay_point_box.png (and IoU in the console when GT is available).
"""
import argparse
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from common import deepest_point, iou, panel, save_png, tight_box   # noqa: E402
from src.models.mobile_sam import MobileSamAdapter                  # noqa: E402


def synthetic(h=480, w=640, seed=0):
    rng = np.random.default_rng(seed)
    base = np.array([90, 130, 170], np.float32)
    noise = cv2.GaussianBlur(rng.normal(0, 12, (h, w, 3)).astype(np.float32), (0, 0), 8)
    img = np.clip(base + noise, 0, 255).astype(np.uint8)
    gt = np.zeros((h, w), np.uint8)
    cv2.ellipse(gt, (330, 250), (120, 80), 20, 0, 360, 255, -1)
    img[gt > 0] = np.array([90, 190, 110], np.uint8)          # Green object makes the red predicted mask easy to distinguish
    return img, gt > 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--image")
    ap.add_argument("--gt", help="Binary GT mask (PNG) matching the image size, used to compute IoU")
    ap.add_argument("--point", nargs=2, type=float, metavar=("X", "Y"))
    ap.add_argument("--box", nargs=4, type=float, metavar=("X0", "Y0", "X1", "Y1"))
    ap.add_argument("--out", default="results/smoke_mobile_sam")
    a = ap.parse_args()

    gt = None
    if a.image:
        from PIL import Image
        img = np.array(Image.open(a.image).convert("RGB"))                   # Pillow always converts to RGB
        if a.gt:
            gt = np.array(Image.open(a.gt).convert("L")) > 0
            assert gt.shape == img.shape[:2], f"GT shape {gt.shape} does not match image shape {img.shape[:2]}"
        point, box = a.point, a.box
        if point is None and box is None:
            sys.exit("Pass --point and/or --box when using --image, or omit --image to use a synthetic image.")
    else:
        img, gt = synthetic()
        point, box = deepest_point(gt), tight_box(gt)

    m = MobileSamAdapter(a.checkpoint, a.device)
    print(f"device={m.device} | image {img.shape[1]}x{img.shape[0]} | encoder img_size = {m.predictor.model.image_encoder.img_size}")
    enc = m.set_image(img)
    panels = []
    for kind, kw in (("point", {"point": point}), ("box", {"box": box})):
        if kw.get(kind) is None:
            continue
        r = m.predict(**kw)
        assert r.mask.dtype == bool and r.mask.shape == img.shape[:2], "Contract violation: mask must be bool with shape (H, W)"
        s = f"{kind:5s}: score_model={r.score:.3f} | decode={r.decode_seconds * 1000:.0f} ms | mask_px={int(r.mask.sum())}"
        v = iou(r.mask, gt) if gt is not None else None
        if v is not None:
            s += f" | IoU_vs_GT={v:.3f}"
        print(s)
        panels.append(panel(img, r.mask, gt, point if kind == "point" else None, box if kind == "box" else None,
                            f"{kind}" + (f"  IoU={v:.3f}" if v is not None else "")))
    print(f"encode={enc:.2f} s (once per image)")
    path = os.path.join(a.out, "overlay_point_box.png")
    save_png(path, np.concatenate(panels, axis=1))
    print(f"-> {path}")


if __name__ == "__main__":
    main()
