"""Shared helpers for smoke scripts: IoU, canonical GT prompts, and overlays. Requires only NumPy and OpenCV."""
import os

import cv2
import numpy as np


def iou(pred, gt):
    """Compute pixel IoU = TP / (TP + FP + FN); two empty masks yield 1.0."""
    pred, gt = np.asarray(pred, bool), np.asarray(gt, bool)
    union = np.logical_or(pred, gt).sum()
    return 1.0 if union == 0 else float(np.logical_and(pred, gt).sum() / union)


def deepest_point(mask):
    """Return the deepest point in the GT as the canonical positive point.

    A one-pixel border is added so image edges are treated as boundaries by the
    distance transform.
    """
    pad = np.pad(np.asarray(mask, bool), 1).astype(np.uint8)
    d = cv2.distanceTransform(pad, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)[1:-1, 1:-1]
    y, x = np.unravel_index(np.argmax(d), d.shape)
    return float(x), float(y)


def tight_box(mask):
    """Return the inclusive tight GT box [x_min, y_min, x_max, y_max]."""
    ys, xs = np.nonzero(mask)
    return [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]


def inside(mask, xy):
    h, w = mask.shape
    x = int(np.clip(round(xy[0]), 0, w - 1))
    y = int(np.clip(round(xy[1]), 0, h - 1))
    return bool(mask[y, x])


def panel(img_rgb, mask, gt, point=None, box=None, title=""):
    """Compose an image with the predicted mask (red), GT contour (cyan), and prompt (yellow)."""
    out = img_rgb.copy()
    layer = out.copy()
    layer[mask] = (255, 40, 40)
    out = cv2.addWeighted(layer, 0.5, out, 0.5, 0)
    if gt is not None:
        cnts, _ = cv2.findContours(gt.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(out, cnts, -1, (0, 229, 255), 2)
    if point is not None:
        cv2.drawMarker(out, (int(round(point[0])), int(round(point[1]))), (255, 214, 10), cv2.MARKER_STAR, 26, 3)
    if box is not None:
        cv2.rectangle(out, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 214, 10), 2)
    cv2.putText(out, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def save_png(path, rgb):
    """Write a PNG with imencode + tofile for Windows paths containing non-ASCII characters."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    assert ok
    buf.tofile(path)
