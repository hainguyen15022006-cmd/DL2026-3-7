"""MobileSAM adapter (Setup 1 baseline) with the same I/O contract as the group's SAM ViT-B adapter.

Contract (per the group assignment):
  image : np.ndarray uint8, (H, W, 3), RGB channel order (NOT BGR as returned by cv2.imread)
  point : (x, y) in original-image pixels, positive label = 1
  box   : [x_min, y_min, x_max, y_max] in original-image pixels
          (COCO bbox [x, y, w, h] must be converted with coco_xywh_to_xyxy before use)
  mask  : np.ndarray bool (H, W), in the original image coordinate system and size
  multimask_output=False: exactly ONE mask is returned per prompt; the adapter NEVER uses GT to select a mask.

Usage:
    from src.models.mobile_sam import MobileSamAdapter
    m = MobileSamAdapter("weights/mobile_sam.pt")           # device="auto": use CUDA when available, otherwise CPU
    m.set_image(image_rgb)                                   # Encode the image ONCE (the most expensive step)
    r1 = m.predict(point=(x, y))                             # Reuse the embedding for multiple prompts
    r2 = m.predict(box=[x0, y0, x1, y1])
    r = m.segment(image_rgb, point=(x, y))                   # Convenience wrapper for set_image + predict
    r.mask, r.score, r.encode_seconds, r.decode_seconds

Install: pip install git+https://github.com/ChaoningZhang/MobileSAM.git   (model_type = 'vit_t')
"""
from __future__ import annotations

import os
from pathlib import Path
import sys
import time
from dataclasses import dataclass

import numpy as np

MODEL_TYPE = "vit_t"          # Original MobileSAM; do NOT use MobileSAMv2


# ----------------------------------------------------------------------------
# Input validation (pure NumPy helpers; testable without a checkpoint)
# ----------------------------------------------------------------------------
def coco_xywh_to_xyxy(bbox) -> list:
    """bbox COCO [x, y, w, h] -> [x_min, y_min, x_max, y_max]."""
    x, y, w, h = [float(v) for v in bbox]
    return [x, y, x + w, y + h]


def check_image(image_rgb) -> np.ndarray:
    a = np.asarray(image_rgb)
    if a.ndim != 3 or a.shape[2] != 3:
        raise ValueError(f"Image must have RGB shape (H, W, 3); received {a.shape}. "
                         "Convert grayscale/RGBA images to RGB first (for example, PIL: Image.open(p).convert('RGB')).")
    if a.dtype != np.uint8:
        raise ValueError(f"Image dtype must be uint8 (0-255); received {a.dtype}.")
    return np.ascontiguousarray(a)


def check_point(point, hw) -> np.ndarray:
    h, w = hw
    p = np.asarray(point, dtype=np.float32).reshape(-1)
    if p.shape != (2,) or not np.isfinite(p).all():
        raise ValueError(f"point must be a finite (x, y) pair; received {point!r}")
    if not (0 <= p[0] <= w and 0 <= p[1] <= h):
        raise ValueError(f"point {tuple(p)} is outside the {w}x{h} image (x first, y second). "
                         "The runner must clip it to the image before calling this adapter.")
    return p


def check_box(box, hw) -> np.ndarray:
    h, w = hw
    b = np.asarray(box, dtype=np.float32).reshape(-1)
    if b.shape != (4,) or not np.isfinite(b).all():
        raise ValueError(f"box must be finite [x_min, y_min, x_max, y_max] coordinates; received {box!r}")
    if not (b[2] > b[0] and b[3] > b[1]):
        raise ValueError(f"box {tuple(b)} is invalid: x_max must exceed x_min and y_max must exceed y_min. "
                         "If this is a COCO bbox [x,y,w,h], convert it with coco_xywh_to_xyxy().")
    if not (b[0] >= 0 and b[1] >= 0 and b[2] <= w and b[3] <= h):
        raise ValueError(f"box {tuple(b)} is outside the {w}x{h} image. The runner must clip it before calling this adapter.")
    return b


# ----------------------------------------------------------------------------
@dataclass
class SegResult:
    mask: np.ndarray            # bool (H, W)
    score: float                # Model-predicted IoU (not IoU against GT; may be slightly greater than 1)
    encode_seconds: float       # set_image time; 0.0 when reusing the previous image embedding
    decode_seconds: float       # Prompt-to-mask time (GPU synchronized when applicable)

    @property
    def seconds(self) -> float:
        return self.encode_seconds + self.decode_seconds


class MobileSamAdapter:
    name = "mobile_sam"
    model_type = MODEL_TYPE

    def __init__(self, checkpoint: str, device: str = "auto"):
        if not os.path.isfile(checkpoint):
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint}. See docs/checkpoints.md (weights/mobile_sam.pt from the "
                "ChaoningZhang/MobileSAM).")
        size = os.path.getsize(checkpoint)
        if size < 1_000_000:
            raise ValueError(f"{checkpoint} is only {size} bytes: it may be a broken file or Git LFS pointer, not a model weight "
                             "(the expected file is about 40.7 MB).")
        # Validate local checkpoint errors before importing heavyweight optional
        # model dependencies. This keeps input-validation tests usable in CI.
        import torch

        try:
            from mobile_sam import SamPredictor, sam_model_registry
        except ModuleNotFoundError as import_error:
            if import_error.name != "mobile_sam":
                raise
            repository_root = Path(__file__).resolve().parents[3]
            sibling_source = repository_root.parent / "MobileSAM"
            if sibling_source.is_dir():
                sys.path.insert(0, str(sibling_source))
                try:
                    from mobile_sam import SamPredictor, sam_model_registry
                except ModuleNotFoundError as nested_error:
                    if nested_error.name != "mobile_sam":
                        raise
                    raise ModuleNotFoundError(
                        "MobileSAM was found but could not be imported. Install its dependencies with "
                        "chuc_mobile_sam/requirements.txt."
                    ) from nested_error
            else:
                raise ModuleNotFoundError(
                    "MobileSAM is not installed and the sibling source directory ../MobileSAM was not found. "
                    "Install chuc_mobile_sam/requirements.txt or clone the pinned MobileSAM source next to the repository."
                ) from import_error
        self._torch = torch
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        sam = sam_model_registry[MODEL_TYPE](checkpoint=checkpoint)
        sam.to(device=self.device)
        sam.eval()                                   # Inference mode (disable dropout and BatchNorm updates)
        self.predictor = SamPredictor(sam)
        self.checkpoint = checkpoint
        self._hw = None

    # --- Internal helpers --------------------------------------------------
    def _sync(self):
        if str(self.device).startswith("cuda"):
            self._torch.cuda.synchronize()           # Synchronize for accurate GPU timing

    # --- API ---------------------------------------------------------------
    def set_image(self, image_rgb) -> float:
        """Encode an image once and return the elapsed time in seconds."""
        img = check_image(image_rgb)
        self._sync()
        t0 = time.perf_counter()
        self.predictor.set_image(img, image_format="RGB")
        self._sync()
        self._hw = img.shape[:2]
        return time.perf_counter() - t0

    def predict(self, point=None, box=None) -> SegResult:
        """Accept exactly one of point=(x, y) or box=[x0, y0, x1, y1]. Call set_image first."""
        if self._hw is None:
            raise RuntimeError("set_image(image_rgb) has not been called.")
        if (point is None) == (box is None):
            raise ValueError("Pass exactly one of point or box.")
        kw = {}
        if point is not None:
            kw["point_coords"] = check_point(point, self._hw).reshape(1, 2)
            kw["point_labels"] = np.ones(1, dtype=np.int32)          # 1 = positive point
        else:
            kw["box"] = check_box(box, self._hw)
        self._sync()
        t0 = time.perf_counter()
        masks, scores, _ = self.predictor.predict(multimask_output=False, **kw)   # Return exactly one mask
        self._sync()
        dt = time.perf_counter() - t0
        mask = np.asarray(masks[0]).astype(bool)
        assert mask.shape == self._hw, f"Mask shape {mask.shape} does not match image shape {self._hw}"
        return SegResult(mask=mask, score=float(scores[0]), encode_seconds=0.0, decode_seconds=dt)

    def segment(self, image_rgb, point=None, box=None) -> SegResult:
        """Convenience method for set_image + predict (encode_seconds is included in the result)."""
        enc = self.set_image(image_rgb)
        r = self.predict(point=point, box=box)
        r.encode_seconds = enc
        return r
