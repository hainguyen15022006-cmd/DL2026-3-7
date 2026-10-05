"""Fake adapter for testing scripts/smoke_manifest.py without a checkpoint.

Box prompts produce filled box masks; point prompts produce circles of radius 10.
"""

import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from chuc_mobile_sam.src.models.mobile_sam import SegResult, check_box, check_image, check_point  # noqa: E402


class FakeAdapter:
    """Minimal deterministic adapter implementing the MobileSAM runner contract."""

    name = "fake"

    def __init__(self, checkpoint, device="auto"):
        self._hw = None

    def set_image(self, image_rgb):
        self._hw = check_image(image_rgb).shape[:2]
        return 0.01

    def predict(self, point=None, box=None):
        height, width = self._hw
        mask = np.zeros((height, width), dtype=bool)
        if box is not None:
            x0, y0, x1, y1 = check_box(box, self._hw).astype(int)
            mask[y0:y1 + 1, x0:x1 + 1] = True
        else:
            x, y = check_point(point, self._hw)
            yy, xx = np.mgrid[:height, :width]
            mask = (xx - x) ** 2 + (yy - y) ** 2 <= 100
        return SegResult(mask=mask, score=0.5, encode_seconds=0.0, decode_seconds=0.002)


class FailOnBoxAdapter(FakeAdapter):
    """Adapter that succeeds for point prompts and fails for box prompts."""

    def predict(self, point=None, box=None):
        if box is not None:
            raise RuntimeError("simulated box prediction failure")
        return super().predict(point=point)
