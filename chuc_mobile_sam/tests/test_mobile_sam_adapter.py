"""Test the MobileSAM adapter I/O contract.

* Input validation tests run WITHOUT a checkpoint.
* Model tests require a checkpoint; set MOBILE_SAM_CKPT or place the file at weights/mobile_sam.pt.
  Otherwise, these tests are skipped.
"""
import os
import sys
import builtins

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(ROOT)
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from common import tight_box  # noqa: E402
from chuc_mobile_sam.src.models.mobile_sam import (  # noqa: E402
    MobileSamAdapter,
    check_box,
    check_image,
    check_point,
    coco_xywh_to_xyxy,
)

CKPT = os.environ.get("MOBILE_SAM_CKPT", os.path.join(ROOT, "weights", "mobile_sam.pt"))
needs_ckpt = pytest.mark.skipif(not os.path.isfile(CKPT), reason=f"Checkpoint not found: {CKPT}")


# ---------------------------------------------------------------- input validation (no checkpoint required)
def test_coco_bbox_conversion():
    assert coco_xywh_to_xyxy([10, 20, 30, 40]) == [10.0, 20.0, 40.0, 60.0]


def test_tight_box_uses_inclusive_pixel_coordinates():
    mask = np.zeros((8, 9), dtype=bool)
    mask[2:7, 3:8] = True
    assert tight_box(mask) == [3.0, 2.0, 7.0, 6.0]


def test_check_image():
    assert check_image(np.zeros((5, 7, 3), np.uint8)).shape == (5, 7, 3)
    for bad in (np.zeros((5, 7), np.uint8), np.zeros((5, 7, 4), np.uint8), np.zeros((5, 7, 3), np.float32)):
        with pytest.raises(ValueError):
            check_image(bad)


def test_check_point_and_box():
    hw = (100, 200)                                   # H=100, W=200
    assert check_point((199, 99), hw).tolist() == [199.0, 99.0]
    for bad in ((-1, 5), (5, 101), (float("nan"), 3), (1, 2, 3)):
        with pytest.raises(ValueError):
            check_point(bad, hw)
    assert check_box([10, 10, 50, 60], hw).tolist() == [10, 10, 50, 60]
    for bad in ([50, 10, 10, 60], [10, 10, 10, 60], [0, 0, 201, 50], [-1, 0, 5, 5], [1, 2, 3]):
        with pytest.raises(ValueError):
            check_box(bad, hw)
    with pytest.raises(ValueError):                   # A COCO [x,y,w,h] bbox passed by mistake is rejected when w,h < x,y
        check_box([100, 80, 30, 20], hw)


def test_missing_or_bad_checkpoint_does_not_require_torch(tmp_path, monkeypatch):
    original_import = builtins.__import__

    def import_without_torch(name, *args, **kwargs):
        if name == "torch":
            raise ModuleNotFoundError("torch intentionally unavailable", name="torch")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_torch)
    with pytest.raises(FileNotFoundError):
        MobileSamAdapter(str(tmp_path / "missing.pt"))
    tiny = tmp_path / "pointer.pt"
    tiny.write_text("version https://git-lfs.github.com/spec/v1")
    with pytest.raises(ValueError):
        MobileSamAdapter(str(tiny))


# ---------------------------------------------------------------- real model
def _disc(h=150, w=210, c=(110, 70), r=40):
    yy, xx = np.mgrid[:h, :w]
    gt = (xx - c[0]) ** 2 + (yy - c[1]) ** 2 <= r ** 2
    img = np.full((h, w, 3), (60, 100, 140), np.uint8)
    img[gt] = (220, 80, 50)
    return img, gt


@pytest.fixture(scope="module")
def model():
    return MobileSamAdapter(CKPT, "cpu")


@needs_ckpt
def test_contract_point_and_box(model):
    img, gt = _disc()                                  # Non-square image (150x210) catches swapped H/W dimensions
    model.set_image(img)
    for r in (model.predict(point=(110, 70)), model.predict(box=[70, 30, 151, 111])):
        assert r.mask.dtype == bool and r.mask.shape == img.shape[:2]
        assert np.isfinite(r.score) and r.decode_seconds > 0 and r.encode_seconds == 0.0
        inter = np.logical_and(r.mask, gt).sum()
        assert inter / np.logical_or(r.mask, gt).sum() > 0.9        # The disc on a flat background should segment very well


@needs_ckpt
def test_deterministic_and_embedding_reuse(model):
    img, _ = _disc()
    enc = model.set_image(img)
    assert enc > 0
    a = model.predict(point=(110, 70)).mask
    b = model.predict(point=(110, 70)).mask
    assert np.array_equal(a, b)
    c = model.segment(img, point=(110, 70))
    assert np.array_equal(a, c.mask) and c.encode_seconds > 0


@needs_ckpt
def test_api_misuse(model):
    img, _ = _disc()
    fresh = MobileSamAdapter(CKPT, "cpu")
    with pytest.raises(RuntimeError):
        fresh.predict(point=(5, 5))                    # set_image has not been called
    model.set_image(img)
    with pytest.raises(ValueError):
        model.predict()                                # Missing prompt
    with pytest.raises(ValueError):
        model.predict(point=(5, 5), box=[1, 1, 9, 9])  # Both prompt types were passed
