"""Tests for smoke_mobile_sam helpers (no model required)."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

# Ensure the repo root is importable
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.smoke_mobile_sam import (  # noqa: E402
    SMOKE_CSV_FIELDS,
    SmokeRow,
    compose_overlay,
    compute_abs_diff,
    gt_contour_coords,
    lookup_committed_iou,
    write_smoke_csv,
)
from src.prompts import PromptRecord  # noqa: E402


def _make_prompt(prompt_type: str = "point") -> PromptRecord:
    """Create a minimal clean prompt for testing."""
    return PromptRecord(
        prompt_id="ann11_point_n00_t0" if prompt_type == "point" else "ann11_box_n00_t0",
        image_id=7,
        annotation_id=11,
        prompt_type=prompt_type,
        noise_level=0,
        trial=0,
        seed=2026,
        point_xy=(4, 3) if prompt_type == "point" else None,
        point_label=1 if prompt_type == "point" else None,
        box_xyxy=(2, 1, 6, 5) if prompt_type == "box" else None,
        sx=0,
        sy=0,
        dx=0,
        dy=0,
        point_inside_gt=True if prompt_type == "point" else None,
        box_iou_gt=1.0 if prompt_type == "box" else None,
    )


class TestComputeAbsDiff(unittest.TestCase):
    def test_both_present(self) -> None:
        self.assertAlmostEqual(compute_abs_diff(0.85, 0.80), 0.05)

    def test_equal(self) -> None:
        self.assertAlmostEqual(compute_abs_diff(0.5, 0.5), 0.0)

    def test_smoke_none(self) -> None:
        self.assertIsNone(compute_abs_diff(None, 0.5))

    def test_committed_none(self) -> None:
        self.assertIsNone(compute_abs_diff(0.5, None))

    def test_both_none(self) -> None:
        self.assertIsNone(compute_abs_diff(None, None))


class TestLookupCommittedIou(unittest.TestCase):
    def test_found(self) -> None:
        committed = {"mobile_sam:ann11_point_n00_t0": {"iou": "0.75", "status": "ok"}}
        result = lookup_committed_iou(committed, "mobile_sam", "ann11_point_n00_t0")
        self.assertAlmostEqual(result, 0.75)

    def test_not_found(self) -> None:
        result = lookup_committed_iou({}, "mobile_sam", "ann11_point_n00_t0")
        self.assertIsNone(result)

    def test_empty_iou(self) -> None:
        committed = {"mobile_sam:ann11_point_n00_t0": {"iou": "", "status": "error"}}
        result = lookup_committed_iou(committed, "mobile_sam", "ann11_point_n00_t0")
        self.assertIsNone(result)


class TestGtContourCoords(unittest.TestCase):
    def test_single_pixel(self) -> None:
        mask = np.zeros((5, 5), dtype=bool)
        mask[2, 2] = True
        coords = gt_contour_coords(mask)
        self.assertEqual(coords, [(2, 2)])

    def test_filled_square_returns_boundary(self) -> None:
        mask = np.zeros((7, 9), dtype=bool)
        mask[1:6, 2:7] = True
        coords = gt_contour_coords(mask)
        # Interior pixels should NOT be in boundary
        self.assertNotIn((4, 3), coords)  # center-ish interior
        # Some boundary pixels should be present
        self.assertIn((2, 1), coords)  # top-left corner


class TestComposeOverlay(unittest.TestCase):
    def test_point_overlay_shape_and_mode(self) -> None:
        image = np.random.randint(0, 256, (20, 30, 3), dtype=np.uint8)
        gt = np.zeros((20, 30), dtype=bool)
        gt[5:15, 10:20] = True
        pred = np.zeros((20, 30), dtype=bool)
        pred[6:14, 11:19] = True
        prompt = _make_prompt("point")
        result = compose_overlay(image, gt, pred, prompt, 0.75)
        self.assertIsInstance(result, Image.Image)
        self.assertEqual(result.size, (30, 20))  # Pillow (w, h)
        self.assertEqual(result.mode, "RGB")
        self.assertEqual(np.asarray(result).shape, (20, 30, 3))
        self.assertEqual(np.asarray(result).dtype, np.uint8)

    def test_box_overlay_shape_and_mode(self) -> None:
        image = np.random.randint(0, 256, (20, 30, 3), dtype=np.uint8)
        gt = np.zeros((20, 30), dtype=bool)
        gt[1:6, 2:7] = True
        pred = gt.copy()
        prompt = _make_prompt("box")
        result = compose_overlay(image, gt, pred, prompt, 1.0)
        self.assertIsInstance(result, Image.Image)
        self.assertEqual(result.size, (30, 20))
        self.assertEqual(result.mode, "RGB")
        self.assertEqual(np.asarray(result).shape, (20, 30, 3))
        self.assertEqual(np.asarray(result).dtype, np.uint8)


class TestWriteSmokeCsv(unittest.TestCase):
    def test_csv_round_trip(self) -> None:
        row = SmokeRow(
            model="mobile_sam",
            image_id=7,
            annotation_id=11,
            prompt_type="point",
            noise_level=0,
            trial=0,
            iou="0.85",
            seconds="0.123",
            status="ok",
            seed=2026,
            prompt_id="ann11_point_n00_t0",
            encode_seconds="1.5",
            predicted_score="0.9",
            error="",
            committed_iou="0.84",
            abs_diff="0.01",
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "smoke.csv"
            write_smoke_csv(path, [row])

            with path.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            self.assertEqual(len(rows), 1)
            self.assertEqual(list(rows[0].keys()), list(SMOKE_CSV_FIELDS))
            self.assertEqual(rows[0]["model"], "mobile_sam")
            self.assertEqual(rows[0]["iou"], "0.85")
            self.assertEqual(rows[0]["status"], "ok")

    def test_error_row_has_empty_iou(self) -> None:
        row = SmokeRow(
            model="mobile_sam",
            image_id=7,
            annotation_id=11,
            prompt_type="point",
            noise_level=0,
            trial=0,
            iou="",
            seconds="",
            status="error",
            seed=2026,
            prompt_id="ann11_point_n00_t0",
            encode_seconds="",
            predicted_score="",
            error="set_image failed",
            committed_iou="",
            abs_diff="",
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "smoke.csv"
            write_smoke_csv(path, [row])

            with path.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            self.assertEqual(rows[0]["status"], "error")
            self.assertEqual(rows[0]["iou"], "")


class TestSmokeWeightsSkip(unittest.TestCase):
    """Model-dependent tests that skip when weights are absent."""

    @unittest.skipUnless(
        (Path(__file__).resolve().parents[1] / "weights" / "mobile_sam.pt").is_file(),
        "mobile_sam.pt not found",
    )
    def test_model_loads(self) -> None:
        """Verify the model can be instantiated (requires weights)."""
        import json

        config = json.loads(
            (ROOT / "configs" / "experiment.json").read_text(encoding="utf-8")
        )
        model = __import__("src.models.factory", fromlist=["create_model"]).create_model(
            "mobile_sam", ROOT, config["models"]["mobile_sam"], "cpu"
        )
        self.assertEqual(model.name, "mobile_sam")


if __name__ == "__main__":
    unittest.main()
