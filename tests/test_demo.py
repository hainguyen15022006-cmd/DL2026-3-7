from __future__ import annotations

import unittest
from pathlib import Path
import json
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

from scripts import demo_image
from src.demo import make_demo_prompt


class DemoPromptTests(unittest.TestCase):
    def test_valid_point_and_box(self) -> None:
        point = make_demo_prompt(100, 80, point=(20, 30))
        box = make_demo_prompt(100, 80, box=(10, 20, 40, 60))
        self.assertEqual(point.point_label, 1)
        self.assertEqual(point.point_xy, (20, 30))
        self.assertEqual(box.box_xyxy, (10, 20, 40, 60))

    def test_rejects_out_of_bounds_or_ambiguous_prompts(self) -> None:
        with self.assertRaisesRegex(ValueError, "outside image"):
            make_demo_prompt(100, 80, point=(100, 30))
        with self.assertRaisesRegex(ValueError, "exactly one"):
            make_demo_prompt(100, 80)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            make_demo_prompt(100, 80, point=(20, 30), box=(1, 1, 2, 2))

    def test_demo_writes_mask_overlay_and_metadata(self) -> None:
        class FakeModel:
            def set_image(self, image: np.ndarray) -> float:
                self.shape = image.shape[:2]
                return 0.1

            def predict(self, prompt):
                mask = np.zeros(self.shape, dtype=bool)
                mask[1:3, 2:5] = True
                return SimpleNamespace(mask=mask, score=0.8, seconds=0.02)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "configs").mkdir()
            (root / "configs" / "experiment.json").write_text(
                json.dumps({"models": {"sam_vit_b": {}}}), encoding="utf-8"
            )
            Image.new("RGB", (10, 8), "white").save(root / "image.png")
            output_dir = root / "demo-output"
            with (
                patch.object(demo_image, "ROOT", root),
                patch.object(demo_image, "create_model", return_value=FakeModel()),
                patch("sys.argv", [
                    "demo_image.py", "--image", str(root / "image.png"),
                    "--point", "3", "2", "--output-dir", str(output_dir),
                ]),
            ):
                self.assertEqual(demo_image.main(), 0)
            self.assertTrue((output_dir / "mask.png").is_file())
            self.assertTrue((output_dir / "overlay.png").is_file())
            details = json.loads((output_dir / "result.json").read_text())
            self.assertEqual(details["prompt_type"], "point")
            self.assertIsNone(details["iou"])


if __name__ == "__main__":
    unittest.main()
