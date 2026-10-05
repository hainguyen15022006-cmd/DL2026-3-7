from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from src.data_contract import DataContractError, load_manifest, validate_dataset


class DataContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        (self.root / "configs").mkdir()
        (self.root / "data" / "coco" / "val2017").mkdir(parents=True)
        (self.root / "data" / "coco" / "gt_masks").mkdir(parents=True)

        self.row = {
            "image_id": 7,
            "annotation_id": 11,
            "file_name": "000000000007.jpg",
            "width": 4,
            "height": 3,
            "category_id": 1,
            "bbox_xywh": [1.0, 1.0, 2.0, 1.0],
            "area": 2.0,
        }
        self._write_manifest([self.row])
        Image.new("RGB", (4, 3), (10, 20, 30)).save(
            self.root / "data" / "coco" / "val2017" / self.row["file_name"]
        )
        mask = np.zeros((3, 4), dtype=np.uint8)
        mask[1, 1:3] = 255
        Image.fromarray(mask, mode="L").save(
            self.root / "data" / "coco" / "gt_masks" / "11.png"
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write_manifest(self, rows: list[dict]) -> None:
        (self.root / "configs" / "eval_manifest.json").write_text(
            json.dumps(rows), encoding="utf-8"
        )

    def test_valid_dataset_passes(self) -> None:
        report = validate_dataset(self.root, expected_count=1)
        self.assertEqual(report.status, "ok")
        self.assertEqual(report.checked_rows, 1)
        self.assertEqual(report.unique_images, 1)

    def test_mask_shape_mismatch_fails(self) -> None:
        Image.fromarray(np.full((2, 4), 255, dtype=np.uint8), mode="L").save(
            self.root / "data" / "coco" / "gt_masks" / "11.png"
        )
        with self.assertRaisesRegex(DataContractError, "mask shape"):
            validate_dataset(self.root, expected_count=1)

    def test_duplicate_image_id_fails(self) -> None:
        duplicate = dict(self.row, annotation_id=12, file_name="000000000008.jpg")
        self._write_manifest([self.row, duplicate])
        with self.assertRaisesRegex(DataContractError, "duplicate image_id"):
            load_manifest(self.root, expected_count=2)

    def test_unexpected_field_fails(self) -> None:
        invalid = dict(self.row, prompt="not part of the data handoff")
        self._write_manifest([invalid])
        with self.assertRaisesRegex(DataContractError, "extra"):
            load_manifest(self.root, expected_count=1)


if __name__ == "__main__":
    unittest.main()
