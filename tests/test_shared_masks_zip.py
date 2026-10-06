"""Focused checks for the shared-run mask archive installer."""

from __future__ import annotations

import io
from pathlib import Path
import tempfile
import unittest
import zipfile

import numpy as np
from PIL import Image

from scripts.extract_shared_masks_zip import (
    archive_prefix,
    binary_mask,
    check_member_paths,
    csv_rows,
    verify_iou,
)


def png(array: np.ndarray) -> bytes:
    stream = io.BytesIO()
    Image.fromarray(array).save(stream, format="PNG")
    return stream.getvalue()


class SharedMaskInstallerTests(unittest.TestCase):
    def test_unsafe_and_duplicate_members_are_rejected(self) -> None:
        for names in (["../escape.png"], ["/absolute.png"], ["a\\b.png"], ["same", "same"]):
            with self.subTest(names=names), self.assertRaises(ValueError):
                check_member_paths(names)

    def test_nested_bundle_prefix(self) -> None:
        expected = {"results/masks/sam_vit_b/a.png"}
        names = {"son_visualization_bundle/results/masks/sam_vit_b/a.png"}
        self.assertEqual(archive_prefix(names, expected), "son_visualization_bundle/")
        with self.assertRaises(ValueError):
            archive_prefix(names, {"results/masks/sam_vit_b/b.png"})

    def test_csv_line_endings_do_not_change_rows(self) -> None:
        self.assertEqual(csv_rows(b"run_id,iou\r\na,0.5\r\n"), csv_rows(b"run_id,iou\na,0.5\n"))

    def test_mask_values_must_be_binary(self) -> None:
        with self.assertRaises(ValueError):
            binary_mask(png(np.array([[0, 128]], dtype=np.uint8)))

    def test_iou_is_recomputed_from_mask_and_gt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gt_dir = root / "data/coco/gt_masks"
            gt_dir.mkdir(parents=True)
            (gt_dir / "42.png").write_bytes(
                png(np.array([[255, 0], [0, 255]], dtype=np.uint8))
            )
            mask_bytes = png(np.array([[255, 0], [255, 0]], dtype=np.uint8))
            mask_path = "results/masks/sam_vit_b/ann42_box_n00_t0.png"
            zip_path = root / "masks.zip"
            with zipfile.ZipFile(zip_path, "w") as archive:
                archive.writestr("bundle/" + mask_path, mask_bytes)
            row = {
                "run_id": "sam_vit_b:ann42_box_n00_t0",
                "annotation_id": "42",
                "mask_path": mask_path,
                "iou": str(1 / 3),
            }
            with zipfile.ZipFile(zip_path) as archive:
                self.assertEqual(verify_iou(archive, "bundle/", row, root), mask_bytes)
                row["iou"] = "0.5"
                with self.assertRaisesRegex(ValueError, "Mask IoU differs"):
                    verify_iou(archive, "bundle/", row, root)


if __name__ == "__main__":
    unittest.main()
