from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from src.data_contract import EvalInstance
from src.prompts import (
    PromptContractError,
    deepest_positive_point,
    generate_prompt_set,
    generate_prompts_for_instance,
    pixel_box_iou,
    tight_box,
    validate_prompt_matrix,
    write_prompt_files,
)


def instance(annotation_id: int = 11, image_id: int = 7) -> EvalInstance:
    return EvalInstance(
        image_id=image_id,
        annotation_id=annotation_id,
        file_name=f"{image_id:012d}.jpg",
        width=9,
        height=7,
        category_id=1,
        bbox_xywh=(2.0, 1.0, 5.0, 5.0),
        area=25.0,
    )


def square_mask() -> np.ndarray:
    mask = np.zeros((7, 9), dtype=bool)
    mask[1:6, 2:7] = True
    return mask


class PromptTests(unittest.TestCase):
    def test_clean_point_and_box(self) -> None:
        mask = square_mask()
        self.assertEqual(deepest_positive_point(mask), (4, 3))
        self.assertEqual(tight_box(mask), (2, 1, 6, 5))

    def test_box_iou_uses_inclusive_pixel_coordinates(self) -> None:
        self.assertEqual(pixel_box_iou((0, 0, 0, 0), (0, 0, 0, 0)), 1.0)
        self.assertEqual(pixel_box_iou((0, 0, 0, 0), (1, 1, 1, 1)), 0.0)

    def test_instance_has_locked_fourteen_prompt_rows(self) -> None:
        records = generate_prompts_for_instance(
            instance(), square_mask(), [(-1, -1), (-1, 1), (1, -1)]
        )
        self.assertEqual(len(records), 14)
        self.assertEqual(sum(row.noise_level == 0 for row in records), 2)
        self.assertEqual(sum(row.noise_level > 0 for row in records), 12)
        validate_prompt_matrix(records, expected_instances=1)

    def test_generation_is_deterministic_and_files_match(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            mask_dir = root / "data" / "coco" / "gt_masks"
            mask_dir.mkdir(parents=True)
            instances = [instance(), instance(annotation_id=12, image_id=8)]
            for row in instances:
                Image.fromarray(square_mask().astype(np.uint8) * 255).save(
                    mask_dir / f"{row.annotation_id}.png"
                )
            first = generate_prompt_set(root, instances, seed=2026)
            second = generate_prompt_set(root, instances, seed=2026)
            self.assertEqual(first, second)
            self.assertEqual(len(first), 28)

            json_path = root / "prompts.json"
            csv_path = root / "prompts.csv"
            write_prompt_files(first, json_path, csv_path)
            self.assertEqual(len(json.loads(json_path.read_text())), 28)
            self.assertEqual(len(csv_path.read_text().splitlines()), 29)

    def test_incomplete_direction_set_fails(self) -> None:
        with self.assertRaisesRegex(PromptContractError, "three unique"):
            generate_prompts_for_instance(
                instance(), square_mask(), [(-1, -1), (-1, -1), (1, 1)]
            )


if __name__ == "__main__":
    unittest.main()
