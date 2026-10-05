from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from src.experiment import (
    binary_iou,
    build_run_plan,
    missing_mask_run_ids,
    missing_model_assets,
    read_results,
    result_summary,
    write_results,
)
from src.prompts import PromptRecord, generate_prompts_for_instance
from tests.test_prompts import instance, square_mask


class ExperimentTests(unittest.TestCase):
    def prompts(self, annotation_id: int, image_id: int) -> list[PromptRecord]:
        return generate_prompts_for_instance(
            instance(annotation_id=annotation_id, image_id=image_id),
            square_mask(),
            [(-1, -1), (-1, 1), (1, -1)],
        )

    def test_full_plan_has_sixteen_unique_runs_per_instance(self) -> None:
        prompts = self.prompts(11, 7) + self.prompts(12, 8)
        plan = build_run_plan(prompts)
        self.assertEqual(len(plan), 32)
        self.assertEqual(len({row.run_id for row in plan}), 32)
        self.assertEqual(sum(row.model == "mobile_sam" for row in plan), 4)
        self.assertEqual(sum(row.model == "sam_vit_b" for row in plan), 28)

    def test_setup_two_reuses_only_sam_clean_rows(self) -> None:
        plan = build_run_plan(self.prompts(11, 7), selected_setups=("setup2",))
        self.assertEqual(len(plan), 2)
        self.assertTrue(all(row.model == "sam_vit_b" for row in plan))

    def test_binary_iou(self) -> None:
        prediction = np.array([[1, 1], [0, 0]], dtype=bool)
        ground_truth = np.array([[1, 0], [1, 0]], dtype=bool)
        self.assertAlmostEqual(binary_iou(prediction, ground_truth), 1 / 3)

    def test_result_round_trip_and_summary(self) -> None:
        plan = build_run_plan(self.prompts(11, 7), selected_setups=("setup2",))
        rows = {
            plan[0].run_id: {
                "run_id": plan[0].run_id,
                "setup": plan[0].setup,
                "model": plan[0].model,
                "image_id": plan[0].image_id,
                "annotation_id": plan[0].annotation_id,
                "prompt_id": plan[0].prompt_id,
                "prompt_type": "point",
                "noise_level": 0,
                "trial": 0,
                "iou": 0.5,
                "seconds": 0.1,
                "encode_seconds": 1.0,
                "predicted_score": 0.9,
                "status": "ok",
                "error": "",
                "seed": 2026,
                "mask_path": "results/masks/example.png",
                "device": "cpu",
                "started_at_utc": "2026-10-05T00:00:00+00:00",
            }
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "results.csv"
            write_results(path, rows)
            loaded = read_results(path)
        summary = result_summary(plan, loaded)
        self.assertEqual(summary["ok_rows"], 1)
        self.assertEqual(summary["missing_rows"], 1)

    def test_missing_mask_check_requires_a_local_file(self) -> None:
        plan = build_run_plan(self.prompts(11, 7), selected_setups=("setup2",))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            mask = root / "results" / "masks" / "example.png"
            mask.parent.mkdir(parents=True)
            mask.write_bytes(b"png placeholder")
            rows = {
                plan[0].run_id: {
                    "status": "ok",
                    "mask_path": "results/masks/example.png",
                },
                plan[1].run_id: {
                    "status": "ok",
                    "mask_path": "../outside.png",
                },
            }
            self.assertEqual(missing_mask_run_ids(plan, rows, root), [plan[1].run_id])

    def test_missing_model_assets_preflight(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = {
                "sam_vit_b": {
                    "source_path": "../segment-anything",
                    "checkpoint_path": "weights/sam_vit_b_01ec64.pth",
                }
            }
            missing = missing_model_assets(root, config, ["sam_vit_b"])
            self.assertEqual(len(missing), 2)
            self.assertEqual(missing_model_assets(root, config, []), [])


if __name__ == "__main__":
    unittest.main()
