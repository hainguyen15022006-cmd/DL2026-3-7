from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image

from scripts.evaluate_masks import evaluate_row
from scripts.make_plots import (
    failure_summary,
    metric_summary,
    paired_rows,
    paired_summary,
    prepare_rows,
    save_plots,
)
from src.metrics import BinaryCounts, binary_counts, binary_dice, binary_iou, median, validate_iou


class BinaryMetricTests(unittest.TestCase):
    def test_hand_calculated_two_by_two_masks(self) -> None:
        ground_truth = [[1, 1], [1, 0]]
        prediction = [[1, 1], [0, 1]]

        self.assertEqual(binary_counts(prediction, ground_truth), BinaryCounts(2, 1, 1, 0))
        self.assertAlmostEqual(binary_iou(prediction, ground_truth), 0.5)
        self.assertAlmostEqual(binary_dice(prediction, ground_truth), 2 / 3)

    def test_both_empty_masks_are_a_perfect_match(self) -> None:
        empty = [[0, 0], [0, 0]]
        self.assertEqual(binary_iou(empty, empty), 1.0)
        self.assertEqual(binary_dice(empty, empty), 1.0)

    def test_empty_prediction_against_non_empty_gt_is_zero(self) -> None:
        self.assertEqual(binary_iou([[0, 0]], [[1, 0]]), 0.0)
        self.assertEqual(binary_dice([[0, 0]], [[1, 0]]), 0.0)

    def test_shape_mismatch_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "same shape"):
            binary_iou([[1, 0]], [[1], [0]])

    def test_invalid_iou_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            validate_iou(1.01)
        with self.assertRaises(ValueError):
            validate_iou("not-a-number")

    def test_even_length_median(self) -> None:
        self.assertEqual(median([0.1, 0.4, 0.2, 0.3]), 0.25)


def result_row(
    model: str,
    annotation_id: int,
    prompt_type: str,
    noise_level: float,
    trial: int,
    iou: float | str,
    status: str = "ok",
) -> dict[str, str]:
    prompt_id = f"ann{annotation_id}_{prompt_type}_n{int(noise_level * 100):02d}_t{trial}"
    return {
        "run_id": f"{model}:{prompt_id}",
        "model": model,
        "image_id": str(annotation_id + 1000),
        "annotation_id": str(annotation_id),
        "prompt_id": prompt_id,
        "prompt_type": prompt_type,
        "noise_level": str(noise_level),
        "trial": str(trial),
        "iou": str(iou),
        "seconds": "0.1",
        "status": status,
        "error": "synthetic error" if status != "ok" else "",
        "seed": "2026",
    }


class AggregationTests(unittest.TestCase):
    def setUp(self) -> None:
        source = [
            result_row("mobile_sam", 10, "point", 0.0, 0, 0.4),
            result_row("sam_vit_b", 10, "point", 0.0, 0, 0.6),
            result_row("sam_vit_b", 10, "box", 0.0, 0, 0.8),
            result_row("sam_vit_b", 10, "point", 0.2, 1, 0.3),
            result_row("sam_vit_b", 10, "box", 0.2, 1, "", status="error"),
        ]
        prompts = {
            row["prompt_id"]: {
                "point_inside_gt": "False" if row["noise_level"] == "0.2" else "True",
                "box_iou_gt": "",
            }
            for row in source
        }
        self.rows = prepare_rows(source, prompts)

    def test_failures_include_low_iou_and_large_clean_drop(self) -> None:
        noisy_point = next(
            row for row in self.rows
            if row["prompt_type"] == "point" and row["noise_level"] == 0.2
        )
        self.assertTrue(noisy_point["low_iou_failure"])
        self.assertTrue(noisy_point["large_drop_failure"])
        self.assertAlmostEqual(noisy_point["iou_drop"], 0.3)

    def test_error_rows_are_counted_not_silently_dropped(self) -> None:
        summary = failure_summary(self.rows)
        noisy_box = next(
            row for row in summary
            if row["prompt_type"] == "box" and row["noise_level"] == 0.2
        )
        self.assertEqual(noisy_box["n_valid"], 0)
        self.assertEqual(noisy_box["n_error"], 1)

    def test_paired_differences_use_same_annotation(self) -> None:
        details = paired_rows(self.rows)
        summaries = paired_summary(details)
        setup1 = next(
            row for row in summaries
            if row["comparison"] == "setup1_sam_vit_b_minus_mobile_sam"
        )
        setup2 = next(row for row in summaries if row["comparison"] == "setup2_box_minus_point")
        setup3 = next(row for row in summaries if row["comparison"] == "setup3_noisy_minus_clean")
        self.assertAlmostEqual(setup1["mean_difference"], 0.2)
        self.assertAlmostEqual(setup2["mean_difference"], 0.2)
        self.assertAlmostEqual(setup3["mean_difference"], -0.3)

    def test_all_three_figures_are_created(self) -> None:
        with TemporaryDirectory() as directory:
            figure_dir = Path(directory)
            save_plots(metric_summary(self.rows), figure_dir)
            self.assertTrue((figure_dir / "model_prompt_clean.png").is_file())
            self.assertTrue((figure_dir / "sam_point_vs_box.png").is_file())
            self.assertTrue((figure_dir / "sam_noise_robustness.png").is_file())


class MaskFileEvaluationTests(unittest.TestCase):
    def test_png_masks_are_recomputed_and_checked(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            prediction_path = root / "prediction.png"
            gt_dir = root / "gt"
            gt_dir.mkdir()
            Image.new("L", (2, 2)).save(prediction_path)
            prediction = Image.new("L", (2, 2))
            prediction.putdata([255, 255, 0, 255])
            prediction.save(prediction_path)
            ground_truth = Image.new("L", (2, 2))
            ground_truth.putdata([255, 255, 255, 0])
            ground_truth.save(gt_dir / "10.png")

            row = result_row("sam_vit_b", 10, "point", 0.0, 0, 0.5)
            row["mask_path"] = "prediction.png"
            evaluated = evaluate_row(row, root, gt_dir, tolerance=1e-9)

            self.assertEqual(evaluated["metric_status"], "ok")
            self.assertAlmostEqual(evaluated["iou"], 0.5)
            self.assertAlmostEqual(evaluated["dice"], 2 / 3)
            self.assertEqual(evaluated["true_positive"], 2)
            self.assertEqual(evaluated["false_positive"], 1)
            self.assertEqual(evaluated["false_negative"], 1)


if __name__ == "__main__":
    unittest.main()
