# Slide 3: Key Results

## Prompt type and shift sensitivity on SAM ViT-B

![SAM ViT-B mean IoU under clean, 10%, and 20% prompt shifts](../results/metrics/figures/sam_noise_robustness.png)

### On-slide findings

- **Clean prompts:** box reached **0.783 mean IoU**, compared with **0.468** for point. The paired box-minus-point difference was **+0.315**, and box performed better on **43 of 50** instances.
- **20% shift:** mean IoU fell to **0.372** for box and **0.335** for point. The paired noisy-minus-clean changes were **-0.412** and **-0.134**, respectively.
- **Prompt quality check:** at 20% shift, **73.3%** of point prompts remained inside GT. Noisy boxes retained **0.498 mean IoU** with the clean box.

> **Evaluation scope:** 50 fixed COCO val2017 instances, three shifted trials per noise level, seed 2026. All 800 scheduled model-prompt runs have `status=ok`.

## Suggested slide layout

- Place the title at the top.
- Use the robustness figure on the left, covering about two-thirds of the slide.
- Put the three findings on the right.
- Keep the evaluation scope as a small footer.

## Number traceability

| Slide value | CSV source | Row or calculation |
|---|---|---|
| 0.783 clean box mean IoU | `results/metrics/summary_metrics.csv` | `sam_vit_b`, `box`, `noise_level=0.0`, `mean_iou` |
| 0.468 clean point mean IoU | `results/metrics/summary_metrics.csv` | `sam_vit_b`, `point`, `noise_level=0.0`, `mean_iou` |
| +0.315 paired box-minus-point | `results/metrics/paired_summary.csv` | `setup2_box_minus_point`, `mean_difference` |
| 43 of 50 pairs | `results/metrics/paired_summary.csv` | `setup2_box_minus_point`, `n_positive` and `n_pairs` |
| 0.372 box IoU at 20% | `results/metrics/summary_metrics.csv` | `sam_vit_b`, `box`, `noise_level=0.2`, `mean_iou` |
| 0.335 point IoU at 20% | `results/metrics/summary_metrics.csv` | `sam_vit_b`, `point`, `noise_level=0.2`, `mean_iou` |
| -0.412 box change | `results/metrics/paired_summary.csv` | `setup3_noisy_minus_clean`, `box`, `noise_level=0.2` |
| -0.134 point change | `results/metrics/paired_summary.csv` | `setup3_noisy_minus_clean`, `point`, `noise_level=0.2` |
| 73.3% point inside GT | `results/metrics/summary_metrics.csv` | `sam_vit_b`, `point`, `noise_level=0.2`, `point_inside_gt_rate` |
| 0.498 noisy-box overlap | `results/metrics/summary_metrics.csv` | `sam_vit_b`, `box`, `noise_level=0.2`, `mean_box_iou_gt` |
| 800 successful runs | `results/metrics/per_run_with_metrics.csv` | 800 rows with `metric_valid=True` |

<details>
<summary>Speaker notes</summary>

The clean-prompt comparison uses the same 50 annotation IDs for point and box. Box prompts produced higher IoU on most instances, although point prompts were higher in seven cases. Under the 20% shift, both prompt types lost mean IoU. The average reduction was larger for boxes. Prompt-quality measurements confirm that a larger shift moved more points outside the object and reduced box overlap. These results describe a small fixed COCO subset with simulated prompts, not a real user study.

</details>

## Data provenance

All values come from the 800-row experiment log identified in [`results/metrics/SOURCE.md`](../results/metrics/SOURCE.md). No placeholder or manually invented result is included.
