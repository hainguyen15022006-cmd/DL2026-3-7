# Metrics and Results Handoff

## Owner and scope

Owner: Dinh Cong Thanh

This component checks binary segmentation metrics and turns experiment logs into traceable tables, paired comparisons, failure counts, figures and an English Results/Discussion draft. It does not create prompts, run model inference, select qualitative examples or change failed runs.

## Agreed input contract

The scripts follow the common team convention:

- One result row contains `model`, `image_id`, `annotation_id`, `prompt_type`, `noise_level`, `trial`, `iou`, `seconds`, `status` and `seed`.
- `status=ok` requires a numeric IoU in `[0, 1]`. Error rows remain in all counts and are never silently removed.
- A prediction mask is a Boolean-compatible `H x W` PNG at `mask_path`.
- Its GT mask is `data/coco/gt_masks/{annotation_id}.png` with the same `H x W` shape.
- Prompt metadata is joined by `prompt_id` from `results/prompts.csv` to obtain `point_inside_gt` or `box_iou_gt`.
- Clean prompts use `noise_level=0` and `trial=0`. Noisy prompts use `noise_level=0.1` or `0.2` and trials 1-3.

## Metric policy

For binary masks:

- `IoU = TP / (TP + FP + FN)`
- `Dice = 2TP / (2TP + FP + FN)`
- Non-zero pixels are foreground.
- A shape mismatch is an error.
- If both masks are empty, IoU and Dice are defined as 1.0. The data pipeline should still reject empty GT masks before experiments.
- If the source run failed, mask evaluation is skipped and the row is preserved with its error.

The following failure thresholds are used consistently for descriptive
analysis. The group did **not** fix them before the project started; do not
describe them as preregistered:

- F1: `IoU < 0.5`.
- F2: for a noisy prompt, `clean IoU - noisy IoU >= 0.2` on the same model, annotation and prompt type.
- A failure satisfies F1 or F2.

## Commands

Install only the metrics dependencies:

```powershell
python -m pip install -r requirements-metrics.txt
```

Run the hand-calculated unit tests:

```powershell
python -m unittest discover -s tests -v
```

Reproduce the committed result tables and three figures from the tracked
800-row runner log (the path available on a fresh clone):

```powershell
python scripts/make_plots.py `
  --input results/raw_predictions.csv `
  --prompts results/prompts.csv `
  --output-dir results/metrics
```

If all **800 masks from the same shared run** become available, first
independently recompute IoU and Dice from those masks and compare them with
the runner's IoU:

```powershell
python scripts/evaluate_masks.py `
  --input results/raw_predictions.csv `
  --gt-mask-dir data/coco/gt_masks `
  --output results/per_run_metrics.csv
```

The audit exits non-zero if a prediction mask is missing, malformed, has the
wrong shape or disagrees with the recorded IoU beyond `1e-9`. Only after it passes
should `make_plots.py --input results/per_run_metrics.csv` be used. Dương's
separate 700-mask run is not a substitute for the shared run's SAM masks.

Audit row counts and numeric traceability against the original run log:

```powershell
python scripts/check_metrics_outputs.py `
  --raw results/raw_predictions.csv `
  --metrics-dir results/metrics
```

This check verifies that run IDs are unchanged, group counts match, mean IoU values can be recomputed from per-run rows, failure counts match their flags and paired sample counts match the paired-detail CSV.

For a read-only check of a CSV supplied through a pipe, use `--input -`.

The committed tables use runner-recorded IoU because full masks were
unavailable **when the tables were generated**. After handoff, the group
received the matching shared-run archive and verified all 800 IoUs against
the fixed GT. See `results/metrics/SOURCE.md` for the dated audit and archive
checksum. Do not imply that mask verification had occurred during the
original table-generation step.

## Outputs

- `per_run_with_metrics.csv`: every input row plus prompt quality, clean baseline, IoU drop and failure flags.
- `summary_metrics.csv`: `n_total`, `n_valid`, `n_error`, mean/median IoU, timing and prompt-quality evidence for each model x prompt x noise condition.
- `paired_differences.csv`: one traceable row for every matched annotation comparison.
- `paired_summary.csv`: mean/median paired difference and positive/tie/negative counts.
- `failure_summary.csv`: failure counts and rates without hiding run errors.
- `figures/model_prompt_clean.png`: clean model and prompt overview.
- `figures/sam_point_vs_box.png`: Setup 2 clean point vs box.
- `figures/sam_noise_robustness.png`: Setup 3 clean/10%/20% trend.

Difference signs are fixed as follows:

- Setup 1: `SAM ViT-B - MobileSAM`; positive favors SAM ViT-B.
- Setup 2: `box - point`; positive favors the clean box prompt.
- Setup 3: `noisy - clean`; negative means degradation from prompt shift.

## Required handoffs from other members

The integrated branch contains the fixed manifest, prompt table and shared
800-row log. COCO images/GT and full masks remain outside Git; the separate
shared-run archive was received and audited on 2026-10-06. A third party
still needs that archive or a fresh inference run to repeat the mask-level
audit. The English Results/Discussion draft is in
`report/results_discussion.md`; it must be reviewed with the final report.
