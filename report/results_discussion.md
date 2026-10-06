# Results and Discussion (draft for the final report)

This section uses the shared 800-row experiment log, not Dương's separate
SAM-only run. The evaluation contains 50 fixed COCO val2017 image-instance
pairs. Each instance receives one clean point and one clean box for both
pretrained models; only SAM ViT-B is additionally evaluated at 10% and 20%
prompt shifts, with three trials per instance, prompt type and shift level.
The plots and all per-run values are in `results/metrics/`, with exact source
identifiers in `results/metrics/SOURCE.md`.

## Setup 1: MobileSAM baseline versus SAM ViT-B

On the 50 clean box prompts, mean IoU was 0.803 for MobileSAM and 0.783 for
SAM ViT-B. On the 50 clean point prompts, the corresponding means were 0.538
and 0.468. Paired by annotation ID, SAM-minus-MobileSAM mean differences
were -0.019 for boxes and -0.070 for points. SAM ViT-B exceeded MobileSAM
in 20/50 box cases and 12/50 point cases. Thus the main model did **not**
outperform the lightweight baseline on this particular fixed sample. This
is a descriptive comparison, not evidence that MobileSAM is generally
better: there are only 50 selected instances, the models have different
pretraining histories, and no repeated dataset sampling or uncertainty
interval is provided. The recorded prediction times should not be used to
claim a hardware-fair speed advantage because the model runs partly
overlapped on the group machine.

## Setup 2: Clean point versus clean box

For SAM ViT-B, mean IoU rose from 0.468 with a clean point to 0.783 with a
clean box. The paired box-minus-point mean difference was +0.315, and the
box gave higher IoU on 43 of 50 instances. These are matched comparisons:
the annotation and image are the same for each point-box pair. A box conveys
the target's approximate extent, while one positive point only indicates a
location inside it; this is a plausible explanation for the observed gap,
not proof that all real users should always prefer boxes. Clean prompts
were generated from the ground-truth mask, so the experiment measures model
responses to controlled input geometry rather than natural clicking skill.

## Setup 3: Prompt-shift robustness of SAM ViT-B

At 10% and 20% shift, box mean IoU decreased from 0.783 (clean) to 0.571
and 0.372. The paired noisy-minus-clean mean changes were -0.212 and
-0.412. Point mean IoU changed from 0.468 (clean) to 0.453 and 0.335, with
paired changes of -0.015 and -0.134. At 20% shift, 73.3% of generated
points still fell inside the target mask; shifted boxes overlapped their
clean box by 0.498 IoU on average. This prompt-quality evidence supports
the interpretation that increasing geometric displacement often reduces
segmentation quality. Box performance fell more in absolute mean IoU from
its higher clean starting value. The 150 noisy runs per condition are three
trials on the same 50 instances, **not 150 independent images**.

The descriptive failure flags (`IoU < 0.5` or, for noisy prompts, an IoU
drop of at least 0.2) are tabulated in `failure_summary.csv`. They indicate
where qualitative inspection should focus, but do not by themselves
identify a causal failure mechanism. These thresholds were not fixed before
the project started; see `docs/failure_definition.md`. The final report should integrate
the separately authored error/qualitative analysis and show example masks
for selected cases.

## Measurement and generalization limits

The committed aggregates were regenerated and checked against the shared
runner CSV, which contains 800 `status=ok` rows. At the time these tables
were made, the full prediction masks were unavailable. The group later
received the matching shared-run mask bundle and independently checked all
800 recorded IoUs against those masks and the fixed GT; see
`results/metrics/SOURCE.md`. The matching masks are now downloadable as
`results.zip` from the repository. Dương's 700
delivered masks belong to a separate SAM-only run and were not substituted.
The sample is small and not class-balanced; COCO polygon-derived masks can also differ
from perceived object boundaries. Only pretrained inference was performed:
neither model was trained or fine-tuned by the group. These findings should
therefore be stated as observations on this controlled evaluation set, not
as universal or statistically significant performance claims.

Source tables: `results/metrics/summary_metrics.csv`,
`paired_summary.csv`, `failure_summary.csv` and
`per_run_with_metrics.csv`. Suggested figures:
`figures/model_prompt_clean.png` and `figures/sam_noise_robustness.png`.
