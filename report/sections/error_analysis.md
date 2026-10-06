## Error and Qualitative Analysis

### Failure definition
We define a failure as an IoU below 0.5 (F1) or, for shifted prompts, an IoU
drop of at least 0.2 relative to the clean prompt on the same model, instance
and prompt type (F2). We apply these thresholds consistently to all rows as
descriptive labels; the repository does not establish preregistration of
the thresholds. See `docs/failure_definition.md`. All 800 planned runs
completed with status ok, so no run was excluded.

### Failure rates by condition
The table below reports results on 50 COCO val2017 instances (one per image). Each
shifted condition contains 150 runs (3 trials x 50 instances).

| Model | Prompt | Shift | n | Mean IoU | Median IoU | Failures | Rate |
|---|---|---|---|---|---|---|---|
| SAM ViT-B | box | 0% | 50 | 0.783 | 0.844 | 4 | 8.0% |
| SAM ViT-B | box | 10% | 150 | 0.571 | 0.694 | 61 | 40.7% |
| SAM ViT-B | box | 20% | 150 | 0.372 | 0.375 | 108 | 72.0% |
| SAM ViT-B | point | 0% | 50 | 0.468 | 0.456 | 27 | 54.0% |
| SAM ViT-B | point | 10% | 150 | 0.453 | 0.431 | 82 | 54.7% |
| SAM ViT-B | point | 20% | 150 | 0.335 | 0.192 | 103 | 68.7% |
| MobileSAM | box | 0% | 50 | 0.803 | 0.869 | 4 | 8.0% |
| MobileSAM | point | 0% | 50 | 0.538 | 0.627 | 20 | 40.0% |

With clean prompts, SAM ViT-B fails on 8.0% of instances with a box but on
54.0% with a single point. In a paired comparison over the same 50 instances,
the box IoU exceeds the point IoU by 0.315 on average, and the box is better
on 43 of 50 instances. Shifting the prompt lowers IoU for both prompt types.
The box degrades more in absolute terms (mean IoU 0.783, 0.571, 0.372 at 0%,
10%, 20% shift) than the point (0.468, 0.453, 0.335), but its mean IoU stays
above the point at every level. At 20% shift the failure rates are similar
(72.0% for box, 68.7% for point), and the median point IoU (0.192) is lower
than the median box IoU (0.375).

On clean prompts the two models are close. The paired mean difference
(SAM ViT-B minus MobileSAM) is -0.019 for box and -0.070 for point, so
MobileSAM scores slightly higher on this sample. With n = 50 and no
significance test, we do not claim that either model is better. Shifted
prompts were evaluated on SAM ViT-B only.

### Prompt quality and its relation to IoU
For every shifted prompt we recorded whether the point stays inside the
ground-truth mask (point_inside_gt) and how much the shifted box overlaps the
tight ground-truth box (box_iou_gt).

Point prompts (SAM ViT-B). A 10% shift moved the point outside the object in
12 of 150 runs (8.0%) and a 20% shift in 40 of 150 runs (26.7%). Runs whose
point left the object had a mean IoU of 0.152 (10% shift) and 0.082 (20%
shift), compared with 0.479 and 0.426 when the point stayed inside. Leaving
the object is therefore associated with near-total failure. However, points
that stayed inside the object also reach only 0.43-0.48 mean IoU, and clean
points, which lie inside the object by construction, already fail in 54% of
cases. Displacement alone therefore does not explain the point failures. A
single positive point may be ambiguous (for example, part of an object versus
the whole object), but we did not verify this and treat it as a hypothesis.

Box prompts (SAM ViT-B). Mean IoU increases with the overlap between the
prompt box and the tight ground-truth box: 0.386 for overlap in (0, 0.5]
(n = 101), 0.518 for (0.5, 0.8] (n = 196) and 0.756 for (0.8, 1.0] (n = 53,
including the 50 clean boxes).

Zero-IoU runs. Across all 800 runs, 8 runs had an IoU of exactly 0, all on
SAM ViT-B with shifted prompts: 3 box runs at 20% shift, 1 point run at 10%
shift and 4 point runs at 20% shift. The predicted quality score is not a
reliable indicator of correctness in these cases: both degraded examples
below had scores above 0.8.

### Qualitative examples
The saved qualitative figure in `results/examples/figure_examples.png` shows
four examples chosen by a reproducible post-hoc rule: the two
highest-IoU clean-box cases of SAM ViT-B and the two largest IoU drops at 20%
shift. Each panel shows the image, ground truth, prompt and predicted mask,
with annotation_id and IoU.
- Good case 1: annotation 36588, clean box, IoU 0.968.
- Good case 2: annotation 1050658, clean box, IoU 0.961.
- Degraded case 1: annotation 36588, 20% shift, trial 3. IoU fell from 0.968
  to approximately 0 (below 0.001) although the shifted box still overlaps
  the tight box by 0.494. The predicted score remained high (0.81), so the
  mask was produced with a high *model-predicted quality score* but did not
  match the target object. In the overlay, the shifted-box prediction covers
  foliage above the bird rather than the bird selected by the GT mask.
- Degraded case 2: annotation 339546, 20% shift, trial 2. IoU fell from 0.950
  to 0.000 with box overlap 0.466 and a predicted score of 0.84.
  The overlay shows a prediction concentrated around the clock's rim or
  adjacent structure rather than the annotated clock face. This describes
  the visible mismatch; the image alone cannot establish why SAM chose it.

These examples are illustrative and do not represent overall frequency.
Descriptions are based on visual inspection and are not causal claims. The
saved overlay panels are available in `results/examples/`; regenerating them
requires the selected prediction masks from the **shared** 800-row run.

## Conclusion and Limitations
- On SAM ViT-B, a clean box gave much higher IoU than a clean point (mean
  paired difference 0.315; 8.0% versus 54.0% failure rate).
- Shifting prompts by 10-20% of the object's bounding-box dimensions lowered
  IoU, more strongly for boxes in absolute terms. Among shifted point prompts
  that left the object, 10/12 runs at 10% shift and 36/40 at 20% shift met
  the descriptive failure criterion; the trials are not independent images.
- SAM ViT-B and MobileSAM differ little on clean prompts in this sample.
- Prompts are simulated from ground-truth masks and are cleaner than real
  user input; shifts follow a fixed rule and are not a user study.
- Only 50 instances (one per image) from COCO val2017 were used, and the
  three trials per instance are not independent, so results may not
  generalize. We report no significance tests.
- Robustness was tested on SAM ViT-B only, not on MobileSAM.
- COCO polygon-derived GT may not match every visually perceived boundary;
  this can affect absolute IoU, especially for thin or occluded regions.
- We used pretrained checkpoints only; no training or fine-tuning was
  performed.
- The shared run recorded CPU inference in `results/environment.json`.
  MobileSAM and SAM ViT-B execution windows overlapped, so these logs do not
  support a hardware-fair speed comparison.
