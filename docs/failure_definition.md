# Failure labels used in the qualitative analysis

This document describes the thresholds **applied consistently to the shared
800-row result table**. The earlier version of this file was empty, so the
repository does not establish that these thresholds were registered before
the results were inspected. Treat the labels as descriptive analysis, not a
preregistered hypothesis test.

- **F1, low overlap:** predicted-mask IoU with the selected COCO ground-truth
  mask is below 0.5.
- **F2, large decline:** for a shifted prompt, the clean-prompt IoU minus its
  shifted-prompt IoU is at least 0.2. The reference uses the same model,
  annotation ID and prompt type.
- **Failure:** F1 or F2. A run satisfying both is counted once in the total.
- Runs with `status != ok` must be counted as execution errors separately;
  they must not be silently removed or called segmentation failures. All 800
  rows in the recorded shared run have `status=ok`.

The four example panels were selected **after** reading the result table by
a reproducible rule: the two highest-IoU clean SAM ViT-B box rows and the two
largest clean-to-shifted IoU drops among SAM ViT-B 20% prompts. This is an
illustration strategy, not a random sample or evidence of failure frequency.
See `scripts/analyze_failures.py`, `results/selected_examples.csv`, and the
numeric outputs in `results/metrics/`.
