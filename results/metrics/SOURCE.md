# Metric Result Provenance

These metric tables and figures were generated on branch `thanh/metrics` from the completed 800-row experiment log and deterministic prompt table stored at:

- Source branch: `origin/son-prompt-runner`
- Source commit: `cc35e5a563ad744b87e22dcf8fc24d2e5bcd822c`
- `results/raw_predictions.csv` Git blob: `6393a5e757a3575d7b010cde0997d202fb1cc7c1`
- `results/prompts.csv` Git blob: `7ac4d8e485c8da9ff22604339b8cf5df836e03f6`
- Generation command: `python scripts/make_plots.py --input results/raw_predictions.csv --prompts results/prompts.csv --output-dir results/metrics`

The input contains 800 successful rows: 700 SAM ViT-B runs and 100 MobileSAM runs. The SAM ViT-B results originate from the model branch ending at `0320c2847e446c4254af05d511e365f674eb29ba`; the current MobileSAM branch ends at `92deae94c0d89dd5debdbea8cbaf2aedea0ca170`.

The ignored prediction-mask archive is not present on `thanh/metrics`, so `scripts/evaluate_masks.py` was unit-tested with known PNG masks but was not run over all 800 experiment masks here. The committed aggregate outputs therefore use the IoU values recorded by the experiment runner. Re-run mask-level verification after the data/model handoff provides `data/coco/gt_masks/` and the prediction files referenced by `mask_path`.
