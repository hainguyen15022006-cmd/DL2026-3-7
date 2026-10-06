# Metric Result Provenance

These metric tables and figures were generated on branch `thanh/metrics` from the completed 800-row experiment log and deterministic prompt table stored at:

- Source branch: `origin/son-prompt-runner`
- Source commit: `cc35e5a563ad744b87e22dcf8fc24d2e5bcd822c`
- `results/raw_predictions.csv` Git blob: `6393a5e757a3575d7b010cde0997d202fb1cc7c1`
- `results/prompts.csv` Git blob: `7ac4d8e485c8da9ff22604339b8cf5df836e03f6`
- Generation command: `python scripts/make_plots.py --input results/raw_predictions.csv --prompts results/prompts.csv --output-dir results/metrics`

The input contains 800 successful rows: 700 SAM ViT-B runs and 100 MobileSAM runs. The SAM ViT-B handoff branch ends at `0320c2847e446c4254af05d511e365f674eb29ba`. The MobileSAM clean handoff branch ends at `b2ce2209a905d0ff3ec82f268ff74d56237b2693`; its 20-row smoke run checked selected clean-prompt IoUs against the shared log, but did not generate or mask-verify the entire 100-row MobileSAM portion.

The ignored prediction-mask archive is not present on `thanh/metrics`, so `scripts/evaluate_masks.py` was unit-tested with known PNG masks but was not run over all 800 experiment masks here. The committed aggregate outputs therefore use the IoU values recorded by the experiment runner. Re-run mask-level verification after the data/model handoff provides `data/coco/gt_masks/` and the prediction files referenced by `mask_path`.

## Post-handoff audit on 2026-10-06

The group later received `son_visualization_bundle_2026-10-06.zip` (SHA-256
`6edca33ca626a7aec86400f9fa15dca4f2c3f65a8209dfe173f4eed940c34575`).
Its manifest, prompt CSV and 800-row result CSV match the tracked data by
parsed content. It contains 700 SAM ViT-B and 100 MobileSAM prediction masks.
`scripts/extract_shared_masks_zip.py --check-only` recomputed IoU for every
mask against the fixed local GT and found **800/800 matches**. Independently,
`scripts/evaluate_masks.py` returned 800 `ok` rows. This verifies the
recorded IoUs against the delivered mask files; it does **not** prove that a
fresh inference run on another machine will be byte-identical. The archive
is an optional handoff artifact and is not committed to GitHub.
