# DL2026-3-7
Effects of User Prompts on Interactive Image Segmentation.

## Metrics and results

The metrics component for the fixed COCO val2017 evaluation protocol is documented in `docs/metrics_handoff.md`.

```powershell
python -m pip install -r requirements-metrics.txt
python -m unittest discover -s tests -v
python scripts/evaluate_masks.py --input results/raw_predictions.csv --output results/per_run_metrics.csv
python scripts/make_plots.py --input results/per_run_metrics.csv --prompts results/prompts.csv --output-dir results/metrics
python scripts/check_metrics_outputs.py --raw results/raw_predictions.csv --metrics-dir results/metrics
```

The scripts preserve failed runs, use paired comparisons on the same `annotation_id`, and generate no placeholder performance values.

Generated tables, figures and their exact input provenance are available under `results/metrics/`.

The GitHub-readable content for the team's third slide is in [`slides/slide_3_results.md`](slides/slide_3_results.md).
