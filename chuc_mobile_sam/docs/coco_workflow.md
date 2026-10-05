# COCO Workflow for the Model Owners

The group evaluates COCO val2017 using a fixed manifest of 50 instances (one
instance per image, seed 2026; use 20 only if the group explicitly reduces the
sample size). Model owners must use the same manifest created by the data owner;
do not select separate samples.

## Required inputs

| Input | Requirement |
|---|---|
| Manifest | `configs/eval_manifest.json` or CSV, with at least `image_id`, `annotation_id`, and `file_name` |
| Images | Images referenced by the manifest, under `data/coco/val2017` |
| Ground truth | `data/coco/annotations/instances_val2017.json` for pycocotools, or pre-exported binary PNG masks |
| MobileSAM checkpoint | `weights/mobile_sam.pt` from the official MobileSAM repository |

Compare the manifest SHA-256 with the group before running:

```powershell
Get-FileHash configs\eval_manifest.json -Algorithm SHA256
```

## Run the MobileSAM smoke test

Run this command from the repository root in PowerShell:

```powershell
python chuc_mobile_sam\scripts\smoke_manifest.py `
  --adapter chuc_mobile_sam.src.models.mobile_sam:MobileSamAdapter `
  --checkpoint weights\mobile_sam.pt `
  --manifest configs\eval_manifest.json `
  --image-dir data\coco\val2017 `
  --ann-file data\coco\annotations\instances_val2017.json `
  --n 10 --device cpu `
  --out chuc_mobile_sam\results\smoke_coco_mobile
```

Use `--n 50` to smoke-test all instances. To use pre-exported masks, replace
`--ann-file ...` with `--gt-dir <mask-directory>` and optionally set
`--gt-pattern` (default: `{annotation_id}.png`). Exactly one GT source is required.

## Script behavior and outputs

- Loads images as RGB and computes a canonical positive point from the deepest
  foreground pixel in GT.
- Computes an **inclusive** tight box `[x_min, y_min, x_max, y_max]` from GT,
  matching the group prompt convention.
- Runs one point prompt and one box prompt per instance with
  `multimask_output=False`; GT is used for prompt generation and evaluation, never
  to select a predicted mask.
- Writes `results.csv` with the group schema as its first ten columns:
  `model, image_id, annotation_id, prompt_type, noise_level, trial, iou, seconds,
  status, seed`. Extra columns are `score`, `encode_seconds`, and `error`.
  `seconds` measures prompt decoding; image encoding is recorded separately.
- Writes prompt coordinates to `prompts.csv` and overlays to
  `overlay_<annotation_id>.png` for the first instances.
- If an instance fails, it still writes both prompt rows with `status=error` and
  the diagnostic in the `error` column, then continues.

Smoke runs validate the pipeline; they are not the group's final results. Use the
repository's `scripts/run_experiments.py` for the full locked experiment matrix.
