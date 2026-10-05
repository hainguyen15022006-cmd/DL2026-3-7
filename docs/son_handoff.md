# Prompt and Experiment Runner Handoff

## Owner and scope

Owner: Ngô Đức Minh Sơn

This component converts the locked evaluation manifest and GT masks into a
deterministic prompt set, schedules Setup 1-3, calls model adapters and records
every expected run. It does not select the COCO subset, train a model, choose a
prediction using GT, create final metric plots or merge the final report.

## Inputs

- `configs/eval_manifest.json`: 50 fixed COCO instances.
- `data/coco/val2017/`: original RGB images.
- `data/coco/gt_masks/`: binary GT masks.
- `weights/sam_vit_b_01ec64.pth`: SAM ViT-B checkpoint.
- `weights/mobile_sam.pt`: MobileSAM checkpoint.
- Official source folders at `../segment-anything` and `../MobileSAM`.

Run this gate before any prompt or model job:

```powershell
.\.venv\Scripts\python.exe scripts\validate_inputs.py
```

## Generated prompt files

```text
results/prompts.json
results/prompts.csv
results/prompt_summary.json
```

The files contain 700 rows: 100 clean prompt rows and 600 shifted prompt rows.
The JSON file is the runner input. The CSV file exposes flat coordinate columns
for review. Re-running the generator with the same manifest and seed must
produce byte-identical files.

```powershell
.\.venv\Scripts\python.exe scripts\generate_prompts.py
```

## Model adapter contract

`src/models/base.py` defines two operations:

```text
set_image(image_rgb) -> encode_seconds
predict(prompt) -> Boolean H x W mask, predicted score, predict seconds
```

The RGB array must use the original image dimensions and `uint8` dtype. Both
adapters call their official `SamPredictor` with `multimask_output=False`.
`seconds` excludes `set_image`; `encode_seconds` records it separately.

## Runner behavior

The full plan contains 800 unique `run_id` values. The runner groups work by
model and image so it encodes each image once and evaluates every associated
prompt before continuing. It writes results atomically after every prompt.

```powershell
# Inspect counts only
.\.venv\Scripts\python.exe scripts\run_experiments.py --dry-run

# First ten manifest rows
.\.venv\Scripts\python.exe scripts\run_experiments.py --smoke

# Full run or resume
.\.venv\Scripts\python.exe scripts\run_experiments.py

# Explicit CPU thread count, recorded in results/environment.json
.\.venv\Scripts\python.exe scripts\run_experiments.py --device cpu --torch-threads 4

# Retry rows that already contain status=error
.\.venv\Scripts\python.exe scripts\run_experiments.py --retry-errors
```

Existing `status=ok` rows are skipped only when their mask file still exists.
Existing error rows remain visible and are skipped unless `--retry-errors` is
provided. Model initialization and image-encoding failures create an error row
for every affected scheduled job.

## Outputs

- `results/raw_predictions.csv`: one row per model-prompt job.
- `results/environment.json`: package, device, checkpoint and source metadata.
- `results/run_state.json`: final completeness summary.
- `results/masks/{model}/{prompt_id}.png`: local binary predictions.

The mask directory is intentionally ignored by Git. The team must use the raw
CSV for tables and keep or share local masks when qualitative analysis needs
them.

## Completeness gate

```powershell
.\.venv\Scripts\python.exe scripts\check_experiment_matrix.py
```

The command passes only when all 800 expected rows exist and have `status=ok`.
For a partial run, use `--allow-incomplete` to print the missing and failed run
IDs without treating incompleteness as a command failure.

## Tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The tests cover data schema checks, clean prompt geometry, deterministic noise,
prompt matrix completeness, experiment counts, paired Setup 2 reuse, IoU and
durable result round-trips.

## Report and slide material

- `docs/experimental_setup.md`: English report section for Setup 1-3.
- `slides/slide_2_method_experiments.pptx`: editable one-slide PowerPoint.

The slide reports design counts only. It does not claim model performance
before the full run and metric review are complete.
