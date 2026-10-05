# Experimental Setup 1-3

## Research scope

The experiments evaluate how prompt type and prompt displacement affect
interactive instance segmentation. SAM ViT-B is the main model. MobileSAM is
the baseline used for the clean-prompt comparison. Both models use pretrained
checkpoints and inference only. The project does not train or fine-tune either
model.

The primary comparison uses the same COCO instance under a clean positive
point and a clean tight box. The robustness comparison shifts those prompts by
10% or 20% of the ground-truth COCO bounding-box width and height. All model
comparisons use the same image, instance and stored prompt coordinates.

## Evaluation subset

The fixed subset contains 50 instances from 50 different images in COCO 2017
`val2017`. The selection uses `seed=2026`, excludes crowd annotations and
invalid or empty masks, and chooses one valid instance per image before
sampling 50 images. The first ten manifest rows form the smoke subset. No run
may replace an instance after inspecting a prediction or IoU.

`configs/eval_manifest.json` stores exactly eight fields per row:

```text
image_id, annotation_id, file_name, width, height,
category_id, bbox_xywh, area
```

Images remain at their original resolution. Ground-truth masks are 8-bit PNG
files with values 0 and 255 and shape `(height, width)`. The command below
checks all 50 image-mask pairs before an experiment:

```powershell
.\.venv\Scripts\python.exe scripts\validate_inputs.py
```

## Clean prompts

Each instance receives one positive point and one box in original-image pixel
coordinates.

- The clean point is the first row-major foreground pixel that has the maximum
  Euclidean distance to the background. The implementation uses
  `scipy.ndimage.distance_transform_edt` and assigns label `1`.
- The clean box is the inclusive tight box of the GT mask in
  `[x_min, y_min, x_max, y_max]` form. COCO `bbox_xywh` remains unchanged in
  the manifest and is not passed directly to SAM.

Clean prompts use `noise_level=0` and `trial=0`.

## Shifted prompts

For each instance, a single `random.Random(2026)` stream samples three unique
direction pairs from `(-1,-1)`, `(-1,+1)`, `(+1,-1)` and `(+1,+1)`. Iteration
follows manifest order. The same three directions are reused for point and box
prompts and for both noise levels, preserving paired comparisons.

For a noise level `a` in `{0.10, 0.20}` and COCO GT bounding-box dimensions
`w` and `h`, the integer displacement is:

```text
dx = round(sx * a * w)
dy = round(sy * a * h)
```

The point or all four box coordinates receive the same displacement and are
then clipped coordinate by coordinate to the image. A shifted point that lands
outside the GT mask remains unchanged. `point_inside_gt` records this event.
For boxes, `box_iou_gt` stores the pixel-coordinate IoU between the shifted box
and the clean tight box.

The stored prompt set contains 700 rows:

| Condition | Rows per instance | Rows for 50 instances |
|---|---:|---:|
| Clean point and clean box | 2 | 100 |
| Two prompt types, two noise levels, three trials | 12 | 600 |
| Total | 14 | 700 |

Generate and verify the prompt files with:

```powershell
.\.venv\Scripts\python.exe scripts\generate_prompts.py
```

The command writes `results/prompts.json` for the runner and
`results/prompts.csv` for manual inspection. Prompt IDs are deterministic and
include annotation ID, prompt type, noise level and trial.

## Experiment matrix

### Setup 1: baseline versus main model

MobileSAM and SAM ViT-B receive the same 50 clean point prompts and the same 50
clean box prompts. The setup produces 100 predictions per model and 200 model
predictions in total. IoU is the primary measure. Timing is secondary and only
supports a within-machine comparison.

### Setup 2: point versus box

The comparison uses the 100 clean SAM ViT-B rows already produced by Setup 1.
It does not run a duplicate inference job. Point and box IoU are paired by
`annotation_id`.

### Setup 3: prompt robustness

SAM ViT-B receives point and box prompts shifted by 10% and 20%, with three
trials at each level. This setup produces 600 predictions. Each noisy row pairs
with the clean row that has the same `annotation_id` and `prompt_type`.

Across all three setups, the runner schedules 800 unique model-prompt jobs:

| Model | Clean jobs | Noisy jobs | Total |
|---|---:|---:|---:|
| MobileSAM | 100 | 0 | 100 |
| SAM ViT-B | 100 | 600 | 700 |
| Total | 200 | 600 | 800 |

## Inference contract

Both adapters receive an RGB `uint8` image with shape `H x W x 3`. The runner
calls `set_image` once per model and image, then reuses the encoded image for
all prompts associated with that instance. Every call uses
`multimask_output=False`; therefore, each prompt produces exactly one Boolean
mask with the original `H x W` shape. GT data never selects among model
outputs.

The runner uses official source snapshots and local checkpoints:

- SAM ViT-B: `model_type='vit_b'`, checkpoint
  `weights/sam_vit_b_01ec64.pth`, official source commit
  `dca509fe793f601edb92606367a655c15ac00fdf`.
- MobileSAM: `model_type='vit_t'`, checkpoint `weights/mobile_sam.pt`, official
  source commit `f706ad9c4eb7f219c00d9050e46328518ffb65d2`.

The adapters are isolated under `src/models/`. A team member can replace an
adapter implementation while preserving `set_image(image_rgb)` and
`predict(prompt)`.

## Timing and result logging

`seconds` measures one prompt-specific `predict()` call and excludes image
encoding. `encode_seconds` measures the preceding `set_image()` call and is
stored separately. The environment record identifies the device, platform,
PyTorch version, checkpoint checksums and official source commits. Timing
comparisons are valid only when both models run on the same machine under the
same conditions.

`results/raw_predictions.csv` contains one durable row per `run_id`:

```text
run_id, setup, model, image_id, annotation_id, prompt_id,
prompt_type, noise_level, trial, iou, seconds, encode_seconds,
predicted_score, status, error, seed, mask_path, device, started_at_utc
```

Successful masks are saved locally under `results/masks/{model}/`. The full
mask directory is ignored by Git because it contains large generated
artifacts. The raw CSV, prompt files and environment metadata remain small and
traceable.

If model initialization, image encoding or prediction fails, the runner keeps
the expected row with `status=error`, a diagnostic message and blank metric
fields. It never removes a hard case. After every prompt, it atomically rewrites
the CSV so an interruption loses at most the active prediction. A resumed run
skips successful rows whose mask still exists. `--retry-errors` explicitly
retries recorded failures.

## Commands

Plan the full matrix without loading a model:

```powershell
.\.venv\Scripts\python.exe scripts\run_experiments.py --dry-run
```

Run the first ten instances:

```powershell
.\.venv\Scripts\python.exe scripts\run_experiments.py --smoke
```

Run or resume the complete experiment:

```powershell
.\.venv\Scripts\python.exe scripts\run_experiments.py
```

For a CPU-only machine, an explicit thread count can be recorded and reused:

```powershell
.\.venv\Scripts\python.exe scripts\run_experiments.py --device cpu --torch-threads 4
```

Check completeness after a full run:

```powershell
.\.venv\Scripts\python.exe scripts\check_experiment_matrix.py
```

## Reproducibility limits

The clean prompts are derived from GT masks and may be easier than prompts from
real users. The shifted conditions model coordinate errors but do not represent
a user study. The 50-instance subset is small and not balanced by category.
CPU and GPU timing must not be mixed. Any failed job remains visible in the
result log and must be reported in the final sample count.
