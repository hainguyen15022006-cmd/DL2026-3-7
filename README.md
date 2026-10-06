# DL2026-3-7 — Effects of User Prompts on Interactive Image Segmentation

This project studies how clean and displaced **point and box prompts** change
instance-segmentation quality. It compares pretrained MobileSAM (baseline)
with pretrained SAM ViT-B (main model) on the same fixed 50 COCO 2017
validation image-instance pairs. SAM ViT-B is also evaluated under 10% and 20%
prompt displacement. See [the complete experimental protocol](docs/experimental_setup.md)
and [dataset specification](DATA.md).

**Training code/instructions: not applicable.** As confirmed by the course
instructor for this topic, the group uses pretrained checkpoints for inference
only; it does not train or fine-tune either model. No training dataset or
training command is needed. The repository supplies data preparation,
evaluation, inference, and a single-image demo instead. See
[TRAINING.md](TRAINING.md) for the explicit training-code declaration.

## What is already in the repository?

- `configs/eval_manifest.json`: locked 50-instance evaluation set (seed 2026).
- `results/prompts.json` and `.csv`: 700 reproducible point/box prompts.
- `results/raw_predictions.csv`: the shared 800-row experiment log (100
  MobileSAM and 700 SAM ViT-B predictions), intended as the **primary** result
  table because both models were run under the same recorded environment.
- `results/duong/raw_predictions.csv`: a **separate** 700-row SAM ViT-B run in
  a different environment. Do not combine its timing with MobileSAM timing
  from the shared CSV. Small numeric differences between the two SAM runs are
  expected; choose one complete run consistently for each reported table.
- `results/duong/verification/`: three fixed qualitative examples and a
  historical audit record. The full-resolution prediction masks used in that
  audit are **not stored on GitHub**. They must be installed from Dương's
  separate handoff ZIP or reproduced by inference before the audit can be
  repeated on a new checkout.
- `results/smoke/mobile_sam/`: 20 fixed clean-prompt MobileSAM smoke results
  and overlays for the first ten instances; these are not the full 100
  prediction masks used by Setup 1.
- `results/metrics/`: tables and figures derived from the **shared** 800-row
  CSV. See [result provenance](results/metrics/SOURCE.md) and the
  [Results/Discussion draft](report/results_discussion.md).
- `scripts/extract_shared_masks_zip.py`: verifies and installs the optional
  shared-run mask bundle without overwriting tracked CSVs or COCO data.

**Download the shared-run results:** [results.zip](results.zip) (800 predicted
masks: 100 MobileSAM and 700 SAM ViT-B, plus the raw results and prompt CSVs).
Its SHA-256 is
`f3e2b6dd7b1640eedfd35bf5b1aede9fd76c66ff5aa11057a65ffd3943a7c073`.
The ZIP contains binary masks, not the original COCO photographs. For a quick
visual inspection, open the [qualitative examples](results/examples/figure_examples.png)
and the [metric figures](results/metrics/figures/) in this repository.

COCO images/GT masks, pretrained checkpoints, virtual environments, and the
loose full-resolution prediction-mask directory are not committed. The
processed **evaluation dataset** has the public download link in `DATA.md`;
the shared-run prediction masks are downloadable in `results.zip` above. The
`status=ok` CSV rows alone do not prove that the prediction PNGs have been
installed on a new machine.

## 1. Get the repository and Python dependencies

Use Python 3.11 or 3.12. Commands below run from the repository root. To
reproduce inference, install a compatible **PyTorch + TorchVision pair** for
your platform using the [official PyTorch installer](https://pytorch.org/get-started/locally/)
before installing the remaining packages. On Apple Silicon, use CPU for the
documented runs; the current adapter supports `cpu` and `cuda`, not `mps`.

macOS/Linux:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install torch torchvision
.venv/bin/python -m pip install -r requirements-inference.txt
.venv/bin/python -m pip install -r requirements-metrics.txt
```

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch torchvision
.\.venv\Scripts\python.exe -m pip install -r requirements-inference.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-metrics.txt
```

For a CSV/mask review, PyTorch and model checkpoints are unnecessary: install
`requirements.txt` and `requirements-metrics.txt` for the full test suite and
plots, then run the checks in section 4. Install `requirements-inference.txt`
(and a suitable PyTorch/TorchVision pair) when running either pretrained model.
The exact package versions observed for Sơn's run are in `requirements-son.txt` and
`results/environment.json`; Dương's versions are in
`results/duong/requirements-used.txt` and `results/duong/environment.json`.
Those recorded environments differ, so byte-identical model outputs across
machines are not promised.

## 2. Install the evaluation data

Download the processed ZIP from the link and checksum in [DATA.md](DATA.md).
The official COCO source/version, fixed split, preparation algorithm, and
commands for rebuilding the set from COCO annotations are documented there.
Do **not** unzip the package directly over the repository.

macOS/Linux (replace the ZIP path with your actual download):

```bash
.venv/bin/python scripts/extract_data_zip.py /path/to/coco_eval_seed2026.zip --check-only
.venv/bin/python scripts/extract_data_zip.py /path/to/coco_eval_seed2026.zip
.venv/bin/python scripts/validate_inputs.py
```

Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe scripts\extract_data_zip.py C:\path\to\coco_eval_seed2026.zip --check-only
.\.venv\Scripts\python.exe scripts\extract_data_zip.py C:\path\to\coco_eval_seed2026.zip
.\.venv\Scripts\python.exe scripts\validate_inputs.py
```

Validation must report 50 distinct images, 50 annotations and `status: ok`.

## 3. Install the two official model sources and checkpoints

The adapter expects both official source checkouts **next to** this repository:

```text
parent/
  DL2026-3-7/
  segment-anything/
  MobileSAM/
```

These Git commands work in both Bash and PowerShell:

```bash
git clone https://github.com/facebookresearch/segment-anything.git ../segment-anything
git -C ../segment-anything checkout dca509fe793f601edb92606367a655c15ac00fdf
git clone https://github.com/ChaoningZhang/MobileSAM.git ../MobileSAM
git -C ../MobileSAM checkout f706ad9c4eb7f219c00d9050e46328518ffb65d2
```

If either directory already exists, inspect its commit with `git -C <path>
rev-parse HEAD` instead of cloning over it. Obtain the official
[SAM ViT-B checkpoint](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)
and the [MobileSAM checkpoint](https://github.com/ChaoningZhang/MobileSAM/blob/f706ad9c4eb7f219c00d9050e46328518ffb65d2/weights/mobile_sam.pt).
Copy them to these exact locations inside this repository:

```text
weights/sam_vit_b_01ec64.pth
weights/mobile_sam.pt
```

The MobileSAM checkout at the pinned commit contains `weights/mobile_sam.pt`;
copy that file into this repository's `weights/` directory. The recorded
SHA-256 values are `ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912`
for SAM ViT-B and `6dbb90523a35330fedd7f1d3dfc66f995213d81b29a5ca8108dbcdd4e37d6c2f`
for MobileSAM. These are observed run checksums, not publisher signatures.
Check them with `shasum -a 256 weights/*` on macOS/Linux or
`Get-FileHash weights\... -Algorithm SHA256` in PowerShell. A missing/wrong
checkpoint prevents inference.

For macOS/Linux, the checkpoint-copy commands are:

```bash
mkdir -p weights
curl -fL --retry 3 -o weights/sam_vit_b_01ec64.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
cp ../MobileSAM/weights/mobile_sam.pt weights/mobile_sam.pt
shasum -a 256 weights/sam_vit_b_01ec64.pth weights/mobile_sam.pt
```

Stop if either recorded checksum does not match. The source directories and
checkpoints are **local dependencies**, not files that Git pushes with this
repository.

## 4. Validate the experiment without rerunning 800 predictions

macOS/Linux:

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/validate_inputs.py
.venv/bin/python scripts/check_experiment_matrix.py
.venv/bin/python scripts/check_experiment_matrix.py --config configs/experiment_duong.json
.venv/bin/python scripts/verify_duong_results.py --csv-only
.venv/bin/python scripts/run_experiments.py --dry-run
```

On Windows, replace `.venv/bin/python` with `.\.venv\Scripts\python.exe` and
use the same script arguments. The default matrix check expects 800 rows;
the Dương configuration correctly expects 700 SAM-only rows. These checks
validate IDs, conditions and recorded values, **not** the prediction
masks. To require local PNGs too, add `--require-masks` to the matrix check.
That option should fail on a fresh clone until the masks are restored or
inference has been rerun. `--dry-run` prints the planned matrix without
loading the neural networks.

Download [results.zip](results.zip) from this repository, or use the complete
`son_visualization_bundle_2026-10-06.zip` if you have it. Install
**only** the 800 masks from the shared run after installing the COCO data:

```bash
.venv/bin/python scripts/extract_shared_masks_zip.py results.zip --check-only
.venv/bin/python scripts/extract_shared_masks_zip.py results.zip
.venv/bin/python scripts/check_experiment_matrix.py --require-masks
```

The installer checks all 800 mask IoUs against the tracked CSV and refuses
to overwrite a different local mask. It compares both CSVs in either ZIP
with the tracked files; the complete bundle's manifest is also checked. It
does not extract CSV, manifest, COCO images, GT masks or macOS sidecar files.
The smaller `results.zip` is hosted in this GitHub repository; someone
reproducing the project can either install it or rerun model inference from
the documented checkpoints.
The locally audited `results.zip` has SHA-256
`f3e2b6dd7b1640eedfd35bf5b1aede9fd76c66ff5aa11057a65ffd3943a7c073`;
the complete bundle has SHA-256
`6edca33ca626a7aec86400f9fa15dca4f2c3f65a8209dfe173f4eed940c34575`.
The complete bundle is a separate local handoff and is not required to
repeat the shared-run mask audit.

If you have `DL2026-3-7-duong-sam-vit-b-handoff.zip`, install **only** its 700
prediction masks without overwriting source files or CSVs:

```bash
.venv/bin/python scripts/extract_duong_masks_zip.py /path/to/DL2026-3-7-duong-sam-vit-b-handoff.zip --check-only
.venv/bin/python scripts/extract_duong_masks_zip.py /path/to/DL2026-3-7-duong-sam-vit-b-handoff.zip
.venv/bin/python scripts/verify_duong_results.py --check-only
```

For Windows, use `.\.venv\Scripts\python.exe` in place of `.venv/bin/python`.
The ZIP is a separate handoff artifact; the group must provide its download
link if third parties need to repeat the mask-level audit without rerunning
inference. `--check-only` on the verifier recomputes all 700 IoUs without
rewriting the historical audit or overlays.

To regenerate prompt files, run `scripts/generate_prompts.py` after installing
the data. It uses the same manifest and seed; do not edit the evaluation set
after inspecting model performance.

## 5. Reproduce inference and evaluate

For a quick end-to-end check on the first ten instances:

```bash
.venv/bin/python scripts/run_experiments.py --smoke --device cpu --torch-threads 4
```

For the complete **shared** 800-job run:

```bash
.venv/bin/python scripts/run_experiments.py --device cpu --torch-threads 4
.venv/bin/python scripts/check_experiment_matrix.py --require-masks
```

For Dương's **SAM-only** 700-job run, isolated from the shared CSV and
metadata:

```bash
.venv/bin/python scripts/run_experiments.py --config configs/experiment_duong.json --models sam_vit_b --device cpu --torch-threads 4
.venv/bin/python scripts/check_experiment_matrix.py --config configs/experiment_duong.json --require-masks
.venv/bin/python scripts/verify_duong_results.py --check-only
```

The runner saves one binary mask per prediction, IoU against GT, a model
quality score, prediction time, separate image-encoding time and failures.
It can resume only when an `ok` row **and its mask file** both exist. A fresh
clone contains historical CSVs but no full masks, so these commands will
actually rerun the jobs. Before replacing historical `ok` rows whose masks
are missing, the runner saves a local copy of the original CSV under ignored
`results/backups/`. Do not start the 800-job command just to inspect an
existing CSV; SAM ViT-B can take substantial time on CPU. `--retry-errors`
retries recorded failed rows. See [runner handoff](docs/son_handoff.md) and
[SAM setup details](docs/sam_vit_b_checkpoints.md).

## 6. Demo on your own image

No COCO annotation or GT mask is needed. Select **one** positive point `(x,y)`
on the desired object, or a box `[x_min,y_min,x_max,y_max]`, in pixels of the
original image. Example using an installed COCO image and its known point:

```bash
.venv/bin/python scripts/demo_image.py --image data/coco/val2017/000000521282.jpg --model sam_vit_b --point 177 435 --device cpu
```

Replace `--point 177 435` with `--box 69 298 285 628` to test a box. Replace
`--image` with an absolute path to your own JPG/PNG and choose coordinates for
that image. The command prints an output directory under `results/demo/`
containing `mask.png` (white object on black background), `overlay.png` (green
prediction and yellow prompt), and `result.json`. It does not compute IoU for
an arbitrary image because there is no ground-truth mask. This is a
command-line demo, **not** a web upload interface.

## 7. Reproduce result tables and figures

The committed tables use the runner-recorded IoU values in the shared
`results/raw_predictions.csv`. The matching full prediction masks are now
available in `results.zip`, but the original table-generation step used the
CSV values. Install the plotting dependencies and regenerate the tables and
three figures from the tracked CSVs:

```bash
.venv/bin/python -m pip install -r requirements-metrics.txt
.venv/bin/python scripts/make_plots.py --input results/raw_predictions.csv --prompts results/prompts.csv --output-dir results/metrics
.venv/bin/python scripts/check_metrics_outputs.py --raw results/raw_predictions.csv --metrics-dir results/metrics
```

On Windows, replace `.venv/bin/python` with `.\.venv\Scripts\python.exe`.
The check verifies row IDs, group counts, mean IoU, failure counts and paired
counts against the original log. It does **not** prove that every prediction
mask is installed or independently re-run inference. After installing the
800 masks from the **same shared run**, `scripts/evaluate_masks.py` can first
recompute IoU and Dice from each mask and its GT. Dương's separate 700-mask
archive cannot replace the SAM masks of the shared run. See the
[metrics handoff](docs/metrics_handoff.md) for both paths and limitations.

## 8. Reproduce the error and qualitative analysis

The tracked failure tables and four example panels are described in
[`report/sections/error_analysis.md`](report/sections/error_analysis.md).
Install the additional analysis dependency, then regenerate the failure
counts and prompt-quality checks from the **shared** 800-row log:

```bash
.venv/bin/python -m pip install -r requirements-analysis.txt
.venv/bin/python scripts/analyze_failures.py
.venv/bin/python scripts/check_prompt_quality.py
.venv/bin/python scripts/check_zero_iou.py
```

These commands do not run SAM again. `analyze_failures.py` refreshes the
tracked CSV analysis artifacts under `results/`; the numeric values must
agree with `results/metrics/`. The four saved overlays can be viewed on a
fresh clone, but regenerating them with `scripts/make_overlays.py` additionally
requires the five relevant full-resolution prediction masks from the
**same shared 800-row run**, as well as the COCO images/GT. The separate
Dương masks must not be substituted. The overlay script checks mask paths
and recorded IoUs before changing any saved example images. A fresh clone
without those masks will print a clear missing-artifact message instead.

The thresholds in [the failure definition](docs/failure_definition.md) are
descriptive; the repository does not establish that they were preregistered
before results were inspected. Example selection is reproducible but
post-hoc, not a random sample.

## Reporting and limitations

For Setup 1, compare MobileSAM and SAM ViT-B only on the same 100 clean
prompts in the shared run. For Setup 2, pair clean point and box by
`annotation_id` on SAM ViT-B. For Setup 3, pair each shifted SAM row with its
clean row of the same instance and prompt type. The clean prompts are derived
from GT; noisy prompts are simulated coordinate shifts, not observed human
clicks. The 50-instance sample is small and not class-balanced. The predicted
SAM quality score is **not** ground-truth IoU. See
[`docs/experimental_setup.md`](docs/experimental_setup.md) for exact counts,
timing scope and other interpretation limits.
