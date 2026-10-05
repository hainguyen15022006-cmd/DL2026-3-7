# SAM ViT-B Checkpoint and Local Setup

## Recorded execution

- Official source: https://github.com/facebookresearch/segment-anything
- Source commit: `dca509fe793f601edb92606367a655c15ac00fdf`
- Model registry key: `vit_b`
- Checkpoint: `weights/sam_vit_b_01ec64.pth`
- Download: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
- Recorded checkpoint SHA256: `ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912`
- The checksum was observed in the submitted execution metadata and user hash output; it is not presented as a publisher signature.
- Python 3.11.9; NumPy 1.26.4; Pillow 11.1.0; SciPy 1.15.2; pycocotools 2.0.11; PyTorch 2.6.0+cpu; TorchVision 0.21.0+cpu.
- CPU execution with four PyTorch threads. The recorded platform string is `Windows-10-10.0.26200-SP0`; this is runtime metadata, not a marketing-version identification.

## Setup on Windows PowerShell

Run from the group repository root. Create a fresh environment if one does not exist:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install numpy==1.26.4 Pillow==11.1.0 scipy==1.15.2 pycocotools==2.0.11
.\.venv\Scripts\python.exe -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
```

Clone the official source into the directory adjacent to the group repository if it is not already there. A detached HEAD at the recorded commit is expected.

```powershell
git clone https://github.com/facebookresearch/segment-anything.git "..\segment-anything"
git -C "..\segment-anything" checkout dca509fe793f601edb92606367a655c15ac00fdf
.\.venv\Scripts\python.exe -m pip install -e "..\segment-anything"
New-Item -ItemType Directory -Force weights
curl.exe -L --fail --output "weights\sam_vit_b_01ec64.pth" "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
Get-FileHash weights\sam_vit_b_01ec64.pth -Algorithm SHA256
```

The environment's observed full package list is `results/duong/requirements-used.txt`. It includes a source reference for the editable SAM installation. The minimal explicit setup above is the supported entry point; record the installed versions if a fresh installation resolves different transitive dependencies.

Images and GT masks must be installed from the team's data ZIP with `scripts/extract_data_zip.py`. Use the tracked manifest and existing shared `results/prompts.json`. Do not regenerate or change prompts to improve individual predictions.

## Run and verify

```powershell
.\.venv\Scripts\python.exe scripts\validate_inputs.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe scripts\run_experiments.py --config configs\experiment_duong.json --models sam_vit_b --device cpu --torch-threads 4 --smoke
.\.venv\Scripts\python.exe scripts\run_experiments.py --config configs\experiment_duong.json --models sam_vit_b --device cpu --torch-threads 4
```

The full run has 700 expected rows. The runner reuses successful rows only when their mask files still exist. To retry logged error rows, append `--retry-errors`. With `configs/experiment_duong.json`, runner metadata is written directly to `results/duong/environment.json` and `results/duong/run_state.json`; the shared metadata files are not replaced.

```powershell
.\.venv\Scripts\python.exe scripts\check_experiment_matrix.py --config configs\experiment_duong.json --require-masks
.\.venv\Scripts\python.exe scripts\verify_duong_results.py --check-only
```

The verification command checks IDs/conditions, binary PNGs, dimensions and all IoU values. With `--check-only`, it does not rewrite the existing audit or fixed first-three smoke overlays. Omit `--check-only` only when regenerating those artifacts is intended. The command requires the team's shared prompts and local COCO images/GT masks. It does not load the model or checkpoint. If the prediction masks are absent, install only those masks from Dương's separate ZIP using `scripts/extract_duong_masks_zip.py`, or use `--csv-only` for a weaker metadata check.

Do not commit COCO images, GT masks, checkpoint files or `.venv`. Prediction masks can be shared in the result ZIP for qualitative analysis; commit only the small verification overlays and logs selected for review.
