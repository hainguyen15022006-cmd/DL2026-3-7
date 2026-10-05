# DL2026-3-7

Effects of user prompts on interactive image segmentation.

This repository contains a fixed COCO evaluation subset, deterministic prompt
generation, SAM / MobileSAM adapters, experiment runners, and result-analysis
helpers. The models are evaluated with pretrained checkpoints; the project
does not train or fine-tune them.

## Requirements

- Python 3.12 (recommended; the recorded experiment environment uses Python
  3.12).
- Git.
- COCO evaluation files and pretrained checkpoints for model inference. Large
  local data and model files are intentionally excluded from Git.

Create an environment in PowerShell from the repository root:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-test.txt
```

The test dependencies are sufficient for the automated, model-independent test
suite. Install a PyTorch build compatible with your CPU/CUDA setup, then install
the model dependencies for the full experiment:

```powershell
python -m pip install torch torchvision
python -m pip install -r requirements-son.txt
python -m pip install -r chuc_mobile_sam\requirements.txt
```

The SAM source repositories and checkpoints are pinned/documented in
[`configs/experiment.json`](configs/experiment.json),
[`docs/experimental_setup.md`](docs/experimental_setup.md), and
[`docs/checkpoints_mobile_sam.md`](docs/checkpoints_mobile_sam.md). Do not add
checkpoint files or the COCO dataset to Git.

## Dataset and checkpoints

Download or prepare the agreed COCO files using the instructions in
[`DATA.md`](DATA.md). The experiment expects the 50-row manifest at
`configs/eval_manifest.json`, images under `data/coco/val2017/`, and GT masks
under `data/coco/gt_masks/`.

Place checkpoints at the paths specified in `configs/experiment.json`:

- `weights/sam_vit_b_01ec64.pth`
- `weights/mobile_sam.pt`

The MobileSAM adapter also needs the official source at the sibling path
`../MobileSAM` (pinned commit is documented in the checkpoint guide). The SAM
ViT-B adapter expects `../segment-anything` at its pinned commit.

## Verify the repository

Run the automated tests without loading a model or downloading data:

```powershell
python -m pytest -q
```

When the dataset is present, validate it and regenerate deterministic prompts:

```powershell
python scripts\validate_inputs.py
python scripts\generate_prompts.py
python scripts\run_experiments.py --dry-run
```

Run the first ten manifest instances as a model smoke test:

```powershell
python scripts\run_experiments.py --smoke
```

Run the full experiment with `python scripts\run_experiments.py`. See
[`docs/experimental_setup.md`](docs/experimental_setup.md) for the locked
protocol and output schema.

## MobileSAM contributor module

The standalone MobileSAM smoke adapter and its tests are documented in
[`chuc_mobile_sam/README.md`](chuc_mobile_sam/README.md). This module uses the
same fixed manifest and does not replace the official experiment runner.

## Project documentation

- [`DATA.md`](DATA.md): dataset provenance, selection, licensing notes, and
  preparation instructions.
- [`docs/experimental_setup.md`](docs/experimental_setup.md): experimental
  protocol and run commands.
- [`docs/son_handoff.md`](docs/son_handoff.md): prompt and experiment runner
  implementation handoff.
- [`docs/checkpoints_mobile_sam.md`](docs/checkpoints_mobile_sam.md): model
  checkpoint/source verification.

## License

No repository-wide software license has been selected yet. Third-party models,
checkpoints, COCO images, and annotations retain their own licenses and terms;
see the linked source documentation before redistribution.
