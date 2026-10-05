# MobileSAM Baseline

This folder contains the MobileSAM adapter, smoke-test scripts, and tests for the
group's segmentation project. The model uses the official pretrained `vit_t`
checkpoint. It is not fine-tuned. Keep model weights and COCO data outside Git.

## Environment setup

Use Python 3.12 and install the PyTorch build that matches your CPU or CUDA
environment. From the repository root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch torchvision
python -m pip install -r chuc_mobile_sam\requirements.txt
```

Place the official checkpoint at `weights/mobile_sam.pt`. The group dataset should
be available at `data/coco/val2017`, with annotations at
`data/coco/annotations/instances_val2017.json`, and the fixed manifest at
`configs/eval_manifest.json`.

## Smoke test on the group dataset

Run from the repository root. This uses the real MobileSAM adapter on the first
10 manifest instances, with one clean point and one clean box per instance:

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

Use `--n 50` to smoke-test all fixed instances. This script is for validating the
adapter and data path; the repository's `scripts/run_experiments.py` is the
official experiment runner for the full locked protocol.

## Tests

```powershell
python -m pytest chuc_mobile_sam\tests -q
```

The manifest tests use `tests/fake_adapter.py` to validate data handling without
loading a checkpoint. The fake adapter must never be used for experiment results.
