# MobileSAM Checkpoint Guide

## Official Sources

| Item | URL |
|---|---|
| MobileSAM repository | https://github.com/ChaoningZhang/MobileSAM |
| MobileSAM paper | Zhang et al., "Faster Segment Anything: Towards Lightweight SAM for Mobile Applications", 2023 — https://arxiv.org/abs/2306.14289 |
| Checkpoint at pinned commit | https://github.com/ChaoningZhang/MobileSAM/blob/f706ad9c4eb7f219c00d9050e46328518ffb65d2/weights/mobile_sam.pt |
| SAM repository | https://github.com/facebookresearch/segment-anything |
| SAM paper | Kirillov et al., "Segment Anything", 2023 — https://arxiv.org/abs/2304.02643 |

## Model identity

- **Model type string:** `vit_t`
- **Package name:** `mobile_sam` (from the MobileSAM repository)
- MobileSAM replaces SAM's heavy ViT-H image encoder (~611 M parameters) with a
  TinyViT encoder (~5 M parameters) and keeps the original SAM prompt encoder and
  mask decoder (3.876 M parameters). Total: ~9.66 M parameters.
- Training: decoupled distillation on 1 % of SA-1B (100 k images), single GPU,
  under one day.

## Pinned source commit

```
f706ad9c4eb7f219c00d9050e46328518ffb65d2
```

Clone the exact commit:

```powershell
git clone https://github.com/ChaoningZhang/MobileSAM.git ..\MobileSAM
cd ..\MobileSAM
git checkout f706ad9c4eb7f219c00d9050e46328518ffb65d2
cd ..\DL2026-3-7
```

## Expected checkpoint fingerprint

| Property | Expected value |
|---|---|
| File | `weights/mobile_sam.pt` |
| Size (bytes) | **40,728,226** |
| SHA-256 | `6dbb90523a35330fedd7f1d3dfc66f995213d81b29a5ca8108dbcdd4e37d6c2f` |

### Local verification (2026-10-06)

The local file `weights/mobile_sam.pt` was verified on the group machine:

| Property | Local value | Match? |
|---|---|---|
| Size (bytes) | 40,728,226 | ✅ |
| SHA-256 | `6dbb90523a35330fedd7f1d3dfc66f995213d81b29a5ca8108dbcdd4e37d6c2f` | ✅ |

Both values match `results/environment.json`.
These are fingerprints of the checkpoint used in the recorded run, not
cryptographic signatures published by the model authors.

## Windows install steps

1. **Clone MobileSAM** next to the project repository (sibling directory):

   ```powershell
   cd <parent-of-DL2026-3-7>
   git clone https://github.com/ChaoningZhang/MobileSAM.git
   cd MobileSAM
   git checkout f706ad9c4eb7f219c00d9050e46328518ffb65d2
   ```

2. **Copy the checkpoint** into the project `weights/` directory:

   ```powershell
   cd ..\DL2026-3-7
   mkdir weights -ErrorAction SilentlyContinue
   Copy-Item ..\MobileSAM\weights\mobile_sam.pt weights\mobile_sam.pt
   ```

3. **Verify size and hash:**

   ```powershell
   # Size
   (Get-Item weights\mobile_sam.pt).Length
   # Expected: 40728226

   # SHA-256
   certutil -hashfile weights\mobile_sam.pt SHA256
   # Expected: 6dbb90523a35330fedd7f1d3dfc66f995213d81b29a5ca8108dbcdd4e37d6c2f
   ```

   Python one-liner:

   ```python
   import hashlib, pathlib
   h = hashlib.sha256(pathlib.Path("weights/mobile_sam.pt").read_bytes()).hexdigest()
   print(h)
   # Expected: 6dbb90523a35330fedd7f1d3dfc66f995213d81b29a5ca8108dbcdd4e37d6c2f
   ```

## Warnings

> **⚠ MobileSAMv2 is a different project.** The MobileSAM repository also hosts
> MobileSAMv2 code and checkpoints. This project uses only the original MobileSAM
> (`vit_t`). Do not load MobileSAMv2 checkpoints.

> **⚠ Do not use ViT-H or ViT-L checkpoints.** MobileSAM's TinyViT encoder is
> incompatible with SAM's ViT-H/ViT-L/ViT-B weights. Always use `mobile_sam.pt`
> with `model_type='vit_t'`.

> **⚠ Weights are not committed to Git.** The `.gitignore` excludes `weights/`.
> Every team member must download or copy the checkpoint locally and verify the
> hash before running experiments.
