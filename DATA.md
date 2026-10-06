# Dataset and Data Preparation

## Scope and version

The experiments use a fixed evaluation subset of **COCO 2017 val2017**:
50 images, one selected instance per image, and `seed=2026`. The first ten
entries of `configs/eval_manifest.json` form the smoke-test subset. All three
experimental setups use the same 50 entries. The group does not create a
training split or train/fine-tune a model on these images.

The original images and COCO-derived ground-truth (GT) masks are distributed
in a separate processed-data ZIP, not committed to Git. The manifest, data
preparation code, validation code, three small alignment overlays and this
document are committed. Model predictions are documented in `results/`.

## Official sources, licensing and processed-data link

- Official COCO download page: <https://cocodataset.org/#download>.
- Dataset version and split: **COCO 2017, val2017**, with instance-segmentation
  annotations from
  <http://images.cocodataset.org/annotations/annotations_trainval2017.zip>.
  The preparation script extracts only `annotations/instances_val2017.json`
  from this archive. The archive measured 252,907,541 bytes when downloaded
  on 2026-10-05 (Vietnam time).
- Official full validation-image archive:
  <http://images.cocodataset.org/zips/val2017.zip>. To avoid downloading
  unrelated images, the script fetches only the selected files from
  `http://images.cocodataset.org/val2017/{file_name}`. It does not resize or
  recompress them.
- COCO API: <https://github.com/cocodataset/cocoapi>. The script converts
  polygon/RLE annotations into masks with `pycocotools.COCO.annToMask`.
- Terms of use: <https://cocodataset.org/#termsofuse> and the
  [official site repository](https://github.com/cocodataset/cocodataset.github.io/blob/master/dataset/termsofuse.htm).
- Processed 50-image/50-GT-mask ZIP:
  [direct download](https://drive.google.com/uc?export=download&id=1WbxCsxLFAPNl1HjxBa-_02kNHiUH-zu8)
  or [Google Drive page](https://drive.google.com/file/d/1WbxCsxLFAPNl1HjxBa-_02kNHiUH-zu8/view?usp=sharing).
  The group checked anonymous download access on 2026-10-05. The ZIP contains
  106 entries, is 9,856,902 bytes, and has SHA-256
  `10366a6fe77cf343bb23f704ed575d53151f7ce83ce78cfc9077a4d20c55d6fa`.

COCO annotations are attributed to the COCO Consortium under CC BY 4.0.
Lê Tấn Thành converted the selected annotations into binary GT PNGs for
Group 3, Project 7 without manually changing the labels. The image
copyrights remain with their respective owners; **do not assign one blanket
CC BY 4.0 license to every photograph**. `data/coco/preparation_stats.json`
records each selected image's source/Flickr URL, license metadata and
checksums. The three overlays under `results/examples/data_*.png` are
illustrations of the original image plus GT.

## Fixed selection and preprocessing procedure

1. Read all COCO val2017 instance annotations. Visit `image_id` in ascending
   order and `annotation_id` in ascending order within each image, avoiding
   dependence on the JSON file's original order.
2. Reject annotations with `iscrowd != 0`, a missing/non-finite/non-positive
   box or area, an undecodable segmentation, a mask whose shape differs from
   the image, or an empty mask. Rejection reasons are counted in that order,
   once per annotation. There is no object-size, category or visual-ease
   filter.
3. Construct exactly one Python `random.Random(2026)` generator. For every
   image with at least one valid instance, call `rng.choice` on its
   ID-sorted valid annotations to select one instance.
4. Call `rng.sample(candidates, 50)` using the **same** generator. Preserve
   the returned order in the manifest.
5. Write/check the manifest before downloading selected images or drawing
   overlays. If a rerun produces a different list, the script stops instead
   of silently replacing the fixed evaluation set.
6. Use `manifest[:10]` for smoke tests and `manifest[:3]` for three GT
   alignment overlays. Do not reselect examples based on model IoU.

The seed alone does not fully specify the sample; the ordered algorithm,
source version and software environment above are part of the protocol.
Selection was completed before model predictions were inspected.

| Data-preparation count | Value |
|---|---:|
| Images in val2017 annotations | 5,000 |
| Initial instance annotations | 36,781 |
| Rejected because `iscrowd != 0` | 446 |
| Rejected by other validity checks | 0 |
| Remaining valid annotations | 36,335 |
| Images with at least one valid instance | 4,952 |
| Images without a valid instance | 48 |
| Selected images / instances | 50 / 50 |
| Categories represented in the subset | 30 |
| Smoke instances / alignment overlays | 10 / 3 |

These are **dataset counts, not model results**. The 50-instance sample is
not class-balanced and does not represent all COCO categories. The
machine-readable counts are in `data/coco/preparation_stats.json`; an absent
key in `rejected_annotations` means zero occurrences.

## Files and data contract

```text
scripts/prepare_data.py
scripts/extract_data_zip.py
scripts/validate_inputs.py
configs/eval_manifest.json
data/coco/annotations/instances_val2017.json     # only after official download
data/coco/val2017/{file_name}
data/coco/gt_masks/{annotation_id}.png
data/coco/preparation_stats.json
results/examples/data_{image_id}_{annotation_id}.png  # three tracked previews
```

The manifest is a JSON **list**. Each entry has exactly these eight fields:

| Field | Meaning |
|---|---|
| `image_id` | Integer COCO image ID |
| `annotation_id` | Integer COCO instance ID |
| `file_name` | Original JPEG filename under `data/coco/val2017/` |
| `width`, `height` | Original image dimensions in pixels |
| `category_id` | Original COCO category ID; not remapped |
| `bbox_xywh` | Original annotation box `[x, y, width, height]` |
| `area` | Original COCO annotation area, not the bounding-box area |

Each GT mask represents **only the selected instance**, not every object of
the same category. It is an 8-bit single-channel PNG with values 0 and 255
and shape `(height, width)`. The COCO `area` field may differ from the number
of foreground pixels after rasterizing a polygon. Images and masks are kept
at original resolution; the data-preparation stage performs no crop, resize,
augmentation or pixel normalization. RGB conversion occurs when loading an
image for a model.

Example from the repository root:

```python
import json
from pathlib import Path

import numpy as np
from PIL import Image

root = Path(".")
manifest = json.loads((root / "configs/eval_manifest.json").read_text(encoding="utf-8"))
row = manifest[0]  # The first ten entries form the smoke subset.
image = np.asarray(Image.open(root / "data/coco/val2017" / row["file_name"]).convert("RGB"))
gt = np.asarray(Image.open(root / "data/coco/gt_masks" / f"{row['annotation_id']}.png")) > 0
assert image.shape[:2] == gt.shape == (row["height"], row["width"])
assert gt.dtype == np.bool_
```

The prompt generator derives one positive point and one tight box from each
GT mask according to `docs/experimental_setup.md`. If using the manifest's
COCO `bbox_xywh` directly as a box prompt, first convert it to
`[x, y, x + width, y + height]`; do not pass XYWH coordinates to SAM as XYXY.
GT is used to define clean prompts and calculate metrics. It is **not** used
to choose whichever predicted mask looks best.

The first ten smoke entries, in manifest order, are:

| # | `image_id` | `annotation_id` |
|---:|---:|---:|
| 1 | 521282 | 1152444 |
| 2 | 260925 | 144786 |
| 3 | 54592 | 613127 |
| 4 | 455085 | 206430 |
| 5 | 220310 | 1161337 |
| 6 | 474854 | 1042325 |
| 7 | 261732 | 658626 |
| 8 | 134689 | 595505 |
| 9 | 32811 | 37495 |
| 10 | 378284 | 506923 |

## Reproduce or install the evaluation data

Use Python 3.12 from the repository root. The recorded preparation
environment used Python 3.12.14, NumPy 2.5.3, Pillow 12.3.0 and
pycocotools 2.0.11. Only CPU is needed. The root `requirements.txt` records
compatible dependency ranges; exact observed versions are above.

### Fast path: install the published processed ZIP

Download the processed ZIP linked above. **Do not unpack the whole archive
over the repository:** it also contains an older copy of `DATA.md` and a
manifest whose line endings can differ from Git. The extraction script
checks the ZIP and installs only the required files under `data/coco/`.

macOS/Linux:

```bash
.venv/bin/python scripts/extract_data_zip.py /path/to/coco_eval_seed2026.zip --check-only
.venv/bin/python scripts/extract_data_zip.py /path/to/coco_eval_seed2026.zip
.venv/bin/python scripts/validate_inputs.py
```

Windows PowerShell: replace `.venv/bin/python` with
`.\.venv\Scripts\python.exe` and use the downloaded ZIP's Windows path.
The `--check-only` call does not write files. The installer checks the fixed
manifest, the 50 image/GT-mask pairs and archive integrity. It does not
need to download the complete COCO annotation archive.

### From the official COCO annotations

Create a Python 3.12 environment and install `requirements.txt` if needed,
then run:

```bash
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/prepare_data.py --download
.venv/bin/python scripts/prepare_data.py --verify-only
```

`--download` obtains the official annotation archive and the **same 50**
selected images, then prepares the masks. It resumes by retaining files
already downloaded; it does not silently replace the manifest. If the
annotation JSON and source images were placed manually at the paths above,
`scripts/prepare_data.py` can run without `--download`. `--verify-only`
checks the fixed sample against the original annotations, all image/mask
pairs, binary mask values, image/mask checksums and the three overlays. It
requires `instances_val2017.json`, which is **not** in the processed ZIP.
For Windows, use `py -3.12` or `.\.venv\Scripts\python.exe`.

To package a freshly prepared and verified subset:

```bash
.venv/bin/python scripts/prepare_data.py --verify-only --package
```

Do not replace the linked processed ZIP without publishing its new URL and
checksum. The current Drive ZIP was created before the latest `DATA.md`,
but its images, masks and manifest contents match the tracked protocol.

## Visual checks and annotation limits

The three tracked alignment overlays use the **first three manifest rows**;
they were not selected by model performance:

- `data_521282_1152444.png`: **vase**; the GT marks the flower vase.
- `data_260925_144786.png`: **car**; the COCO mask is relatively coarse and
  extends partly over a cat in front of the car. We retain the original
  annotation and discuss this as a GT limitation when interpreting IoU.
- `data_54592_613127.png`: **skis**; the GT marks the selected skis.

The original image is on the left and a red GT overlay at 45% opacity is on
the right. Visual inspection of these three examples found no coordinate or
dimension shift introduced by preprocessing. Three checks cannot establish
that every COCO label is perfect.

The downloaded `instances_val2017.json` had SHA-256
`e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f`.
The original ZIP's CRLF manifest had byte-level SHA-256
`de36a8b929c27bc14872969d091b72ad54f2bd62013b2c342d80d7ac0bd0b4a2`;
the same tracked JSON with LF line endings had byte-level SHA-256
`650aa13384930d21c107ba2e4c3da9fbe93a24e8df91219e5b9b33de7498f87d`.
The normalized JSON content hash was
`220a2ec1053848996e4b325e60f3acb3a7bc36b9ef77a3dd7dfc201df749602f`.
Different line endings alone do not mean the sample changed. These observed
checksums are provenance records, not publisher signatures; individual
image/mask hashes are in `preparation_stats.json`.

The local `data/` directory is ignored by Git, as are model checkpoints and
full prediction masks. The code, fixed manifest, preview overlays and this
document remain in the repository. The sample's small size, class imbalance
and occasionally coarse COCO boundaries limit generalization.
