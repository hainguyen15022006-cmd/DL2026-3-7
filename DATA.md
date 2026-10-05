# Evaluation Data - Lê Tấn Thành

## Handoff Scope

COCO 2017 **val2017**, 50 images and 50 instances, one instance per image,
with `seed=2026`. The first ten entries in the manifest form the smoke-test
subset. All three experimental setups use this same list. This is an
**evaluation dataset**: there is no training or fine-tuning, and no separate
train/validation split is created for the team.

The data handoff includes the original images, ground-truth (GT) masks,
manifest, preparation script, three verification overlays, and this document.
Prompts, models, IoU, and analysis of model results are the responsibility of
other team members. This handoff contains no inference results.

## Sources and Terms of Use

- COCO download page: <https://cocodataset.org/#download>
- Official annotations: <http://images.cocodataset.org/annotations/annotations_trainval2017.zip>
  (252,907,541 bytes when downloaded on October 5, 2026, Vietnam time).
  The script extracts only `annotations/instances_val2017.json`.
- Complete validation image archive: <http://images.cocodataset.org/zips/val2017.zip>.
  To avoid downloading images outside the evaluation subset, the script
  downloads the 50 fixed images directly from
  `http://images.cocodataset.org/val2017/{file_name}`, without resizing or
  recompressing them.
- COCO API: <https://github.com/cocodataset/cocoapi>;
  polygons/RLE are decoded using `pycocotools.COCO.annToMask`.
- Original terms: <https://cocodataset.org/#termsofuse>;
  [copy in the official repository](https://github.com/cocodataset/cocodataset.github.io/blob/master/dataset/termsofuse.htm).

The annotations belong to the COCO Consortium and are licensed under
CC BY 4.0: <https://creativecommons.org/licenses/by/4.0/>. The GT PNG files
were converted from COCO annotations by Lê Tấn Thành for Group 3 / Topic 7;
no labels were manually edited. Image copyright belongs to the respective
image owners, not the COCO Consortium. Consult the Flickr terms and each
image's individual license. Do not apply CC BY 4.0 to all images as a group.
`preparation_stats.json` records each image's `flickr_url`, source URL,
license ID/name/URL from COCO metadata, and checksums. Overlays are
illustrations with a colored GT layer added to the original image.

## Fixed Sampling Rules

1. Read all val2017 annotations. Iterate over `image_id` in ascending order
   and, within each image, over `annotation_id` in ascending order to avoid
   depending on JSON ordering.
2. Exclude annotations with `iscrowd != 0`; non-finite bounding boxes,
   bounding boxes without exactly four elements, or non-positive width or
   height; non-finite or non-positive area; undecodable segmentation;
   incorrect H×W mask dimensions; or empty masks. Do not impose an object
   size threshold or filter by category or image difficulty. Rejection
   reasons are counted in this order, with each rejected annotation counted
   once.
3. Initialize **one** Python `random.Random(2026)` generator. For each image
   with valid instances, use `rng.choice` to select one instance from the
   list sorted by ID.
4. After selecting one instance for every eligible image, call
   `rng.sample(candidates, 50)` using the same generator. Preserve the
   returned order in the manifest.
5. Write the manifest before downloading images or creating overlays. If
   a subsequent run produces a list that differs from the existing
   manifest, the script stops instead of changing the list automatically.
6. Use `manifest[:10]` for smoke tests and `manifest[:3]` for verification
   overlays. Do not reselect samples based on predicted masks or IoU.
   This handoff uses all 50 samples; the 20-sample fallback is not used.

Recording the seed alone is not sufficient for reproducibility: preserve
the processing order, algorithm above, source data, and environment
versions. The exported manifest is the official list for all team members.

## Actual Counts After Preparation

| Item | Count |
|---|---:|
| Images in the val2017 annotations | 5,000 |
| Original annotations | 36,781 |
| Excluded because `iscrowd != 0` | 446 |
| Excluded because of invalid bbox/area | 0 |
| Excluded because of decoding errors, incorrect dimensions, or empty masks | 0 |
| Remaining valid annotations | 36,335 |
| Images with at least one valid instance | 4,952 |
| Images without a valid instance | 48 |
| Selected images / instances | 50 / 50 |
| Categories represented by the 50 instances | 30 |
| Smoke-test instances / verification overlays | 10 / 3 |

These are dataset statistics, not model results. The small subset is not
class-balanced and does not fully represent COCO. Machine-readable
statistics are stored in `data/coco/preparation_stats.json`. Rejection
reasons absent from the `rejected_annotations` dictionary have a count
of zero.

## Handoff Files and Data Interface

```text
scripts/prepare_data.py
configs/eval_manifest.json
DATA.md
data/coco/annotations/instances_val2017.json
data/coco/val2017/{file_name}
data/coco/gt_masks/{annotation_id}.png
data/coco/preparation_stats.json
results/examples/data_{image_id}_{annotation_id}.png  (3 images)
data/coco_eval_seed2026.zip
```

The manifest is a JSON **list**, with exactly eight agreed fields per entry:

| Field | Meaning / format |
|---|---|
| `image_id` | COCO image ID, integer |
| `annotation_id` | COCO instance ID, integer |
| `file_name` | Original JPEG filename, joined with `data/coco/val2017/` |
| `width`, `height` | Original image dimensions in pixels |
| `category_id` | Original COCO category ID, without remapping |
| `bbox_xywh` | `[x, y, width, height]` from the annotation, with its format unchanged |
| `area` | Area provided by the COCO annotation, not replaced by bounding-box area |

The original `area` may differ from the number of foreground pixels after
polygon rasterization. GT masks are saved as single-channel, 8-bit PNG
files with values of **0 or 255**; convert them to bool using `> 0` when
loading. Each mask identifies the exact instance in its manifest entry,
without combining objects of the same category. Every mask has the
original image shape `(height, width)`. No resizing, cropping, augmentation,
or pixel normalization is performed during data preparation.

Example of loading data from the repository root:

```python
import json
from pathlib import Path
import numpy as np
from PIL import Image

root = Path('.')
manifest = json.loads((root / 'configs/eval_manifest.json').read_text(encoding='utf-8'))
row = manifest[0]  # Use manifest[:10] for smoke tests
image = np.asarray(Image.open(root / 'data/coco/val2017' / row['file_name']).convert('RGB'))

gt = np.asarray(Image.open(root / 'data/coco/gt_masks' / f"{row['annotation_id']}.png")) > 0
assert image.shape[:2] == gt.shape == (row['height'], row['width'])
assert gt.dtype == np.bool_
```

The team member responsible for prompts generates positive points and
tight bounding boxes from GT masks according to the team's protocol.
`bbox_xywh` in the manifest retains the original COCO format; if used as
a box input, convert it to `[x, y, x+w, y+h]`. Do not pass xywh directly
to SAM. GT is used to generate reference prompts and evaluate predictions,
not to select the best predicted mask.

The ten smoke-test samples, in manifest order:

| No. | image_id | annotation_id |
|---|---:|---:|
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

## Installation and Reproduction

Environment used: Python **3.12.14**, NumPy **2.5.3**, Pillow **12.3.0**,
and pycocotools **2.0.11**. Data preparation requires only a CPU.
Run the following commands in PowerShell from the repository root.

On a new machine with Python 3.12:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install numpy==2.5.3 Pillow==12.3.0 pycocotools==2.0.11
.\.venv\Scripts\python.exe scripts\prepare_data.py --download
```

On Thành's machine, where `.venv` already exists, run the final command
directly. That environment was created using the Python runtime bundled
with Codex; do not share `.venv` across machines. The versions above are
the data-preparation dependencies for the team leader to include in the
shared `requirements.txt`.

The script downloads only missing files. If an image download fails, it
keeps the fixed ID list and stops. Rerunning the same command resumes using
the files already present; no images are automatically substituted.
If you downloaded the data manually, place the JSON and images at the
paths above, then run:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_data.py
```

To verify the data without modifying it:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_data.py --verify-only
```

This command regenerates the list from the annotations, compares it with
the fixed manifest, checks all 50 image/mask pairs and binary PNG values,
compares masks pixel by pixel with `annToMask`, and checks checksums and
the three overlay files. It does not use the network.

To recreate the ZIP after preparation or verification:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_data.py --verify-only --package
```

## Overlay Inspection and Annotation Limitations

The three examples come from the first three manifest entries; they were
not selected based on model quality:

- `data_521282_1152444.png`: category **vase**; GT marks the vase.
- `data_260925_144786.png`: category **car**; GT marks the car. The COCO
  annotation is relatively coarse and partly covers the cat in front.
  The original annotation is retained, and this GT limitation should be
  considered when interpreting IoU later.
- `data_54592_613127.png`: category **skis**; GT marks the selected ski
  instance.

All three were visually inspected: the original image is on the left,
and the GT overlay in red at 45% opacity is on the right. No coordinate
shifts or swapped dimensions caused by data processing were observed.
Inspecting three samples does not establish that all COCO annotations
are perfect.

SHA256 of the downloaded `instances_val2017.json`:
`e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f`.

SHA256 of the fixed manifest:
`de36a8b929c27bc14872969d091b72ad54f2bd62013b2c342d80d7ac0bd0b4a2`.

These checksums were recorded during preparation; they are not publisher
authentication signatures. Individual image and mask checksums are stored
in `preparation_stats.json`.

## Prepared Data Handoff Package

Local file: **`data/coco_eval_seed2026.zip`**. The package contains
50 original images, 50 GT masks, the manifest, statistics/source/license
metadata, three overlays, and `DATA.md`. Extract it at the repository
root to use the images and masks immediately. The package does not contain
the complete COCO annotations. To run `--verify-only` on a new machine,
download and place `instances_val2017.json` at the required path, or run
`--download` first.

- [Download ZIP directly](https://drive.google.com/uc?export=download&id=1WbxCsxLFAPNl1HjxBa-_02kNHiUH-zu8)
- [Open file on Google Drive](https://drive.google.com/file/d/1WbxCsxLFAPNl1HjxBa-_02kNHiUH-zu8/view?usp=sharing)

Verified on October 5, 2026: the file could be downloaded without signing
in. The ZIP contained 9,856,902 bytes and 106 files and passed the integrity
check. ZIP SHA256:
`10366a6fe77cf343bb23f704ed575d53151f7ce83ce78cfc9077a4d20c55d6fa`.

The ZIP on Drive retains the verified data package. Its copy of
`DATA.md` was created before the sharing link was added; use
`DATA.md` on the `thanh-data` branch for the current documentation.
The images, masks, and manifest remain unchanged.

Git ignores the `/data/` directory, including images, masks, annotations,
and the ZIP. The code, manifest, three overlays, and `DATA.md` are the
lightweight files shared through the repository on the `thanh-data`
branch. COCO images and model checkpoints do not need to be added to Git.

## Dataset and Data Preparation - Report Handoff Text

We constructed a fixed evaluation subset from the COCO 2017 validation
split (val2017), using the official instance segmentation annotations.
The source contains 5,000 images and 36,781 annotations. We excluded 446
crowd annotations (`iscrowd != 0`), leaving 36,335 valid instances across
4,952 images. We also checked bounding-box and area validity, segmentation
decoding, non-empty masks and mask dimensions; these checks rejected no
additional annotations. No category-specific or minimum-size filtering
was applied.

Using Python's `random.Random(2026)`, we selected one valid instance per
eligible image in ascending image-ID order, with annotations sorted by ID,
and then sampled 50 image-instance pairs using the same random generator.
The subset contains 30 categories and is not class-balanced. All three
experimental setups must use this fixed manifest; its first ten entries
form the smoke-test subset. Selection was completed before model inference
or inspection of IoU scores.

Ground-truth segmentations were decoded with pycocotools 2.0.11 and saved
as binary PNG masks at the original image resolution (0 for background,
255 for the selected instance). Images were not resized or augmented;
RGB conversion is performed when loading them. Three overlays were
visually inspected to check image-mask alignment. We retained the original
COCO annotations, including coarse boundaries or occlusion-related label
limitations. The dataset is used only for evaluation of pretrained models,
with no training, fine-tuning or additional train/validation split. The
small sample and annotation imperfections limit generalization. Sources,
selection rules, software versions and checksums are documented in DATA.md
and the accompanying preparation statistics.

Sources for the Dataset section: the COCO download page, terms of use, and
COCO API linked at the beginning of this document. The team leader should
incorporate this text into the shared report and use the prepared data
download link in the handoff section above.
