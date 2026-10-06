# Methods and Related Work — MobileSAM Baseline

## Methods — MobileSAM baseline

We include MobileSAM (Zhang et al., 2023) as a lightweight pretrained baseline
alongside SAM ViT-B (Kirillov et al., 2023). MobileSAM replaces SAM's heavy
ViT-H image encoder (about 611 million parameters) with a TinyViT encoder (about
5 million parameters), while retaining SAM's original prompt encoder and mask
decoder. Its decoupled distillation trains the smaller encoder to reproduce
ViT-H image embeddings without retraining the decoder. The MobileSAM README
reports 3.876 million parameters for the mask decoder and 9.66 million parameters
for the complete model. It also reports training on 100,000 images (1% of SA-1B)
on one GPU in under one day. The full model has about 1.6% of ViT-H's parameter count, motivating its use as a compact reference model; this size difference alone does not establish comparable segmentation behavior.

We use the official pretrained `mobile_sam.pt` checkpoint from MobileSAM source
commit `f706ad9c4eb7f219c00d9050e46328518ffb65d2`, with `model_type='vit_t'`.
No training or fine-tuning is performed. Each image is converted to RGB
`uint8` with shape H × W × 3 and passed to `set_image` once. For each point or
box prompt, `predict` is called with `multimask_output=False` and returns one
binary mask at the original image size. Ground truth is used only to construct
the fixed prompts and calculate IoU; it never selects the predicted mask. We
record `seconds` for the `predict` call alone and record image encoding separately
in `encode_seconds`. Inference is run on CPU. MobileSAM receives exactly the
same stored prompts as SAM ViT-B, allowing a direct comparison under matched
prompt conditions. The checkpoint is not fine-tuned for this evaluation.

## Why this baseline and limits of the comparison

- **Shared family and prompts:** Both models use the SAM promptable segmentation
  setup and receive identical stored prompts.
- **Lightweight encoder:** MobileSAM replaces the ViT-H image encoder with TinyViT.
- **Different checkpoints:** MobileSAM is distilled from ViT-H, not ViT-B. Encoder
  and decoder weights differ, so IoU gaps cannot be attributed to encoder size
  alone.
- **Small sample:** The fixed evaluation set contains 50 instances; small gaps
  are not conclusive.
- **Simulated prompts:** Prompts are generated from ground truth and do not model
  natural user behavior.
- **Timing caveat:** `started_at_utc` shows MobileSAM ran from 06:44:38 to 06:51:44
  UTC and SAM ViT-B from 06:47:04 to 07:23:54 UTC on 2026-10-05. The intervals
  overlap, so timing comparisons are confounded by partial concurrency and apply
  only to the group machine.

## Related Work (draft)

### Segment Anything Model (SAM)

Kirillov et al. (2023) introduced SAM as a model for promptable image
segmentation. Its architecture combines an image encoder, a prompt encoder and a
lightweight mask decoder. This promptable design provides a shared framework for
studying segmentation with user inputs. Our project evaluates SAM ViT-B using
point and box prompts generated from a fixed set of ground-truth masks, then
examines how prompt type and controlled spatial shifts affect the output.
[SAM paper](https://arxiv.org/abs/2304.02643)

[SAM ViT-B owner: add paragraph]

### MobileSAM

Zhang et al. (2023) introduced MobileSAM as a lightweight member of the SAM
family. It replaces the heavy ViT-H image encoder with TinyViT and retains SAM's
original prompt encoder and mask decoder. Decoupled distillation trains the small
encoder to reproduce ViT-H embeddings without retraining the decoder. The README
reports about 5 million image-encoder parameters and 9.66 million parameters
overall, compared with about 611 million and 615 million for SAM ViT-H. It also
reports training on 100,000 images (1% of SA-1B) on one GPU in under a day. We use
the official `vit_t` checkpoint as a lightweight baseline under the same prompts.
[MobileSAM paper](https://arxiv.org/abs/2306.14289) · [official
repository](https://github.com/ChaoningZhang/MobileSAM)

### Promptable segmentation

Promptable segmentation uses user-provided inputs to guide a model toward a
segmented region. SAM offers a common promptable model framework, and MobileSAM
studies a smaller image encoder within that framework. Here we use point and box
prompts and controlled shifts from the ground-truth bounding box to measure
sensitivity to prompt geometry across fixed COCO val2017 instances. Because the
prompts are simulated from ground truth, this experiment measures responses to
controlled inputs rather than natural user behavior. [COCO dataset](https://cocodataset.org/#download)

