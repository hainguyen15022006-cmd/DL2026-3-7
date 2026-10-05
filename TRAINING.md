# Training and fine-tuning: not applicable

Both study models are used as **pretrained, frozen inference models**:
MobileSAM (`vit_t`) is the baseline and SAM ViT-B (`vit_b`) is the main model.
The group does not update model weights, run an optimizer, or use COCO images
for training/fine-tuning. The fixed COCO `val2017` subset is used only for
prompt generation and evaluation.

For this project topic, the course instructor confirmed by email that the
repository's training-code section may explicitly say training and
fine-tuning are not applicable. Therefore, there is no training script or
training command to reproduce. Reproduction instead requires the pretrained
checkpoints, the fixed data/prompt protocol, and the evaluation/inference
commands in [README.md](README.md). Checkpoint sources and recorded hashes
are also documented there.
