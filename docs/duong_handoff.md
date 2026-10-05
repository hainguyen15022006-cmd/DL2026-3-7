# Hoàng Công Dương: SAM ViT-B Handoff

## Completed execution and audit

- Fixed shared COCO manifest: 50 distinct images/instances.
- Shared seed: 2026; clean point/box and 10%/20% shifts, three noisy trials.
- SAM ViT-B smoke run: first ten manifest entries, 140 successful predictions.
- Full run: 700 successful predictions, zero missing/error rows.
- All 700 delivered PNG masks were checked for binary values, original dimensions and correspondence with CSV/prompt IDs.
- IoU was independently recomputed from all predictions and GT masks. Maximum absolute difference from CSV: 0.0.
- Source commit and checkpoint hash are recorded in the accompanying metadata. The checkpoint binary was not included in the submitted ZIP and was not independently rehashed in this audit.
- Local inference was performed by Dương; the artifact audit did not repeat inference.

## Files

| File | Purpose |
|---|---|
| `configs/experiment_duong.json` | SAM-only config with separate CSV/mask paths |
| `results/duong/raw_predictions.csv` | 700 individual measured predictions |
| `results/duong/masks/sam_vit_b/` | 700 masks for independent metrics and qualitative analysis; distributed in the separate handoff ZIP, not tracked by Git |
| `results/duong/environment.json` | Recorded execution environment and checkpoint hash |
| `results/duong/run_state.json` | Runner completeness summary |
| `results/duong/requirements-used.txt` | Observed local package versions |
| `results/duong/sam-source-commit.txt` | Actual local SAM checkout commit |
| `results/duong/verification/verification.json` | Independent artifact audit |
| `results/duong/verification/smoke_*.png` | GT, clean point prediction and clean box prediction overlays |
| `scripts/verify_duong_results.py` | Reproducible audit and smoke rendering |
| `docs/sam_vit_b_methods.md` | English Methods text and contribution/source notes |
| `docs/sam_vit_b_checkpoints.md` | English checkpoint/setup/reproduction guide |

The repo already contains `src/models/sam_vit_b.py` and `src/models/official_sam.py` from the merged runner branch. They were reused without modification for this run. This handoff does not add a second adapter or modify another member's model implementation.

## Smoke observations for the qualitative-analysis owner

Selection is the first three manifest entries, fixed independently of model scores. Red denotes GT, green denotes prediction, yellow denotes point/box prompt. Panels are resized only for display; mask evaluation uses original resolution.

- Image 521282, annotation 1152444: clean point IoU 0.1412, clean box IoU 0.8848. The point mask covers a narrower region of the vase than GT; the box mask covers much more of it.
- Image 260925, annotation 144786: clean point IoU 0.0019, clean box IoU 0.9236. The point mask covers a very small region. The GT car boundary is coarse and extends over the foreground cat, a label limitation also noted in the data component.
- Image 54592, annotation 613127: clean point IoU 0.7105, clean box IoU 0.7307. Both prompts recover much of the selected ski region.

These are observations on three smoke examples, not a population failure rate or causal proof. Low IoU is retained as a valid experimental observation. Final tables and figures are owned by Công Thành; final failure categorization is owned by Hoàng Anh. MobileSAM timing from a different machine must not be used for a direct hardware-fair speed comparison with this CPU run.

## Integration

This contribution has been merged with the runner and data handoff. The SAM-specific docs, verification script, config and CSV are tracked; only the 700 full-resolution prediction masks remain in the separate handoff ZIP. Do not extract that ZIP over the repository: `scripts/extract_duong_masks_zip.py` validates the archive and installs only the masks. It does not include the local COCO dataset or checkpoint. Source images/GT remain available in the existing team data ZIP.

Keep the shared `results/raw_predictions.csv` from Sơn separate from `results/duong/raw_predictions.csv`; they are independent runs. The runner now writes Dương's environment and state directly under `results/duong/`. Do not force-add full-resolution masks ignored by Git. The result ZIP must remain available separately if others need to reproduce the mask-level audit without rerunning inference. Hải Anh integrates Methods with the jointly authored Related Work section.
