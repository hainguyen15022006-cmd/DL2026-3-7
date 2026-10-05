"""Experiment planning, result persistence and completeness checks."""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from src.prompts import PromptRecord


RESULT_FIELDS = (
    "run_id",
    "setup",
    "model",
    "image_id",
    "annotation_id",
    "prompt_id",
    "prompt_type",
    "noise_level",
    "trial",
    "iou",
    "seconds",
    "encode_seconds",
    "predicted_score",
    "status",
    "error",
    "seed",
    "mask_path",
    "device",
    "started_at_utc",
)


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    setup: str
    model: str
    image_id: int
    annotation_id: int
    prompt_id: str


def _memberships(prompt: PromptRecord, model: str) -> tuple[str, ...]:
    if prompt.noise_level == 0:
        if model == "mobile_sam":
            return ("setup1",)
        if model == "sam_vit_b":
            return ("setup1", "setup2")
    if prompt.noise_level > 0 and model == "sam_vit_b":
        return ("setup3",)
    return ()


def build_run_plan(
    prompts: Iterable[PromptRecord],
    selected_setups: Iterable[str] = ("setup1", "setup2", "setup3"),
    selected_models: Iterable[str] = ("mobile_sam", "sam_vit_b"),
) -> list[RunSpec]:
    selected_setups = set(selected_setups)
    selected_models = set(selected_models)
    unknown_setups = selected_setups - {"setup1", "setup2", "setup3"}
    if unknown_setups:
        raise ValueError(f"Unknown setup values: {sorted(unknown_setups)}")

    plan: list[RunSpec] = []
    for prompt in prompts:
        for model in ("mobile_sam", "sam_vit_b"):
            if model not in selected_models:
                continue
            membership = _memberships(prompt, model)
            if not selected_setups.intersection(membership):
                continue
            plan.append(
                RunSpec(
                    run_id=f"{model}:{prompt.prompt_id}",
                    setup=",".join(membership),
                    model=model,
                    image_id=prompt.image_id,
                    annotation_id=prompt.annotation_id,
                    prompt_id=prompt.prompt_id,
                )
            )
    run_ids = [item.run_id for item in plan]
    if len(run_ids) != len(set(run_ids)):
        raise ValueError("Run plan contains duplicate run_id values")
    return plan


def binary_iou(prediction: np.ndarray, ground_truth: np.ndarray) -> float:
    prediction = np.asarray(prediction, dtype=bool)
    ground_truth = np.asarray(ground_truth, dtype=bool)
    if prediction.shape != ground_truth.shape:
        raise ValueError(
            f"IoU masks must have the same shape, got {prediction.shape} and {ground_truth.shape}"
        )
    intersection = np.logical_and(prediction, ground_truth).sum(dtype=np.int64)
    union = np.logical_or(prediction, ground_truth).sum(dtype=np.int64)
    if union == 0:
        return 1.0
    return float(intersection / union)


def read_results(path: str | Path) -> dict[str, dict[str, str]]:
    path = Path(path)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    results: dict[str, dict[str, str]] = {}
    for row_number, row in enumerate(rows, start=2):
        run_id = row.get("run_id", "")
        if not run_id:
            raise ValueError(f"Result row {row_number} has no run_id")
        if run_id in results:
            raise ValueError(f"Duplicate run_id in results: {run_id}")
        results[run_id] = row
    return results


def write_results(path: str | Path, results: dict[str, dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=RESULT_FIELDS, extrasaction="raise")
        writer.writeheader()
        for run_id in sorted(results):
            writer.writerow(results[run_id])
    # On Windows, a short-lived reader (for example VS Code's CSV preview or an
    # antivirus scanner) can momentarily lock the destination. Keep the atomic
    # replace semantics, but tolerate that transient condition instead of
    # terminating a long experiment run.
    for attempt in range(8):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(0.25 * (attempt + 1))


def result_summary(
    plan: Iterable[RunSpec], results: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    plan = list(plan)
    expected = {item.run_id for item in plan}
    present = expected.intersection(results)
    ok = {run_id for run_id in present if results[run_id].get("status") == "ok"}
    errors = present - ok
    unexpected = set(results) - expected
    return {
        "expected_rows": len(expected),
        "present_rows": len(present),
        "ok_rows": len(ok),
        "error_rows": len(errors),
        "missing_rows": len(expected - present),
        "unexpected_rows": len(unexpected),
        "complete": len(present) == len(expected),
        "all_ok": len(ok) == len(expected),
        "missing_run_ids": sorted(expected - present),
        "error_run_ids": sorted(errors),
        "unexpected_run_ids": sorted(unexpected),
    }


def missing_mask_run_ids(
    plan: Iterable[RunSpec], results: dict[str, dict[str, Any]], root: str | Path
) -> list[str]:
    """Find successful rows whose saved prediction mask is unavailable or unsafe."""
    root = Path(root).resolve()
    missing: list[str] = []
    for spec in plan:
        row = results.get(spec.run_id)
        if row is None or row.get("status") != "ok":
            continue
        mask_ref = row.get("mask_path", "")
        if not mask_ref:
            missing.append(spec.run_id)
            continue
        mask_path = Path(mask_ref)
        if mask_path.is_absolute():
            missing.append(spec.run_id)
            continue
        resolved = (root / mask_path).resolve()
        if not resolved.is_relative_to(root) or not resolved.is_file():
            missing.append(spec.run_id)
    return sorted(missing)


def missing_model_assets(
    root: str | Path, model_configs: dict[str, dict[str, Any]], model_names: Iterable[str]
) -> list[str]:
    """List absent source/checkpoint paths before a run can change existing CSVs."""
    root = Path(root).resolve()
    missing: list[str] = []
    for name in sorted(set(model_names)):
        config = model_configs[name]
        source = (root / config["source_path"]).resolve()
        checkpoint = (root / config["checkpoint_path"]).resolve()
        if not source.is_dir():
            missing.append(f"{name} source directory: {source}")
        if not checkpoint.is_file():
            missing.append(f"{name} checkpoint: {checkpoint}")
    return missing
