"""Run Setup 1-3 with deterministic prompts, resume support and durable logs."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import traceback
from typing import Any

import numpy as np
from PIL import Image
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_contract import load_manifest, validate_dataset  # noqa: E402
from src.experiment import (  # noqa: E402
    RESULT_FIELDS,
    RunSpec,
    binary_iou,
    build_run_plan,
    read_results,
    result_summary,
    write_results,
)
from src.models.factory import MODEL_NAMES, create_model  # noqa: E402
from src.prompts import PromptRecord, load_gt_mask, load_prompt_records  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    if config.get("seed") != 2026:
        raise ValueError("The locked experiment seed must be 2026")
    return config


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def base_result(
    spec: RunSpec,
    prompt: PromptRecord,
    *,
    device: str,
    started_at_utc: str,
) -> dict[str, Any]:
    return {
        "run_id": spec.run_id,
        "setup": spec.setup,
        "model": spec.model,
        "image_id": spec.image_id,
        "annotation_id": spec.annotation_id,
        "prompt_id": spec.prompt_id,
        "prompt_type": prompt.prompt_type,
        "noise_level": prompt.noise_level,
        "trial": prompt.trial,
        "iou": "",
        "seconds": "",
        "encode_seconds": "",
        "predicted_score": "",
        "status": "error",
        "error": "",
        "seed": prompt.seed,
        "mask_path": "",
        "device": device,
        "started_at_utc": started_at_utc,
    }


def save_mask(root: Path, directory: Path, model: str, prompt_id: str, mask: np.ndarray) -> str:
    destination = root / directory / model / f"{prompt_id}.png"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".png.tmp")
    Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255, mode="L").save(
        temporary, format="PNG"
    )
    temporary.replace(destination)
    return destination.relative_to(root).as_posix()


def write_environment(root: Path, config: dict[str, Any], device: str) -> None:
    model_metadata: dict[str, Any] = {}
    for name, model_config in config["models"].items():
        checkpoint = root / model_config["checkpoint_path"]
        model_metadata[name] = {
            **model_config,
            "checkpoint_sha256": sha256(checkpoint) if checkpoint.is_file() else None,
        }
    payload = {
        "created_at_utc": utc_now(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "requested_device": device,
        "torch_threads": torch.get_num_threads(),
        "models": model_metadata,
    }
    path = root / "results" / "environment.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def should_skip(
    row: dict[str, str] | None, root: Path, retry_errors: bool
) -> bool:
    if row is None:
        return False
    if row.get("status") == "error":
        return not retry_errors
    if row.get("status") != "ok":
        return False
    mask_path = row.get("mask_path", "")
    return bool(mask_path and (root / mask_path).is_file())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.json"))
    parser.add_argument(
        "--setup",
        choices=("all", "setup1", "setup2", "setup3"),
        default="all",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_NAMES,
        default=list(MODEL_NAMES),
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--torch-threads",
        type=int,
        help="Set torch CPU threads explicitly and record the value in environment.json",
    )
    parser.add_argument("--smoke", action="store_true", help="Run the first 10 instances")
    parser.add_argument("--limit-instances", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--retry-errors", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.torch_threads is not None:
        if args.torch_threads <= 0:
            raise ValueError("--torch-threads must be a positive integer")
        torch.set_num_threads(args.torch_threads)
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    config = load_config(config_path)
    manifest_path = Path(config["manifest_path"])
    validate_dataset(ROOT, manifest_path, expected_count=50)
    _, instances = load_manifest(ROOT, manifest_path, expected_count=50)

    limit = 10 if args.smoke else args.limit_instances
    if limit is not None:
        if limit <= 0 or limit > len(instances):
            raise ValueError(f"--limit-instances must be between 1 and {len(instances)}")
        instances = instances[:limit]
    selected_annotations = {row.annotation_id for row in instances}

    prompt_path = ROOT / config["prompt_json_path"]
    prompts = [
        prompt
        for prompt in load_prompt_records(prompt_path)
        if prompt.annotation_id in selected_annotations
    ]
    prompts_by_id = {prompt.prompt_id: prompt for prompt in prompts}
    selected_setups = (
        ("setup1", "setup2", "setup3") if args.setup == "all" else (args.setup,)
    )
    plan = build_run_plan(prompts, selected_setups, args.models)

    expected_by_model: dict[str, int] = defaultdict(int)
    for spec in plan:
        expected_by_model[spec.model] += 1
    planning_summary = {
        "status": "planned",
        "instances": len(instances),
        "selected_setups": selected_setups,
        "selected_models": args.models,
        "expected_rows": len(plan),
        "expected_by_model": dict(sorted(expected_by_model.items())),
    }
    print(json.dumps(planning_summary, indent=2))
    if args.dry_run:
        return 0

    results_path = ROOT / config["raw_results_path"]
    mask_directory = Path(config["mask_directory"])
    results = read_results(results_path)
    write_environment(ROOT, config, args.device)
    instance_by_image = {row.image_id: row for row in instances}

    grouped: dict[str, dict[int, list[RunSpec]]] = defaultdict(lambda: defaultdict(list))
    for spec in plan:
        grouped[spec.model][spec.image_id].append(spec)

    for model_name in sorted(grouped):
        pending_model_specs = [
            spec
            for image_specs in grouped[model_name].values()
            for spec in image_specs
            if not should_skip(results.get(spec.run_id), ROOT, args.retry_errors)
        ]
        if not pending_model_specs:
            print(f"{model_name}: no pending rows")
            continue

        try:
            model = create_model(
                model_name,
                ROOT,
                config["models"][model_name],
                args.device,
            )
        except Exception as error:
            message = f"model initialization failed: {type(error).__name__}: {error}"
            for spec in pending_model_specs:
                prompt = prompts_by_id[spec.prompt_id]
                row = base_result(
                    spec, prompt, device=args.device, started_at_utc=utc_now()
                )
                row["error"] = message
                results[spec.run_id] = row
            write_results(results_path, results)
            print(message, file=sys.stderr)
            continue

        for image_index, image_id in enumerate(sorted(grouped[model_name]), start=1):
            specs = grouped[model_name][image_id]
            pending = [
                spec
                for spec in specs
                if not should_skip(results.get(spec.run_id), ROOT, args.retry_errors)
            ]
            if not pending:
                continue

            instance = instance_by_image[image_id]
            image_path = ROOT / "data" / "coco" / "val2017" / instance.file_name
            try:
                with Image.open(image_path) as image_source:
                    image = np.asarray(image_source.convert("RGB"))
                encode_seconds = model.set_image(image)
            except Exception as error:
                message = f"set_image failed: {type(error).__name__}: {error}"
                for spec in pending:
                    prompt = prompts_by_id[spec.prompt_id]
                    row = base_result(
                        spec, prompt, device=model.device, started_at_utc=utc_now()
                    )
                    row["error"] = message
                    results[spec.run_id] = row
                write_results(results_path, results)
                print(f"{model_name} image_id={image_id}: {message}", file=sys.stderr)
                continue

            ground_truth = load_gt_mask(ROOT, instance.annotation_id)
            ok_count = 0
            for spec in pending:
                prompt = prompts_by_id[spec.prompt_id]
                row = base_result(
                    spec, prompt, device=model.device, started_at_utc=utc_now()
                )
                row["encode_seconds"] = encode_seconds
                try:
                    prediction = model.predict(prompt)
                    mask_path = save_mask(
                        ROOT, mask_directory, model_name, prompt.prompt_id, prediction.mask
                    )
                    row.update(
                        {
                            "iou": binary_iou(prediction.mask, ground_truth),
                            "seconds": prediction.seconds,
                            "predicted_score": prediction.score,
                            "status": "ok",
                            "error": "",
                            "mask_path": mask_path,
                        }
                    )
                    ok_count += 1
                except Exception as error:
                    row["error"] = (
                        f"predict failed: {type(error).__name__}: {error}; "
                        f"traceback={traceback.format_exc(limit=1).strip()}"
                    )
                results[spec.run_id] = row
                write_results(results_path, results)
            print(
                f"{model_name} [{image_index}/{len(grouped[model_name])}] "
                f"image_id={image_id}: {ok_count}/{len(pending)} pending rows OK"
            )

    summary = result_summary(plan, results)
    state_path = ROOT / "results" / "run_state.json"
    state_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if summary["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
