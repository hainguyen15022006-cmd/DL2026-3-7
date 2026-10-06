"""Create metric tables, paired comparisons, failure counts and plots."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
import os
from pathlib import Path
import sys
import tempfile

_MATPLOTLIB_CACHE = Path(tempfile.gettempdir()) / "dl2026-matplotlib"
_MATPLOTLIB_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MATPLOTLIB_CACHE))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.metrics import mean, median, validate_iou


REQUIRED_FIELDS = {
    "model",
    "image_id",
    "annotation_id",
    "prompt_type",
    "noise_level",
    "trial",
    "iou",
    "seconds",
    "status",
    "seed",
}
LOW_IOU_THRESHOLD = 0.5
LARGE_DROP_THRESHOLD = 0.2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="results/raw_predictions.csv")
    parser.add_argument("--prompts", default="results/prompts.csv")
    parser.add_argument("--output-dir", default="results/metrics")
    return parser.parse_args()


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if str(path) == "-":
        reader = csv.DictReader(sys.stdin)
        if reader.fieldnames is None:
            raise ValueError("Standard input CSV has no header")
        return list(reader.fieldnames), list(reader)
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader.fieldnames), list(reader)


def write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_bool(value: object) -> bool | None:
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def optional_float(value: object) -> float | None:
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def row_key(row: dict[str, object]) -> tuple[str, str, str, str, str]:
    return (
        str(row.get("model", "")),
        str(row.get("annotation_id", "")),
        str(row.get("prompt_type", "")),
        str(row.get("noise_level", "")),
        str(row.get("trial", "")),
    )


def load_prompt_quality(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        print(f"Prompt CSV not found; prompt-quality columns will be empty: {path}")
        return {}
    _, rows = read_csv(path)
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        prompt_id = row.get("prompt_id", "")
        if not prompt_id:
            continue
        if prompt_id in result:
            raise ValueError(f"Duplicate prompt_id in prompt CSV: {prompt_id}")
        result[prompt_id] = row
    return result


def prepare_rows(
    rows: list[dict[str, str]], prompt_quality: dict[str, dict[str, str]]
) -> list[dict[str, object]]:
    seen: set[str | tuple[str, str, str, str, str]] = set()
    prepared: list[dict[str, object]] = []
    for line_number, source in enumerate(rows, start=2):
        unique_key: str | tuple[str, str, str, str, str]
        unique_key = source.get("run_id", "") or row_key(source)
        if unique_key in seen:
            raise ValueError(f"Duplicate result row at line {line_number}: {unique_key}")
        seen.add(unique_key)

        row: dict[str, object] = dict(source)
        row["noise_level"] = float(source["noise_level"])
        row["trial"] = int(source["trial"])
        row["annotation_id"] = int(source["annotation_id"])
        row["image_id"] = int(source["image_id"])
        row["metric_valid"] = False
        row["metric_issue"] = ""

        source_ok = source.get("status", "").strip().lower() == "ok"
        recompute_status = source.get("metric_status", "").strip().lower()
        metric_ok = recompute_status in {"", "ok"}
        if source_ok and metric_ok:
            try:
                row["iou"] = validate_iou(source.get("iou"))
                row["seconds"] = float(source["seconds"])
                row["metric_valid"] = True
            except (TypeError, ValueError) as error:
                row["metric_issue"] = str(error)
        else:
            row["metric_issue"] = source.get("error", "") or source.get(
                "metric_error", ""
            ) or "Run status is not ok"

        prompt = prompt_quality.get(source.get("prompt_id", ""), {})
        row["point_inside_gt"] = prompt.get("point_inside_gt", "")
        row["box_iou_gt"] = prompt.get("box_iou_gt", "")
        prepared.append(row)

    clean_lookup = {
        (str(row["model"]), int(row["annotation_id"]), str(row["prompt_type"])): float(row["iou"])
        for row in prepared
        if row["metric_valid"]
        and float(row["noise_level"]) == 0.0
        and int(row["trial"]) == 0
    }
    for row in prepared:
        row["clean_iou"] = ""
        row["iou_drop"] = ""
        row["low_iou_failure"] = ""
        row["large_drop_failure"] = ""
        row["failure"] = ""
        if not row["metric_valid"]:
            continue
        key = (str(row["model"]), int(row["annotation_id"]), str(row["prompt_type"]))
        clean_iou = clean_lookup.get(key)
        row["clean_iou"] = "" if clean_iou is None else clean_iou
        drop = None if clean_iou is None else clean_iou - float(row["iou"])
        row["iou_drop"] = "" if drop is None else drop
        low = float(row["iou"]) < LOW_IOU_THRESHOLD
        large_drop = float(row["noise_level"]) > 0 and drop is not None and drop >= LARGE_DROP_THRESHOLD
        row["low_iou_failure"] = low
        row["large_drop_failure"] = large_drop
        row["failure"] = low or large_drop
    return prepared


def metric_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, float], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["model"]), str(row["prompt_type"]), float(row["noise_level"]))].append(row)

    output = []
    for (model, prompt_type, noise), group in sorted(groups.items()):
        valid = [row for row in group if row["metric_valid"]]
        ious = [float(row["iou"]) for row in valid]
        seconds = [float(row["seconds"]) for row in valid]
        point_values = [
            value
            for value in (parse_bool(row["point_inside_gt"]) for row in valid)
            if value is not None
        ]
        box_values = [
            value
            for value in (optional_float(row["box_iou_gt"]) for row in valid)
            if value is not None
        ]
        output.append(
            {
                "model": model,
                "prompt_type": prompt_type,
                "noise_level": noise,
                "n_total": len(group),
                "n_valid": len(valid),
                "n_error": len(group) - len(valid),
                "mean_iou": mean(ious),
                "median_iou": median(ious),
                "mean_seconds": mean(seconds),
                "median_seconds": median(seconds),
                "point_inside_gt_rate": mean(float(value) for value in point_values),
                "mean_box_iou_gt": mean(box_values),
            }
        )
    return output


def failure_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, float], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["model"]), str(row["prompt_type"]), float(row["noise_level"]))].append(row)
    output = []
    for (model, prompt_type, noise), group in sorted(groups.items()):
        valid = [row for row in group if row["metric_valid"]]
        n_low = sum(row["low_iou_failure"] is True for row in valid)
        n_drop = sum(row["large_drop_failure"] is True for row in valid)
        n_failure = sum(row["failure"] is True for row in valid)
        output.append(
            {
                "model": model,
                "prompt_type": prompt_type,
                "noise_level": noise,
                "n_valid": len(valid),
                "n_error": len(group) - len(valid),
                "n_low_iou": n_low,
                "n_large_drop": n_drop,
                "n_failure": n_failure,
                "failure_rate": n_failure / len(valid) if valid else None,
            }
        )
    return output


def paired_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    valid = [row for row in rows if row["metric_valid"]]
    clean = {
        (str(row["model"]), int(row["annotation_id"]), str(row["prompt_type"])): row
        for row in valid
        if float(row["noise_level"]) == 0 and int(row["trial"]) == 0
    }
    output: list[dict[str, object]] = []
    annotation_ids = sorted({int(row["annotation_id"]) for row in valid})

    for annotation_id in annotation_ids:
        for prompt_type in ("point", "box"):
            mobile = clean.get(("mobile_sam", annotation_id, prompt_type))
            sam = clean.get(("sam_vit_b", annotation_id, prompt_type))
            if mobile and sam:
                output.append(
                    {
                        "comparison": "setup1_sam_vit_b_minus_mobile_sam",
                        "annotation_id": annotation_id,
                        "prompt_type": prompt_type,
                        "noise_level": 0.0,
                        "trial": 0,
                        "reference_iou": mobile["iou"],
                        "comparison_iou": sam["iou"],
                        "difference": float(sam["iou"]) - float(mobile["iou"]),
                    }
                )

        point = clean.get(("sam_vit_b", annotation_id, "point"))
        box = clean.get(("sam_vit_b", annotation_id, "box"))
        if point and box:
            output.append(
                {
                    "comparison": "setup2_box_minus_point",
                    "annotation_id": annotation_id,
                    "prompt_type": "point_vs_box",
                    "noise_level": 0.0,
                    "trial": 0,
                    "reference_iou": point["iou"],
                    "comparison_iou": box["iou"],
                    "difference": float(box["iou"]) - float(point["iou"]),
                }
            )

    for row in valid:
        if str(row["model"]) != "sam_vit_b" or float(row["noise_level"]) <= 0:
            continue
        baseline = clean.get(("sam_vit_b", int(row["annotation_id"]), str(row["prompt_type"])))
        if baseline:
            output.append(
                {
                    "comparison": "setup3_noisy_minus_clean",
                    "annotation_id": row["annotation_id"],
                    "prompt_type": row["prompt_type"],
                    "noise_level": row["noise_level"],
                    "trial": row["trial"],
                    "reference_iou": baseline["iou"],
                    "comparison_iou": row["iou"],
                    "difference": float(row["iou"]) - float(baseline["iou"]),
                }
            )
    return output


def paired_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, float], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["comparison"]), str(row["prompt_type"]), float(row["noise_level"]))].append(row)
    output = []
    for (comparison, prompt_type, noise), group in sorted(groups.items()):
        differences = [float(row["difference"]) for row in group]
        output.append(
            {
                "comparison": comparison,
                "prompt_type": prompt_type,
                "noise_level": noise,
                "n_pairs": len(group),
                "mean_difference": mean(differences),
                "median_difference": median(differences),
                "n_positive": sum(value > 0 for value in differences),
                "n_tie": sum(value == 0 for value in differences),
                "n_negative": sum(value < 0 for value in differences),
            }
        )
    return output


def save_plots(summary: list[dict[str, object]], figure_dir: Path) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    clean = [row for row in summary if float(row["noise_level"]) == 0 and row["mean_iou"] is not None]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    labels = [f"{row['model']}\n{row['prompt_type']}" for row in clean]
    values = [float(row["mean_iou"]) for row in clean]
    ax.bar(labels, values, color=["#4C78A8", "#72B7B2", "#F58518", "#E45756"][: len(values)])
    ax.set_ylabel("Mean instance IoU")
    ax.set_ylim(0, 1)
    ax.set_title("Clean prompts: model and prompt comparison")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_dir / "model_prompt_clean.png", dpi=180)
    plt.close(fig)

    sam_clean = {str(row["prompt_type"]): row for row in clean if row["model"] == "sam_vit_b"}
    if sam_clean:
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        labels = [item for item in ("point", "box") if item in sam_clean]
        ax.bar(labels, [sam_clean[item]["mean_iou"] for item in labels], color=["#4C78A8", "#F58518"])
        ax.set_ylabel("Mean instance IoU")
        ax.set_ylim(0, 1)
        ax.set_title("SAM ViT-B: clean point vs clean box")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(figure_dir / "sam_point_vs_box.png", dpi=180)
        plt.close(fig)

    sam = [row for row in summary if row["model"] == "sam_vit_b" and row["mean_iou"] is not None]
    if sam:
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        for prompt_type, color in (("point", "#4C78A8"), ("box", "#F58518")):
            series = sorted(
                (row for row in sam if row["prompt_type"] == prompt_type),
                key=lambda row: float(row["noise_level"]),
            )
            if series:
                ax.plot(
                    [100 * float(row["noise_level"]) for row in series],
                    [float(row["mean_iou"]) for row in series],
                    marker="o",
                    label=prompt_type,
                    color=color,
                )
        ax.set_xlabel("Prompt shift (% of GT box size)")
        ax.set_ylabel("Mean instance IoU")
        ax.set_ylim(0, 1)
        ax.set_title("SAM ViT-B robustness to prompt shift")
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(figure_dir / "sam_noise_robustness.png", dpi=180)
        plt.close(fig)


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    prompt_path = Path(args.prompts)
    output_dir = Path(args.output_dir)
    fields, source_rows = read_csv(input_path)
    missing = REQUIRED_FIELDS - set(fields)
    if missing:
        raise ValueError(f"Input CSV is missing columns: {sorted(missing)}")

    rows = prepare_rows(source_rows, load_prompt_quality(prompt_path))
    summary = metric_summary(rows)
    failures = failure_summary(rows)
    pair_details = paired_rows(rows)
    pair_summary = paired_summary(pair_details)

    extra_fields = [
        "metric_valid",
        "metric_issue",
        "point_inside_gt",
        "box_iou_gt",
        "clean_iou",
        "iou_drop",
        "low_iou_failure",
        "large_drop_failure",
        "failure",
    ]
    write_csv(output_dir / "per_run_with_metrics.csv", fields + extra_fields, rows)
    write_csv(
        output_dir / "summary_metrics.csv",
        [
            "model", "prompt_type", "noise_level", "n_total", "n_valid", "n_error",
            "mean_iou", "median_iou", "mean_seconds", "median_seconds",
            "point_inside_gt_rate", "mean_box_iou_gt",
        ],
        summary,
    )
    write_csv(
        output_dir / "failure_summary.csv",
        [
            "model", "prompt_type", "noise_level", "n_valid", "n_error", "n_low_iou",
            "n_large_drop", "n_failure", "failure_rate",
        ],
        failures,
    )
    pair_fields = [
        "comparison", "annotation_id", "prompt_type", "noise_level", "trial",
        "reference_iou", "comparison_iou", "difference",
    ]
    write_csv(output_dir / "paired_differences.csv", pair_fields, pair_details)
    write_csv(
        output_dir / "paired_summary.csv",
        [
            "comparison", "prompt_type", "noise_level", "n_pairs", "mean_difference",
            "median_difference", "n_positive", "n_tie", "n_negative",
        ],
        pair_summary,
    )
    save_plots(summary, output_dir / "figures")
    print(f"Processed {len(rows)} run rows")
    print(f"Wrote metric tables and figures to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
