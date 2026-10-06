"""Check that metric tables remain traceable to the experiment run log."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from math import isclose
from pathlib import Path
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", default="results/raw_predictions.csv")
    parser.add_argument("--metrics-dir", default="results/metrics")
    return parser.parse_args()


def read_csv(path: str | Path) -> list[dict[str, str]]:
    if str(path) == "-":
        return list(csv.DictReader(sys.stdin))
    with Path(path).open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def require_unique_run_ids(rows: list[dict[str, str]], name: str) -> set[str]:
    run_ids = [row.get("run_id", "") for row in rows]
    if any(not run_id for run_id in run_ids):
        raise ValueError(f"{name} contains an empty run_id")
    if len(run_ids) != len(set(run_ids)):
        raise ValueError(f"{name} contains duplicate run_id values")
    return set(run_ids)


def group_key(row: dict[str, str]) -> tuple[str, str, float]:
    return row["model"], row["prompt_type"], float(row["noise_level"])


def main() -> int:
    args = parse_args()
    metrics_dir = Path(args.metrics_dir)
    raw = read_csv(args.raw)
    per_run = read_csv(metrics_dir / "per_run_with_metrics.csv")
    summary = read_csv(metrics_dir / "summary_metrics.csv")
    failures = read_csv(metrics_dir / "failure_summary.csv")
    pair_details = read_csv(metrics_dir / "paired_differences.csv")
    pair_summary = read_csv(metrics_dir / "paired_summary.csv")

    raw_ids = require_unique_run_ids(raw, "raw log")
    metric_ids = require_unique_run_ids(per_run, "per-run metric table")
    if raw_ids != metric_ids:
        raise ValueError(
            f"Run ID mismatch: missing={len(raw_ids - metric_ids)}, extra={len(metric_ids - raw_ids)}"
        )

    raw_by_id = {row["run_id"]: row for row in raw}
    for row in per_run:
        source = raw_by_id[row["run_id"]]
        if source["status"] == "ok" and not isclose(
            float(source["iou"]), float(row["iou"]), rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(f"IoU changed for run_id={row['run_id']}")

    groups: dict[tuple[str, str, float], list[dict[str, str]]] = defaultdict(list)
    for row in per_run:
        groups[group_key(row)].append(row)
    summary_by_key = {group_key(row): row for row in summary}
    failure_by_key = {group_key(row): row for row in failures}
    if set(groups) != set(summary_by_key) or set(groups) != set(failure_by_key):
        raise ValueError("Summary group keys do not match per-run metric groups")

    for key, rows in groups.items():
        valid = [row for row in rows if row["metric_valid"].lower() == "true"]
        invalid_count = len(rows) - len(valid)
        summary_row = summary_by_key[key]
        if int(summary_row["n_total"]) != len(rows):
            raise ValueError(f"n_total mismatch for {key}")
        if int(summary_row["n_valid"]) != len(valid):
            raise ValueError(f"n_valid mismatch for {key}")
        if int(summary_row["n_error"]) != invalid_count:
            raise ValueError(f"n_error mismatch for {key}")
        if valid:
            expected_mean = sum(float(row["iou"]) for row in valid) / len(valid)
            if not isclose(
                float(summary_row["mean_iou"]), expected_mean, rel_tol=0.0, abs_tol=1e-12
            ):
                raise ValueError(f"mean_iou mismatch for {key}")

        failure_row = failure_by_key[key]
        expected_failures = sum(row["failure"].lower() == "true" for row in valid)
        if int(failure_row["n_valid"]) != len(valid):
            raise ValueError(f"Failure n_valid mismatch for {key}")
        if int(failure_row["n_error"]) != invalid_count:
            raise ValueError(f"Failure n_error mismatch for {key}")
        if int(failure_row["n_failure"]) != expected_failures:
            raise ValueError(f"Failure count mismatch for {key}")

    if sum(int(row["n_total"]) for row in summary) != len(raw):
        raise ValueError("Sum of summary n_total does not match the run log")
    if sum(int(row["n_pairs"]) for row in pair_summary) != len(pair_details):
        raise ValueError("Sum of paired n_pairs does not match paired_differences.csv")

    print(f"PASS: {len(raw)} run rows match per_run_with_metrics.csv")
    print(f"PASS: {len(summary)} summary groups preserve all run counts")
    print(f"PASS: {len(pair_details)} paired rows match paired_summary.csv")
    print("PASS: IoU means and failure counts are traceable to per-run CSV rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
