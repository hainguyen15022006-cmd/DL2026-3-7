"""Generate the fixed clean and shifted prompt set for all evaluation instances."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_contract import load_manifest  # noqa: E402
from src.prompts import generate_prompt_set, write_prompt_files  # noqa: E402


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--manifest", type=Path, default=Path("configs/eval_manifest.json"))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--json", type=Path, default=Path("results/prompts.json"))
    parser.add_argument("--csv", type=Path, default=Path("results/prompts.csv"))
    parser.add_argument(
        "--summary", type=Path, default=Path("results/prompt_summary.json")
    )
    args = parser.parse_args()

    root = args.root.resolve()
    _, instances = load_manifest(root, args.manifest, expected_count=50)
    records = generate_prompt_set(root, instances, seed=args.seed)
    json_path = args.json if args.json.is_absolute() else root / args.json
    csv_path = args.csv if args.csv.is_absolute() else root / args.csv
    summary_path = args.summary if args.summary.is_absolute() else root / args.summary
    write_prompt_files(records, json_path, csv_path)

    summary = {
        "status": "ok",
        "seed": args.seed,
        "instances": len(instances),
        "prompt_rows": len(records),
        "clean_rows": sum(record.noise_level == 0 for record in records),
        "noisy_rows": sum(record.noise_level > 0 for record in records),
        "point_outside_gt_rows": sum(
            record.point_inside_gt is False for record in records
        ),
        "json_path": json_path.relative_to(root).as_posix(),
        "json_sha256": file_sha256(json_path),
        "csv_path": csv_path.relative_to(root).as_posix(),
        "csv_sha256": file_sha256(csv_path),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = summary_path.with_suffix(summary_path.suffix + ".tmp")
    temporary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    temporary.replace(summary_path)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
