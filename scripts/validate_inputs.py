"""Validate the fixed manifest, source images and GT masks before experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_contract import DataContractError, validate_dataset  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT,
        help="Repository root (default: inferred from this script)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("configs/eval_manifest.json"),
        help="Manifest path, absolute or relative to --root",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Validate file contents for the first 10 rows; schema-check all 50 rows",
    )
    args = parser.parse_args()

    try:
        report = validate_dataset(
            root=args.root,
            manifest_path=args.manifest,
            expected_count=50,
            limit=10 if args.smoke else None,
        )
    except DataContractError as error:
        print(json.dumps({"status": "error", "error": str(error)}, ensure_ascii=False, indent=2))
        return 1

    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
