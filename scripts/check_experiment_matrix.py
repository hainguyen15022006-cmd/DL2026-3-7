"""Check expected experiment rows against durable raw result logs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.experiment import build_run_plan, read_results, result_summary  # noqa: E402
from src.prompts import load_prompt_records  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.json"))
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()

    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    config = json.loads(config_path.read_text(encoding="utf-8"))
    prompts = load_prompt_records(ROOT / config["prompt_json_path"])
    plan = build_run_plan(prompts)
    results = read_results(ROOT / config["raw_results_path"])
    summary = result_summary(plan, results)
    print(json.dumps(summary, indent=2))
    if args.allow_incomplete:
        return 0
    return 0 if summary["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
