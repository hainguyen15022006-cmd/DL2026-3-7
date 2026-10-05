"""Safely install only Dương's 700 prediction masks from his handoff ZIP."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path, PurePosixPath
import zipfile


ROOT = Path(__file__).resolve().parents[1]
MASK_PREFIX = "results/duong/masks/sam_vit_b/"


def expected_mask_paths(root: Path) -> set[str]:
    with (root / "results/duong/raw_predictions.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 700 or any(row["status"] != "ok" for row in rows):
        raise ValueError("Dương's CSV must contain exactly 700 successful rows")
    paths = {row["mask_path"] for row in rows}
    if len(paths) != 700 or any(
        not path.startswith(MASK_PREFIX) or not path.endswith(".png") for path in paths
    ):
        raise ValueError("Dương's CSV has duplicate or unexpected mask paths")
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zip_file", type=Path)
    parser.add_argument("--check-only", action="store_true", help="Validate without extracting")
    args = parser.parse_args()
    expected = expected_mask_paths(ROOT)

    with zipfile.ZipFile(args.zip_file) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("ZIP contains duplicate member names")
        unsafe = [
            name for name in names
            if PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts or "\\" in name
        ]
        if unsafe:
            raise ValueError(f"ZIP contains an unsafe path: {unsafe[0]}")
        masks = {name for name in names if name.startswith(MASK_PREFIX)}
        if masks != expected:
            raise ValueError(
                f"ZIP mask set differs from CSV: missing={len(expected - masks)}, "
                f"extra={len(masks - expected)}"
            )
        corrupt = archive.testzip()
        if corrupt is not None:
            raise ValueError(f"ZIP CRC validation failed: {corrupt}")

        already_present = 0
        installed = 0
        for name in sorted(expected):
            destination = (ROOT / name).resolve()
            if not destination.is_relative_to(ROOT.resolve()):
                raise ValueError(f"Unsafe destination: {name}")
            contents = archive.read(name)
            if destination.exists():
                if destination.read_bytes() != contents:
                    raise ValueError(f"Existing mask differs; will not overwrite: {destination}")
                already_present += 1
                continue
            if args.check_only:
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(destination.suffix + ".tmp")
            temporary.write_bytes(contents)
            temporary.replace(destination)
            installed += 1

    print(json.dumps({
        "status": "ok",
        "zip_masks": len(expected),
        "already_present": already_present,
        "installed": installed,
        "check_only": args.check_only,
        "note": "Only prediction masks were considered; source files and CSV were not extracted.",
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
