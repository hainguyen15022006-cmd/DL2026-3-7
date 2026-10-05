"""Validate and install only COCO data from the team's handoff ZIP.

The original handoff ZIP also contains older copies of tracked files. This
script deliberately never extracts DATA.md, the manifest or example overlays.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[1]


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="Downloaded coco_eval_seed2026.zip")
    parser.add_argument("--check-only", action="store_true", help="Validate without writing files")
    args = parser.parse_args()

    manifest_path = ROOT / "configs" / "eval_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if len(manifest) != 50:
        raise ValueError(f"Expected 50 instances in {manifest_path}, found {len(manifest)}")

    with zipfile.ZipFile(args.archive) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("ZIP contains duplicate member names")
        bad_member = archive.testzip()
        if bad_member is not None:
            raise ValueError(f"ZIP integrity check failed: {bad_member}")
        archived_manifest_bytes = archive.read("configs/eval_manifest.json")
        archived_manifest = json.loads(archived_manifest_bytes)
        if archived_manifest != manifest:
            raise ValueError("ZIP manifest differs from the repository manifest")

        stats_name = "data/coco/preparation_stats.json"
        stats_bytes = archive.read(stats_name)
        stats = json.loads(stats_bytes)
        if stats["manifest_sha256"] != digest(archived_manifest_bytes):
            raise ValueError("ZIP manifest byte checksum differs from its statistics")
        canonical_manifest = json.dumps(
            archived_manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        if ("manifest_content_sha256" in stats
                and stats["manifest_content_sha256"] != digest(canonical_manifest)):
            raise ValueError("ZIP manifest content checksum differs from its statistics")
        sources = stats["selected_sources"]
        if len(sources) != len(manifest):
            raise ValueError("ZIP source statistics do not cover all 50 instances")

        # Validate everything before writing any file. Paths are built from the
        # tracked manifest rather than untrusted ZIP member names.
        data_files: list[tuple[str, bytes]] = [(stats_name, stats_bytes)]
        for row, source in zip(manifest, sources):
            if (row["image_id"], row["annotation_id"]) != (
                source["image_id"], source["annotation_id"]
            ):
                raise ValueError("ZIP source statistics do not match manifest IDs")
            file_name = row["file_name"]
            if Path(file_name).name != file_name:
                raise ValueError(f"Invalid image filename in manifest: {file_name}")
            image_name = f"data/coco/val2017/{file_name}"
            mask_name = f"data/coco/gt_masks/{row['annotation_id']}.png"
            for name, expected_hash in (
                (image_name, source["image_sha256"]),
                (mask_name, source["mask_sha256"]),
            ):
                content = archive.read(name)
                if digest(content) != expected_hash:
                    raise ValueError(f"Checksum mismatch for {name}")
                data_files.append((name, content))

    if args.check_only:
        print(f"PASS: ZIP contains {len(manifest)} image/mask pairs matching the tracked manifest.")
        return

    # Refuse a conflicting local dataset before writing any of the new files.
    for name, content in data_files:
        destination = ROOT / name
        if destination.exists():
            if not destination.is_file() or digest(destination.read_bytes()) != digest(content):
                raise FileExistsError(f"Existing data differs; refusing to overwrite {destination}")

    for name, content in data_files:
        destination = ROOT / name
        if destination.exists():
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".part")
        temporary.write_bytes(content)
        temporary.replace(destination)
    print(f"Installed {len(manifest)} images and {len(manifest)} GT masks in {ROOT / 'data/coco'}.")
    print("Tracked DATA.md, manifest and example overlays were not changed.")


if __name__ == "__main__":
    main()
