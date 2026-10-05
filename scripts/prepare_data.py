"""Prepare the team's fixed COCO val2017 evaluation subset (no model inference)."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone, timedelta
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import shutil
import sys
import time
import urllib.request
import zipfile

import numpy as np
from PIL import Image, ImageDraw
from pycocotools.coco import COCO

ROOT = Path(__file__).resolve().parents[1]
ANNOTATION_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
IMAGE_URL = "http://images.cocodataset.org/val2017/"
SEED = 2026
COUNT = 50


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_content_sha256(rows):
    """Hash manifest values, independent of checkout line endings/indentation."""
    canonical = json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def download(url, path):
    """Retry interrupted downloads; only expose a completed file at its final path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=90) as response:
                expected = response.headers.get("Content-Length")
                with temporary.open("wb") as output:
                    shutil.copyfileobj(response, output)
            if expected is not None and temporary.stat().st_size != int(expected):
                raise OSError("Incomplete download")
            temporary.replace(path)
            return
        except (OSError, TimeoutError):
            if attempt == 2:
                raise
            time.sleep(attempt + 1)


def load_coco(data_dir, allow_download):
    annotations = data_dir / "annotations" / "instances_val2017.json"
    if not annotations.exists():
        if not allow_download:
            raise FileNotFoundError(f"Missing {annotations}; run with --download")
        archive = annotations.parent / "annotations_trainval2017.zip"
        if not archive.exists():
            print("Downloading official COCO annotations...", flush=True)
            download(ANNOTATION_URL, archive)
        annotations.parent.mkdir(parents=True, exist_ok=True)
        # Extract only the needed member, not arbitrary archive paths.
        with zipfile.ZipFile(archive) as source:
            with source.open("annotations/instances_val2017.json") as stream:
                with annotations.with_suffix(".json.part").open("wb") as target:
                    shutil.copyfileobj(stream, target)
        annotations.with_suffix(".json.part").replace(annotations)
    return COCO(str(annotations)), annotations


def select_instances(coco):
    """Filter all annotations before sampling, using stable ID ordering."""
    rejected = Counter()
    candidates = []
    rng = random.Random(SEED)
    valid_count = 0
    for image_id in sorted(coco.imgs):
        info = coco.imgs[image_id]
        valid = []
        for ann in sorted(coco.imgToAnns.get(image_id, []), key=lambda row: row["id"]):
            if ann.get("iscrowd", 0) != 0:
                rejected["iscrowd"] += 1
                continue
            box = ann.get("bbox", [])
            if (len(box) != 4 or not np.isfinite(box).all()
                    or box[2] <= 0 or box[3] <= 0
                    or not np.isfinite(ann.get("area", -1)) or ann.get("area", -1) <= 0):
                rejected["invalid_bbox_or_area"] += 1
                continue
            try:
                mask = coco.annToMask(ann)
            except (TypeError, ValueError, IndexError, KeyError, OverflowError):
                rejected["decode_error"] += 1
                continue
            if mask.shape != (info["height"], info["width"]):
                rejected["wrong_mask_shape"] += 1
            elif not mask.any():
                rejected["empty_mask"] += 1
            else:
                valid.append(ann)
        valid_count += len(valid)
        if valid:
            candidates.append(rng.choice(valid))
    if len(candidates) < COUNT:
        raise ValueError(f"Only {len(candidates)} eligible images; need {COUNT}")
    selected = rng.sample(candidates, COUNT)
    manifest = []
    for ann in selected:
        info = coco.imgs[ann["image_id"]]
        manifest.append({
            "image_id": info["id"], "annotation_id": ann["id"],
            "file_name": info["file_name"], "width": info["width"],
            "height": info["height"], "category_id": ann["category_id"],
            "bbox_xywh": ann["bbox"], "area": ann["area"],
        })
    stats = {
        "images_before_filter": len(coco.imgs),
        "annotations_before_filter": len(coco.anns),
        "annotations_after_filter": valid_count,
        "rejected_annotations": dict(sorted(rejected.items())),
        "eligible_images": len(candidates),
        "images_without_valid_instance": len(coco.imgs) - len(candidates),
        "selected_images": COUNT, "selected_instances": COUNT,
        "smoke_annotation_ids": [row["annotation_id"] for row in manifest[:10]],
        "overlay_annotation_ids": [row["annotation_id"] for row in manifest[:3]],
    }
    assert valid_count + sum(rejected.values()) == len(coco.anns)
    return manifest, stats


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def overlay(image, mask, row, category, path):
    """Original and GT overlay side by side, with explicit instance IDs."""
    pixels = np.asarray(image).copy()
    pixels[mask] = (0.55 * pixels[mask] + 0.45 * np.array([255, 50, 50])).astype(np.uint8)
    marked = Image.fromarray(pixels)
    width, height = image.size
    canvas = Image.new("RGB", (2 * width, height + 64), "white")
    canvas.paste(image, (0, 64))
    canvas.paste(marked, (width, 64))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 6), f"image_id={row['image_id']}  annotation_id={row['annotation_id']}  category={category}", fill="black")
    draw.text((8, 28), "Original RGB image", fill="black")
    draw.text((width + 8, 28), "Red = selected instance GT (45% opacity)", fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def process(root, allow_download, verify_only):
    data_dir = root / "data" / "coco"
    manifest_path = root / "configs" / "eval_manifest.json"
    coco, annotation_path = load_coco(data_dir, allow_download and not verify_only)
    manifest, selection_stats = select_instances(coco)
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise ValueError("Existing manifest differs. Refusing to change the team's locked IDs.")
    elif verify_only:
        raise FileNotFoundError(manifest_path)
    else:
        # Freeze IDs before image downloads, overlays or model evaluation.
        write_json(manifest_path, manifest)
    print(f"Locked {COUNT} instances; first 10 are the smoke subset.", flush=True)
    licenses = {row["id"]: row for row in coco.dataset["licenses"]}
    selected_sources = []
    for index, row in enumerate(manifest):
        image_path = data_dir / "val2017" / row["file_name"]
        mask_path = data_dir / "gt_masks" / f"{row['annotation_id']}.png"
        if not image_path.exists():
            if not allow_download or verify_only:
                raise FileNotFoundError(image_path)
            download(IMAGE_URL + row["file_name"], image_path)
        with Image.open(image_path) as source:
            source.load()
            if source.size != (row["width"], row["height"]):
                raise ValueError(f"Image dimension mismatch: {image_path}")
            image = source.convert("RGB")
        mask = coco.annToMask(coco.anns[row["annotation_id"]]).astype(bool)
        if mask.shape != (row["height"], row["width"]) or not mask.any():
            raise ValueError(f"Invalid selected mask: {row['annotation_id']}")
        if verify_only:
            with Image.open(mask_path) as source:
                saved = np.asarray(source)
            if not np.isin(saved, [0, 255]).all() or not np.array_equal(saved > 0, mask):
                raise ValueError(f"Saved mask differs from COCO GT: {mask_path}")
        else:
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(mask.astype(np.uint8) * 255).save(mask_path)
        if index < 3:
            overlay_path = root / "results" / "examples" / f"data_{row['image_id']}_{row['annotation_id']}.png"
            if verify_only:
                with Image.open(overlay_path) as check:
                    check.verify()
            else:
                overlay(image, mask, row, coco.cats[row["category_id"]]["name"], overlay_path)
        info = coco.imgs[row["image_id"]]
        selected_sources.append({
            "image_id": row["image_id"], "annotation_id": row["annotation_id"],
            "image_url": IMAGE_URL + row["file_name"],
            "flickr_url": info.get("flickr_url"),
            "license": licenses[info["license"]],
            "image_sha256": sha256(image_path), "mask_sha256": sha256(mask_path),
        })
        print(f"[{index + 1}/{COUNT}] image={row['image_id']} ann={row['annotation_id']} OK", flush=True)
    stats = {
        "dataset": "COCO 2017", "split": "val2017", "purpose": "evaluation only; no training or fine-tuning",
        "seed": SEED, "selection": selection_stats,
        "annotation_url": ANNOTATION_URL,
        "annotation_sha256": sha256(annotation_path),
        "manifest_sha256": sha256(manifest_path),
        "manifest_content_sha256": manifest_content_sha256(manifest),
        "python": sys.version.split()[0],
        "dependencies": {name: importlib.metadata.version(name) for name in ("numpy", "Pillow", "pycocotools")},
        "selected_sources": selected_sources,
    }
    stats_path = data_dir / "preparation_stats.json"
    if verify_only:
        existing = json.loads(stats_path.read_text(encoding="utf-8"))
        for key in ("seed", "selection", "annotation_sha256", "selected_sources"):
            if existing[key] != stats[key]:
                raise ValueError(f"Verification mismatch: {key}")
        # Older handoff archives contain a raw SHA256 of the Windows CRLF file.
        # The parsed manifest was already compared with regenerated COCO IDs above;
        # a Git checkout may legitimately change only its line endings.
        if ("manifest_content_sha256" in existing
                and existing["manifest_content_sha256"] != stats["manifest_content_sha256"]):
            raise ValueError("Verification mismatch: manifest_content_sha256")
        if existing.get("manifest_sha256") != stats["manifest_sha256"]:
            print("NOTE: Manifest byte checksum differs; parsed content and COCO selection match.", flush=True)
        print("PASS: regenerated IDs, all 50 image/mask pairs, binary masks, hashes and 3 overlays.", flush=True)
    else:
        stats["prepared_at_vietnam"] = datetime.now(timezone(timedelta(hours=7))).isoformat(timespec="seconds")
        write_json(stats_path, stats)
        print(json.dumps(selection_stats, indent=2), flush=True)
    return manifest


def package(root, manifest):
    """Package data with a reference manifest, not docs or tracked overlays."""
    manifest_path = root / "configs" / "eval_manifest.json"
    stats_path = root / "data" / "coco" / "preparation_stats.json"
    paths = [manifest_path, stats_path]
    for row in manifest:
        paths += [root / "data" / "coco" / "val2017" / row["file_name"],
                  root / "data" / "coco" / "gt_masks" / f"{row['annotation_id']}.png"]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    packaged_stats = json.loads(stats_path.read_text(encoding="utf-8"))
    packaged_stats["manifest_sha256"] = sha256(manifest_path)
    packaged_stats["manifest_content_sha256"] = manifest_content_sha256(manifest)
    target = root / "data" / "coco_eval_seed2026.zip"
    temporary = target.with_suffix(".zip.part")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            member_name = path.relative_to(root).as_posix()
            if path == stats_path:
                archive.writestr(member_name, json.dumps(packaged_stats, ensure_ascii=False, indent=2) + "\n")
            else:
                archive.write(path, member_name)
    with zipfile.ZipFile(temporary) as archive:
        if archive.testzip() is not None:
            raise ValueError("ZIP integrity check failed")
    temporary.replace(target)
    print(f"ZIP: {target}\nSHA256: {sha256(target)}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="Download missing official annotations and the 50 selected images")
    parser.add_argument("--verify-only", action="store_true", help="Read-only validation and regeneration of fixed IDs")
    parser.add_argument("--package", action="store_true", help="Create local handoff ZIP after successful preparation/verification")
    args = parser.parse_args()
    if args.verify_only and args.download:
        parser.error("--verify-only cannot be combined with --download")
    manifest = process(ROOT, args.download, args.verify_only)
    if args.package:
        package(ROOT, manifest)


if __name__ == "__main__":
    main()
