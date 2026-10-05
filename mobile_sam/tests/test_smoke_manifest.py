"""Test scripts/smoke_manifest.py with a valid synthetic COCO set and fake adapter (no checkpoint or network required)."""
import csv
import json
import os
import subprocess
import sys

import cv2
import numpy as np
import pytest

pytest.importorskip("pycocotools.mask")
from pycocotools import mask as mask_util   # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "scripts", "smoke_manifest.py")
H, W = 160, 200


def _write_png(path, arr_bgr_or_gray):
    ok, buf = cv2.imencode(".png", arr_bgr_or_gray)
    assert ok
    buf.tofile(str(path))


def build_mock(tmp_path, n=4, with_missing=False):
    """Create n synthetic 160x200 COCO images, each containing an ellipse, in a Unicode-named directory."""
    imgs = tmp_path / "coco images"
    gtdir = tmp_path / "gt"
    imgs.mkdir()
    gtdir.mkdir()
    rng = np.random.default_rng(0)
    images, anns, manifest = [], [], []
    for i in range(n):
        iid, aid = 1000 + i, 5000 + i
        img = np.clip(rng.normal(90, 10, (H, W, 3)), 0, 255).astype(np.uint8)
        gt = np.zeros((H, W), np.uint8)
        cv2.ellipse(gt, (100, 80), (50 + 5 * i, 30 + 4 * i), 15 * i, 0, 360, 1, -1)
        img[gt > 0] = (200, 120, 60)
        name = f"{iid:012d}.png"
        _write_png(imgs / name, img)
        _write_png(gtdir / f"{aid}.png", gt * 255)
        rle = mask_util.encode(np.asfortranarray(gt))
        rle["counts"] = rle["counts"].decode()
        images.append({"id": iid, "file_name": name, "height": H, "width": W})
        anns.append({"id": aid, "image_id": iid, "category_id": 1, "segmentation": rle, "iscrowd": 0,
                     "area": float(mask_util.area(mask_util.encode(np.asfortranarray(gt)))), "bbox": [0, 0, 1, 1]})
        manifest.append({"image_id": iid, "annotation_id": aid, "file_name": name, "width": W, "height": H,
                         "category_id": 1, "bbox_xywh": "0 0 1 1", "area": anns[-1]["area"]})
    if with_missing:
        manifest.insert(1, {**manifest[0], "image_id": 9, "annotation_id": 9999, "file_name": "missing.png"})
    ann_file = tmp_path / "instances.json"
    ann_file.write_text(json.dumps({"images": images, "annotations": anns, "categories": [{"id": 1, "name": "x"}]}))
    mpath = tmp_path / "manifest.csv"
    with open(mpath, "w", newline="", encoding="utf-8-sig") as f:        # BOM as used by Excel-exported files
        w = csv.DictWriter(f, list(manifest[0]))
        w.writeheader()
        w.writerows(manifest)
    return imgs, gtdir, ann_file, mpath


def run(tmp_path, extra, out="out"):
    imgs, gtdir, ann_file, mpath = extra["paths"]
    adapter = extra.get("adapter", "tests.fake_adapter:FakeAdapter")
    cmd = [sys.executable, SCRIPT, "--adapter", adapter, "--checkpoint", "not_required.pt",
           "--manifest", str(mpath), "--image-dir", str(imgs), "--out", str(tmp_path / out), "--n", "10"] + extra["gt"]
    return subprocess.run(cmd, capture_output=True, cwd=ROOT, text=True, encoding="utf-8")


def read(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_runs_with_annotation_file(tmp_path):
    paths = build_mock(tmp_path)
    r = run(tmp_path, {"paths": paths, "gt": ["--ann-file", str(paths[2])]})
    assert r.returncode == 0, r.stderr[-800:]
    res = read(tmp_path / "out" / "results.csv")
    assert len(res) == 8 and {x["prompt_type"] for x in res} == {"point", "box"}      # 4 instance x (point, box)
    assert all(x["status"] == "ok" and 0 <= float(x["iou"]) <= 1 for x in res)
    assert list(res[0])[:10] == ["model", "image_id", "annotation_id", "prompt_type", "noise_level", "trial", "iou",
                                 "seconds", "status", "seed"]                          # Expected group schema
    assert all(x["model"] == "fake" and x["seed"] == "2026" for x in res)
    box = [float(x["iou"]) for x in res if x["prompt_type"] == "box"]
    assert all(0.6 < v < 0.95 for v in box)                  # Tight ellipse boxes cover about pi/4 of their area
    pr = read(tmp_path / "out" / "prompts.csv")
    pts = [p for p in pr if p["prompt_type"] == "point"]
    assert len(pts) == 4 and all(p["point_inside_gt"] == "True" for p in pts)     # The deepest point is always inside GT
    assert len([f for f in os.listdir(tmp_path / "out") if f.startswith("overlay_")]) == 4


def test_gt_dir_gives_same_iou_as_annotation_file(tmp_path):
    paths = build_mock(tmp_path)
    assert run(tmp_path, {"paths": paths, "gt": ["--ann-file", str(paths[2])]}, "a").returncode == 0
    assert run(tmp_path, {"paths": paths, "gt": ["--gt-dir", str(paths[1])]}, "b").returncode == 0
    a = [x["iou"] for x in read(tmp_path / "a" / "results.csv")]
    b = [x["iou"] for x in read(tmp_path / "b" / "results.csv")]
    assert a == b


def test_error_is_logged_not_skipped_and_run_continues(tmp_path):
    paths = build_mock(tmp_path, with_missing=True)
    r = run(tmp_path, {"paths": paths, "gt": ["--ann-file", str(paths[2])]})
    assert r.returncode == 0, r.stderr[-800:]
    res = read(tmp_path / "out" / "results.csv")
    assert len(res) == 10                                    # 5 instances (one fails) x 2 prompt types: no combination is missing
    bad = [x for x in res if x["status"] != "ok"]
    assert len(bad) == 2 and all(x["annotation_id"] == "9999" and x["status"] == "error" for x in bad)
    assert all(x["error"].startswith("error:") for x in bad)
    assert sum(x["status"] == "ok" for x in res) == 8        # The remaining instances still run


def test_prompt_failure_does_not_duplicate_the_other_prompt_row(tmp_path):
    paths = build_mock(tmp_path, n=1)
    r = run(tmp_path, {"paths": paths, "gt": ["--ann-file", str(paths[2])],
                       "adapter": "tests.fake_adapter:FailOnBoxAdapter"})
    assert r.returncode == 0, r.stderr[-800:]
    res = read(tmp_path / "out" / "results.csv")
    assert len(res) == 2
    by_type = {row["prompt_type"]: row for row in res}
    assert by_type["point"]["status"] == "ok"
    assert by_type["box"]["status"] == "error"
    assert "simulated box prediction failure" in by_type["box"]["error"]


def test_requires_exactly_one_gt_source(tmp_path):
    paths = build_mock(tmp_path)
    r = run(tmp_path, {"paths": paths, "gt": []})
    assert r.returncode != 0
    r = run(tmp_path, {"paths": paths, "gt": ["--ann-file", str(paths[2]), "--gt-dir", str(paths[1])]})
    assert r.returncode != 0
