"""Object-aware (subject) blur and exposure measures for D5 (fallback path).

Used only if the CLIP-neighbor-relative NIQE/blur/exposure ranking still shows
strong category bias or fails the 95% defect-precision audit. Train-only.

Measures are taken inside COCO train2017 object boxes rather than over the whole
frame, so flat backgrounds (sky, snow, studio walls) do not count as defects:

* subject mask: union of non-crowd boxes with area >= MIN_BOX_AREA native pixels;
* blur: log variance of the full-image Laplacian restricted to the mask
  (computed on the whole image first, so box borders add no artificial edges),
  for the union mask and for the largest qualifying box alone;
* exposure: near-black / near-white fraction inside the union mask.

Images with no qualifying box fall back to whole-image values and are flagged.

    PYTHONPATH=src pixi run -e quality python scripts/score_subject_quality.py
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from vlm_data.config import load_config
from vlm_data.io import sha256_file, write_json_atomic, write_jsonl_atomic
from vlm_data.manifest import read_manifest
from vlm_data.visual_embeddings import image_ids_sha256

ALGORITHM_VERSION = "subject-box-blur-exposure-v1"
MIN_BOX_AREA = 32 * 32  # COCO's small/medium object boundary
DARK_LEVEL = 4
BRIGHT_LEVEL = 251

_ROOT: Path
_BOXES: dict[int, list[tuple[int, int, int, int]]]


def _init(root: Path, boxes: dict) -> None:
    global _ROOT, _BOXES
    _ROOT, _BOXES = root, boxes


def _log_var(values: np.ndarray) -> float:
    return float(np.log(values.var() + 1e-6))


def score(image_id: int) -> dict:
    with Image.open(_ROOT / f"{image_id:012d}.jpg") as image:
        gray = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    height, width = gray.shape
    boxes = _BOXES.get(image_id, [])
    mask = np.zeros(gray.shape, dtype=bool)
    largest, largest_area = None, 0
    for x0, y0, x1, y1 in boxes:
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(width, x1), min(height, y1)
        if x1 <= x0 or y1 <= y0:
            continue
        mask[y0:y1, x0:x1] = True
        if (x1 - x0) * (y1 - y0) > largest_area:
            largest, largest_area = (x0, y0, x1, y1), (x1 - x0) * (y1 - y0)
    fallback = not mask.any()
    if fallback:
        mask[:] = True
        largest = (0, 0, width, height)
    x0, y0, x1, y1 = largest
    pixels = gray[mask]
    return {
        "image_id": image_id,
        "subject_fallback": fallback,
        "subject_box_count": len(boxes),
        "subject_fraction": float(mask.mean()) if not fallback else 1.0,
        "subject_log_laplacian_var": _log_var(laplacian[mask]),
        "largest_box_log_laplacian_var": _log_var(laplacian[y0:y1, x0:x1]),
        "subject_dark_fraction": float(np.mean(pixels <= DARK_LEVEL)),
        "subject_bright_fraction": float(np.mean(pixels >= BRIGHT_LEVEL)),
    }


def load_boxes(instances: Path) -> dict[int, list[tuple[int, int, int, int]]]:
    with instances.open() as handle:
        annotations = json.load(handle)["annotations"]
    boxes: dict[int, list] = defaultdict(list)
    for annotation in annotations:
        x, y, w, h = annotation["bbox"]
        if annotation.get("iscrowd") or w * h < MIN_BOX_AREA:
            continue
        boxes[int(annotation["image_id"])].append(
            (int(np.floor(x)), int(np.floor(y)), int(np.ceil(x + w)), int(np.ceil(y + h)))
        )
    return dict(boxes)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config = load_config()
    root = config.project_root
    instances = config.annotations_dir / "instances_train2017.json"
    image_ids = [record.image_id for record in read_manifest(config.train_manifest)]
    if args.limit:
        image_ids = image_ids[: args.limit]
    suffix = f"_limit{args.limit}" if args.limit else ""
    output = root / f"data/interim/quality/subject_quality_train2017{suffix}.jsonl"
    if output.exists() and not args.overwrite:
        raise SystemExit(f"{output} exists; pass --overwrite to replace it atomically")

    started = time.perf_counter()
    boxes = load_boxes(instances)
    with Pool(args.workers, initializer=_init, initargs=(config.train_images_dir, boxes)) as pool:
        rows = pool.map(score, image_ids, chunksize=64)
    count, digest = write_jsonl_atomic(output, rows)
    write_json_atomic(
        output.with_suffix(".meta.json"),
        {
            "schema_version": 1,
            "kind": "subject_quality_scores",
            "split": "train2017",
            "algorithm_version": ALGORITHM_VERSION,
            "record_count": count,
            "scores_sha256": digest,
            "source_manifest": config.train_manifest.name,
            "source_manifest_sha256": sha256_file(config.train_manifest),
            "instances_sha256": sha256_file(instances),
            "image_ids_sha256": image_ids_sha256(image_ids),
            "min_box_area": MIN_BOX_AREA,
            "exclude_crowd": True,
            "exposure": {"dark_level": DARK_LEVEL, "bright_level": BRIGHT_LEVEL},
            "fallback_count": sum(row["subject_fallback"] for row in rows),
            "versions": {"python": platform.python_version(), "opencv": cv2.__version__},
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "seconds": round(time.perf_counter() - started, 1),
        },
    )
    print({"output": str(output.relative_to(root)), "records": count}, flush=True)


if __name__ == "__main__":
    main()
