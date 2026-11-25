"""Train-only audit of P3 crop ranges: how often a crop keeps the objects a caption names.

Uses COCO train2017 instance boxes on the shared random selection (D1); no images are read
and val2017 is never touched. For each image it takes the crops of training epochs 0-7
(the real ``sample_crop_box`` for the image-relative rows) and checks every caption:

* a caption is checkable if it names a COCO category annotated in the image, matched by
  category name, plural, or a short synonym list;
* a crop is caption-consistent if every named category keeps >= 50% of at least one box.

The fixed near-square rows reproduce the usual torchvision-style sampler for comparison.

    PYTHONPATH=src pixi run python scripts/audit_crop_consistency.py
"""

from __future__ import annotations

from collections import defaultdict
import json
import math
import re

import numpy as np

from vlm_data.config import load_config
from vlm_data.io import sha256_file, write_json_atomic
from vlm_data.manifest import read_manifest
from vlm_train.data import CROP_ALGORITHM_VERSION, sample_crop_box

SELECTION = "data/processed/selections/random_seed5942_n15000.jsonl"
INSTANCES = "data/raw/coco/annotations/instances_train2017.json"
EPOCHS = 8
VISIBILITY = 0.5
SMALL_OBJECT_AREA = 0.05  # named object's largest box is under 5% of the image
SEED = 5942
SYNONYMS = {
    "person": "person people man men woman women boy boys girl girls child children kid kids "
    "guy guys lady ladies player players skier skiers surfer surfers snowboarder "
    "skateboarder rider riders",
    "airplane": "airplane airplanes plane planes jet jets aircraft",
    "motorcycle": "motorcycle motorcycles motorbike motorbikes",
    "couch": "couch couches sofa sofas",
    "tv": "tv tvs television televisions",
    "bicycle": "bicycle bicycles bike bikes",
    "laptop": "laptop laptops computer",
    "skis": "ski skis",
    "remote": "remote remotes",
    "donut": "donut donuts doughnut doughnuts",
}
PHRASES = {
    "cell phone": ["cell phone", "cell phones", "cellphone", "cellphones", "phone", "phones"],
    "dining table": ["dining table", "dining tables", "table", "tables"],
    "potted plant": ["potted plant", "potted plants", "houseplant"],
    "wine glass": ["wine glass", "wine glasses"],
    "sports ball": ["ball", "balls", "soccer ball", "tennis ball", "baseball", "basketball"],
    "hot dog": ["hot dog", "hot dogs", "hotdog", "hotdogs"],
    "mouse": ["computer mouse"],
}


def category_patterns(categories: list[dict]) -> dict[int, re.Pattern]:
    patterns = {}
    for category in categories:
        name = category["name"]
        terms = PHRASES.get(name) or (
            SYNONYMS[name].split() if name in SYNONYMS else [name, name + "s", name + "es"]
        )
        patterns[category["id"]] = re.compile(r"\b(?:" + "|".join(map(re.escape, terms)) + r")\b")
    return patterns


def fixed_ratio_crops(rng, width, height, min_scale, ratio, count):
    """torchvision-style RandomResizedCrop boxes (absolute ratio), for comparison only."""

    crops, fallbacks = [], 0
    log_low, log_high = math.log(ratio[0]), math.log(ratio[1])
    for _ in range(count):
        for _attempt in range(10):
            area = width * height * rng.uniform(min_scale, 1)
            r = math.exp(rng.uniform(log_low, log_high))
            w, h = round(math.sqrt(area * r)), round(math.sqrt(area / r))
            if 0 < w <= width and 0 < h <= height:
                crops.append(
                    (rng.integers(0, width - w + 1), rng.integers(0, height - h + 1), w, h)
                )
                break
        else:
            fallbacks += 1
            target = min(max(width / height, ratio[0]), ratio[1])
            w, h = (
                (round(height * target), height)
                if width / height > target
                else (
                    width,
                    round(width / target),
                )
            )
            crops.append(((width - w) // 2, (height - h) // 2, w, h))
    return np.asarray(crops, float), fallbacks


def retained_fraction(crops: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """(crops, boxes) fraction of each box's area inside each crop."""

    x0 = np.maximum(crops[:, None, 0], boxes[:, 0])
    y0 = np.maximum(crops[:, None, 1], boxes[:, 1])
    x1 = np.minimum(crops[:, None, 0] + crops[:, None, 2], boxes[:, 0] + boxes[:, 2])
    y1 = np.minimum(crops[:, None, 1] + crops[:, None, 3], boxes[:, 1] + boxes[:, 3])
    inside = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    return inside / (boxes[:, 2] * boxes[:, 3])


def main() -> None:
    config = load_config()
    root = config.project_root
    records = list(read_manifest(root / SELECTION))
    instances = json.loads((root / INSTANCES).read_text(encoding="utf-8"))
    wanted = {record.image_id for record in records}
    boxes: dict[int, dict[int, list]] = defaultdict(lambda: defaultdict(list))
    for annotation in instances["annotations"]:
        x, y, w, h = annotation["bbox"]
        if annotation["image_id"] in wanted and not annotation.get("iscrowd") and w > 0 and h > 0:
            boxes[annotation["image_id"]][annotation["category_id"]].append((x, y, w, h))
    patterns = category_patterns(instances["categories"])

    variants = [("fixed_ratio", 0.08, (0.75, 4 / 3))]
    variants += [("fixed_ratio", s, (0.8, 1.25)) for s in (0.5, 0.7, 0.8)]
    variants += [("image_relative", s, (0.8, 1.25)) for s in (0.5, 0.7, 0.8)]
    results = []
    for mode, min_scale, ratio in variants:
        rng = np.random.default_rng(SEED)
        kept = checked = small_kept = small_checked = fallbacks = 0
        captions = checkable = 0
        stretch, shape_gap, area = [], [], []
        for record in records:
            width, height = record.width, record.height
            if mode == "image_relative":
                crops = []
                for epoch in range(EPOCHS):
                    b = sample_crop_box(
                        width,
                        height,
                        seed=SEED,
                        epoch=epoch,
                        image_id=record.image_id,
                        min_scale=min_scale,
                        ratio_jitter=ratio,
                    )
                    crops.append((b.left, b.top, b.width, b.height))
                    # Counts fallbacks and genuine full-size draws alike.
                    fallbacks += (b.width, b.height) == (width, height)
                crops = np.asarray(crops, float)
            else:
                crops, f = fixed_ratio_crops(rng, width, height, min_scale, ratio, EPOCHS)
                fallbacks += f
            crop_ratio = crops[:, 2] / crops[:, 3]
            stretch.extend(np.exp(np.abs(np.log(crop_ratio))))
            shape_gap.extend(np.exp(np.abs(np.log(crop_ratio / (width / height)))))
            area.extend(crops[:, 2] * crops[:, 3] / (width * height))
            image_boxes = {k: np.asarray(v, float) for k, v in boxes[record.image_id].items()}
            retained = {k: retained_fraction(crops, b).max(1) for k, b in image_boxes.items()}
            for caption in record.captions:
                captions += 1
                text = caption.text.lower()
                named = [k for k in image_boxes if patterns[k].search(text)]
                if not named:
                    continue
                checkable += 1
                good = np.all([retained[k] >= VISIBILITY for k in named], axis=0)
                kept += int(good.sum())
                checked += EPOCHS
                largest = min(
                    float(np.max(image_boxes[k][:, 2] * image_boxes[k][:, 3])) for k in named
                )
                if largest / (width * height) < SMALL_OBJECT_AREA:
                    small_kept += int(good.sum())
                    small_checked += EPOCHS
        row = {
            "mode": mode,
            "min_scale": min_scale,
            "ratio": list(ratio),
            "caption_consistent": kept / checked,
            "small_object_caption_consistent": small_kept / small_checked,
            "small_object_share": small_checked / checked,
            "fallback_or_full_image_rate": fallbacks / (EPOCHS * len(records)),
            "mean_area_kept": float(np.mean(area)),
            "stretch_to_square_p50": float(np.median(stretch)),
            "shape_gap_vs_p1_eval_p50": float(np.median(shape_gap)),
            "shape_gap_vs_p1_eval_p90": float(np.quantile(shape_gap, 0.9)),
            "checkable_caption_share": checkable / captions,
        }
        results.append(row)
        print({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()}, flush=True)

    output = root / "reports/data/crop-audit.json"
    write_json_atomic(
        output,
        {
            "schema_version": 1,
            "kind": "p3_crop_audit",
            "split": "train2017",
            "selection": SELECTION,
            "selection_sha256": sha256_file(root / SELECTION),
            "instances_sha256": sha256_file(root / INSTANCES),
            "crop_algorithm_version": CROP_ALGORITHM_VERSION,
            "epochs_audited": EPOCHS,
            "visibility_threshold": VISIBILITY,
            "small_object_area": SMALL_OBJECT_AREA,
            "comparison_rng_seed": SEED,
            "script_sha256": sha256_file(root / "scripts/audit_crop_consistency.py"),
            "results": results,
        },
    )
    print({"output": str(output.relative_to(root))})


if __name__ == "__main__":
    main()
