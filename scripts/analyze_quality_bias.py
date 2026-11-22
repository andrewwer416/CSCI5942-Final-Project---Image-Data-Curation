"""Measure category bias of quality measures and build defect-audit contact sheets (D5).

Train-only. Compares two ways of ranking each defect measure:

* global: robust z-score against all of train2017;
* relative: robust z-score against the image's 50 nearest frozen-CLIP neighbors
  (the approved "compare against visually similar images" rule; a local
  neighborhood instead of hard k-means clusters, so no cluster count is chosen).

For each measure and ranking, reports how concentrated COCO categories are in the
worst 1% (lift = tail share / population share), then writes contact sheets for
requested rankings in bands 0-0.25%, 0.25-0.5%, 0.5-1%, 1-2% for visual audit.

    PYTHONPATH=src pixi run python scripts/analyze_quality_bias.py
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from vlm_data.category_balance import load_category_names
from vlm_data.config import load_config
from vlm_data.io import sha256_file, write_json_atomic
from vlm_data.manifest import read_manifest
from vlm_data.visual_neighbors import load_neighbor_artifact

NEIGHBORS = 50
BANDS = [(0.0, 0.0025), (0.0025, 0.005), (0.005, 0.01), (0.01, 0.02)]
SHEET_SAMPLE = 24
TILE = 300
SEED = 5942
MIN_SHORT_SIDE = 192  # approved small-image floor

# Each measure is oriented so that larger values mean "more defective".
MEASURES = {
    "niqe": lambda c: c["niqe"],
    "blur": lambda c: -c["log_laplacian_var_s256"],
    "clipped": lambda c: c["dark_fraction"] + c["bright_fraction"],
    # Object-aware fallback: measured inside COCO train boxes (scripts/score_subject_quality.py).
    "subject_blur": lambda c: -c["subject_log_laplacian_var"],
    "largest_box_blur": lambda c: -c["largest_box_log_laplacian_var"],
    "subject_clipped": lambda c: c["subject_dark_fraction"] + c["subject_bright_fraction"],
    # Learned IQA trained on KonIQ-10k human ratings (scripts/score_learned_quality.py);
    # higher model score = better, so negate.
    "topiq": lambda c: -c["topiq_nr"],
    "musiq": lambda c: -c["musiq"],
}


def robust_z(values: np.ndarray, reference: np.ndarray, axis=None) -> np.ndarray:
    # NaN marks images too small for NIQE; they are ignored here and ranked last.
    median = np.nanmedian(reference, axis=axis)
    mad = np.nanmedian(
        np.abs(reference - (median if axis is None else median[:, None])), axis=axis
    )
    scale = 1.4826 * np.maximum(mad, 1e-6)
    return (values - median) / scale


def category_lift(tail: np.ndarray, labels: list[set[int]], names: dict[int, str]) -> dict:
    population = Counter(c for row in labels for c in row)
    in_tail = Counter(c for i in tail for c in labels[i])
    n, t = len(labels), len(tail)
    lifts = {
        names[c]: round((in_tail[c] / t) / (population[c] / n), 2)
        for c in population
        if in_tail[c] >= 5
    }
    top = dict(sorted(lifts.items(), key=lambda kv: -kv[1])[:8])
    # Total variation distance between tail and population category mixes.
    tvd = 0.5 * sum(
        abs(in_tail[c] / max(1, sum(in_tail.values())) - population[c] / sum(population.values()))
        for c in population
    )
    return {"top_lifts": top, "category_tvd": round(tvd, 4)}


def contact_sheet(root: Path, rows: list[dict], indices, path: Path, label) -> None:
    columns = 4
    height = (len(indices) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * TILE, height * (TILE + 20)), "white")
    draw = ImageDraw.Draw(canvas)
    for slot, index in enumerate(indices):
        row = rows[index]
        with Image.open(root / f"{row['image_id']:012d}.jpg") as image:
            tile = image.convert("RGB")
            tile.thumbnail((TILE, TILE))
        x, y = (slot % columns) * TILE, (slot // columns) * (TILE + 20)
        canvas.paste(tile, (x, y))
        draw.text((x + 2, y + TILE + 3), f"#{slot} {row['image_id']} {label(index)}", fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, quality=90)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sheets",
        nargs="*",
        default=[],
        help="measure:ranking pairs to render audit sheets for, e.g. blur:relative",
    )
    args = parser.parse_args()

    config = load_config()
    root = config.project_root
    scores_path = root / "data/interim/quality/quality_scores_train2017.jsonl"
    subject_path = root / "data/interim/quality/subject_quality_train2017.jsonl"
    rows = [json.loads(line) for line in scores_path.open()]
    subject_rows = [json.loads(line) for line in subject_path.open()]
    if [r["image_id"] for r in rows] != [r["image_id"] for r in subject_rows]:
        raise SystemExit("Whole-image and subject score files are not in the same order")
    learned_path = root / "data/interim/quality/learned_quality_train2017.jsonl"
    learned_rows = [json.loads(line) for line in learned_path.open()]
    if [r["image_id"] for r in rows] != [r["image_id"] for r in learned_rows]:
        raise SystemExit("Whole-image and learned score files are not in the same order")
    for row, subject, learned in zip(rows, subject_rows, learned_rows):
        row.update(subject)
        row.update(learned)
    ids = np.array([r["image_id"] for r in rows], dtype=np.int64)
    columns = {
        k: np.array([np.nan if r[k] is None else r[k] for r in rows], dtype=np.float64)
        for k in rows[0]
    }
    # Approved small-image floor: these images are removed before quality filtering,
    # so they take no part in ranking (NaN sorts last) or in tail statistics.
    eligible = np.minimum(columns["width"], columns["height"]) >= MIN_SHORT_SIDE

    graph_meta = config.visual_artifact_dir / "clip_neighbors_k400.meta.json"
    _, _, neighbor_ids = load_neighbor_artifact(graph_meta)
    neighbor_ids = np.asarray(neighbor_ids[:, : NEIGHBORS + 1])
    if not np.array_equal(neighbor_ids[:, 0], ids):
        raise SystemExit("Quality score order does not match the CLIP neighbor graph")
    position = np.empty(ids.max() + 1, dtype=np.int64)
    position[ids] = np.arange(ids.size)
    neighbor_pos = position[neighbor_ids[:, 1:]]

    records = list(read_manifest(config.train_manifest))
    names = load_category_names(config.train_manifest)
    eligible_index = np.flatnonzero(eligible)
    labels = [set(r.category_ids) for r in records]
    eligible_labels = [labels[i] for i in eligible_index]
    m = int(eligible.sum())
    tail = m // 100
    audit_dir = root / "reports/data/quality-audit"
    report = {
        "neighbors": NEIGHBORS,
        "n": int(ids.size),
        "eligible": m,
        "min_short_side": MIN_SHORT_SIDE,
        "measures": {},
    }
    requested = {tuple(item.split(":")) for item in args.sheets}
    report["sheets_requested"] = sorted(args.sheets)
    rng = random.Random(SEED)

    for measure, orient in MEASURES.items():
        value = np.where(eligible, orient(columns), np.nan)
        zs = {
            "global": robust_z(value, value),
            "relative": robust_z(value, value[neighbor_pos], axis=1),
        }
        order = {name: np.argsort(-z, kind="stable")[:m] for name, z in zs.items()}
        entry = {
            "spearman_global_vs_relative": round(
                float(
                    np.corrcoef(
                        zs["global"][eligible].argsort().argsort(),
                        zs["relative"][eligible].argsort().argsort(),
                    )[0, 1]
                ),
                4,
            ),
            "worst1pct_overlap_global_relative": round(
                len(set(order["global"][:tail]) & set(order["relative"][:tail])) / tail, 4
            ),
        }
        position_in_eligible = np.full(ids.size, -1)
        position_in_eligible[eligible_index] = np.arange(m)
        for name in zs:
            worst = position_in_eligible[order[name][:tail]]
            entry[name] = category_lift(worst, eligible_labels, names)
        entry["bands"] = []
        for ranking in ("global", "relative"):
            if (measure, ranking) not in requested:
                continue
            z = zs[ranking]
            for low, high in BANDS:
                band = order[ranking][int(low * m) : int(high * m)].tolist()
                sample = sorted(rng.sample(band, SHEET_SAMPLE))
                sheet = audit_dir / f"{measure}_{ranking}_{low * 100:g}-{high * 100:g}pct.jpg"
                contact_sheet(
                    config.train_images_dir,
                    rows,
                    sample,
                    sheet,
                    lambda i, z=z, v=value: f"z={z[i]:.1f} raw={v[i]:.2f}",
                )
                entry["bands"].append(
                    {
                        "ranking": ranking,
                        "band": [low, high],
                        "z_at_band_start": round(float(z[band[0]]), 3),
                        "sheet": str(sheet.relative_to(root)),
                        "sample_image_ids": [int(ids[i]) for i in sample],
                    }
                )
        report["measures"][measure] = entry
        print(measure, json.dumps({k: v for k, v in entry.items() if k != "bands"}), flush=True)

    report["scores_sha256"] = sha256_file(scores_path)
    report["subject_scores_sha256"] = sha256_file(subject_path)
    report["learned_scores_sha256"] = sha256_file(learned_path)
    report["script_sha256"] = sha256_file(Path(__file__).resolve())
    write_json_atomic(root / "reports/data/quality-bias.json", report)


if __name__ == "__main__":
    main()
