"""Generate the D5 quality-aware diversity selection (train-only).

Approved method (docs/quality-aware-selection.md):

1. Hard filter: drop images whose short side is under 192 px, then drop the worst
   0.25% of the remainder by TOPIQ-NR (KonIQ) score, ranked globally.
2. Objective: D3's sparse facility coverage (frozen OpenCLIP, k=50, 0.99 duplicate
   groups, outlier cap 300) plus a modular quality term,
       F(S) = coverage(S) / N_train + lambda * sum_{i in S} topiq(i) / 15,000.
3. lambda sweep: choose the largest mean quality whose exact mean coverage keeps
   >= 90% of D3's gain over random.
4. Gates (fail rather than force swaps): >= 750 images different from random and
   from D3; mean TOPIQ above D3 with a bootstrap 95% interval excluding zero.

    PYTHONPATH=src pixi run python scripts/calibrate_quality.py
"""

from __future__ import annotations

import json
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np

from vlm_data.config import load_config
from vlm_data.facility_location import select_facility_location_numpy
from vlm_data.io import sha256_file, write_json_atomic, write_jsonl_atomic
from vlm_data.manifest import metadata_path, read_manifest
from vlm_data.selection import select_random
from vlm_data.visual_analysis import (
    build_duplicate_groups,
    compute_visual_density,
    derive_sparse_csr,
    enforce_random_difference,
    evaluate_exact_gpu_coverage,
    map_neighbor_ids_to_positions,
    mark_low_density_outliers,
)
from vlm_data.visual_embeddings import image_ids_sha256, load_embedding_artifact
from vlm_data.visual_neighbors import load_neighbor_artifact

ALGORITHM_VERSION = "quality-aware-facility-topiq-v1"
NEIGHBOR_K = 50  # D3's calibrated k
DUPLICATE_THRESHOLD = 0.99  # D3's audited duplicate cutoff
MIN_SHORT_SIDE = 192
TOPIQ_FILTER_FRACTION = 0.0025
COVERAGE_RETENTION = 0.90
MIN_DIFFERENCE = 750
LAMBDA_GRID = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
BISECTION_STEPS = 8
BOOTSTRAP_RESAMPLES = 2000
D3_MANIFEST = "visual_diversity_k50_dup0p99_seed5942_n15000.jsonl"


def category_mix(records, ids) -> Counter:
    chosen = set(ids)
    return Counter(c for r in records if r.image_id in chosen for c in r.category_ids)


def tvd(a: Counter, b: Counter) -> float:
    ta, tb = sum(a.values()), sum(b.values())
    return 0.5 * sum(abs(a[c] / ta - b[c] / tb) for c in set(a) | set(b))


def bootstrap_difference(a: np.ndarray, b: np.ndarray, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    diffs = [
        rng.choice(a, a.size).mean() - rng.choice(b, b.size).mean()
        for _ in range(BOOTSTRAP_RESAMPLES)
    ]
    return [float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))]


def main() -> None:
    c = load_config()
    v = c.visual_diversity
    root = c.project_root
    budget = c.train_image_budget
    ep = c.visual_artifact_dir / "openclip_vit_b16_datacomp_xl_s13b_b90k_train2017.meta.json"
    gp = c.visual_artifact_dir / "clip_neighbors_k400.meta.json"
    em, embeddings, ids = load_embedding_artifact(ep, source_manifest=c.train_manifest)
    _, scores, neighbors = load_neighbor_artifact(gp, embedding_metadata_path=ep)
    ids = np.asarray(ids)
    n = ids.size

    # Quality inputs, aligned to the embedding order and hash-checked.
    quality_dir = root / "data/interim/quality"
    learned_path = quality_dir / "learned_quality_train2017.jsonl"
    scores_path = quality_dir / "quality_scores_train2017.jsonl"
    learned_meta = json.loads(learned_path.with_suffix(".meta.json").read_text())
    if sha256_file(learned_path) != learned_meta["scores_sha256"]:
        raise SystemExit("Learned quality scores do not match their metadata hash")
    learned = [json.loads(line) for line in learned_path.open()]
    sizes = [json.loads(line) for line in scores_path.open()]
    if [r["image_id"] for r in learned] != ids.tolist() or [
        r["image_id"] for r in sizes
    ] != ids.tolist():
        raise SystemExit("Quality score order does not match the embedding order")
    topiq = np.array([r["topiq_nr"] for r in learned], dtype=np.float64)
    if not bool(np.all(np.isfinite(topiq))) or float(topiq.min()) < 0.0:
        raise SystemExit("TOPIQ scores must be finite and nonnegative")
    short_side = np.array([min(r["width"], r["height"]) for r in sizes])

    # Hard filter.
    size_ok = short_side >= MIN_SHORT_SIDE
    cut_count = int(TOPIQ_FILTER_FRACTION * int(size_ok.sum()))
    order = np.lexsort((ids, topiq))  # ascending TOPIQ, image ID breaks ties
    order = order[size_ok[order]]
    topiq_removed = np.zeros(n, dtype=bool)
    topiq_removed[order[:cut_count]] = True
    topiq_threshold = float(topiq[order[cut_count]])
    eligible = size_ok & ~topiq_removed

    # D3's geometry and constraints.
    positions = map_neighbor_ids_to_positions(ids, neighbors)
    duplicates = build_duplicate_groups(
        embeddings=embeddings,
        source_image_ids=ids,
        neighbor_image_ids=neighbors,
        neighbor_similarities=scores,
        threshold=DUPLICATE_THRESHOLD,
    )
    density = compute_visual_density(ids, neighbors, scores)
    outliers = mark_low_density_outliers(ids, density)
    graph = derive_sparse_csr(positions, scores, k=NEIGHBOR_K)
    allowed = duplicates.candidate_allowed & eligible
    medoids_filtered = int(np.count_nonzero(duplicates.candidate_allowed & ~eligible & (duplicates.group_ids >= 0)))
    max_outliers = math.floor(v.max_selected_outlier_fraction * budget)

    records = list(read_manifest(c.train_manifest))
    random_ids = [r.image_id for r in select_random(records, budget=budget, seed=c.seed)]
    d3_path = c.selection_dir / D3_MANIFEST
    d3_ids = [r.image_id for r in read_manifest(d3_path)]
    position = {int(image_id): i for i, image_id in enumerate(ids)}

    def coverage(selected):
        return evaluate_exact_gpu_coverage(
            embeddings=embeddings,
            source_image_ids=ids,
            selected_image_ids=selected,
            query_batch_size=1024,
        )

    random_cov = coverage(random_ids)
    d3_cov = coverage(d3_ids)
    floor = random_cov["mean_coverage"] + COVERAGE_RETENTION * (
        d3_cov["mean_coverage"] - random_cov["mean_coverage"]
    )
    d3_topiq = topiq[[position[i] for i in d3_ids]]
    random_topiq = topiq[[position[i] for i in random_ids]]

    runs: dict[float, dict] = {}
    selections: dict[float, list[int]] = {}

    def run(lam: float) -> dict:
        if lam in runs:
            return runs[lam]
        started = time.perf_counter()
        result = select_facility_location_numpy(
            image_ids=ids,
            reverse_indptr=graph.reverse_indptr,
            reverse_indices=graph.reverse_indices,
            reverse_similarities=graph.reverse_similarities,
            budget=budget,
            allowed_candidate_mask=allowed,
            duplicate_groups=duplicates.group_ids,
            outlier_mask=outliers.mask,
            max_outliers=max_outliers,
            # coverage/N + lam*quality/budget, rescaled by N for the coverage-sum selector.
            modular_gains=lam * (n / budget) * topiq,
        )
        selected = list(result.selected_image_ids)
        q = topiq[list(result.selected_indices)]
        row = {
            "lambda": lam,
            **{k: coverage(selected)[k] for k in ("mean_coverage", "p95_distance", "p99_distance")},
            "mean_topiq": float(q.mean()),
            "p10_topiq": float(np.quantile(q, 0.10)),
            "difference_from_random": len(set(selected) - set(random_ids)),
            "difference_from_d3": len(set(selected) - set(d3_ids)),
            "outlier_count": result.outlier_count,
            "seconds": round(time.perf_counter() - started, 2),
        }
        row["coverage_gain_retained"] = (row["mean_coverage"] - random_cov["mean_coverage"]) / (
            d3_cov["mean_coverage"] - random_cov["mean_coverage"]
        )
        row["meets_coverage_floor"] = row["mean_coverage"] >= floor
        runs[lam], selections[lam] = row, selected
        print({k: (round(x, 5) if isinstance(x, float) else x) for k, x in row.items()}, flush=True)
        return row

    for lam in LAMBDA_GRID:
        run(lam)
    passing = [lam for lam in LAMBDA_GRID if runs[lam]["meets_coverage_floor"]]
    if not passing:
        raise SystemExit("No lambda keeps 90% of D3's coverage gain; the filter alone costs too much")
    failing = [lam for lam in LAMBDA_GRID if not runs[lam]["meets_coverage_floor"] and lam > max(passing)]
    if failing:
        low, high = max(passing), min(failing)
        for _ in range(BISECTION_STEPS):
            mid = math.sqrt(low * high) if low > 0 else high / 2
            if run(mid)["meets_coverage_floor"]:
                low = mid
            else:
                high = mid
    eligible_runs = [r for r in runs.values() if r["meets_coverage_floor"]]
    chosen = max(eligible_runs, key=lambda r: (r["mean_topiq"], -r["lambda"]))
    lam = chosen["lambda"]
    selected = selections[lam]
    sel_topiq = topiq[[position[i] for i in selected]]

    gates = {
        "random_difference": enforce_random_difference(selected, random_ids, minimum_replacements=MIN_DIFFERENCE),
        "d3_difference": enforce_random_difference(selected, d3_ids, minimum_replacements=MIN_DIFFERENCE),
    }
    interval = bootstrap_difference(sel_topiq, d3_topiq, c.seed)
    gates["quality_gain_vs_d3"] = {
        "mean_difference": float(sel_topiq.mean() - d3_topiq.mean()),
        "bootstrap_95": interval,
        "passed": interval[0] > 0.0,
    }
    if not gates["quality_gain_vs_d3"]["passed"]:
        raise SystemExit(f"Quality gain over D3 is not distinguishable from zero: {interval}")
    if chosen["outlier_count"] > max_outliers:
        raise SystemExit("Outlier cap violated")

    population = Counter(cat for r in records for cat in r.category_ids)
    removed_ids = ids[topiq_removed | ~size_ok].tolist()
    report = {
        "algorithm_version": ALGORITHM_VERSION,
        "objective": "coverage(S)/N + lambda * sum topiq(S)/15000",
        "filter": {
            "min_short_side": MIN_SHORT_SIDE,
            "small_image_count": int((~size_ok).sum()),
            "topiq_fraction": TOPIQ_FILTER_FRACTION,
            "topiq_removed_count": cut_count,
            "topiq_threshold": topiq_threshold,
            "eligible_count": int(eligible.sum()),
            "duplicate_medoids_filtered": medoids_filtered,
            "removed_category_tvd_vs_population": tvd(category_mix(records, removed_ids), population),
            "removed_in_d3": len(set(removed_ids) & set(d3_ids)),
            "removed_in_random": len(set(removed_ids) & set(random_ids)),
        },
        "random": {**random_cov, "mean_topiq": float(random_topiq.mean())},
        "d3": {**d3_cov, "mean_topiq": float(d3_topiq.mean())},
        "coverage_floor": floor,
        "sweep": sorted(runs.values(), key=lambda r: r["lambda"]),
        "chosen_lambda": lam,
        "chosen": chosen,
        "gates": gates,
        "category_tvd_vs_d3": tvd(category_mix(records, selected), category_mix(records, d3_ids)),
        "category_tvd_vs_population": tvd(category_mix(records, selected), population),
        "d3_category_tvd_vs_population": tvd(category_mix(records, d3_ids), population),
        "inputs": {
            "embedding_metadata_sha256": sha256_file(ep),
            "graph_metadata_sha256": sha256_file(gp),
            "learned_quality_sha256": sha256_file(learned_path),
            "quality_scores_sha256": sha256_file(scores_path),
            "topiq_weights_sha256": learned_meta["models"]["topiq_nr"]["weights_sha256"],
            "d3_manifest_sha256": sha256_file(d3_path),
            "checkpoint": em["checkpoint"],
        },
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "config_sha256": sha256_file(root / "configs/data.toml"),
    }
    write_json_atomic(c.report_dir / "quality-calibration.json", report)

    lam_tag = f"{lam:.4g}".replace(".", "p")
    output = c.selection_dir / (
        f"quality_aware_k{NEIGHBOR_K}_dup0p99_topiq_lam{lam_tag}_seed{c.seed}_n{budget}.jsonl"
    )
    chosen_set = set(selected)
    count, digest = write_jsonl_atomic(output, (r.to_dict() for r in records if r.image_id in chosen_set))
    write_json_atomic(
        metadata_path(output),
        {
            "schema_version": 1,
            "kind": "coco_selection_manifest",
            "method": "quality-aware-diversity",
            "algorithm_version": ALGORITHM_VERSION,
            "seed": c.seed,
            "budget": budget,
            "record_count": count,
            "manifest_sha256": digest,
            "source_manifest": c.train_manifest.name,
            "source_manifest_sha256": sha256_file(c.train_manifest),
            "selected_ids_sha256": image_ids_sha256(sorted(chosen_set)),
            "calibration": report,
        },
    )
    print({"accepted_manifest": str(output.relative_to(root)), "lambda": lam, "gates": gates}, flush=True)


if __name__ == "__main__":
    main()
