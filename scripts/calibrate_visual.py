"""Run the approved train-only facility k sweep after the documented duplicate audit."""

import math
from collections import Counter
from pathlib import Path
import time

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
    recommend_neighbor_k,
    write_calibration_report,
)
from vlm_data.visual_embeddings import image_ids_sha256, load_embedding_artifact
from vlm_data.visual_neighbors import load_neighbor_artifact


def main():
    c = load_config()
    v = c.visual_diversity
    p = c.visual_artifact_dir
    ep = p / "openclip_vit_b16_datacomp_xl_s13b_b90k_train2017.meta.json"
    gp = p / "clip_neighbors_k400.meta.json"
    em, e, ids = load_embedding_artifact(ep, source_manifest=c.train_manifest)
    _, scores, neighbors = load_neighbor_artifact(gp, embedding_metadata_path=ep)
    positions = map_neighbor_ids_to_positions(ids, neighbors)
    duplicates = build_duplicate_groups(
        embeddings=e,
        source_image_ids=ids,
        neighbor_image_ids=neighbors,
        neighbor_similarities=scores,
        threshold=0.99,
    )
    density = compute_visual_density(ids, neighbors, scores)
    outliers = mark_low_density_outliers(ids, density)
    records = list(read_manifest(c.train_manifest))
    baseline = [
        r.image_id for r in select_random(records, budget=c.train_image_budget, seed=c.seed)
    ]
    baseline_metrics = evaluate_exact_gpu_coverage(
        embeddings=e, source_image_ids=ids, selected_image_ids=baseline, query_batch_size=1024
    )
    metrics, selections = {}, {}
    for k in v.neighbor_k_candidates:
        started = time.perf_counter()
        graph = derive_sparse_csr(positions, scores, k=k)
        result = select_facility_location_numpy(
            image_ids=ids,
            reverse_indptr=graph.reverse_indptr,
            reverse_indices=graph.reverse_indices,
            reverse_similarities=graph.reverse_similarities,
            budget=c.train_image_budget,
            allowed_candidate_mask=duplicates.candidate_allowed,
            duplicate_groups=duplicates.group_ids,
            outlier_mask=outliers.mask,
            max_outliers=math.floor(v.max_selected_outlier_fraction * c.train_image_budget),
        )
        selected = list(result.selected_image_ids)
        row = evaluate_exact_gpu_coverage(
            embeddings=e, source_image_ids=ids, selected_image_ids=selected, query_batch_size=1024
        )
        row.update(enforce_random_difference(selected, baseline))
        row.update(
            seconds=time.perf_counter() - started,
            outlier_count=result.outlier_count,
            gain_evaluations=result.gain_evaluations,
            sparse_coverage=result.objective_value,
            sparse_uncovered=sum(x == 0 for x in result.best_similarities) / len(ids),
        )
        selected_set = set(selected)
        selected_records = [r for r in records if r.image_id in selected_set]
        category_counts = Counter(cat for r in selected_records for cat in r.category_ids)
        row["category_coverage"] = dict(sorted(category_counts.items()))
        row["mean_distinct_categories"] = sum(len(r.category_ids) for r in selected_records) / len(
            selected
        )
        row["mean_object_count"] = sum(r.object_count for r in selected_records) / len(selected)
        selected_positions = [i for i, image_id in enumerate(ids) if int(image_id) in selected_set]
        row["duplicate_representatives_only"] = bool(
            duplicates.candidate_allowed[selected_positions].all()
        )
        row["hard_requirements_met"] = (
            row["mean_coverage"] > baseline_metrics["mean_coverage"]
            and row["mean_distance"] < baseline_metrics["mean_distance"]
            and row["p95_distance"] < baseline_metrics["p95_distance"]
            and row["duplicate_representatives_only"]
            and result.outlier_count <= 300
        )
        metrics[k], selections[k] = row, selected
        write_json_atomic(
            c.report_dir / f"visual-k{k}.json", {"k": k, "metrics": row, "selected_ids": selected}
        )
        print({"k": k, **row}, flush=True)
        del graph
    recommendation = recommend_neighbor_k(metrics)
    report = {
        "k_recommendation": recommendation,
        "metrics": metrics,
        "random": baseline_metrics,
        "duplicate_threshold": 0.99,
        "embedding_metadata_sha256": sha256_file(ep),
        "graph_metadata_sha256": sha256_file(gp),
        "checkpoint": em["checkpoint"],
        "duplicate_audit_sha256": sha256_file(c.project_root / "docs/visual-duplicate-audit.md"),
        "script_sha256": sha256_file(Path(__file__)),
        "config_sha256": sha256_file(c.project_root / "configs/data.toml"),
    }
    write_calibration_report(
        report=report,
        json_path=c.report_dir / "visual-calibration.json",
        markdown_path=c.report_dir / "visual-calibration.md",
    )
    k = recommendation["recommended_k"]
    chosen = set(selections[k])
    output = c.selection_dir / f"visual_diversity_k{k}_dup0p99_seed{c.seed}_n15000.jsonl"
    count, digest = write_jsonl_atomic(
        output, (r.to_dict() for r in records if r.image_id in chosen)
    )
    write_json_atomic(
        metadata_path(output),
        {
            "schema_version": 1,
            "kind": "coco_selection_manifest",
            "method": "visual-diversity",
            "algorithm_version": "sparse-facility-lazy-numpy-v1",
            "seed": c.seed,
            "budget": c.train_image_budget,
            "record_count": count,
            "manifest_sha256": digest,
            "source_manifest": c.train_manifest.name,
            "source_manifest_sha256": sha256_file(c.train_manifest),
            "selected_ids_sha256": image_ids_sha256(sorted(chosen)),
            "calibration": report,
        },
    )
    print({"accepted_manifest": str(output), "k": k}, flush=True)


if __name__ == "__main__":
    main()
