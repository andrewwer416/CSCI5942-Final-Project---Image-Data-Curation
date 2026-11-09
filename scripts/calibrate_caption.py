"""Train-only caption facility sweep; accepts only candidates beating paired random."""

from collections import Counter
import math
from pathlib import Path
import time

from vlm_data.caption_semantic import generate_embeddings, load_embeddings
from vlm_data.config import load_config
from vlm_data.facility_location import select_facility_location_numpy
from vlm_data.io import sha256_file, write_json_atomic, write_jsonl_atomic
from vlm_data.manifest import metadata_path, read_manifest
from vlm_data.selection import select_random
from vlm_data.visual_analysis import (
    derive_sparse_csr,
    enforce_random_difference,
    evaluate_exact_gpu_coverage,
    map_neighbor_ids_to_positions,
    recommend_neighbor_k,
    write_calibration_report,
)
from vlm_data.visual_embeddings import image_ids_sha256
from vlm_data.visual_neighbors import build_exact_gpu_neighbor_artifact, load_neighbor_artifact


def main():
    c = load_config()
    p = c.project_root / "data/interim/caption_semantic"
    generate_embeddings(c.train_manifest, p, c.model_cache_dir)
    ep, gp = p / "mpnet.meta.json", p / "neighbors.meta.json"
    em, e, ids = load_embeddings(ep, source_manifest=c.train_manifest)
    if not gp.exists():
        build_exact_gpu_neighbor_artifact(
            embedding_metadata_path=ep,
            source_manifest=c.train_manifest,
            similarities_path=p / "neighbors.npy",
            project_root=c.project_root,
            max_k=400,
            query_batch_size=1024,
            embedding_loader=load_embeddings,
        )
    _, scores, neighbors = load_neighbor_artifact(gp, embedding_metadata_path=ep)
    positions = map_neighbor_ids_to_positions(ids, neighbors)
    records = sorted(read_manifest(c.train_manifest), key=lambda r: r.image_id)
    baseline = [
        r.image_id for r in select_random(records, budget=c.train_image_budget, seed=c.seed)
    ]
    baseline_metrics = evaluate_exact_gpu_coverage(
        embeddings=e, source_image_ids=ids, selected_image_ids=baseline, query_batch_size=1024
    )
    metrics, selections = {}, {}
    for k in (50, 100, 200, 400):
        started = time.perf_counter()
        graph = derive_sparse_csr(positions, scores, k=k)
        result = select_facility_location_numpy(
            image_ids=ids,
            reverse_indptr=graph.reverse_indptr,
            reverse_indices=graph.reverse_indices,
            reverse_similarities=graph.reverse_similarities,
            budget=c.train_image_budget,
        )
        selected = list(result.selected_image_ids)
        row = evaluate_exact_gpu_coverage(
            embeddings=e, source_image_ids=ids, selected_image_ids=selected, query_batch_size=1024
        )
        row.update(
            enforce_random_difference(
                selected,
                baseline,
                expected_count=c.train_image_budget,
                minimum_replacements=math.ceil(0.05 * c.train_image_budget),
            )
        )
        row.update(
            seconds=time.perf_counter() - started,
            gain_evaluations=result.gain_evaluations,
            sparse_coverage=result.objective_value,
            sparse_uncovered=sum(x == 0 for x in result.best_similarities) / len(ids),
        )
        row["hard_requirements_met"] = (
            row["mean_coverage"] > baseline_metrics["mean_coverage"]
            and row["mean_distance"] < baseline_metrics["mean_distance"]
            and row["p95_distance"] < baseline_metrics["p95_distance"]
        )
        metrics[k], selections[k] = row, selected
        write_json_atomic(
            c.report_dir / f"caption-k{k}.json", {"k": k, "metrics": row, "selected_ids": selected}
        )
        print({"k": k, **row}, flush=True)
        del graph
    recommendation = recommend_neighbor_k(metrics)
    report = {
        "title": "Caption-semantic diversity calibration",
        "k_recommendation": recommendation,
        "metrics": metrics,
        "random": baseline_metrics,
        "checkpoint": em["checkpoint"],
        "embedding_metadata_sha256": sha256_file(ep),
        "graph_metadata_sha256": sha256_file(gp),
        "script_sha256": sha256_file(Path(__file__)),
        "environment_lock_sha256": sha256_file(c.project_root / "pixi.lock"),
        "implementation_sha256": {
            name: sha256_file(c.project_root / "src/vlm_data" / name)
            for name in (
                "caption_semantic.py",
                "visual_neighbors.py",
                "visual_analysis.py",
                "facility_location.py",
            )
        },
        "seed": c.seed,
        "duplicate_policy": "marginal gain only; no exclusion",
        "outlier_policy": "diagnostic only; no quota",
    }
    write_calibration_report(
        report=report,
        json_path=c.report_dir / "caption-calibration.json",
        markdown_path=c.report_dir / "caption-calibration.md",
    )
    k = recommendation["recommended_k"]
    chosen = set(selections[k])
    output = c.selection_dir / f"caption_semantic_k{k}_seed{c.seed}_n{c.train_image_budget}.jsonl"
    count, digest = write_jsonl_atomic(
        output, (r.to_dict() for r in records if r.image_id in chosen)
    )
    coverage = Counter(cat for r in records if r.image_id in chosen for cat in r.category_ids)
    write_json_atomic(
        metadata_path(output),
        {
            "schema_version": 1,
            "kind": "coco_selection_manifest",
            "method": "caption-semantic",
            "algorithm_version": "mpnet-mean-sparse-facility-lazy-v1",
            "seed": c.seed,
            "budget": c.train_image_budget,
            "record_count": count,
            "manifest_sha256": digest,
            "source_manifest": c.train_manifest.name,
            "source_manifest_sha256": sha256_file(c.train_manifest),
            "selected_ids_sha256": image_ids_sha256(sorted(chosen)),
            "category_coverage": dict(sorted(coverage.items())),
            "calibration": report,
        },
    )
    print({"accepted_manifest": str(output), "k": k}, flush=True)


if __name__ == "__main__":
    main()
