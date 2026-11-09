"""Produce explanatory category, language, dispersion, and cross-view diagnostics."""

import json
import csv
import io
import math
from pathlib import Path

from vlm_data.caption_diagnostics import diagnostic_report
from vlm_data.caption_semantic import load_embeddings
from vlm_data.config import load_config
from vlm_data.io import sha256_file, write_json_atomic, write_text_atomic
from vlm_data.manifest import read_manifest
from vlm_data.selection import select_random
from vlm_data.visual_analysis import evaluate_exact_gpu_coverage
from vlm_data.visual_embeddings import load_embedding_artifact
from vlm_data.visual_neighbors import load_neighbor_artifact


def main():
    import faiss
    import numpy as np

    c = load_config()
    p = c.project_root / "data/interim/caption_semantic"
    calibration = json.loads((c.report_dir / "caption-calibration.json").read_text())
    k = calibration["k_recommendation"]["recommended_k"]
    selected_path = (
        c.selection_dir / f"caption_semantic_k{k}_seed{c.seed}_n{c.train_image_budget}.jsonl"
    )
    selected = [r.image_id for r in read_manifest(selected_path)]
    records = sorted(read_manifest(c.train_manifest), key=lambda r: r.image_id)
    random = [r.image_id for r in select_random(records, budget=c.train_image_budget, seed=c.seed)]
    _, e, ids = load_embeddings(p / "mpnet.meta.json", source_manifest=c.train_manifest)
    _, scores, _ = load_neighbor_artifact(p / "neighbors.meta.json")
    dispersion = np.load(p / "mpnet.dispersion.npy", allow_pickle=False)
    report = diagnostic_report(
        records, e, dispersion, scores, selected, random, c.project_root / "data/raw/nltk"
    )
    _, visual, visual_ids = load_embedding_artifact(
        c.visual_artifact_dir / "openclip_vit_b16_datacomp_xl_s13b_b90k_train2017.meta.json",
        source_manifest=c.train_manifest,
    )
    if not np.array_equal(ids, visual_ids):
        raise ValueError("Cross-view IDs differ")
    report["visual_cross_view"] = {}
    report["selected_redundancy"] = {}
    for name, selection in [("caption", selected), ("random", random)]:
        report["visual_cross_view"][name] = evaluate_exact_gpu_coverage(
            embeddings=visual,
            source_image_ids=ids,
            selected_image_ids=selection,
            query_batch_size=1024,
        )
        positions = np.searchsorted(ids, np.array(selection))
        x = np.ascontiguousarray(e[positions])
        resources = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(resources, 0, faiss.IndexFlatIP(768))
        index.add(x)
        similarities, _ = index.search(x, 2)
        report["selected_redundancy"][name] = {
            "mean_nearest_other_similarity": float(similarities[:, 1].mean()),
            "fraction_nearest_other_at_least_0p99": float((similarities[:, 1] >= 0.99).mean()),
        }
    report["provenance"] = {
        "selection_sha256": sha256_file(selected_path),
        "source_sha256": sha256_file(c.train_manifest),
        "script_sha256": sha256_file(Path(__file__)),
        "nltk_tagger_files": {
            path.name: sha256_file(path)
            for path in (
                c.project_root / "data/raw/nltk/taggers/averaged_perceptron_tagger_eng"
            ).glob("*.json")
        },
    }
    write_json_atomic(c.report_dir / "caption-diagnostics.json", report)
    names = {
        str(x["id"]): x["name"] for x in json.loads(c.train_instances.read_text())["categories"]
    }
    table = io.StringIO()
    rows = [
        {"category_id": cat, "category_name": names[cat], **row}
        for cat, row in report["category_allocation"].items()
    ]
    writer = csv.DictWriter(table, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    write_text_atomic(c.report_dir / "caption-category-allocation.csv", table.getvalue())
    svg = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="900" height="550" viewBox="0 0 900 550">',
        '<rect width="900" height="550" fill="white"/>',
        '<text x="80" y="30" font-family="sans-serif" font-size="18">Caption allocation versus semantic coverage error</text>',
        '<path d="M80 60 V470 H850" fill="none" stroke="black"/>',
        '<text x="280" y="520" font-family="sans-serif">Selected images per category (log scale)</text>',
        '<text x="20" y="450" transform="rotate(-90 20 450)" font-family="sans-serif">Mean cosine coverage distance</text>',
    ]
    max_count = max(r["selected_count"] for r in rows)
    max_error = max(r["selected_mean_distance"] for r in rows) * 1.05
    for r in rows:
        x = 80 + 760 * math.log1p(r["selected_count"]) / math.log1p(max_count)
        y = 470 - 400 * r["selected_mean_distance"] / max_error
        svg.append(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="#2766a6"><title>{r["category_name"]}: {r["selected_count"]} images, distance {r["selected_mean_distance"]:.5f}</title></circle>'
        )
    for tick in (10, 100, 1000, 10000):
        x = 80 + 760 * math.log1p(tick) / math.log1p(max_count)
        if x <= 850:
            svg.append(f'<text x="{x:.1f}" y="490" font-family="sans-serif">{tick}</text>')
    for fraction in (0, 0.25, 0.5, 0.75, 1):
        y = 470 - 400 * fraction
        svg.append(
            f'<text x="35" y="{y:.1f}" font-family="sans-serif">{max_error * fraction:.2f}</text>'
        )
    svg.append("</svg>")
    write_text_atomic(c.report_dir / "caption-category-allocation.svg", "\n".join(svg))
    print(
        {
            "diagnostics": str(c.report_dir / "caption-diagnostics.json"),
            "groups": report["groups"],
            "cross_view": report["visual_cross_view"],
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
