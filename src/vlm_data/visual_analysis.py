"""Train-only calibration and audit helpers for visual-diversity selection.

NumPy and FAISS remain optional at package-import time.  Runtime functions fail
with a direct dependency message when their numerical backend is unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from vlm_data.io import write_json_atomic, write_text_atomic


@dataclass(frozen=True)
class SparseCSR:
    """Forward and reverse CSR arrays in the canonical source-row coordinates."""

    forward_indptr: Any
    forward_indices: Any
    forward_similarities: Any
    reverse_indptr: Any
    reverse_indices: Any
    reverse_similarities: Any


@dataclass(frozen=True)
class OutlierClassification:
    """Deterministic low-density classification."""

    mask: Any
    image_ids: tuple[int, ...]
    count: int
    fraction: float
    cutoff_density: float


@dataclass(frozen=True)
class DuplicateGrouping:
    """Representative-centered near-duplicate groups and selection constraints."""

    threshold: float
    group_ids: Any
    candidate_allowed: Any
    groups: tuple[dict[str, Any], ...]
    audit_pairs: tuple[dict[str, Any], ...]


def _numpy():
    try:
        import numpy as np
    except ImportError as error:  # pragma: no cover - environment-dependent.
        raise RuntimeError("Visual analysis requires the optional numpy package") from error
    return np


def map_neighbor_ids_to_positions(
    source_image_ids: Sequence[int], neighbor_image_ids: Any
):
    """Map COCO neighbor IDs to canonical row positions, failing on any drift."""

    np = _numpy()
    source = np.asarray(source_image_ids, dtype=np.int64)
    neighbors = np.asarray(neighbor_image_ids)
    if source.ndim != 1 or source.size == 0:
        raise ValueError("source_image_ids must be a nonempty one-dimensional array")
    if neighbors.ndim != 2:
        raise ValueError("neighbor_image_ids must be a two-dimensional matrix")
    if neighbors.shape[0] != source.shape[0]:
        raise ValueError("Neighbor row count does not match source image-ID count")
    if not bool(np.all(source[1:] > source[:-1])):
        raise ValueError("Source image IDs must be unique and strictly increasing")
    try:
        parsed_neighbors = neighbors.astype(np.int64, copy=False)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("Neighbor image IDs must be integers") from error
    if not bool(np.array_equal(parsed_neighbors, neighbors)):
        raise ValueError("Neighbor image IDs must be exact integers")
    positions = np.searchsorted(source, parsed_neighbors)
    safe_positions = np.minimum(positions, len(source) - 1)
    known = (positions < len(source)) & (source[safe_positions] == parsed_neighbors)
    if not bool(np.all(known)):
        unknown = np.unique(parsed_neighbors[~known])[:5]
        preview = ", ".join(str(int(value)) for value in unknown)
        raise ValueError(f"Neighbor graph contains unknown image IDs: {preview}")
    return positions.astype(np.int64, copy=False)


def derive_sparse_csr(
    neighbor_positions: Any,
    neighbor_similarities: Any,
    *,
    k: int,
) -> SparseCSR:
    """Derive forward and reverse CSR views from a leading-column k graph."""

    np = _numpy()
    positions = np.asarray(neighbor_positions)
    similarities = np.asarray(neighbor_similarities)
    if positions.ndim != 2 or similarities.ndim != 2 or positions.shape != similarities.shape:
        raise ValueError("Neighbor positions and similarities must be equal-shape matrices")
    row_count, max_k = positions.shape
    if row_count == 0 or not 1 <= k <= max_k:
        raise ValueError(f"k must be between 1 and {max_k}")
    selected_positions = positions[:, :k].astype(np.int64, copy=False)
    selected_similarities = similarities[:, :k].astype(np.float32, copy=False)
    if bool(np.any(selected_positions < 0)) or bool(np.any(selected_positions >= row_count)):
        raise ValueError("Neighbor position is outside the canonical row range")
    if not bool(np.isfinite(selected_similarities).all()):
        raise ValueError("Neighbor similarities must be finite")
    for start in range(0, row_count, 4096):
        ordered = np.sort(selected_positions[start : start + 4096], axis=1)
        if k > 1 and bool(np.any(ordered[:, 1:] == ordered[:, :-1])):
            raise ValueError("A forward neighbor row contains duplicate candidates")

    forward_indptr = np.arange(0, row_count * k + 1, k, dtype=np.int64)
    forward_indices = np.ascontiguousarray(selected_positions.reshape(-1), dtype=np.int64)
    forward_similarities = np.ascontiguousarray(
        np.maximum(selected_similarities.reshape(-1), 0.0), dtype=np.float32
    )
    demands = np.repeat(np.arange(row_count, dtype=np.int64), k)
    reverse_order = np.argsort(forward_indices, kind="stable")
    counts = np.bincount(forward_indices, minlength=row_count)
    reverse_indptr = np.empty(row_count + 1, dtype=np.int64)
    reverse_indptr[0] = 0
    np.cumsum(counts, out=reverse_indptr[1:])
    return SparseCSR(
        forward_indptr=forward_indptr,
        forward_indices=forward_indices,
        forward_similarities=forward_similarities,
        reverse_indptr=reverse_indptr,
        reverse_indices=np.ascontiguousarray(demands[reverse_order], dtype=np.int64),
        reverse_similarities=np.ascontiguousarray(
            forward_similarities[reverse_order], dtype=np.float32
        ),
    )


def compute_visual_density(
    source_image_ids: Sequence[int],
    neighbor_image_ids: Any,
    neighbor_similarities: Any,
    *,
    nonself_count: int = 10,
):
    """Mean cosine similarity to each image's first ``nonself_count`` neighbors."""

    np = _numpy()
    source = np.asarray(source_image_ids, dtype=np.int64)
    neighbor_ids = np.asarray(neighbor_image_ids)
    similarities = np.asarray(neighbor_similarities, dtype=np.float32)
    if nonself_count <= 0:
        raise ValueError("nonself_count must be positive")
    expected_shape = neighbor_ids.shape
    if neighbor_ids.ndim != 2 or similarities.shape != expected_shape:
        raise ValueError("Neighbor IDs and similarities must be equal-shape matrices")
    if expected_shape[0] != len(source) or expected_shape[1] < nonself_count + 1:
        raise ValueError("Neighbor graph does not contain enough non-self columns")
    if not bool(np.array_equal(neighbor_ids[:, 0], source)):
        raise ValueError("Neighbor column zero must be the canonical source image itself")
    if bool(np.any(neighbor_ids[:, 1:] == source[:, None])):
        raise ValueError("Self appears more than once in a neighbor row")
    values = similarities[:, 1 : nonself_count + 1]
    if not bool(np.isfinite(values).all()):
        raise ValueError("Density similarities must be finite")
    return np.mean(values, axis=1, dtype=np.float64).astype(np.float32)


def mark_low_density_outliers(
    image_ids: Sequence[int],
    density: Any,
    *,
    fraction: float = 0.01,
) -> OutlierClassification:
    """Mark exactly ``ceil(fraction * n)`` lowest-density rows deterministically."""

    np = _numpy()
    ids = np.asarray(image_ids, dtype=np.int64)
    values = np.asarray(density, dtype=np.float64)
    if ids.ndim != 1 or values.shape != ids.shape or ids.size == 0:
        raise ValueError("Image IDs and density must be equal nonempty vectors")
    if len(np.unique(ids)) != len(ids):
        raise ValueError("Image IDs must be unique")
    if not bool(np.isfinite(values).all()):
        raise ValueError("Density must contain only finite values")
    if not 0.0 < fraction <= 1.0:
        raise ValueError("Outlier fraction must be in (0, 1]")
    count = math.ceil(fraction * len(ids))
    order = np.lexsort((ids, values))
    chosen = order[:count]
    mask = np.zeros(len(ids), dtype=np.bool_)
    mask[chosen] = True
    chosen_ids = tuple(sorted(int(value) for value in ids[chosen]))
    return OutlierClassification(
        mask=mask,
        image_ids=chosen_ids,
        count=count,
        fraction=float(fraction),
        cutoff_density=float(values[chosen].max()),
    )


def build_duplicate_groups(
    *,
    embeddings: Any,
    source_image_ids: Sequence[int],
    neighbor_image_ids: Any,
    neighbor_similarities: Any,
    threshold: float,
) -> DuplicateGrouping:
    """Build direct, representative-centered duplicate groups.

    Centers are processed by descending threshold-neighbor count and then COCO
    image ID.  A group contains only currently unassigned direct neighbors of
    its center, so an A-B-C-D similarity chain cannot collapse transitively.
    """

    np = _numpy()
    if not -1.0 <= threshold <= 1.0:
        raise ValueError("Duplicate threshold must be in [-1, 1]")
    ids = np.asarray(source_image_ids, dtype=np.int64)
    vectors = np.asarray(embeddings, dtype=np.float32)
    neighbor_ids = np.asarray(neighbor_image_ids)
    similarities = np.asarray(neighbor_similarities, dtype=np.float32)
    if ids.ndim != 1 or not bool(np.all(ids[1:] > ids[:-1])):
        raise ValueError("Source image IDs must be strictly increasing")
    if vectors.ndim != 2 or vectors.shape[0] != len(ids):
        raise ValueError("Embedding rows must align with source image IDs")
    if neighbor_ids.shape != similarities.shape or neighbor_ids.shape[0] != len(ids):
        raise ValueError("Neighbor arrays must align with source image IDs")
    if not bool(np.isfinite(vectors).all()) or not bool(np.isfinite(similarities).all()):
        raise ValueError("Duplicate analysis inputs must be finite")
    neighbor_positions = map_neighbor_ids_to_positions(ids, neighbor_ids)

    adjacency: list[set[int]] = [set() for _ in range(len(ids))]
    qualifying_rows, qualifying_columns = np.nonzero(similarities >= threshold)
    for row, column in zip(qualifying_rows.tolist(), qualifying_columns.tolist(), strict=True):
        neighbor = int(neighbor_positions[row, column])
        if neighbor == row:
            continue
        adjacency[row].add(neighbor)
        adjacency[neighbor].add(row)

    center_order = sorted(
        (index for index, neighbors in enumerate(adjacency) if neighbors),
        key=lambda index: (-len(adjacency[index]), int(ids[index])),
    )
    group_ids = np.full(len(ids), -1, dtype=np.int64)
    candidate_allowed = np.ones(len(ids), dtype=np.bool_)
    assigned = np.zeros(len(ids), dtype=np.bool_)
    groups: list[dict[str, Any]] = []
    audit_pairs: list[dict[str, Any]] = []
    for center in center_order:
        if bool(assigned[center]):
            continue
        members = [center] + sorted(
            (neighbor for neighbor in adjacency[center] if not bool(assigned[neighbor])),
            key=lambda index: int(ids[index]),
        )
        if len(members) < 2:
            continue
        member_array = np.asarray(members, dtype=np.int64)
        member_vectors = vectors[member_array]
        dense_similarities = member_vectors @ member_vectors.T
        medoid_scores = np.mean(dense_similarities, axis=1, dtype=np.float64)
        member_ids = ids[member_array]
        medoid_order = np.lexsort((member_ids, -medoid_scores))
        medoid_local = int(medoid_order[0])
        medoid = int(member_array[medoid_local])
        group_id = len(groups)
        group_ids[member_array] = group_id
        assigned[member_array] = True
        candidate_allowed[member_array] = False
        candidate_allowed[medoid] = True
        group_record = {
            "group_id": group_id,
            "center_image_id": int(ids[center]),
            "medoid_image_id": int(ids[medoid]),
            "member_image_ids": [int(value) for value in sorted(member_ids.tolist())],
            "member_count": len(members),
            "medoid_mean_similarity": float(medoid_scores[medoid_local]),
        }
        groups.append(group_record)
        for member in members:
            if member == center:
                continue
            audit_pairs.append(
                {
                    "group_id": group_id,
                    "center_image_id": int(ids[center]),
                    "member_image_id": int(ids[member]),
                    "cosine_similarity": float(vectors[center] @ vectors[member]),
                    "member_is_medoid": member == medoid,
                }
            )
    return DuplicateGrouping(
        threshold=float(threshold),
        group_ids=group_ids,
        candidate_allowed=candidate_allowed,
        groups=tuple(groups),
        audit_pairs=tuple(audit_pairs),
    )


def evaluate_exact_gpu_coverage(
    *,
    embeddings: Any,
    source_image_ids: Sequence[int],
    selected_image_ids: Iterable[int],
    query_batch_size: int,
    gpu_device: int = 0,
) -> dict[str, Any]:
    """Evaluate exact nearest-selected cosine coverage with GPU IndexFlatIP."""

    np = _numpy()
    try:
        import faiss
    except ImportError as error:  # pragma: no cover - environment-dependent.
        raise RuntimeError("Exact coverage evaluation requires faiss-gpu") from error
    vectors = np.asarray(embeddings, dtype=np.float32)
    ids = np.asarray(source_image_ids, dtype=np.int64)
    selected = np.asarray(list(selected_image_ids), dtype=np.int64)
    if vectors.ndim != 2 or vectors.shape[0] != len(ids):
        raise ValueError("Embedding rows must align with source image IDs")
    if not bool(np.all(ids[1:] > ids[:-1])):
        raise ValueError("Source image IDs must be strictly increasing")
    if selected.size == 0 or len(np.unique(selected)) != len(selected):
        raise ValueError("Selected image IDs must be nonempty and unique")
    if query_batch_size <= 0:
        raise ValueError("query_batch_size must be positive")
    selected_positions = np.searchsorted(ids, selected)
    safe = np.minimum(selected_positions, len(ids) - 1)
    if not bool(np.all((selected_positions < len(ids)) & (ids[safe] == selected))):
        raise ValueError("Selected set contains an unknown image ID")
    if not hasattr(faiss, "StandardGpuResources") or not hasattr(faiss, "index_cpu_to_gpu"):
        raise RuntimeError("Installed FAISS build does not provide GPU IndexFlat support")
    if not 0 <= gpu_device < int(faiss.get_num_gpus()):
        raise RuntimeError(f"FAISS GPU device {gpu_device} is unavailable")

    contiguous = np.ascontiguousarray(vectors, dtype=np.float32)
    selected_vectors = np.ascontiguousarray(contiguous[selected_positions], dtype=np.float32)
    resources = faiss.StandardGpuResources()
    index = faiss.index_cpu_to_gpu(
        resources, gpu_device, faiss.IndexFlatIP(contiguous.shape[1])
    )
    index.add(selected_vectors)
    best = np.empty(len(ids), dtype=np.float32)
    for start in range(0, len(ids), query_batch_size):
        end = min(len(ids), start + query_batch_size)
        similarities, _ = index.search(contiguous[start:end], 1)
        best[start:end] = similarities[:, 0]
    best = np.clip(best, -1.0, 1.0)
    distances = 1.0 - best
    return {
        "selected_count": int(len(selected)),
        "mean_coverage": float(np.mean(best, dtype=np.float64)),
        "facility_coverage_sum": float(np.sum(np.maximum(best, 0.0), dtype=np.float64)),
        "mean_distance": float(np.mean(distances, dtype=np.float64)),
        "p50_distance": float(np.percentile(distances, 50)),
        "p95_distance": float(np.percentile(distances, 95)),
        "p99_distance": float(np.percentile(distances, 99)),
    }


def enforce_random_difference(
    selected_image_ids: Iterable[int],
    random_image_ids: Iterable[int],
    *,
    expected_count: int = 15_000,
    minimum_replacements: int = 750,
) -> dict[str, Any]:
    """Validate equal-size selections and enforce the 5%-style replacement gate."""

    selected = [int(value) for value in selected_image_ids]
    random = [int(value) for value in random_image_ids]
    if expected_count <= 0 or minimum_replacements < 0:
        raise ValueError("Expected count must be positive and replacements nonnegative")
    if len(selected) != expected_count or len(set(selected)) != expected_count:
        raise ValueError(f"Selected set must contain {expected_count} unique image IDs")
    if len(random) != expected_count or len(set(random)) != expected_count:
        raise ValueError(f"Random set must contain {expected_count} unique image IDs")
    selected_set = set(selected)
    random_set = set(random)
    replacements = len(selected_set - random_set)
    result = {
        "selected_count": expected_count,
        "overlap_count": len(selected_set & random_set),
        "replacement_count": replacements,
        "replacement_fraction": replacements / expected_count,
        "minimum_replacements": minimum_replacements,
        "minimum_difference_met": replacements >= minimum_replacements,
    }
    if replacements < minimum_replacements:
        raise ValueError(
            f"Selection replaces {replacements} random images; "
            f"at least {minimum_replacements} are required"
        )
    return result


def recommend_neighbor_k(
    metrics_by_k: Mapping[int, Mapping[str, Any]],
    *,
    mean_coverage_relative_tolerance: float = 0.001,
    p95_distance_tolerance: float = 0.005,
    p99_distance_tolerance: float = 0.010,
) -> dict[str, Any]:
    """Choose the smallest accepted k inside all configured best-metric tolerances."""

    if not metrics_by_k:
        raise ValueError("At least one k candidate is required")
    if min(
        mean_coverage_relative_tolerance,
        p95_distance_tolerance,
        p99_distance_tolerance,
    ) < 0:
        raise ValueError("k recommendation tolerances must be nonnegative")
    rows: list[dict[str, Any]] = []
    for raw_k, metrics in metrics_by_k.items():
        k = int(raw_k)
        if k <= 0:
            raise ValueError("k candidates must be positive")
        try:
            coverage = float(metrics["mean_coverage"])
            p95 = float(metrics["p95_distance"])
            p99 = float(metrics["p99_distance"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Candidate k={k} has incomplete metrics") from error
        if not all(math.isfinite(value) for value in (coverage, p95, p99)):
            raise ValueError(f"Candidate k={k} has non-finite metrics")
        rows.append(
            {
                "k": k,
                "mean_coverage": coverage,
                "p95_distance": p95,
                "p99_distance": p99,
                "hard_requirements_met": bool(metrics.get("hard_requirements_met", True)),
            }
        )
    accepted_rows = [row for row in rows if row["hard_requirements_met"]]
    if not accepted_rows:
        raise ValueError("No k candidate passed the hard requirements")
    best_coverage = max(row["mean_coverage"] for row in accepted_rows)
    best_p95 = min(row["p95_distance"] for row in accepted_rows)
    best_p99 = min(row["p99_distance"] for row in accepted_rows)
    coverage_floor = best_coverage * (1.0 - mean_coverage_relative_tolerance)
    for row in rows:
        row["within_coverage_tolerance"] = row["mean_coverage"] >= coverage_floor
        row["within_p95_tolerance"] = (
            row["p95_distance"] <= best_p95 + p95_distance_tolerance
        )
        row["within_p99_tolerance"] = (
            row["p99_distance"] <= best_p99 + p99_distance_tolerance
        )
        row["eligible"] = bool(
            row["hard_requirements_met"]
            and row["within_coverage_tolerance"]
            and row["within_p95_tolerance"]
            and row["within_p99_tolerance"]
        )
    eligible = sorted(row["k"] for row in rows if row["eligible"])
    if not eligible:
        raise ValueError("No k candidate falls within every configured tolerance")
    return {
        "recommended_k": eligible[0],
        "best_mean_coverage": best_coverage,
        "best_p95_distance": best_p95,
        "best_p99_distance": best_p99,
        "coverage_floor": coverage_floor,
        "mean_coverage_relative_tolerance": mean_coverage_relative_tolerance,
        "p95_distance_tolerance": p95_distance_tolerance,
        "p99_distance_tolerance": p99_distance_tolerance,
        "candidates": sorted(rows, key=lambda row: row["k"]),
    }


def render_calibration_markdown(report: Mapping[str, Any]) -> str:
    """Render the core calibration decision without Markdown math dependencies."""

    recommendation = report.get("k_recommendation", report)
    lines = [
        "# Visual-diversity calibration",
        "",
        f"- Recommended neighbor count: `{recommendation['recommended_k']}`",
        f"- Best mean exact coverage: `{recommendation['best_mean_coverage']:.8f}`",
        f"- Best p95 cosine distance: `{recommendation['best_p95_distance']:.8f}`",
        f"- Best p99 cosine distance: `{recommendation['best_p99_distance']:.8f}`",
        "",
        "| k | Mean coverage | p95 distance | p99 distance | Hard pass | Eligible |",
        "|---:|---:|---:|---:|:---:|:---:|",
    ]
    for row in recommendation["candidates"]:
        lines.append(
            f"| {row['k']} | {row['mean_coverage']:.8f} | "
            f"{row['p95_distance']:.8f} | {row['p99_distance']:.8f} | "
            f"{'yes' if row['hard_requirements_met'] else 'no'} | "
            f"{'yes' if row['eligible'] else 'no'} |"
        )
    return "\n".join(lines) + "\n"


def write_calibration_report(
    *,
    report: Mapping[str, Any],
    json_path: Path,
    markdown_path: Path,
) -> None:
    """Atomically persist machine-readable and human-readable calibration reports."""

    # Round-trip rejects non-JSON report values before either destination changes.
    normalized = json.loads(json.dumps(report, sort_keys=True, allow_nan=False))
    markdown = render_calibration_markdown(normalized)
    write_json_atomic(Path(json_path), normalized)
    write_text_atomic(Path(markdown_path), markdown)
