"""Deterministic training-subset selection algorithms and provenance."""

from __future__ import annotations

from collections import Counter
import hashlib
import heapq
import math
from pathlib import Path
from typing import Callable, Iterable, Any

from vlm_data.category_balance import (
    DEFAULT_FREQUENCY_THRESHOLD,
    DEFAULT_MAX_BOOST,
    DEFAULT_MINIMUM_CHANGED_FRACTION,
    category_counts,
    compute_category_targets,
)
from vlm_data.io import sha256_file, write_json_atomic, write_jsonl_atomic
from vlm_data.manifest import ImageRecord, metadata_path


RANDOM_ALGORITHM_VERSION = "stable_hash_rank_v1"
CATEGORY_BALANCED_ALGORITHM_VERSION = "sqrt_frequency_baseline_repair_v1"


def _rank(seed: int, namespace: str, image_id: int) -> bytes:
    value = f"{namespace}:{seed}:{image_id}".encode("utf-8")
    return hashlib.blake2b(value, digest_size=16).digest()


def select_random(
    records: Iterable[ImageRecord], *, budget: int, seed: int
) -> list[ImageRecord]:
    available = list(records)
    if budget <= 0:
        raise ValueError("budget must be positive")
    if budget > len(available):
        raise ValueError(f"Cannot select {budget} records from only {len(available)}")
    selected = sorted(
        available,
        key=lambda item: (_rank(seed, "random-v1", item.image_id), item.image_id),
    )[:budget]
    return sorted(selected, key=lambda item: item.image_id)


def select_category_balanced(
    records: Iterable[ImageRecord],
    *,
    budget: int,
    seed: int,
    frequency_threshold: float = DEFAULT_FREQUENCY_THRESHOLD,
    max_boost: float = DEFAULT_MAX_BOOST,
) -> list[ImageRecord]:
    """Repair the random baseline until all boosted-category targets are met.

    Beginning from the paired random baseline isolates the rare-category
    intervention. A from-scratch deficit greedy pass strongly favors images with
    many annotated categories, which is a separate density-selection effect.
    """

    available = list(records)
    if budget <= 0:
        raise ValueError("budget must be positive")
    if budget > len(available):
        raise ValueError(f"Cannot select {budget} records from only {len(available)}")
    if len({record.image_id for record in available}) != len(available):
        raise ValueError("Category-balanced selection requires unique image IDs")

    counts = category_counts(available)
    targets = compute_category_targets(
        counts,
        total_images=len(available),
        budget=budget,
        frequency_threshold=frequency_threshold,
        max_boost=max_boost,
    )
    selected = select_random(available, budget=budget, seed=seed)
    selected_by_id = {record.image_id: record for record in selected}
    coverage = category_counts(selected)
    enforced_categories = {
        category_id for category_id, target in targets.items() if target.affected
    }
    if not enforced_categories:
        return selected

    def deficit_score(record: ImageRecord) -> float:
        return sum(
            max(
                0.0,
                (targets[category_id].target - coverage[category_id])
                / targets[category_id].target,
            )
            for category_id in record.category_ids
            if category_id in enforced_categories
        )

    # Candidate gains only decrease as boosted-category deficits are repaired.
    # Stored gains are therefore upper bounds, so lazy greedy avoids rescoring
    # every unselected record after each swap.
    heap: list[tuple[float, bytes, int, ImageRecord]] = [
        (
            -deficit_score(record),
            _rank(seed, "category-deficit-v1", record.image_id),
            record.image_id,
            record,
        )
        for record in available
        if record.image_id not in selected_by_id
    ]
    heapq.heapify(heap)
    blocked_candidates: list[tuple[bytes, int, ImageRecord]] = []

    while any(
        coverage[category_id] < targets[category_id].target
        for category_id in enforced_categories
    ):
        if not heap:
            raise RuntimeError("Category targets are not reachable from the available records")
        _stored_negative_score, tie_rank, image_id, candidate = heapq.heappop(heap)
        current_score = deficit_score(candidate)
        current_key = (-current_score, tie_rank, image_id)
        if heap and current_key > heap[0][:3]:
            heapq.heappush(heap, (-current_score, tie_rank, image_id, candidate))
            continue
        if current_score <= 0.0:
            raise RuntimeError("Category targets remain unmet but no candidate has positive gain")

        post_add_coverage = coverage.copy()
        post_add_coverage.update(candidate.category_ids)
        removable: list[tuple[int, bytes, int, ImageRecord]] = []
        for record in selected_by_id.values():
            if any(
                post_add_coverage[category_id] - 1 < targets[category_id].target
                for category_id in set(record.category_ids) & enforced_categories
            ):
                continue
            affected_membership = len(set(record.category_ids) & enforced_categories)
            removable.append(
                (
                    affected_membership,
                    _rank(seed, "category-swap-out-v1", record.image_id),
                    record.image_id,
                    record,
                )
            )
        if not removable:
            # A single-category candidate may have positive gain but be unable to
            # replace the only image currently protecting another target. Try a
            # multi-label candidate before declaring the target set infeasible.
            blocked_candidates.append((tie_rank, image_id, candidate))
            continue
        _membership, _remove_rank, removed_id, removed = min(removable)

        del selected_by_id[removed_id]
        selected_by_id[candidate.image_id] = candidate
        coverage = post_add_coverage
        coverage.subtract(removed.category_ids)
        heapq.heappush(
            heap,
            (
                -deficit_score(removed),
                _rank(seed, "category-deficit-v1", removed.image_id),
                removed.image_id,
                removed,
            ),
        )
        for blocked_rank, blocked_id, blocked in blocked_candidates:
            heapq.heappush(
                heap,
                (
                    -deficit_score(blocked),
                    blocked_rank,
                    blocked_id,
                    blocked,
                ),
            )
        blocked_candidates.clear()

    return sorted(selected_by_id.values(), key=lambda item: item.image_id)


SELECTION_METHODS: dict[
    str, tuple[str, Callable[..., list[ImageRecord]]]
] = {
    "random": (RANDOM_ALGORITHM_VERSION, select_random),
    "category-balanced": (CATEGORY_BALANCED_ALGORITHM_VERSION, select_category_balanced),
}


def write_selection(
    *,
    records: Iterable[ImageRecord],
    output_path: Path,
    source_manifest: Path,
    method: str,
    seed: int,
    budget: int,
    schema_version: int = 1,
    category_frequency_threshold: float = DEFAULT_FREQUENCY_THRESHOLD,
    category_max_boost: float = DEFAULT_MAX_BOOST,
    category_minimum_changed_fraction: float = DEFAULT_MINIMUM_CHANGED_FRACTION,
) -> dict[str, Any]:
    if method not in SELECTION_METHODS:
        raise ValueError(f"Unknown implemented selection method: {method}")
    algorithm_version, selector = SELECTION_METHODS[method]
    available = list(records)
    selector_options: dict[str, Any] = {}
    if method == "category-balanced":
        if not 0.0 <= category_minimum_changed_fraction <= 1.0:
            raise ValueError("category minimum changed fraction must be in [0, 1]")
        selector_options = {
            "frequency_threshold": category_frequency_threshold,
            "max_boost": category_max_boost,
        }
    selected = selector(available, budget=budget, seed=seed, **selector_options)
    baseline_comparison: dict[str, Any] | None = None
    if method == "category-balanced":
        baseline = select_random(available, budget=budget, seed=seed)
        selected_ids = {record.image_id for record in selected}
        baseline_ids = {record.image_id for record in baseline}
        changed_image_count = len(selected_ids - baseline_ids)
        required_changed_count = math.ceil(category_minimum_changed_fraction * budget)
        if changed_image_count < required_changed_count:
            raise ValueError(
                "Category-balanced selection changes "
                f"{changed_image_count} images, below the required "
                f"{required_changed_count} ({category_minimum_changed_fraction:.2%})"
            )
        baseline_comparison = {
            "baseline_algorithm_version": RANDOM_ALGORITHM_VERSION,
            "overlap_count": len(selected_ids & baseline_ids),
            "overlap_fraction": len(selected_ids & baseline_ids) / budget,
            "changed_image_count": changed_image_count,
            "changed_fraction": changed_image_count / budget,
            "minimum_changed_fraction": category_minimum_changed_fraction,
            "required_changed_count": required_changed_count,
            "minimum_changed_fraction_met": True,
        }
    row_count, manifest_sha256 = write_jsonl_atomic(
        output_path, (record.to_dict() for record in selected)
    )
    coverage: Counter[int] = Counter()
    for record in selected:
        coverage.update(record.category_ids)
    metadata: dict[str, Any] = {
        "schema_version": schema_version,
        "kind": "coco_selection_manifest",
        "method": method,
        "algorithm_version": algorithm_version,
        "seed": seed,
        "budget": budget,
        "record_count": row_count,
        "manifest_sha256": manifest_sha256,
        "source_manifest": source_manifest.name,
        "source_manifest_sha256": sha256_file(source_manifest),
        "category_coverage": {
            str(category_id): count for category_id, count in sorted(coverage.items())
        },
    }
    if method == "category-balanced":
        targets = compute_category_targets(
            category_counts(available),
            total_images=len(available),
            budget=budget,
            frequency_threshold=category_frequency_threshold,
            max_boost=category_max_boost,
        )
        metadata["method_parameters"] = {
            "frequency_threshold": category_frequency_threshold,
            "max_boost": category_max_boost,
            "minimum_changed_fraction": category_minimum_changed_fraction,
        }
        metadata["random_baseline_comparison"] = baseline_comparison
        metadata["category_targets"] = {
            str(category_id): {
                "source_count": target.source_count,
                "frequency": target.frequency,
                "natural_expected": target.natural_expected,
                "raw_boost": target.raw_boost,
                "effective_boost": target.effective_boost,
                "target": target.target,
                "enforced": target.affected,
                "selected_coverage": coverage[category_id],
                "shortfall": (
                    max(0, target.target - coverage[category_id]) if target.affected else 0
                ),
            }
            for category_id, target in targets.items()
        }
    write_json_atomic(metadata_path(output_path), metadata)
    return metadata
