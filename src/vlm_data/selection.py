"""Deterministic training-subset selection algorithms and provenance."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
from pathlib import Path
from typing import Callable, Iterable, Any

from vlm_data.io import sha256_file, write_json_atomic, write_jsonl_atomic
from vlm_data.manifest import ImageRecord, metadata_path


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
    selected = sorted(available, key=lambda item: (_rank(seed, "random-v1", item.image_id), item.image_id))[
        :budget
    ]
    return sorted(selected, key=lambda item: item.image_id)


def select_category_balanced(
    records: Iterable[ImageRecord], *, budget: int, seed: int
) -> list[ImageRecord]:
    """Greedily raise the least-covered category, then fill deterministically.

    This is a transparent first implementation of the proposal's otherwise
    unspecified balancing objective. The algorithm version is recorded in the
    selection metadata so a later scientific-method decision can replace it.
    """

    available = list(records)
    if budget <= 0:
        raise ValueError("budget must be positive")
    if budget > len(available):
        raise ValueError(f"Cannot select {budget} records from only {len(available)}")

    by_category: dict[int, list[ImageRecord]] = defaultdict(list)
    for record in available:
        for category_id in record.category_ids:
            by_category[category_id].append(record)
    if not by_category:
        raise ValueError("Category-balanced selection requires instance category annotations")

    for category_id in by_category:
        by_category[category_id].sort(
            key=lambda item: (_rank(seed, f"category-{category_id}-v1", item.image_id), item.image_id)
        )

    pointers: Counter[int] = Counter()
    selected_ids: set[int] = set()
    selected: list[ImageRecord] = []
    coverage: Counter[int] = Counter()

    while len(selected) < budget:
        chosen: ImageRecord | None = None
        for category_id in sorted(by_category, key=lambda item: (coverage[item], item)):
            candidates = by_category[category_id]
            pointer = pointers[category_id]
            while pointer < len(candidates) and candidates[pointer].image_id in selected_ids:
                pointer += 1
            pointers[category_id] = pointer
            if pointer < len(candidates):
                chosen = candidates[pointer]
                pointers[category_id] += 1
                break
        if chosen is None:
            break
        selected.append(chosen)
        selected_ids.add(chosen.image_id)
        coverage.update(chosen.category_ids)

    if len(selected) < budget:
        unselected = (record for record in available if record.image_id not in selected_ids)
        fill = select_random(unselected, budget=budget - len(selected), seed=seed)
        selected.extend(fill)

    return sorted(selected, key=lambda item: item.image_id)


SELECTION_METHODS: dict[
    str, tuple[str, Callable[..., list[ImageRecord]]]
] = {
    "random": ("stable_hash_rank_v1", select_random),
    "category-balanced": ("least_covered_round_robin_v1", select_category_balanced),
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
) -> dict[str, Any]:
    if method not in SELECTION_METHODS:
        raise ValueError(f"Unknown implemented selection method: {method}")
    algorithm_version, selector = SELECTION_METHODS[method]
    selected = selector(records, budget=budget, seed=seed)
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
    write_json_atomic(metadata_path(output_path), metadata)
    return metadata
