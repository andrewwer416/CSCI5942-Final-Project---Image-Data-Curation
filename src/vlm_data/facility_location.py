"""Dependency-free sparse facility-location selection.

The production visual-diversity pipeline stores a directed sparse neighbor graph:
each forward row is a demand image and names the candidate images allowed to
represent it.  The reverse rows contain the same edges grouped by candidate,
which makes marginal-gain updates efficient.

This module deliberately depends only on the Python standard library.  Array
implementations may pass any integer/float sequences (including memory-mapped
arrays) without this module importing their owning package.
"""

from __future__ import annotations

import heapq
import math
import operator
from collections.abc import Collection, Hashable, Mapping, Sequence
from dataclasses import dataclass


def _integer(value: object, *, label: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{label} must be an integer, not a boolean")
    try:
        return operator.index(value)  # type: ignore[arg-type]
    except TypeError as error:
        raise TypeError(f"{label} must be an integer") from error


def _validate_csr(
    *,
    name: str,
    row_count: int,
    column_count: int,
    indptr: Sequence[int],
    indices: Sequence[int],
    similarities: Sequence[float],
) -> None:
    if len(indptr) != row_count + 1:
        raise ValueError(f"{name} indptr has length {len(indptr)}; expected {row_count + 1}")
    if len(indices) != len(similarities):
        raise ValueError(f"{name} indices and similarities must have equal length")
    if _integer(indptr[0], label=f"{name} indptr[0]") != 0:
        raise ValueError(f"{name} indptr must start at zero")

    previous = 0
    for row in range(row_count):
        start = _integer(indptr[row], label=f"{name} indptr[{row}]")
        stop = _integer(indptr[row + 1], label=f"{name} indptr[{row + 1}]")
        if start != previous or stop < start or stop > len(indices):
            raise ValueError(f"{name} indptr is not a valid nondecreasing CSR pointer array")
        seen: set[int] = set()
        for offset in range(start, stop):
            column = _integer(indices[offset], label=f"{name} indices[{offset}]")
            if not 0 <= column < column_count:
                raise ValueError(
                    f"{name} index {column} at offset {offset} is outside [0, {column_count})"
                )
            if column in seen:
                raise ValueError(f"{name} row {row} contains duplicate index {column}")
            seen.add(column)
            similarity = float(similarities[offset])
            if not math.isfinite(similarity) or similarity < 0.0:
                raise ValueError(
                    f"{name} similarity at offset {offset} must be finite and nonnegative"
                )
        previous = stop
    if previous != len(indices):
        raise ValueError(f"{name} indptr ends at {previous}; expected {len(indices)} entries")


@dataclass(frozen=True)
class SparseFacilityGraph:
    """Forward and reverse CSR views of one sparse facility graph.

    Candidate and demand indices share the same ``image_ids`` coordinate system.
    ``forward_indices`` therefore contain candidate indices, while
    ``reverse_indices`` contain demand indices.
    """

    image_ids: Sequence[int]
    forward_indptr: Sequence[int]
    forward_indices: Sequence[int]
    forward_similarities: Sequence[float]
    reverse_indptr: Sequence[int]
    reverse_indices: Sequence[int]
    reverse_similarities: Sequence[float]

    def __post_init__(self) -> None:
        image_ids = [
            _integer(image_id, label=f"image_ids[{index}]")
            for index, image_id in enumerate(self.image_ids)
        ]
        if len(set(image_ids)) != len(image_ids):
            raise ValueError("image_ids must be unique")
        count = len(image_ids)
        _validate_csr(
            name="forward",
            row_count=count,
            column_count=count,
            indptr=self.forward_indptr,
            indices=self.forward_indices,
            similarities=self.forward_similarities,
        )
        _validate_csr(
            name="reverse",
            row_count=count,
            column_count=count,
            indptr=self.reverse_indptr,
            indices=self.reverse_indices,
            similarities=self.reverse_similarities,
        )
        if len(self.forward_indices) != len(self.reverse_indices):
            raise ValueError("forward and reverse graphs must contain the same edge count")

    @classmethod
    def from_forward_csr(
        cls,
        *,
        image_ids: Sequence[int],
        indptr: Sequence[int],
        indices: Sequence[int],
        similarities: Sequence[float],
    ) -> SparseFacilityGraph:
        """Validate a forward CSR graph and construct its reverse CSR view."""

        count = len(image_ids)
        _validate_csr(
            name="forward",
            row_count=count,
            column_count=count,
            indptr=indptr,
            indices=indices,
            similarities=similarities,
        )
        degrees = [0] * count
        for value in indices:
            degrees[_integer(value, label="forward index")] += 1

        reverse_indptr = [0] * (count + 1)
        for candidate, degree in enumerate(degrees):
            reverse_indptr[candidate + 1] = reverse_indptr[candidate] + degree
        cursors = reverse_indptr[:-1].copy()
        reverse_indices = [0] * len(indices)
        reverse_similarities = [0.0] * len(similarities)
        for demand in range(count):
            start = _integer(indptr[demand], label=f"forward indptr[{demand}]")
            stop = _integer(indptr[demand + 1], label=f"forward indptr[{demand + 1}]")
            for offset in range(start, stop):
                candidate = _integer(indices[offset], label=f"forward indices[{offset}]")
                destination = cursors[candidate]
                reverse_indices[destination] = demand
                reverse_similarities[destination] = float(similarities[offset])
                cursors[candidate] += 1

        return cls(
            image_ids=image_ids,
            forward_indptr=indptr,
            forward_indices=indices,
            forward_similarities=similarities,
            reverse_indptr=reverse_indptr,
            reverse_indices=reverse_indices,
            reverse_similarities=reverse_similarities,
        )


@dataclass(frozen=True)
class FacilitySelectionResult:
    """Greedy selection order and auditable objective state."""

    selected_indices: tuple[int, ...]
    selected_image_ids: tuple[int, ...]
    marginal_gains: tuple[float, ...]
    objective_value: float
    best_similarities: tuple[float, ...]
    outlier_count: int
    gain_evaluations: int


@dataclass(frozen=True)
class _Constraints:
    group_by_index: Mapping[int, Hashable]
    outlier_indices: frozenset[int]
    max_outliers: int


def _constraints(
    graph: SparseFacilityGraph,
    *,
    duplicate_group_by_image_id: Mapping[int, Hashable] | None,
    outlier_image_ids: Collection[int] | None,
    max_outliers: int | None,
) -> _Constraints:
    id_to_index = {int(image_id): index for index, image_id in enumerate(graph.image_ids)}
    groups: dict[int, Hashable] = {}
    for raw_image_id, group in (duplicate_group_by_image_id or {}).items():
        image_id = _integer(raw_image_id, label="duplicate-group image ID")
        if image_id not in id_to_index:
            raise ValueError(f"Duplicate group refers to unknown image ID {image_id}")
        try:
            hash(group)
        except TypeError as error:
            raise ValueError("Duplicate group identifiers must be hashable") from error
        groups[id_to_index[image_id]] = group

    outliers: set[int] = set()
    for raw_image_id in outlier_image_ids or ():
        image_id = _integer(raw_image_id, label="outlier image ID")
        if image_id not in id_to_index:
            raise ValueError(f"Outlier set refers to unknown image ID {image_id}")
        outliers.add(id_to_index[image_id])

    if max_outliers is None:
        cap = len(outliers)
    else:
        cap = _integer(max_outliers, label="max_outliers")
        if cap < 0:
            raise ValueError("max_outliers must be nonnegative")
    return _Constraints(groups, frozenset(outliers), cap)


def _maximum_feasible_count(graph: SparseFacilityGraph, constraints: _Constraints) -> int:
    # Every duplicate group is one selection slot. Ungrouped candidates each own
    # a unique private slot. A slot containing a non-outlier can be filled without
    # consuming the outlier allowance.
    slot_has_non_outlier: dict[object, bool] = {}
    private_prefix = object()
    for candidate in range(len(graph.image_ids)):
        slot: object
        if candidate in constraints.group_by_index:
            slot = ("group", constraints.group_by_index[candidate])
        else:
            slot = (private_prefix, candidate)
        has_non_outlier = candidate not in constraints.outlier_indices
        slot_has_non_outlier[slot] = slot_has_non_outlier.get(slot, False) or has_non_outlier
    unrestricted = sum(slot_has_non_outlier.values())
    outlier_only = len(slot_has_non_outlier) - unrestricted
    return unrestricted + min(constraints.max_outliers, outlier_only)


def _validate_budget(graph: SparseFacilityGraph, budget: int, constraints: _Constraints) -> int:
    parsed = _integer(budget, label="budget")
    if parsed < 0:
        raise ValueError("budget must be nonnegative")
    feasible = _maximum_feasible_count(graph, constraints)
    if parsed > feasible:
        raise ValueError(
            f"budget {parsed} exceeds the {feasible} selections allowed by the constraints"
        )
    return parsed


def _is_feasible(
    candidate: int,
    *,
    constraints: _Constraints,
    selected_groups: set[Hashable],
    selected_outliers: int,
) -> bool:
    if (
        candidate in constraints.group_by_index
        and constraints.group_by_index[candidate] in selected_groups
    ):
        return False
    return not (
        candidate in constraints.outlier_indices and selected_outliers >= constraints.max_outliers
    )


def _marginal_gain(
    graph: SparseFacilityGraph, candidate: int, best_similarities: Sequence[float]
) -> float:
    start = _integer(graph.reverse_indptr[candidate], label="reverse row start")
    stop = _integer(graph.reverse_indptr[candidate + 1], label="reverse row stop")
    gain = 0.0
    for offset in range(start, stop):
        demand = _integer(graph.reverse_indices[offset], label="reverse demand index")
        similarity = float(graph.reverse_similarities[offset])
        if similarity > best_similarities[demand]:
            gain += similarity - best_similarities[demand]
    return gain


def _select_candidate(
    graph: SparseFacilityGraph,
    candidate: int,
    best_similarities: list[float],
) -> None:
    start = _integer(graph.reverse_indptr[candidate], label="reverse row start")
    stop = _integer(graph.reverse_indptr[candidate + 1], label="reverse row stop")
    for offset in range(start, stop):
        demand = _integer(graph.reverse_indices[offset], label="reverse demand index")
        similarity = float(graph.reverse_similarities[offset])
        best_similarities[demand] = max(best_similarities[demand], similarity)


def _result(
    graph: SparseFacilityGraph,
    selected: list[int],
    gains: list[float],
    best_similarities: list[float],
    constraints: _Constraints,
    gain_evaluations: int,
) -> FacilitySelectionResult:
    return FacilitySelectionResult(
        selected_indices=tuple(selected),
        selected_image_ids=tuple(int(graph.image_ids[index]) for index in selected),
        marginal_gains=tuple(gains),
        objective_value=math.fsum(best_similarities),
        best_similarities=tuple(best_similarities),
        outlier_count=sum(index in constraints.outlier_indices for index in selected),
        gain_evaluations=gain_evaluations,
    )


def select_facility_location_naive(
    graph: SparseFacilityGraph,
    *,
    budget: int,
    duplicate_group_by_image_id: Mapping[int, Hashable] | None = None,
    outlier_image_ids: Collection[int] | None = None,
    max_outliers: int | None = None,
) -> FacilitySelectionResult:
    """Reference greedy maximization that rescans every feasible candidate."""

    constraints = _constraints(
        graph,
        duplicate_group_by_image_id=duplicate_group_by_image_id,
        outlier_image_ids=outlier_image_ids,
        max_outliers=max_outliers,
    )
    budget = _validate_budget(graph, budget, constraints)
    best_similarities = [0.0] * len(graph.image_ids)
    remaining = set(range(len(graph.image_ids)))
    selected: list[int] = []
    gains: list[float] = []
    selected_groups: set[Hashable] = set()
    selected_outliers = 0
    gain_evaluations = 0

    while len(selected) < budget:
        winner: int | None = None
        winner_gain = -1.0
        for candidate in remaining:
            if not _is_feasible(
                candidate,
                constraints=constraints,
                selected_groups=selected_groups,
                selected_outliers=selected_outliers,
            ):
                continue
            gain = _marginal_gain(graph, candidate, best_similarities)
            gain_evaluations += 1
            image_id = int(graph.image_ids[candidate])
            if (
                winner is None
                or gain > winner_gain
                or (gain == winner_gain and image_id < int(graph.image_ids[winner]))
            ):
                winner = candidate
                winner_gain = gain
        if winner is None:  # Defensive: feasibility was checked before selection.
            raise RuntimeError("No feasible facility candidate remains")
        remaining.remove(winner)
        selected.append(winner)
        gains.append(winner_gain)
        _select_candidate(graph, winner, best_similarities)
        if winner in constraints.group_by_index:
            selected_groups.add(constraints.group_by_index[winner])
        if winner in constraints.outlier_indices:
            selected_outliers += 1

    return _result(
        graph,
        selected,
        gains,
        best_similarities,
        constraints,
        gain_evaluations,
    )


def select_facility_location_lazy(
    graph: SparseFacilityGraph,
    *,
    budget: int,
    duplicate_group_by_image_id: Mapping[int, Hashable] | None = None,
    outlier_image_ids: Collection[int] | None = None,
    max_outliers: int | None = None,
) -> FacilitySelectionResult:
    """Memoized lazy greedy maximization of sparse facility coverage.

    Cached gains are valid upper bounds because facility-location marginal gains
    only decrease as coverage improves.  The heap key includes image ID, so this
    returns the same sequence as :func:`select_facility_location_naive`, including
    exact-gain ties, while usually performing fewer marginal-gain evaluations.
    """

    constraints = _constraints(
        graph,
        duplicate_group_by_image_id=duplicate_group_by_image_id,
        outlier_image_ids=outlier_image_ids,
        max_outliers=max_outliers,
    )
    budget = _validate_budget(graph, budget, constraints)
    best_similarities = [0.0] * len(graph.image_ids)
    heap: list[tuple[float, int, int]] = []
    gain_evaluations = 0
    for candidate, raw_image_id in enumerate(graph.image_ids):
        gain = _marginal_gain(graph, candidate, best_similarities)
        gain_evaluations += 1
        heap.append((-gain, int(raw_image_id), candidate))
    heapq.heapify(heap)

    selected: list[int] = []
    gains: list[float] = []
    selected_groups: set[Hashable] = set()
    selected_outliers = 0

    def prune_infeasible() -> None:
        while heap and not _is_feasible(
            heap[0][2],
            constraints=constraints,
            selected_groups=selected_groups,
            selected_outliers=selected_outliers,
        ):
            heapq.heappop(heap)

    while len(selected) < budget:
        prune_infeasible()
        if not heap:  # Defensive: feasibility was checked before selection.
            raise RuntimeError("No feasible facility candidate remains")
        _negative_bound, image_id, candidate = heapq.heappop(heap)
        gain = _marginal_gain(graph, candidate, best_similarities)
        gain_evaluations += 1
        prune_infeasible()
        current_key = (-gain, image_id, candidate)
        if heap and current_key > heap[0]:
            heapq.heappush(heap, current_key)
            continue

        selected.append(candidate)
        gains.append(gain)
        _select_candidate(graph, candidate, best_similarities)
        if candidate in constraints.group_by_index:
            selected_groups.add(constraints.group_by_index[candidate])
        if candidate in constraints.outlier_indices:
            selected_outliers += 1

    return _result(
        graph,
        selected,
        gains,
        best_similarities,
        constraints,
        gain_evaluations,
    )


def facility_location_value(
    graph: SparseFacilityGraph, selected_image_ids: Collection[int]
) -> float:
    """Evaluate sparse facility coverage using the forward CSR view."""

    selected_list = [
        _integer(image_id, label="selected image ID") for image_id in selected_image_ids
    ]
    if len(set(selected_list)) != len(selected_list):
        raise ValueError("selected_image_ids must be unique")
    id_to_index = {int(image_id): index for index, image_id in enumerate(graph.image_ids)}
    unknown = sorted(set(selected_list) - set(id_to_index))
    if unknown:
        raise ValueError(f"Selection refers to unknown image IDs: {unknown}")
    selected_indices = {id_to_index[image_id] for image_id in selected_list}

    best: list[float] = []
    for demand in range(len(graph.image_ids)):
        start = _integer(graph.forward_indptr[demand], label="forward row start")
        stop = _integer(graph.forward_indptr[demand + 1], label="forward row stop")
        value = 0.0
        for offset in range(start, stop):
            candidate = _integer(graph.forward_indices[offset], label="forward candidate index")
            if candidate in selected_indices:
                value = max(value, float(graph.forward_similarities[offset]))
        best.append(value)
    return math.fsum(best)


def select_facility_location_numpy(
    *,
    image_ids: object,
    reverse_indptr: object,
    reverse_indices: object,
    reverse_similarities: object,
    budget: int,
    allowed_candidate_mask: object | None = None,
    duplicate_groups: object | None = None,
    outlier_mask: object | None = None,
    max_outliers: int | None = None,
) -> FacilitySelectionResult:
    """Scalable NumPy implementation of deterministic lazy greedy selection.

    NumPy is imported inside this function so importing the base package and
    running its standard-library data-foundation commands do not require the
    visual-selection environment.  The reverse CSR arrays are treated as a
    trusted transpose of the forward neighbor graph: shapes, dtypes, pointer
    monotonicity, index bounds, and similarity values are checked with vectorized
    operations, but edge-by-edge transpose consistency is intentionally not
    revalidated here.

    ``duplicate_groups`` is an integer array aligned with ``image_ids``.  Every
    nonnegative value identifies a group from which at most one member may be
    selected; negative values denote independent, ungrouped candidates.
    """

    try:
        import numpy as np
    except ImportError as error:  # pragma: no cover - depends on optional environment.
        raise RuntimeError(
            "NumPy is required for accelerated facility-location selection"
        ) from error

    ids = np.asarray(image_ids)
    indptr = np.asarray(reverse_indptr)
    indices = np.asarray(reverse_indices)
    similarities = np.asarray(reverse_similarities)
    if ids.ndim != 1:
        raise ValueError("image_ids must be one-dimensional")
    if not np.issubdtype(ids.dtype, np.integer):
        raise ValueError("image_ids must have an integer dtype")
    count = int(ids.size)
    if np.unique(ids).size != count:
        raise ValueError("image_ids must be unique")
    if indptr.ndim != 1 or indptr.size != count + 1:
        raise ValueError(f"reverse_indptr must have shape ({count + 1},)")
    if not np.issubdtype(indptr.dtype, np.integer):
        raise ValueError("reverse_indptr must have an integer dtype")
    if indices.ndim != 1 or similarities.ndim != 1:
        raise ValueError("reverse indices and similarities must be one-dimensional")
    if indices.size != similarities.size:
        raise ValueError("reverse indices and similarities must have equal length")
    if not np.issubdtype(indices.dtype, np.integer):
        raise ValueError("reverse_indices must have an integer dtype")
    if not np.issubdtype(similarities.dtype, np.number):
        raise ValueError("reverse_similarities must have a numeric dtype")
    edge_count = int(indices.size)
    if (
        int(indptr[0]) != 0
        or int(indptr[-1]) != edge_count
        or bool(np.any(indptr[1:] < indptr[:-1]))
    ):
        raise ValueError("reverse_indptr is not a valid nondecreasing CSR pointer array")
    if edge_count and (int(indices.min()) < 0 or int(indices.max()) >= count):
        raise ValueError("reverse_indices contains an out-of-range demand index")
    if not bool(np.all(np.isfinite(similarities))) or bool(np.any(similarities < 0.0)):
        raise ValueError("reverse_similarities must be finite and nonnegative")

    if allowed_candidate_mask is None:
        allowed = np.ones(count, dtype=np.bool_)
    else:
        allowed_raw = np.asarray(allowed_candidate_mask)
        if allowed_raw.ndim != 1 or allowed_raw.size != count:
            raise ValueError(f"allowed_candidate_mask must have shape ({count},)")
        if not np.issubdtype(allowed_raw.dtype, np.bool_):
            raise ValueError("allowed_candidate_mask must have a boolean dtype")
        allowed = allowed_raw

    if duplicate_groups is None:
        groups = np.full(count, -1, dtype=np.int64)
    else:
        groups = np.asarray(duplicate_groups)
        if groups.ndim != 1 or groups.size != count:
            raise ValueError(f"duplicate_groups must have shape ({count},)")
        if not np.issubdtype(groups.dtype, np.integer):
            raise ValueError("duplicate_groups must have an integer dtype")

    if outlier_mask is None:
        outliers = np.zeros(count, dtype=np.bool_)
    else:
        outlier_raw = np.asarray(outlier_mask)
        if outlier_raw.ndim != 1 or outlier_raw.size != count:
            raise ValueError(f"outlier_mask must have shape ({count},)")
        if not np.issubdtype(outlier_raw.dtype, np.bool_):
            raise ValueError("outlier_mask must have a boolean dtype")
        outliers = outlier_raw

    parsed_budget = _integer(budget, label="budget")
    if parsed_budget < 0:
        raise ValueError("budget must be nonnegative")
    if max_outliers is None:
        outlier_cap = int(np.count_nonzero(outliers & allowed))
    else:
        outlier_cap = _integer(max_outliers, label="max_outliers")
        if outlier_cap < 0:
            raise ValueError("max_outliers must be nonnegative")

    # Determine exact cardinality feasibility under the allowed mask, duplicate
    # partition, and outlier cap without constructing Python objects per edge.
    allowed_indices = np.flatnonzero(allowed)
    allowed_groups = groups[allowed_indices]
    allowed_outliers = outliers[allowed_indices]
    ungrouped = allowed_groups < 0
    unrestricted_slots = int(np.count_nonzero(ungrouped & ~allowed_outliers))
    outlier_only_slots = int(np.count_nonzero(ungrouped & allowed_outliers))
    grouped = ~ungrouped
    if bool(np.any(grouped)):
        _group_values, inverse = np.unique(allowed_groups[grouped], return_inverse=True)
        group_has_non_outlier = np.zeros(_group_values.size, dtype=np.bool_)
        np.logical_or.at(
            group_has_non_outlier,
            inverse,
            ~allowed_outliers[grouped],
        )
        unrestricted_slots += int(np.count_nonzero(group_has_non_outlier))
        outlier_only_slots += int(
            group_has_non_outlier.size - np.count_nonzero(group_has_non_outlier)
        )
    feasible_count = unrestricted_slots + min(outlier_cap, outlier_only_slots)
    if parsed_budget > feasible_count:
        raise ValueError(
            f"budget {parsed_budget} exceeds the {feasible_count} selections "
            "allowed by the constraints"
        )

    # Sum each reverse row once in compiled code.  Supplying only nonempty row
    # starts to reduceat handles empty candidate rows without allocating an
    # edge-sized prefix-sum array.
    initial_gains = np.zeros(count, dtype=np.float64)
    nonempty = indptr[1:] > indptr[:-1]
    if edge_count:
        starts = indptr[:-1][nonempty]
        initial_gains[nonempty] = np.add.reduceat(similarities, starts, dtype=np.float64)

    heap: list[tuple[float, int, int]] = [
        (-float(initial_gains[candidate]), int(ids[candidate]), int(candidate))
        for candidate in allowed_indices
    ]
    heapq.heapify(heap)
    gain_evaluations = len(heap)
    best_similarities = np.zeros(count, dtype=similarities.dtype)
    selected: list[int] = []
    marginal_gains: list[float] = []
    selected_groups: set[int] = set()
    selected_outliers = 0

    def feasible(candidate: int) -> bool:
        group = int(groups[candidate])
        if group >= 0 and group in selected_groups:
            return False
        return not (bool(outliers[candidate]) and selected_outliers >= outlier_cap)

    def prune_infeasible() -> None:
        while heap and not feasible(heap[0][2]):
            heapq.heappop(heap)

    def marginal_gain(candidate: int) -> float:
        start = int(indptr[candidate])
        stop = int(indptr[candidate + 1])
        if start == stop:
            return 0.0
        demands = indices[start:stop]
        improvements = similarities[start:stop] - best_similarities[demands]
        return float(np.maximum(improvements, 0.0).sum(dtype=np.float64))

    while len(selected) < parsed_budget:
        prune_infeasible()
        if not heap:  # Defensive: exact feasibility was checked above.
            raise RuntimeError("No feasible facility candidate remains")
        _negative_bound, image_id, candidate = heapq.heappop(heap)
        gain = marginal_gain(candidate)
        gain_evaluations += 1
        prune_infeasible()
        current_key = (-gain, image_id, candidate)
        if heap and current_key > heap[0]:
            heapq.heappush(heap, current_key)
            continue

        selected.append(candidate)
        marginal_gains.append(gain)
        start = int(indptr[candidate])
        stop = int(indptr[candidate + 1])
        if start != stop:
            demands = indices[start:stop]
            np.maximum.at(
                best_similarities,
                demands,
                similarities[start:stop],
            )
        group = int(groups[candidate])
        if group >= 0:
            selected_groups.add(group)
        if bool(outliers[candidate]):
            selected_outliers += 1

    return FacilitySelectionResult(
        selected_indices=tuple(selected),
        selected_image_ids=tuple(int(ids[index]) for index in selected),
        marginal_gains=tuple(marginal_gains),
        objective_value=float(best_similarities.sum(dtype=np.float64)),
        best_similarities=tuple(float(value) for value in best_similarities),
        outlier_count=selected_outliers,
        gain_evaluations=gain_evaluations,
    )


__all__ = [
    "FacilitySelectionResult",
    "SparseFacilityGraph",
    "facility_location_value",
    "select_facility_location_lazy",
    "select_facility_location_naive",
    "select_facility_location_numpy",
]
