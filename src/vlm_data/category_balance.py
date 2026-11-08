"""Train-only calibration and reporting for category-balanced COCO selection."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence, Any

from vlm_data.io import sha256_file, write_json_atomic, write_text_atomic
from vlm_data.manifest import ImageRecord, metadata_path


ANALYSIS_VERSION = "category_balance_sensitivity_v2"
DEFAULT_FREQUENCY_THRESHOLD = 0.025
DEFAULT_MAX_BOOST = 4.0
DEFAULT_MINIMUM_CHANGED_FRACTION = 0.05


@dataclass(frozen=True)
class CategoryTarget:
    """Frequency-derived target for one category under one parameter scenario."""

    category_id: int
    source_count: int
    frequency: float
    natural_expected: float
    raw_boost: float
    effective_boost: float
    target: int
    affected: bool
    capped: bool

    def to_dict(self, name: str) -> dict[str, Any]:
        value = asdict(self)
        value["name"] = name
        return value


def category_counts(records: Iterable[ImageRecord]) -> Counter[int]:
    """Count distinct image membership for every annotated category."""

    counts: Counter[int] = Counter()
    for record in records:
        counts.update(set(record.category_ids))
    return counts


def _validate_parameters(
    *, total_images: int, budget: int, frequency_threshold: float, max_boost: float | None
) -> None:
    if total_images <= 0:
        raise ValueError("category analysis requires at least one image")
    if budget <= 0:
        raise ValueError("budget must be positive")
    if budget > total_images:
        raise ValueError(f"Cannot select {budget} records from only {total_images}")
    if not 0.0 < frequency_threshold <= 1.0:
        raise ValueError("frequency threshold must be in (0, 1]")
    if max_boost is not None and max_boost < 1.0:
        raise ValueError("maximum boost must be at least 1")


def compute_category_targets(
    counts: Mapping[int, int],
    *,
    total_images: int,
    budget: int,
    frequency_threshold: float,
    max_boost: float | None,
) -> dict[int, CategoryTarget]:
    """Compute square-root frequency targets from actual manifest counts.

    ``max_boost=None`` represents an uncapped sensitivity scenario. Targets are
    integer image counts and can never exceed the number of available images for
    a category.
    """

    _validate_parameters(
        total_images=total_images,
        budget=budget,
        frequency_threshold=frequency_threshold,
        max_boost=max_boost,
    )
    targets: dict[int, CategoryTarget] = {}
    for category_id, source_count in sorted(counts.items()):
        if source_count <= 0:
            continue
        frequency = source_count / total_images
        natural_expected = budget * frequency
        raw_boost = max(1.0, math.sqrt(frequency_threshold / frequency))
        effective_boost = raw_boost if max_boost is None else min(max_boost, raw_boost)
        target = min(source_count, math.ceil(natural_expected * effective_boost))
        targets[category_id] = CategoryTarget(
            category_id=category_id,
            source_count=source_count,
            frequency=frequency,
            natural_expected=natural_expected,
            raw_boost=raw_boost,
            effective_boost=effective_boost,
            target=target,
            affected=raw_boost > 1.0 + 1e-12,
            capped=(max_boost is not None and raw_boost > max_boost + 1e-12),
        )
    if not targets:
        raise ValueError("category analysis requires instance category annotations")
    return targets


def load_category_names(source_manifest: Path) -> dict[int, str]:
    """Load category names from the canonical manifest metadata when available."""

    sidecar = metadata_path(source_manifest)
    if not sidecar.is_file():
        return {}
    try:
        value = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid manifest metadata {sidecar}: {error}") from error
    categories = value.get("categories", {})
    return {
        int(category_id): str(details.get("name", category_id))
        for category_id, details in categories.items()
        if isinstance(details, dict)
    }


def _scenario(
    counts: Mapping[int, int],
    names: Mapping[int, str],
    *,
    total_images: int,
    budget: int,
    frequency_threshold: float,
    max_boost: float | None,
) -> dict[str, Any]:
    targets = compute_category_targets(
        counts,
        total_images=total_images,
        budget=budget,
        frequency_threshold=frequency_threshold,
        max_boost=max_boost,
    )
    # This is category-image incidence pressure, not a count of distinct images.
    # It deliberately excludes integer rounding so the zero-boost case has zero
    # added mass.
    extra_target_mass = sum(
        item.natural_expected * (item.effective_boost - 1.0) for item in targets.values()
    )
    return {
        "frequency_threshold": frequency_threshold,
        "max_boost": max_boost,
        "affected_category_count": sum(item.affected for item in targets.values()),
        "capped_category_count": sum(item.capped for item in targets.values()),
        "extra_target_mass": extra_target_mass,
        "extra_target_fraction_of_budget": extra_target_mass / budget,
        "max_raw_boost": max(item.raw_boost for item in targets.values()),
        "natural_category_incidence_expectation": sum(
            item.natural_expected for item in targets.values()
        ),
        "integer_target_incidence_total": sum(item.target for item in targets.values()),
        "categories": [
            item.to_dict(names.get(category_id, str(category_id)))
            for category_id, item in targets.items()
        ],
    }


def _scenario_key(threshold: float, cap: float | None) -> tuple[float, float | None]:
    return (threshold, cap)


def analyze_category_balance(
    records: Sequence[ImageRecord],
    *,
    source_manifest: Path,
    budget: int,
    proposed_threshold: float,
    proposed_max_boost: float,
    threshold_candidates: Sequence[float],
    max_boost_candidates: Sequence[float],
    max_extra_target_fraction: float,
) -> dict[str, Any]:
    """Run a deterministic train-only sensitivity analysis and recommend parameters.

    The recommendation deliberately controls the *size* of the intervention; it
    does not claim to optimize retrieval accuracy. The largest candidate threshold
    that adds no more than the configured category-incidence mass is selected.
    The smallest candidate cap that is inactive at that threshold is then selected
    as a safety guard.
    """

    if not records:
        raise ValueError("category analysis requires at least one manifest record")
    if not 0.0 < max_extra_target_fraction <= 1.0:
        raise ValueError("max extra target fraction must be in (0, 1]")

    thresholds = sorted({float(value) for value in threshold_candidates} | {proposed_threshold})
    caps = sorted({float(value) for value in max_boost_candidates} | {proposed_max_boost})
    for threshold in thresholds:
        _validate_parameters(
            total_images=len(records),
            budget=budget,
            frequency_threshold=threshold,
            max_boost=None,
        )
    for cap in caps:
        if cap < 1.0:
            raise ValueError("maximum boost candidates must be at least 1")

    counts = category_counts(records)
    names = load_category_names(source_manifest)
    scenarios: list[dict[str, Any]] = []
    scenario_lookup: dict[tuple[float, float | None], dict[str, Any]] = {}
    for threshold in thresholds:
        for cap in [*caps, None]:
            scenario = _scenario(
                counts,
                names,
                total_images=len(records),
                budget=budget,
                frequency_threshold=threshold,
                max_boost=cap,
            )
            scenarios.append(scenario)
            scenario_lookup[_scenario_key(threshold, cap)] = scenario

    eligible_thresholds = [
        threshold
        for threshold in thresholds
        if (
            scenario_lookup[_scenario_key(threshold, None)]["affected_category_count"] > 0
            and scenario_lookup[_scenario_key(threshold, None)]["extra_target_fraction_of_budget"]
            <= max_extra_target_fraction + 1e-12
        )
    ]
    if not eligible_thresholds:
        raise ValueError(
            "No candidate threshold affects a category within the extra-target-mass limit"
        )
    recommended_threshold = max(eligible_thresholds)
    nonbinding_caps = [
        cap
        for cap in caps
        if scenario_lookup[_scenario_key(recommended_threshold, cap)]["capped_category_count"] == 0
    ]
    if not nonbinding_caps:
        raise ValueError("No maximum-boost candidate is nonbinding at the recommended threshold")
    recommended_max_boost = min(nonbinding_caps)

    proposed_scenario = scenario_lookup[_scenario_key(proposed_threshold, proposed_max_boost)]
    recommended_scenario = scenario_lookup[
        _scenario_key(recommended_threshold, recommended_max_boost)
    ]
    return {
        "schema_version": 1,
        "kind": "category_balance_analysis",
        "analysis_version": ANALYSIS_VERSION,
        "source_manifest": source_manifest.name,
        "source_manifest_sha256": sha256_file(source_manifest),
        "train_image_count": len(records),
        "category_count": len(counts),
        "budget": budget,
        "candidate_thresholds": thresholds,
        "candidate_max_boosts": caps,
        "recommendation_rule": {
            "threshold": (
                "largest candidate that affects at least one category while uncapped "
                "extra target mass is at most the configured fraction of the image budget"
            ),
            "max_extra_target_fraction": max_extra_target_fraction,
            "cap": "smallest candidate cap that does not bind at the chosen threshold",
            "scope": (
                "calibrates intervention size from train2017 counts; "
                "does not optimize model quality"
            ),
        },
        "proposed_parameters": {
            "frequency_threshold": proposed_threshold,
            "max_boost": proposed_max_boost,
        },
        "recommendation": {
            "frequency_threshold": recommended_threshold,
            "max_boost": recommended_max_boost,
            "matches_proposal": (
                recommended_threshold == proposed_threshold
                and recommended_max_boost == proposed_max_boost
            ),
        },
        "proposed_scenario": proposed_scenario,
        "recommended_scenario": recommended_scenario,
        "scenarios": scenarios,
    }


def add_selection_simulation(
    report: dict[str, Any],
    *,
    selected: Sequence[ImageRecord],
    random_baseline: Sequence[ImageRecord],
    seed: int,
    balanced_algorithm_version: str,
    baseline_algorithm_version: str,
    minimum_changed_fraction: float = 0.0,
) -> None:
    """Attach achieved target coverage for real deterministic selections."""

    if not 0.0 <= minimum_changed_fraction <= 1.0:
        raise ValueError("minimum changed fraction must be in [0, 1]")

    selected_coverage = category_counts(selected)
    baseline_coverage = category_counts(random_baseline)
    target_rows = report["recommended_scenario"]["categories"]
    per_category: list[dict[str, Any]] = []
    for row in target_rows:
        category_id = int(row["category_id"])
        target = int(row["target"])
        achieved = selected_coverage[category_id]
        per_category.append(
            {
                "category_id": category_id,
                "name": row["name"],
                "enforced": bool(row["affected"]),
                "target": target,
                "random_coverage": baseline_coverage[category_id],
                "selected_coverage": achieved,
                "shortfall": max(0, target - achieved) if row["affected"] else 0,
            }
        )
    selected_ids = {record.image_id for record in selected}
    baseline_ids = {record.image_id for record in random_baseline}
    changed_image_count = len(selected_ids - baseline_ids)
    changed_fraction = changed_image_count / len(selected_ids)
    required_changed_count = math.ceil(minimum_changed_fraction * len(selected_ids))

    def image_id_sha256(image_ids: set[int]) -> str:
        payload = "".join(f"{image_id}\n" for image_id in sorted(image_ids)).encode("ascii")
        return hashlib.sha256(payload).hexdigest()

    report["selection_simulation"] = {
        "seed": seed,
        "balanced_algorithm_version": balanced_algorithm_version,
        "baseline_algorithm_version": baseline_algorithm_version,
        "selected_count": len(selected),
        "random_baseline_count": len(random_baseline),
        "selected_image_ids_sha256": image_id_sha256(selected_ids),
        "random_image_ids_sha256": image_id_sha256(baseline_ids),
        "overlap_count": len(selected_ids & baseline_ids),
        "overlap_fraction": len(selected_ids & baseline_ids) / len(selected_ids),
        "changed_image_count": changed_image_count,
        "changed_fraction": changed_fraction,
        "minimum_changed_fraction": minimum_changed_fraction,
        "required_changed_count": required_changed_count,
        "minimum_changed_fraction_met": changed_image_count >= required_changed_count,
        "random_mean_distinct_categories_per_image": (
            sum(len(record.category_ids) for record in random_baseline) / len(random_baseline)
        ),
        "selected_mean_distinct_categories_per_image": (
            sum(len(record.category_ids) for record in selected) / len(selected)
        ),
        "enforced_category_count": sum(row["enforced"] for row in per_category),
        "enforced_categories_meeting_target": sum(
            row["enforced"] and row["shortfall"] == 0 for row in per_category
        ),
        "enforced_categories_with_shortfall": sum(
            row["enforced"] and row["shortfall"] > 0 for row in per_category
        ),
        "total_enforced_target_shortfall": sum(row["shortfall"] for row in per_category),
        "per_category": per_category,
    }


def _format_cap(value: float | None) -> str:
    return "uncapped" if value is None else f"{value:g}×"


def render_category_balance_markdown(report: Mapping[str, Any]) -> str:
    """Render the machine-readable analysis as a human-auditable Markdown report."""

    recommendation = report["recommendation"]
    proposed = report["proposed_parameters"]
    recommended_scenario = report["recommended_scenario"]
    lines = [
        "# Category-Balance Sensitivity Analysis",
        "",
        "> Generated from the canonical `train2017` manifest. Do not edit by hand.",
        "",
        f"- Training images: {report['train_image_count']:,}",
        f"- Categories: {report['category_count']}",
        f"- Selection budget: {report['budget']:,}",
        f"- Source SHA-256: `{report['source_manifest_sha256']}`",
        "",
        "## Recommendation",
        "",
        (
            f"The count-based rule recommends threshold "
            f"`{recommendation['frequency_threshold']:g}` and maximum boost "
            f"`{recommendation['max_boost']:g}×`."
        ),
        "",
        (
            f"The checked-in proposal is threshold `{proposed['frequency_threshold']:g}` and "
            f"maximum boost `{proposed['max_boost']:g}×`; "
            f"**match: {'yes' if recommendation['matches_proposal'] else 'no'}**."
        ),
        "",
        "The threshold is the largest tested value whose uncapped added category-image target mass "
        f"is at most {report['recommendation_rule']['max_extra_target_fraction']:.1%} of the "
        f"{report['budget']:,}-image budget. The cap is the smallest tested integer cap that does "
        "not alter any "
        "current COCO target. This controls intervention magnitude; it does not prove retrieval "
        "optimality.",
        "",
        "## Candidate sensitivity",
        "",
        "| Threshold | Cap | Affected categories | Capped categories | Added target mass | "
        "Budget share | Max raw boost |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for scenario in report["scenarios"]:
        lines.append(
            (
                "| {threshold:g} | {cap} | {affected} | {capped} | {mass:.2f} | "
                "{share:.3%} | {raw:.3f}× |"
            ).format(
                threshold=scenario["frequency_threshold"],
                cap=_format_cap(scenario["max_boost"]),
                affected=scenario["affected_category_count"],
                capped=scenario["capped_category_count"],
                mass=scenario["extra_target_mass"],
                share=scenario["extra_target_fraction_of_budget"],
                raw=scenario["max_raw_boost"],
            )
        )

    simulation = report.get("selection_simulation")
    if simulation:
        feasibility_text = (
            "multi-label image can satisfy several targets, so the real selector is simulated "
            "below."
        )
    else:
        feasibility_text = (
            "multi-label image can satisfy several targets. Selection simulation was skipped, so "
            "joint feasibility is not established by this report."
        )
    lines.extend(
        [
            "",
            "`Added target mass` is a sum of category-image incidences, not distinct images. One "
            + feasibility_text,
            "",
            "## Categories affected by the recommendation",
            "",
            f"| ID | Category | Source images | Frequency | Natural {report['budget']:,} "
            "expectation | Boost | Target |",
            "|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in recommended_scenario["categories"]:
        if not row["affected"]:
            continue
        lines.append(
            "| {category_id} | {name} | {source_count:,} | {frequency:.3%} | "
            "{natural_expected:.2f} | {effective_boost:.3f}× | {target} |".format(**row)
        )

    if simulation:
        lines.extend(
            [
                "",
                "## Real-selection simulation",
                "",
                f"- Simulation seed: {simulation['seed']}",
                (
                    f"- Algorithms: `{simulation['balanced_algorithm_version']}` vs. "
                    f"`{simulation['baseline_algorithm_version']}`"
                ),
                f"- Recommended selection size: {simulation['selected_count']:,}",
                (
                    f"- Enforced categories meeting target: "
                    f"{simulation['enforced_categories_meeting_target']} / "
                    f"{simulation['enforced_category_count']}"
                ),
                (
                    f"- Total residual enforced-target shortfall: "
                    f"{simulation['total_enforced_target_shortfall']}"
                ),
                (
                    "- Overlap with deterministic random baseline: "
                    f"{simulation['overlap_count']:,} "
                    f"images ({simulation['overlap_fraction']:.2%})"
                ),
                (
                    f"- Images swapped relative to baseline: "
                    f"{simulation['changed_image_count']:,} "
                    f"({simulation['changed_fraction']:.2%}); required minimum: "
                    f"{simulation['required_changed_count']:,} "
                    f"({simulation['minimum_changed_fraction']:.2%}); "
                    f"**met: "
                    f"{'yes' if simulation['minimum_changed_fraction_met'] else 'no'}**"
                ),
                (
                    f"- Mean distinct categories per image: "
                    f"{simulation['random_mean_distinct_categories_per_image']:.3f} random vs. "
                    f"{simulation['selected_mean_distinct_categories_per_image']:.3f} balanced"
                ),
                "",
                "| Category | Target | Random coverage | Balanced coverage | Shortfall |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for row in simulation["per_category"]:
            if not row["enforced"]:
                continue
            lines.append(
                (
                    "| {name} | {target} | {random_coverage} | {selected_coverage} | "
                    "{shortfall} |"
                ).format(**row)
            )

    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            "This analysis uses only `train2017` category incidence. It measures how aggressive "
            "the balancing intervention is and whether its targets are jointly reachable. It does "
            "not measure caption mention, object size, visual redundancy, or model-specific "
            "difficulty. Those questions require separate train-only audits; `val2017` must not "
            "select these parameters.",
            "",
        ]
    )
    return "\n".join(lines)


def write_category_balance_analysis(
    report: Mapping[str, Any], *, json_output: Path, markdown_output: Path
) -> None:
    write_json_atomic(json_output, report)
    write_text_atomic(markdown_output, render_category_balance_markdown(report))
