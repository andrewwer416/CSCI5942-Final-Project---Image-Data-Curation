"""Command-line interface for acquiring, indexing, selecting, and validating data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import zipfile

from vlm_data.category_balance import (
    add_selection_simulation,
    analyze_category_balance,
    write_category_balance_analysis,
)
from vlm_data.coco import build_coco_manifest
from vlm_data.config import DataConfig, load_config
from vlm_data.download import download_components
from vlm_data.manifest import read_manifest
from vlm_data.selection import (
    CATEGORY_BALANCED_ALGORITHM_VERSION,
    RANDOM_ALGORITHM_VERSION,
    SELECTION_METHODS,
    select_category_balanced,
    select_random,
    write_selection,
)
from vlm_data.validation import validate_project


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vlm-data",
        description="Prepare reproducible COCO manifests for the VLM retrieval study.",
    )
    parser.add_argument("--config", default="configs/data.toml", help="Path to data TOML config")
    subcommands = parser.add_subparsers(dest="command", required=True)

    download = subcommands.add_parser(
        "download", help="Download and extract official COCO archives"
    )
    download.add_argument(
        "components",
        nargs="+",
        choices=["annotations", "train", "val", "all"],
        help="Archives to download (the train image archive is large)",
    )
    download.add_argument(
        "--confirm-large-download",
        action="store_true",
        help="Required when downloading train2017 images or all components",
    )

    prepare = subcommands.add_parser("prepare", help="Build canonical manifests from annotations")
    prepare.add_argument("--split", choices=["train", "val", "all"], default="all")

    select = subcommands.add_parser("select", help="Build a 15,000-image selection manifest")
    select.add_argument("--method", choices=sorted(SELECTION_METHODS), default="random")
    select.add_argument("--seed", type=int, help="Override the configured seed")
    select.add_argument("--budget", type=int, help="Override the configured image budget")
    select.add_argument("--output", type=Path, help="Override the output JSONL path")
    select.add_argument(
        "--category-frequency-threshold",
        type=float,
        help="Override the configured threshold for category-balanced selection",
    )
    select.add_argument(
        "--category-max-boost",
        type=float,
        help="Override the configured maximum boost for category-balanced selection",
    )

    analyze = subcommands.add_parser(
        "analyze-category-balance",
        help="Calibrate category-balance parameters from the canonical train manifest",
    )
    analyze.add_argument("--seed", type=int, help="Override the seed used in selection simulation")
    analyze.add_argument("--budget", type=int, help="Override the configured image budget")
    analyze.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        help="Candidate frequency thresholds (default: configured sensitivity grid)",
    )
    analyze.add_argument(
        "--caps",
        type=float,
        nargs="+",
        help="Candidate maximum boosts (default: configured sensitivity grid)",
    )
    analyze.add_argument(
        "--max-extra-target-fraction",
        type=float,
        help="Maximum added category-incidence mass as a fraction of the image budget",
    )
    analyze.add_argument("--json-output", type=Path, help="Override the JSON report path")
    analyze.add_argument("--markdown-output", type=Path, help="Override the Markdown report path")
    analyze.add_argument(
        "--skip-selection-simulation",
        action="store_true",
        help="Skip constructing the recommended and random 15k selections",
    )

    validate = subcommands.add_parser("validate", help="Validate canonical and selection manifests")
    validate.add_argument(
        "--check-files",
        action="store_true",
        help="Also verify every referenced image exists on disk",
    )
    return parser


def _prepare(config: DataConfig, split: str) -> list[dict[str, object]]:
    outputs: list[dict[str, object]] = []
    if split in {"train", "all"}:
        outputs.append(
            build_coco_manifest(
                split="train2017",
                image_directory_name="train2017",
                captions_path=config.train_captions,
                instances_path=config.train_instances,
                output_path=config.train_manifest,
                schema_version=config.schema_version,
                captions_per_image=config.captions_per_image,
            )
        )
    if split in {"val", "all"}:
        outputs.append(
            build_coco_manifest(
                split="val2017",
                image_directory_name="val2017",
                captions_path=config.val_captions,
                instances_path=config.val_instances,
                output_path=config.val_manifest,
                schema_version=config.schema_version,
                captions_per_image=config.captions_per_image,
            )
        )
    return outputs


def _select(config: DataConfig, args: argparse.Namespace) -> dict[str, object]:
    if not config.train_manifest.is_file():
        raise FileNotFoundError(
            f"Canonical training manifest is missing: {config.train_manifest}. Run prepare first."
        )
    seed = config.seed if args.seed is None else args.seed
    budget = config.train_image_budget if args.budget is None else args.budget
    frequency_threshold = (
        config.category_balance.frequency_threshold
        if args.category_frequency_threshold is None
        else args.category_frequency_threshold
    )
    max_boost = (
        config.category_balance.max_boost
        if args.category_max_boost is None
        else args.category_max_boost
    )
    method_file = args.method.replace("-", "_")
    if args.method == "category-balanced":
        threshold_slug = f"{frequency_threshold:g}".replace(".", "p")
        boost_slug = f"{max_boost:g}".replace(".", "p")
        method_file += f"_t{threshold_slug}_cap{boost_slug}"
    output = args.output or config.selection_dir / f"{method_file}_seed{seed}_n{budget}.jsonl"
    return write_selection(
        records=read_manifest(config.train_manifest),
        output_path=output,
        source_manifest=config.train_manifest,
        method=args.method,
        seed=seed,
        budget=budget,
        schema_version=config.schema_version,
        category_frequency_threshold=frequency_threshold,
        category_max_boost=max_boost,
        category_minimum_changed_fraction=(
            config.category_balance.minimum_changed_fraction
            if args.method == "category-balanced"
            else 0.0
        ),
    )


def _analyze_category_balance(config: DataConfig, args: argparse.Namespace) -> dict[str, object]:
    if not config.train_manifest.is_file():
        raise FileNotFoundError(
            f"Canonical training manifest is missing: {config.train_manifest}. Run prepare first."
        )
    records = list(read_manifest(config.train_manifest))
    budget = config.train_image_budget if args.budget is None else args.budget
    seed = config.seed if args.seed is None else args.seed
    thresholds = args.thresholds or config.category_balance.threshold_candidates
    caps = args.caps or config.category_balance.max_boost_candidates
    max_extra_fraction = (
        config.category_balance.max_extra_target_fraction
        if args.max_extra_target_fraction is None
        else args.max_extra_target_fraction
    )
    report = analyze_category_balance(
        records,
        source_manifest=config.train_manifest,
        budget=budget,
        proposed_threshold=config.category_balance.frequency_threshold,
        proposed_max_boost=config.category_balance.max_boost,
        threshold_candidates=thresholds,
        max_boost_candidates=caps,
        max_extra_target_fraction=max_extra_fraction,
    )
    if not args.skip_selection_simulation:
        recommendation = report["recommendation"]
        selected = select_category_balanced(
            records,
            budget=budget,
            seed=seed,
            frequency_threshold=float(recommendation["frequency_threshold"]),
            max_boost=float(recommendation["max_boost"]),
        )
        random_baseline = select_random(records, budget=budget, seed=seed)
        add_selection_simulation(
            report,
            selected=selected,
            random_baseline=random_baseline,
            seed=seed,
            balanced_algorithm_version=CATEGORY_BALANCED_ALGORITHM_VERSION,
            baseline_algorithm_version=RANDOM_ALGORITHM_VERSION,
            minimum_changed_fraction=config.category_balance.minimum_changed_fraction,
        )
        if not report["selection_simulation"]["minimum_changed_fraction_met"]:
            simulation = report["selection_simulation"]
            raise ValueError(
                "Recommended category-balanced selection changes "
                f"{simulation['changed_image_count']} images, below the required "
                f"{simulation['required_changed_count']} "
                f"({simulation['minimum_changed_fraction']:.2%})"
            )

    json_output = args.json_output or config.report_dir / "category-balance-analysis.json"
    markdown_output = (
        args.markdown_output or config.report_dir / "category-balance-analysis.md"
    )
    write_category_balance_analysis(
        report,
        json_output=json_output,
        markdown_output=markdown_output,
    )
    return {
        "analysis_version": report["analysis_version"],
        "json_report": str(json_output),
        "markdown_report": str(markdown_output),
        "recommendation": report["recommendation"],
        "recommended_scenario": {
            key: report["recommended_scenario"][key]
            for key in (
                "affected_category_count",
                "capped_category_count",
                "extra_target_mass",
                "extra_target_fraction_of_budget",
                "max_raw_boost",
            )
        },
        "selection_simulation": (
            {
                key: report["selection_simulation"][key]
                for key in (
                    "selected_count",
                    "seed",
                    "balanced_algorithm_version",
                    "baseline_algorithm_version",
                    "enforced_category_count",
                    "enforced_categories_meeting_target",
                    "total_enforced_target_shortfall",
                    "overlap_count",
                    "overlap_fraction",
                    "changed_image_count",
                    "changed_fraction",
                    "minimum_changed_fraction",
                    "required_changed_count",
                    "minimum_changed_fraction_met",
                    "random_mean_distinct_categories_per_image",
                    "selected_mean_distinct_categories_per_image",
                )
            }
            if "selection_simulation" in report
            else None
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        if args.command == "download":
            if ({"train", "all"} & set(args.components)) and not args.confirm_large_download:
                raise ValueError(
                    "train2017 is a large download; re-run with --confirm-large-download "
                    "after checking available disk space"
                )
            result = download_components(config, args.components)
        elif args.command == "prepare":
            result = _prepare(config, args.split)
        elif args.command == "select":
            result = _select(config, args)
        elif args.command == "analyze-category-balance":
            result = _analyze_category_balance(config, args)
        elif args.command == "validate":
            result = validate_project(config, check_files=args.check_files)
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0 if result["valid"] else 1
        else:  # pragma: no cover - argparse enforces the command choices.
            parser.error(f"Unknown command: {args.command}")
            return 2
    except (OSError, ValueError, RuntimeError, zipfile.BadZipFile) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0
