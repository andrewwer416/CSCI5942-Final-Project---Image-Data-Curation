"""Command-line interface for acquiring, indexing, selecting, and validating data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import zipfile

from vlm_data.coco import build_coco_manifest
from vlm_data.config import DataConfig, load_config
from vlm_data.download import download_components
from vlm_data.manifest import read_manifest
from vlm_data.selection import SELECTION_METHODS, write_selection
from vlm_data.validation import validate_project


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vlm-data",
        description="Prepare reproducible COCO manifests for the VLM retrieval study.",
    )
    parser.add_argument("--config", default="configs/data.toml", help="Path to data TOML config")
    subcommands = parser.add_subparsers(dest="command", required=True)

    download = subcommands.add_parser("download", help="Download and extract official COCO archives")
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
    method_file = args.method.replace("-", "_")
    output = args.output or config.selection_dir / f"{method_file}_seed{seed}_n{budget}.jsonl"
    return write_selection(
        records=read_manifest(config.train_manifest),
        output_path=output,
        source_manifest=config.train_manifest,
        method=args.method,
        seed=seed,
        budget=budget,
        schema_version=config.schema_version,
    )


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
