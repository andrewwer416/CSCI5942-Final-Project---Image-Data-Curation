"""Integrity checks for canonical and selected COCO manifests."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any

from vlm_data.config import DataConfig
from vlm_data.io import sha256_file, write_json_atomic
from vlm_data.manifest import ImageRecord, metadata_path, read_manifest


def _issue(code: str, message: str) -> dict[str, str]:
    return {"code": code, "message": message}


def validate_manifest(
    path: Path,
    *,
    expected_split: str,
    expected_captions: int,
    expected_count: int | None = None,
    raw_root: Path | None = None,
    check_files: bool = False,
    source_records: dict[int, ImageRecord] | None = None,
    source_manifest_sha256: str | None = None,
) -> tuple[dict[str, Any], list[ImageRecord]]:
    errors: list[dict[str, str]] = []
    if not path.is_file():
        return (
            {
                "path": str(path),
                "valid": False,
                "record_count": 0,
                "errors": [_issue("missing_manifest", f"Manifest does not exist: {path}")],
            },
            [],
        )

    try:
        records = list(read_manifest(path))
    except ValueError as error:
        return (
            {
                "path": str(path),
                "valid": False,
                "record_count": 0,
                "errors": [_issue("invalid_jsonl", str(error))],
            },
            [],
        )

    image_ids: set[int] = set()
    file_names: set[str] = set()
    caption_ids: set[int] = set()
    caption_distribution: Counter[int] = Counter()
    category_coverage: Counter[int] = Counter()

    for record in records:
        if record.image_id in image_ids:
            errors.append(_issue("duplicate_image_id", f"Duplicate image id {record.image_id}"))
        image_ids.add(record.image_id)
        if record.file_name in file_names:
            errors.append(_issue("duplicate_file_name", f"Duplicate file name {record.file_name}"))
        file_names.add(record.file_name)
        if record.split != expected_split:
            errors.append(
                _issue(
                    "wrong_split",
                    f"Image {record.image_id} has split {record.split!r}; expected {expected_split!r}",
                )
            )
        if record.width <= 0 or record.height <= 0:
            errors.append(_issue("invalid_dimensions", f"Image {record.image_id} has invalid dimensions"))
        relative_path = Path(record.relative_path)
        expected_relative_path = Path(expected_split) / record.file_name
        if (
            relative_path.is_absolute()
            or ".." in relative_path.parts
            or relative_path != expected_relative_path
        ):
            errors.append(
                _issue(
                    "unsafe_relative_path",
                    f"Image {record.image_id} has unexpected path {record.relative_path!r}; "
                    f"expected {expected_relative_path.as_posix()!r}",
                )
            )
        caption_distribution[len(record.captions)] += 1
        if len(record.captions) != expected_captions:
            errors.append(
                _issue(
                    "caption_count",
                    f"Image {record.image_id} has {len(record.captions)} captions; "
                    f"expected {expected_captions}",
                )
            )
        for caption in record.captions:
            if caption.caption_id in caption_ids:
                errors.append(
                    _issue("duplicate_caption_id", f"Duplicate caption id {caption.caption_id}")
                )
            caption_ids.add(caption.caption_id)
            if not caption.text:
                errors.append(
                    _issue("empty_caption", f"Caption {caption.caption_id} has no text")
                )
        category_coverage.update(record.category_ids)
        if source_records is not None:
            source_record = source_records.get(record.image_id)
            if source_record is None:
                errors.append(
                    _issue(
                        "not_in_source",
                        f"Image {record.image_id} does not occur in the canonical training manifest",
                    )
                )
            elif record != source_record:
                errors.append(
                    _issue(
                        "source_record_mismatch",
                        f"Image {record.image_id} differs from its canonical training record",
                    )
                )
        if check_files:
            if raw_root is None:
                raise ValueError("raw_root is required when check_files=True")
            image_path = raw_root / record.relative_path
            if not image_path.is_file():
                errors.append(_issue("missing_image", f"Image file does not exist: {image_path}"))

    if expected_count is not None and len(records) != expected_count:
        errors.append(
            _issue(
                "record_count",
                f"Manifest contains {len(records)} records; expected {expected_count}",
            )
        )

    actual_sha256 = sha256_file(path)
    meta_path = metadata_path(path)
    if not meta_path.is_file():
        errors.append(_issue("missing_metadata", f"Metadata does not exist: {meta_path}"))
    else:
        try:
            with meta_path.open("r", encoding="utf-8") as handle:
                metadata = json.load(handle)
            recorded_sha256 = metadata.get("manifest_sha256")
            if recorded_sha256 != actual_sha256:
                errors.append(
                    _issue(
                        "manifest_hash_mismatch",
                        f"Metadata hash {recorded_sha256!r} does not match {actual_sha256!r}",
                    )
                )
            if source_manifest_sha256 is not None:
                recorded_source_sha256 = metadata.get("source_manifest_sha256")
                if recorded_source_sha256 != source_manifest_sha256:
                    errors.append(
                        _issue(
                            "source_hash_mismatch",
                            f"Recorded source hash {recorded_source_sha256!r} does not match "
                            f"canonical manifest hash {source_manifest_sha256!r}",
                        )
                    )
        except (OSError, ValueError, TypeError) as error:
            errors.append(_issue("invalid_metadata", f"Could not read {meta_path}: {error}"))

    report: dict[str, Any] = {
        "path": str(path),
        "valid": not errors,
        "record_count": len(records),
        "unique_image_count": len(image_ids),
        "caption_count_distribution": {
            str(count): images for count, images in sorted(caption_distribution.items())
        },
        "category_coverage": {
            str(category): count for category, count in sorted(category_coverage.items())
        },
        "manifest_sha256": actual_sha256,
        "errors": errors,
    }
    return report, records


def validate_project(config: DataConfig, *, check_files: bool = False) -> dict[str, Any]:
    train_report, train_records = validate_manifest(
        config.train_manifest,
        expected_split="train2017",
        expected_captions=config.captions_per_image,
        expected_count=config.expected_train_images,
        raw_root=config.raw_root,
        check_files=check_files,
    )
    val_report, val_records = validate_manifest(
        config.val_manifest,
        expected_split="val2017",
        expected_captions=config.captions_per_image,
        expected_count=config.expected_val_images,
        raw_root=config.raw_root,
        check_files=check_files,
    )

    train_by_id = {record.image_id: record for record in train_records}
    train_ids = set(train_by_id)
    val_ids = {record.image_id for record in val_records}
    project_errors: list[dict[str, str]] = []
    overlap = train_ids & val_ids
    if overlap:
        project_errors.append(
            _issue(
                "split_leakage",
                f"Training and validation manifests share {len(overlap)} image ids",
            )
        )

    selections: list[dict[str, Any]] = []
    if config.selection_dir.exists():
        for selection_path in sorted(config.selection_dir.glob("*.jsonl")):
            selection_report, _ = validate_manifest(
                selection_path,
                expected_split="train2017",
                expected_captions=config.captions_per_image,
                expected_count=config.train_image_budget,
                raw_root=config.raw_root,
                check_files=check_files,
                source_records=train_by_id,
                source_manifest_sha256=(
                    train_report.get("manifest_sha256") if train_report["valid"] else None
                ),
            )
            selections.append(selection_report)

    warnings: list[dict[str, str]] = []
    if not config.baseline_manifest().is_file():
        warnings.append(
            _issue(
                "baseline_not_generated",
                "The shared random 15,000-image baseline has not been generated yet",
            )
        )

    valid = (
        train_report["valid"]
        and val_report["valid"]
        and all(report["valid"] for report in selections)
        and not project_errors
    )
    report = {
        "schema_version": config.schema_version,
        "valid": valid,
        "check_files": check_files,
        "canonical_manifests": [train_report, val_report],
        "selection_manifests": selections,
        "errors": project_errors,
        "warnings": warnings,
    }
    config.report_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(config.report_dir / "validation.json", report)
    return report
