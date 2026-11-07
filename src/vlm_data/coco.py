"""Build deterministic manifests from official COCO annotation files."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any

from vlm_data.io import sha256_file, write_json_atomic, write_jsonl_atomic
from vlm_data.manifest import Caption, ImageRecord, metadata_path


def _load_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required COCO annotation file is missing: {path}")
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def build_coco_manifest(
    *,
    split: str,
    image_directory_name: str,
    captions_path: Path,
    instances_path: Path,
    output_path: Path,
    schema_version: int = 1,
    captions_per_image: int = 5,
) -> dict[str, Any]:
    """Join captions, image metadata, and object categories into one JSONL file."""

    if captions_per_image <= 0:
        raise ValueError("captions_per_image must be positive")
    captions_document = _load_object(captions_path)

    images: dict[int, dict[str, Any]] = {}
    for image in captions_document.get("images", []):
        image_id = int(image["id"])
        if image_id in images:
            raise ValueError(f"Duplicate image id {image_id} in {captions_path}")
        images[image_id] = image

    captions_by_image: dict[int, list[Caption]] = defaultdict(list)
    caption_ids: set[int] = set()
    for annotation in captions_document.get("annotations", []):
        image_id = int(annotation["image_id"])
        caption_id = int(annotation["id"])
        if image_id not in images:
            raise ValueError(f"Caption {caption_id} refers to unknown image {image_id}")
        if caption_id in caption_ids:
            raise ValueError(f"Duplicate caption id {caption_id} in {captions_path}")
        caption_ids.add(caption_id)
        captions_by_image[image_id].append(
            Caption(caption_id=caption_id, text=str(annotation["caption"]).strip())
        )
    del captions_document

    raw_caption_count_distribution = Counter(
        len(captions_by_image[image_id]) for image_id in images
    )
    too_few = [
        image_id
        for image_id in images
        if len(captions_by_image[image_id]) < captions_per_image
    ]
    if too_few:
        preview = ", ".join(str(image_id) for image_id in too_few[:5])
        raise ValueError(
            f"{len(too_few)} images have fewer than {captions_per_image} captions "
            f"(first ids: {preview})"
        )
    # Official COCO annotations contain a small number of images with more than
    # five captions. The study contract is exactly five, so retain the five
    # lowest annotation IDs deterministically and record the policy in metadata.
    for image_id, captions in captions_by_image.items():
        captions_by_image[image_id] = sorted(
            captions, key=lambda item: item.caption_id
        )[:captions_per_image]

    # Load the much larger instances document only after releasing the raw
    # captions JSON object. This keeps real-COCO peak memory substantially lower.
    instances_document = _load_object(instances_path)
    category_ids_by_image: dict[int, set[int]] = defaultdict(set)
    object_counts: Counter[int] = Counter()
    for annotation in instances_document.get("annotations", []):
        image_id = int(annotation["image_id"])
        if image_id not in images:
            raise ValueError(f"Instance annotation refers to unknown image {image_id}")
        category_ids_by_image[image_id].add(int(annotation["category_id"]))
        object_counts[image_id] += 1

    categories = {
        int(category["id"]): {
            "name": str(category["name"]),
            "supercategory": str(category.get("supercategory", "")),
        }
        for category in instances_document.get("categories", [])
    }
    del instances_document

    caption_count_distribution: Counter[int] = Counter()
    def rows():
        for image_id in sorted(images):
            image = images[image_id]
            captions = tuple(captions_by_image[image_id])
            caption_count_distribution[len(captions)] += 1
            file_name = str(image["file_name"])
            yield ImageRecord(
                image_id=image_id,
                split=split,
                file_name=file_name,
                relative_path=f"{image_directory_name}/{file_name}",
                width=int(image["width"]),
                height=int(image["height"]),
                captions=captions,
                category_ids=tuple(sorted(category_ids_by_image[image_id])),
                object_count=object_counts[image_id],
            ).to_dict()

    row_count, manifest_sha256 = write_jsonl_atomic(output_path, rows())
    metadata: dict[str, Any] = {
        "schema_version": schema_version,
        "kind": "canonical_coco_manifest",
        "split": split,
        "record_count": row_count,
        "caption_count_distribution": {
            str(count): images_with_count
            for count, images_with_count in sorted(caption_count_distribution.items())
        },
        "raw_caption_count_distribution": {
            str(count): images_with_count
            for count, images_with_count in sorted(raw_caption_count_distribution.items())
        },
        "caption_policy": {
            "captions_per_image": captions_per_image,
            "extra_caption_policy": "retain_lowest_caption_ids",
            "missing_caption_policy": "fail",
        },
        "category_count": len(categories),
        "categories": {str(key): value for key, value in sorted(categories.items())},
        "manifest_sha256": manifest_sha256,
        "sources": {
            "captions": {
                "file_name": captions_path.name,
                "sha256": sha256_file(captions_path),
            },
            "instances": {
                "file_name": instances_path.name,
                "sha256": sha256_file(instances_path),
            },
        },
    }
    write_json_atomic(metadata_path(output_path), metadata)
    return metadata
