"""Canonical on-disk data contract for COCO image-caption records."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterator, Any


@dataclass(frozen=True)
class Caption:
    caption_id: int
    text: str

    def to_dict(self) -> dict[str, Any]:
        return {"caption_id": self.caption_id, "text": self.text}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Caption":
        return cls(caption_id=int(value["caption_id"]), text=str(value["text"]))


@dataclass(frozen=True)
class ImageRecord:
    image_id: int
    split: str
    file_name: str
    relative_path: str
    width: int
    height: int
    captions: tuple[Caption, ...]
    category_ids: tuple[int, ...]
    object_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_id": self.image_id,
            "split": self.split,
            "file_name": self.file_name,
            "relative_path": self.relative_path,
            "width": self.width,
            "height": self.height,
            "captions": [caption.to_dict() for caption in self.captions],
            "category_ids": list(self.category_ids),
            "object_count": self.object_count,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ImageRecord":
        return cls(
            image_id=int(value["image_id"]),
            split=str(value["split"]),
            file_name=str(value["file_name"]),
            relative_path=str(value["relative_path"]),
            width=int(value["width"]),
            height=int(value["height"]),
            captions=tuple(Caption.from_dict(item) for item in value["captions"]),
            category_ids=tuple(int(item) for item in value.get("category_ids", [])),
            object_count=int(value.get("object_count", 0)),
        )


def read_manifest(path: Path) -> Iterator[ImageRecord]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield ImageRecord.from_dict(json.loads(line))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(f"Invalid manifest row {path}:{line_number}: {error}") from error


def metadata_path(manifest_path: Path) -> Path:
    return manifest_path.with_suffix(".meta.json")
