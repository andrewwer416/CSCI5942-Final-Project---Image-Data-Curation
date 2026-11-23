"""PyTorch datasets over the project's canonical and selection manifests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from vlm_data.caption_sampling import sample_caption
from vlm_data.manifest import ImageRecord
from vlm_train.tokenizer import Vocabulary


STANDARD_RGB = "standard_rgb"
ASPECT_RATIO_PRESERVING_RGB = "aspect_ratio_preserving_rgb"
PREPROCESSING_CHOICES = (STANDARD_RGB, ASPECT_RATIO_PRESERVING_RGB)
# Preprocessing configurations whose padding is hidden from the image encoder's attention.
MASKED_PREPROCESSING = frozenset({ASPECT_RATIO_PRESERVING_RGB})


def _to_signed_unit(image: Image.Image) -> torch.Tensor:
    """Fixed, unfitted scaling of 8-bit RGB to [-1, 1]; shared by every preprocessing arm."""

    pixels = torch.from_numpy(np.array(image)).permute(2, 0, 1).float()
    return pixels / 127.5 - 1.0


def load_standard_rgb(path: Path, image_size: int) -> torch.Tensor:
    """P1 preprocessing: resize directly to a square and scale pixels to [-1, 1].

    The fixed [-1, 1] scaling is not fitted to any data, which keeps it distinct from the
    separate color-and-contrast-normalization configuration (P4).
    """

    with Image.open(path) as image:
        resized = image.convert("RGB").resize((image_size, image_size), Image.Resampling.BILINEAR)
    return _to_signed_unit(resized)


@dataclass(frozen=True)
class Letterbox:
    """Where the resized image sits on the square canvas, in canvas pixels."""

    left: int
    top: int
    width: int
    height: int


def letterbox_geometry(width: int, height: int, image_size: int) -> Letterbox:
    """P2 geometry: scale the long side to exactly ``image_size`` and center the result.

    Images whose long side is shorter are upscaled, as in P1. The short side is rounded to
    the nearest pixel. When the padding splits unevenly, the extra pixel goes to the bottom
    or right. Content is not snapped to the patch grid, which would distort the aspect ratio.
    """

    if width <= 0 or height <= 0 or image_size <= 0:
        raise ValueError("width, height, and image_size must be positive")
    scale = image_size / max(width, height)
    new_width = image_size if width >= height else max(1, min(image_size, round(width * scale)))
    new_height = image_size if height >= width else max(1, min(image_size, round(height * scale)))
    return Letterbox(
        left=(image_size - new_width) // 2,
        top=(image_size - new_height) // 2,
        width=new_width,
        height=new_height,
    )


def padding_patch_mask(box: Letterbox, image_size: int, patch_size: int) -> torch.Tensor:
    """Row-major boolean mask over patches; True where a patch holds no image pixel.

    A patch on the content border holds real pixels and stays visible. The mask comes from
    the resize geometry, never from pixel values, so gray image content is never masked.
    """

    if image_size % patch_size != 0:
        raise ValueError("image_size must be a multiple of patch_size")
    starts = torch.arange(0, image_size, patch_size)
    ends = starts + patch_size
    padded_rows = (ends <= box.top) | (starts >= box.top + box.height)
    padded_columns = (ends <= box.left) | (starts >= box.left + box.width)
    return (padded_rows[:, None] | padded_columns[None, :]).flatten()


def load_aspect_preserving_rgb(
    path: Path, image_size: int, patch_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """P2 preprocessing: letterbox onto a mid-gray square canvas and mark padded patches.

    Uses the same PIL bilinear resize and [-1, 1] scaling as P1, so aspect ratio is the
    only image difference. Padding is 0 after scaling (8-bit mid-gray, 127.5), which needs
    no fitted statistics. Returns the image and the padding mask for the image encoder.
    """

    with Image.open(path) as image:
        rgb = image.convert("RGB")
        box = letterbox_geometry(rgb.width, rgb.height, image_size)
        resized = rgb.resize((box.width, box.height), Image.Resampling.BILINEAR)
    canvas = torch.zeros(3, image_size, image_size)
    canvas[:, box.top : box.top + box.height, box.left : box.left + box.width] = _to_signed_unit(
        resized
    )
    return canvas, padding_patch_mask(box, image_size, patch_size)


def load_image(
    path: Path, *, preprocessing: str, image_size: int, patch_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(image, padding_patch_mask)``; the mask is all False for unpadded arms."""

    if preprocessing == STANDARD_RGB:
        mask = torch.zeros((image_size // patch_size) ** 2, dtype=torch.bool)
        return load_standard_rgb(path, image_size), mask
    if preprocessing == ASPECT_RATIO_PRESERVING_RGB:
        return load_aspect_preserving_rgb(path, image_size, patch_size)
    raise ValueError(f"Unknown preprocessing {preprocessing!r}; expected {PREPROCESSING_CHOICES}")


class TrainRetrievalDataset(Dataset):
    """Yields one image and one deterministically sampled caption per epoch."""

    def __init__(
        self,
        records: Sequence[ImageRecord],
        *,
        image_root: Path,
        vocabulary: Vocabulary,
        seed: int,
        image_size: int,
        max_length: int = 77,
        preprocessing: str = STANDARD_RGB,
        patch_size: int = 16,
    ) -> None:
        if preprocessing not in PREPROCESSING_CHOICES:
            raise ValueError(f"Unknown preprocessing {preprocessing!r}")
        self.records = records
        self.image_root = image_root
        self.vocabulary = vocabulary
        self.seed = seed
        self.image_size = image_size
        self.max_length = max_length
        self.preprocessing = preprocessing
        self.patch_size = patch_size
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        # DataLoader workers copy this object at the start of each epoch, so persistent
        # workers must stay off or they would keep sampling the first epoch's captions.
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        record = self.records[index]
        caption = sample_caption(
            record.captions, seed=self.seed, epoch=self.epoch, image_id=record.image_id
        )
        image, mask = load_image(
            self.image_root / record.relative_path,
            preprocessing=self.preprocessing,
            image_size=self.image_size,
            patch_size=self.patch_size,
        )
        tokens = self.vocabulary.encode(caption.text, self.max_length)
        return image, mask, torch.tensor(tokens, dtype=torch.long)


class EvalRetrievalDataset(Dataset):
    """Yields one image and all of its reference captions.

    Evaluation uses the same deterministic preprocessing as training for each arm.
    """

    def __init__(
        self,
        records: Sequence[ImageRecord],
        *,
        image_root: Path,
        vocabulary: Vocabulary,
        image_size: int,
        max_length: int = 77,
        preprocessing: str = STANDARD_RGB,
        patch_size: int = 16,
    ) -> None:
        if preprocessing not in PREPROCESSING_CHOICES:
            raise ValueError(f"Unknown preprocessing {preprocessing!r}")
        self.records = records
        self.image_root = image_root
        self.vocabulary = vocabulary
        self.image_size = image_size
        self.max_length = max_length
        self.preprocessing = preprocessing
        self.patch_size = patch_size

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        record = self.records[index]
        image, mask = load_image(
            self.image_root / record.relative_path,
            preprocessing=self.preprocessing,
            image_size=self.image_size,
            patch_size=self.patch_size,
        )
        tokens = [self.vocabulary.encode(item.text, self.max_length) for item in record.captions]
        return image, mask, torch.tensor(tokens, dtype=torch.long)
