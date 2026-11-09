"""PyTorch datasets over the project's canonical and selection manifests."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from vlm_data.caption_sampling import sample_caption
from vlm_data.manifest import ImageRecord
from vlm_train.tokenizer import Vocabulary


def load_standard_rgb(path: Path, image_size: int) -> torch.Tensor:
    """P1 preprocessing: resize directly to a square and scale pixels to [-1, 1].

    The fixed [-1, 1] scaling is not fitted to any data, which keeps it distinct from the
    separate color-and-contrast-normalization configuration (P4).
    """

    with Image.open(path) as image:
        resized = image.convert("RGB").resize((image_size, image_size), Image.Resampling.BILINEAR)
    pixels = torch.from_numpy(np.array(resized)).permute(2, 0, 1).float()
    return pixels / 127.5 - 1.0


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
    ) -> None:
        self.records = records
        self.image_root = image_root
        self.vocabulary = vocabulary
        self.seed = seed
        self.image_size = image_size
        self.max_length = max_length
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        # DataLoader workers copy this object at the start of each epoch, so persistent
        # workers must stay off or they would keep sampling the first epoch's captions.
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        record = self.records[index]
        caption = sample_caption(
            record.captions, seed=self.seed, epoch=self.epoch, image_id=record.image_id
        )
        image = load_standard_rgb(self.image_root / record.relative_path, self.image_size)
        tokens = self.vocabulary.encode(caption.text, self.max_length)
        return image, torch.tensor(tokens, dtype=torch.long)


class EvalRetrievalDataset(Dataset):
    """Yields one image and all of its reference captions."""

    def __init__(
        self,
        records: Sequence[ImageRecord],
        *,
        image_root: Path,
        vocabulary: Vocabulary,
        image_size: int,
        max_length: int = 77,
    ) -> None:
        self.records = records
        self.image_root = image_root
        self.vocabulary = vocabulary
        self.image_size = image_size
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        record = self.records[index]
        image = load_standard_rgb(self.image_root / record.relative_path, self.image_size)
        tokens = [self.vocabulary.encode(item.text, self.max_length) for item in record.captions]
        return image, torch.tensor(tokens, dtype=torch.long)
