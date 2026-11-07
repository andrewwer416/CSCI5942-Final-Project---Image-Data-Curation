"""Stateless, reproducible caption sampling for training epochs."""

from __future__ import annotations

import hashlib
from typing import Sequence, TypeVar


T = TypeVar("T")


def caption_index(*, seed: int, epoch: int, image_id: int, caption_count: int) -> int:
    if caption_count <= 0:
        raise ValueError("caption_count must be positive")
    if epoch < 0:
        raise ValueError("epoch must be non-negative")
    payload = f"caption-v1:{seed}:{epoch}:{image_id}".encode("utf-8")
    value = int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")
    return value % caption_count


def sample_caption(
    captions: Sequence[T], *, seed: int, epoch: int, image_id: int
) -> T:
    return captions[
        caption_index(
            seed=seed,
            epoch=epoch,
            image_id=image_id,
            caption_count=len(captions),
        )
    ]
