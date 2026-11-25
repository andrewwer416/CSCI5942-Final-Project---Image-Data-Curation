"""Image-text retrieval metrics.

``similarity`` is always shaped ``(num_images, num_captions)``, and
``caption_to_image[c]`` is the row index of the image that caption ``c`` describes.
Ranks are 1-indexed: rank 1 means a correct item was retrieved first.

Full val2017 is 5,000 x 25,000 scores and sorting it needs several GB, so call these
functions on CPU tensors unless the GPU has memory to spare.
"""

from __future__ import annotations

import re
from typing import Mapping, Sequence

import torch


# Basic color terms for the color-caption breakdown (docs/preprocessing-p4.md).
COLOR_WORDS = (
    "red orange yellow green blue purple pink brown black white gray grey silver gold "
    "golden tan beige"
).split()
_COLOR_PATTERN = re.compile(r"\b(?:" + "|".join(COLOR_WORDS) + r")\b")


def mentions_color(text: str) -> bool:
    """True if a caption names a basic color word (whole word, case-insensitive)."""

    return _COLOR_PATTERN.search(text.lower()) is not None


def recall_at_k(ranks: torch.Tensor, k: int) -> float:
    """Return the fraction of queries whose first correct item is within the top ``k``."""

    hits = ranks <= k
    return hits.float().mean().item()


def _first_correct_rank(is_correct: torch.Tensor, dim: int, query_name: str) -> torch.Tensor:
    # argmax would report position 0 for a query with no correct item at all, silently
    # counting it as a perfect hit, so reject that case first.
    if not is_correct.any(dim=dim).all():
        raise ValueError(f"Every {query_name} needs at least one correct match")
    return is_correct.int().argmax(dim=dim) + 1


def image_to_text_ranks(similarity: torch.Tensor, caption_to_image: torch.Tensor) -> torch.Tensor:
    """Return, per image, the rank of its best-ranked correct caption."""

    order = similarity.argsort(dim=1, descending=True, stable=True)
    image_indices = torch.arange(similarity.size(0), device=similarity.device).unsqueeze(1)
    is_correct = caption_to_image[order] == image_indices
    return _first_correct_rank(is_correct, dim=1, query_name="image")


def text_to_image_ranks(similarity: torch.Tensor, caption_to_image: torch.Tensor) -> torch.Tensor:
    """Return, per caption, the rank of the one image it describes."""

    order = similarity.argsort(dim=0, descending=True, stable=True)
    is_correct = order == caption_to_image.unsqueeze(0)
    return _first_correct_rank(is_correct, dim=0, query_name="caption")


def retrieval_metrics(
    similarity: torch.Tensor,
    caption_to_image: torch.Tensor,
    ks: Sequence[int] = (1, 5, 10),
    caption_subsets: Mapping[str, torch.Tensor] | None = None,
) -> dict[str, float]:
    """Compute Recall@K in both retrieval directions.

    ``caption_subsets`` maps a name to a boolean mask over captions. For each, text-to-image
    Recall@K is also reported over just those caption queries, as
    ``t2i_recall@{k}_{name}``, with the query count as ``{name}_caption_count``.
    """

    if similarity.size(1) != caption_to_image.size(0):
        raise ValueError(
            f"similarity has {similarity.size(1)} captions but caption_to_image has "
            f"{caption_to_image.size(0)} entries"
        )
    image_ranks = image_to_text_ranks(similarity, caption_to_image)
    caption_ranks = text_to_image_ranks(similarity, caption_to_image)
    metrics: dict[str, float] = {}
    for k in ks:
        metrics[f"i2t_recall@{k}"] = recall_at_k(image_ranks, k)
        metrics[f"t2i_recall@{k}"] = recall_at_k(caption_ranks, k)
    for name, mask in (caption_subsets or {}).items():
        if mask.shape != caption_ranks.shape or mask.dtype != torch.bool:
            raise ValueError(f"caption subset {name!r} must be a bool mask over captions")
        if not mask.any():
            raise ValueError(f"caption subset {name!r} is empty")
        metrics[f"{name}_caption_count"] = float(mask.sum())
        for k in ks:
            metrics[f"t2i_recall@{k}_{name}"] = recall_at_k(caption_ranks[mask], k)
    return metrics
