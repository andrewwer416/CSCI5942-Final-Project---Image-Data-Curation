"""Image-text retrieval metrics.

``similarity`` is always shaped ``(num_images, num_captions)``, and
``caption_to_image[c]`` is the row index of the image that caption ``c`` describes.
Ranks are 1-indexed: rank 1 means a correct item was retrieved first.

Full val2017 is 5,000 x 25,000 scores and sorting it needs several GB, so call these
functions on CPU tensors unless the GPU has memory to spare.
"""

from __future__ import annotations

from typing import Sequence

import torch


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
) -> dict[str, float]:
    """Compute Recall@K in both retrieval directions."""

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
    return metrics
