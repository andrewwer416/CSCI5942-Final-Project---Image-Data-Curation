"""The from-scratch vision-language model assumed by the proposal's compute estimate."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F


# Learnable temperature, starting at CLIP's 0.07 and capped at a scale of 100.
INITIAL_LOGIT_SCALE = math.log(1 / 0.07)
MAX_LOGIT_SCALE = math.log(100)


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    pad_id: int
    eos_id: int
    image_size: int = 640
    patch_size: int = 16
    image_dim: int = 768
    image_layers: int = 12
    image_heads: int = 12
    image_feedforward: int = 3072
    max_length: int = 77
    text_dim: int = 512
    text_layers: int = 12
    text_heads: int = 8
    text_feedforward: int = 2048
    embed_dim: int = 512


def _transformer(dim: int, heads: int, feedforward: int, layers: int) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(
        d_model=dim,
        nhead=heads,
        dim_feedforward=feedforward,
        dropout=0.0,
        activation="gelu",
        batch_first=True,
        norm_first=True,
    )
    return nn.TransformerEncoder(layer, num_layers=layers, enable_nested_tensor=False)


class ImageEncoder(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.image_size % config.patch_size != 0:
            raise ValueError("image_size must be a multiple of patch_size")
        patches_per_side = config.image_size // config.patch_size
        self.patch_embedding = nn.Conv2d(
            3, config.image_dim, kernel_size=config.patch_size, stride=config.patch_size
        )
        self.class_token = nn.Parameter(torch.zeros(1, 1, config.image_dim))
        self.position_embedding = nn.Parameter(
            torch.zeros(1, patches_per_side**2 + 1, config.image_dim)
        )
        self.transformer = _transformer(
            config.image_dim, config.image_heads, config.image_feedforward, config.image_layers
        )
        self.norm = nn.LayerNorm(config.image_dim)
        self.projection = nn.Linear(config.image_dim, config.embed_dim)
        nn.init.trunc_normal_(self.class_token, std=0.02)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        patches = self.patch_embedding(images).flatten(2).transpose(1, 2)
        class_tokens = self.class_token.expand(patches.size(0), -1, -1)
        tokens = torch.cat([class_tokens, patches], dim=1) + self.position_embedding
        encoded = self.transformer(tokens)
        return self.projection(self.norm(encoded[:, 0]))


class TextEncoder(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.pad_id = config.pad_id
        self.eos_id = config.eos_id
        self.token_embedding = nn.Embedding(
            config.vocab_size, config.text_dim, padding_idx=config.pad_id
        )
        self.position_embedding = nn.Parameter(
            torch.zeros(1, config.max_length, config.text_dim)
        )
        self.transformer = _transformer(
            config.text_dim, config.text_heads, config.text_feedforward, config.text_layers
        )
        self.norm = nn.LayerNorm(config.text_dim)
        self.projection = nn.Linear(config.text_dim, config.embed_dim)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        embedded = self.token_embedding(tokens) + self.position_embedding[:, : tokens.size(1)]
        encoded = self.transformer(embedded, src_key_padding_mask=tokens == self.pad_id)
        # Like CLIP, summarize each caption with the output at its <eos> position.
        eos_positions = (tokens == self.eos_id).int().argmax(dim=1)
        pooled = encoded[torch.arange(tokens.size(0), device=tokens.device), eos_positions]
        return self.projection(self.norm(pooled))


class RetrievalModel(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.image_encoder = ImageEncoder(config)
        self.text_encoder = TextEncoder(config)
        self.logit_scale = nn.Parameter(torch.tensor(INITIAL_LOGIT_SCALE))

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.image_encoder(images), dim=-1)

    def encode_text(self, tokens: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.text_encoder(tokens), dim=-1)

    def forward(self, images: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        """Return ``(batch, batch)`` logits: entry ``[i, j]`` scores image i against caption j."""

        image_embeddings = self.encode_image(images)
        text_embeddings = self.encode_text(tokens)
        return self.logit_scale.exp() * image_embeddings @ text_embeddings.t()

    def clamp_logit_scale(self) -> None:
        with torch.no_grad():
            self.logit_scale.clamp_(0.0, MAX_LOGIT_SCALE)


def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    """Symmetric cross-entropy: each image's match is the caption at the same batch index."""

    labels = torch.arange(logits.size(0), device=logits.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)) / 2
