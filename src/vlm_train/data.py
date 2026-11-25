"""PyTorch datasets over the project's canonical and selection manifests."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
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
RANDOM_RESIZED_CROP = "random_resized_crop"
COLOR_CONTRAST_NORMALIZATION = "color_contrast_normalization"
RGB_PLUS_CLIP_FEATURES = "rgb_plus_clip_features"
PREPROCESSING_CHOICES = (
    STANDARD_RGB,
    ASPECT_RATIO_PRESERVING_RGB,
    RANDOM_RESIZED_CROP,
    COLOR_CONTRAST_NORMALIZATION,
    RGB_PLUS_CLIP_FEATURES,
)
# Configurations that add the frozen CLIP image vector (P5); their RGB path is P1's.
CLIP_PREPROCESSING = frozenset({RGB_PLUS_CLIP_FEATURES})
# Preprocessing configurations whose padding is hidden from the image encoder's attention.
MASKED_PREPROCESSING = frozenset({ASPECT_RATIO_PRESERVING_RGB})
# Deterministic transform used on val2017 for each arm; training-only randomness is dropped.
EVAL_PREPROCESSING = {RANDOM_RESIZED_CROP: STANDARD_RGB}

# P3 random resized crop (docs/preprocessing-p3.md).
CROP_ALGORITHM_VERSION = "crop-v1"
CROP_MIN_SCALE = 0.5
CROP_RATIO_JITTER = (0.8, 1.25)  # multiplies the image's own width/height ratio
CROP_ATTEMPTS = 10

# P4 per-image appearance normalization (docs/preprocessing-p4.md). Nothing is fitted.
APPEARANCE_ALGORITHM_VERSION = "sog6-luma-stretch-v1"
SHADES_OF_GRAY_P = 6
WHITE_BALANCE_GAIN_CAP = 2.0
CONTRAST_PERCENTILES = (1.0, 99.0)
CONTRAST_GAIN_CAP = 2.0
LUMA_WEIGHTS = (0.299, 0.587, 0.114)  # ITU-R BT.601, as in JPEG's YCbCr


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


def normalize_appearance(rgb: np.ndarray) -> np.ndarray:
    """P4: per-image white balance, then a global brightness-contrast stretch.

    ``rgb`` is an (H, W, 3) uint8 array of stored pixels; returns uint8 of the same shape.

    1. Shades-of-Gray white balance (Minkowski p = 6): each channel is scaled so its
       p-norm mean equals the mean of the three, leaving overall level unchanged. Each
       gain is capped to [1/2, 2]. Grayscale images get gains of exactly 1.
    2. Brightness (BT.601 luma) is stretched linearly so its 1st and 99th percentiles map
       to 0 and 1, with the gain capped at 2x; under the cap the stretch is centered on
       the percentile midpoint. The same luma change is added to all three channels, so
       chroma (Cb, Cr) is preserved exactly except where values clip at 0 or 1.
    """

    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("expected an (H, W, 3) uint8 RGB array")
    pixels = rgb.astype(np.float64) / 255.0
    channel_norms = np.mean(pixels.reshape(-1, 3) ** SHADES_OF_GRAY_P, axis=0) ** (
        1 / SHADES_OF_GRAY_P
    )
    if np.all(channel_norms > 0):
        gains = channel_norms.mean() / channel_norms
    else:
        gains = np.ones(3)  # an all-black channel gives no color-cast estimate
    gains = np.clip(gains, 1 / WHITE_BALANCE_GAIN_CAP, WHITE_BALANCE_GAIN_CAP)
    balanced = np.clip(pixels * gains, 0.0, 1.0)

    luma = balanced @ np.asarray(LUMA_WEIGHTS)
    low, high = np.percentile(luma, CONTRAST_PERCENTILES)
    spread = high - low
    gain = CONTRAST_GAIN_CAP if spread <= 1 / CONTRAST_GAIN_CAP else 1.0 / spread
    stretched = (luma - (low + high) / 2) * gain + 0.5
    output = np.clip(balanced + (stretched - luma)[..., None], 0.0, 1.0)
    return np.rint(output * 255.0).astype(np.uint8)


def load_color_contrast_normalized(path: Path, image_size: int) -> torch.Tensor:
    """P4 preprocessing: normalize appearance on stored pixels, then resize like P1.

    Uses P1's PIL bilinear resize and fixed [-1, 1] scaling, so the appearance change is
    the only difference. It is deterministic and unfitted, so val2017 uses it unchanged.
    """

    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"))
    normalized = Image.fromarray(normalize_appearance(rgb))
    resized = normalized.resize((image_size, image_size), Image.Resampling.BILINEAR)
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


@dataclass(frozen=True)
class CropBox:
    """A crop of the stored image, in source pixels."""

    left: int
    top: int
    width: int
    height: int


def _uniforms(*, seed: int, epoch: int, image_id: int, count: int) -> list[float]:
    """Stateless uniforms in [0, 1) from ``(seed, epoch, image_id)``, like caption sampling.

    They do not depend on worker count, global RNG state, or the model seed, so every run
    and replicate sees the same crop for an image in a given epoch.
    """

    values = []
    for index in range(count):
        payload = f"{CROP_ALGORITHM_VERSION}:{seed}:{epoch}:{image_id}:{index}".encode("utf-8")
        digest = hashlib.blake2b(payload, digest_size=8).digest()
        values.append(int.from_bytes(digest, "big") / 2**64)
    return values


def sample_crop_box(
    width: int,
    height: int,
    *,
    seed: int,
    epoch: int,
    image_id: int,
    min_scale: float = CROP_MIN_SCALE,
    ratio_jitter: tuple[float, float] = CROP_RATIO_JITTER,
    attempts: int = CROP_ATTEMPTS,
) -> CropBox:
    """P3 crop: area uniform in [min_scale, 1] of the image, aspect ratio log-uniform in
    ``ratio_jitter`` times the image's own ratio, position uniform.

    Sampling relative to the image's ratio means the later stretch to a square matches P1's,
    so cropping is the variable P3 changes. If no attempt fits, the full image is used.
    """

    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    if not 0 < min_scale <= 1 or not 0 < ratio_jitter[0] <= ratio_jitter[1]:
        raise ValueError("invalid crop range")
    draws = _uniforms(seed=seed, epoch=epoch, image_id=image_id, count=4 * attempts)
    area = width * height
    log_low, log_high = math.log(ratio_jitter[0]), math.log(ratio_jitter[1])
    for attempt in range(attempts):
        u_area, u_ratio, u_left, u_top = draws[4 * attempt : 4 * attempt + 4]
        target_area = area * (min_scale + (1 - min_scale) * u_area)
        ratio = (width / height) * math.exp(log_low + (log_high - log_low) * u_ratio)
        crop_width = round(math.sqrt(target_area * ratio))
        crop_height = round(math.sqrt(target_area / ratio))
        if 0 < crop_width <= width and 0 < crop_height <= height:
            return CropBox(
                left=int(u_left * (width - crop_width + 1)),
                top=int(u_top * (height - crop_height + 1)),
                width=crop_width,
                height=crop_height,
            )
    return CropBox(0, 0, width, height)


def load_random_resized_crop(
    path: Path, image_size: int, *, seed: int, epoch: int, image_id: int
) -> torch.Tensor:
    """P3 training preprocessing: crop the stored image, then resize like P1.

    The crop is taken from stored pixels before any resizing, then PIL bilinear stretches it
    to a square and the fixed [-1, 1] scaling applies. Evaluation uses P1 unchanged.
    """

    with Image.open(path) as image:
        rgb = image.convert("RGB")
        box = sample_crop_box(rgb.width, rgb.height, seed=seed, epoch=epoch, image_id=image_id)
        cropped = rgb.crop((box.left, box.top, box.left + box.width, box.top + box.height))
        resized = cropped.resize((image_size, image_size), Image.Resampling.BILINEAR)
    return _to_signed_unit(resized)


@dataclass(frozen=True)
class CropKey:
    """What a stochastic training transform is keyed on."""

    seed: int
    epoch: int
    image_id: int


def load_image(
    path: Path,
    *,
    preprocessing: str,
    image_size: int,
    patch_size: int,
    crop_key: CropKey | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(image, padding_patch_mask)``; the mask is all False for unpadded arms."""

    empty_mask = torch.zeros((image_size // patch_size) ** 2, dtype=torch.bool)
    if preprocessing in (STANDARD_RGB, RGB_PLUS_CLIP_FEATURES):
        return load_standard_rgb(path, image_size), empty_mask
    if preprocessing == ASPECT_RATIO_PRESERVING_RGB:
        return load_aspect_preserving_rgb(path, image_size, patch_size)
    if preprocessing == RANDOM_RESIZED_CROP:
        if crop_key is None:
            raise ValueError("random_resized_crop is training-only and needs a crop_key")
        image = load_random_resized_crop(
            path,
            image_size,
            seed=crop_key.seed,
            epoch=crop_key.epoch,
            image_id=crop_key.image_id,
        )
        return image, empty_mask
    if preprocessing == COLOR_CONTRAST_NORMALIZATION:
        return load_color_contrast_normalized(path, image_size), empty_mask
    raise ValueError(f"Unknown preprocessing {preprocessing!r}; expected {PREPROCESSING_CHOICES}")


class ClipFeatureTable:
    """Frozen CLIP image vectors looked up by COCO image ID (P5).

    Wraps a verified embedding cache (``vlm_data.visual_embeddings``): float32, unit length,
    strictly increasing image IDs. ``deranged`` returns a copy in which every image gets a
    different image's vector, for the shuffled-CLIP evaluation diagnostic.
    """

    def __init__(self, image_ids: np.ndarray, vectors: np.ndarray) -> None:
        image_ids = np.asarray(image_ids, dtype=np.int64)
        if vectors.ndim != 2 or len(vectors) != len(image_ids):
            raise ValueError("vectors must be (len(image_ids), dimension)")
        if len(image_ids) > 1 and not np.all(image_ids[1:] > image_ids[:-1]):
            raise ValueError("image_ids must be strictly increasing")
        self.image_ids = image_ids
        self.vectors = vectors
        self.dimension = int(vectors.shape[1])

    @classmethod
    def from_artifact(cls, metadata_path: Path, *, source_manifest: Path) -> "ClipFeatureTable":
        from vlm_data.visual_embeddings import load_embedding_artifact

        _metadata, vectors, image_ids = load_embedding_artifact(
            metadata_path, source_manifest=source_manifest, mmap_mode=None
        )
        return cls(image_ids, vectors)

    def __getitem__(self, image_id: int) -> torch.Tensor:
        index = int(np.searchsorted(self.image_ids, image_id))
        if index >= len(self.image_ids) or self.image_ids[index] != image_id:
            raise KeyError(f"No CLIP vector for image {image_id}")
        return torch.from_numpy(np.array(self.vectors[index], dtype=np.float32))

    def deranged(self, image_ids: Sequence[int], seed: int) -> "ClipFeatureTable":
        """Reassign vectors among ``image_ids`` so that no image keeps its own vector."""

        ids = np.sort(np.asarray(list(image_ids), dtype=np.int64))
        if len(ids) < 2:
            raise ValueError("a derangement needs at least two images")
        order = np.random.default_rng(seed).permutation(len(ids))
        source = np.empty(len(ids), dtype=np.int64)
        source[order] = order[np.r_[1 : len(ids), 0]]  # each image takes the next one's vector
        vectors = np.stack([self[int(ids[index])].numpy() for index in source])
        return ClipFeatureTable(ids, vectors)


def _no_clip() -> torch.Tensor:
    return torch.zeros(0)


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
        clip_features: ClipFeatureTable | None = None,
    ) -> None:
        if preprocessing not in PREPROCESSING_CHOICES:
            raise ValueError(f"Unknown preprocessing {preprocessing!r}")
        if (preprocessing in CLIP_PREPROCESSING) != (clip_features is not None):
            raise ValueError("clip_features must be given exactly for CLIP-feature preprocessing")
        self.records = records
        self.image_root = image_root
        self.vocabulary = vocabulary
        self.clip_features = clip_features
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

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(image, padding_mask, clip_vector, tokens)``; the CLIP vector is empty
        (shape ``(0,)``) outside P5."""

        record = self.records[index]
        caption = sample_caption(
            record.captions, seed=self.seed, epoch=self.epoch, image_id=record.image_id
        )
        image, mask = load_image(
            self.image_root / record.relative_path,
            preprocessing=self.preprocessing,
            image_size=self.image_size,
            patch_size=self.patch_size,
            crop_key=CropKey(seed=self.seed, epoch=self.epoch, image_id=record.image_id),
        )
        tokens = self.vocabulary.encode(caption.text, self.max_length)
        clip = self.clip_features[record.image_id] if self.clip_features is not None else _no_clip()
        return image, mask, clip, torch.tensor(tokens, dtype=torch.long)


class EvalRetrievalDataset(Dataset):
    """Yields one image and all of its reference captions.

    Evaluation uses each arm's deterministic transform (``EVAL_PREPROCESSING``): the same
    as training, except that random resized crop evaluates on the full P1 image.
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
        clip_features: ClipFeatureTable | None = None,
    ) -> None:
        if preprocessing not in PREPROCESSING_CHOICES:
            raise ValueError(f"Unknown preprocessing {preprocessing!r}")
        if (preprocessing in CLIP_PREPROCESSING) != (clip_features is not None):
            raise ValueError("clip_features must be given exactly for CLIP-feature preprocessing")
        self.records = records
        self.image_root = image_root
        self.vocabulary = vocabulary
        self.clip_features = clip_features
        self.image_size = image_size
        self.max_length = max_length
        self.preprocessing = EVAL_PREPROCESSING.get(preprocessing, preprocessing)
        self.patch_size = patch_size

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        record = self.records[index]
        image, mask = load_image(
            self.image_root / record.relative_path,
            preprocessing=self.preprocessing,
            image_size=self.image_size,
            patch_size=self.patch_size,
        )
        tokens = [self.vocabulary.encode(item.text, self.max_length) for item in record.captions]
        clip = self.clip_features[record.image_id] if self.clip_features is not None else _no_clip()
        return image, mask, clip, torch.tensor(tokens, dtype=torch.long)
