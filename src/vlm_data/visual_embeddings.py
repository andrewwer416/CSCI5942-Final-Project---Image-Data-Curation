"""Frozen OpenCLIP image embeddings with auditable, fail-closed artifacts.

Heavy optional dependencies are imported only by the runtime entry points.  The
module can therefore be imported by the standard-library-only data foundation.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Sequence

from vlm_data.io import sha256_file, write_json_atomic
from vlm_data.manifest import ImageRecord, read_manifest


EMBEDDING_ALGORITHM_VERSION = "frozen-openclip-image-v1"
MODEL_NAME = "ViT-B-16"
PRETRAINED_TAG = "datacomp_xl_s13b_b90k"
HF_REPOSITORY = "laion/CLIP-ViT-B-16-DataComp.XL-s13B-b90K"
HF_REVISION = "767e6d14be23c46e4f80af31e6b4e73f323e72f5"
WEIGHT_FILENAME = "open_clip_pytorch_model.bin"
CONFIG_FILENAME = "open_clip_config.json"
WEIGHT_SHA256 = "80905176486d952914c1deb3b753b8e653ab54aa4ffb14208bc5007cf7643a16"
EXPECTED_EMBEDDING_DIMENSION = 512


@dataclass(frozen=True)
class EmbeddingArtifactPaths:
    """Committed files belonging to one embedding artifact."""

    embeddings: Path
    image_ids: Path
    metadata: Path


@dataclass(frozen=True)
class PinnedCheckpoint:
    """Resolved files for the immutable OpenCLIP checkpoint."""

    weights: Path
    config: Path
    config_sha256: str


def embedding_artifact_paths(embeddings_path: Path) -> EmbeddingArtifactPaths:
    """Return deterministic companion paths for an ``.npy`` embedding cache."""

    embeddings_path = Path(embeddings_path)
    if embeddings_path.suffix != ".npy":
        raise ValueError("The embedding output path must end in .npy")
    stem = embeddings_path.name[: -len(embeddings_path.suffix)]
    return EmbeddingArtifactPaths(
        embeddings=embeddings_path,
        image_ids=embeddings_path.with_name(f"{stem}.image_ids.npy"),
        metadata=embeddings_path.with_name(f"{stem}.meta.json"),
    )


def image_ids_sha256(image_ids: Iterable[int]) -> str:
    """Hash ordered image IDs using the repository's newline-delimited convention."""

    digest = hashlib.sha256()
    for image_id in image_ids:
        digest.update(f"{int(image_id)}\n".encode("ascii"))
    return digest.hexdigest()


def checkpoint_descriptor() -> dict[str, Any]:
    """Return the immutable checkpoint identity recorded in every artifact."""

    return {
        "library": "open_clip",
        "model_name": MODEL_NAME,
        "pretrained_tag": PRETRAINED_TAG,
        "hf_repository": HF_REPOSITORY,
        "hf_revision": HF_REVISION,
        "weight_filename": WEIGHT_FILENAME,
        "weight_sha256": WEIGHT_SHA256,
    }


def assert_frozen_eval_model(model: Any) -> None:
    """Fail unless a model is in evaluation mode with every parameter frozen."""

    if bool(getattr(model, "training", True)):
        raise RuntimeError("The CLIP model must be in eval mode before embedding")
    parameters = getattr(model, "parameters", None)
    if not callable(parameters):
        raise RuntimeError("The CLIP model does not expose parameters()")
    if any(bool(getattr(parameter, "requires_grad", True)) for parameter in parameters()):
        raise RuntimeError("Every CLIP parameter must have requires_grad=False")


def canonical_training_records(
    records: Iterable[ImageRecord],
    *,
    raw_root: Path,
    expected_split: str = "train2017",
    check_files: bool = True,
) -> list[ImageRecord]:
    """Validate and sort the exact image rows used by an embedding artifact."""

    ordered = sorted(records, key=lambda record: record.image_id)
    if not ordered:
        raise ValueError("The source manifest contains no image records")
    seen: set[int] = set()
    for record in ordered:
        if record.image_id in seen:
            raise ValueError(f"Duplicate image ID in source manifest: {record.image_id}")
        seen.add(record.image_id)
        if record.split != expected_split:
            raise ValueError(
                f"Image {record.image_id} belongs to {record.split!r}; "
                f"expected {expected_split!r}"
            )
        relative = Path(record.relative_path)
        expected = Path(expected_split) / record.file_name
        if relative.is_absolute() or ".." in relative.parts or relative != expected:
            raise ValueError(
                f"Image {record.image_id} has unsafe or unexpected path "
                f"{record.relative_path!r}"
            )
        if check_files and not (raw_root / relative).is_file():
            raise FileNotFoundError(f"Missing source image: {raw_root / relative}")
    return ordered


def _project_relative(path: Path, project_root: Path) -> str:
    resolved_root = project_root.resolve()
    try:
        return path.resolve().relative_to(resolved_root).as_posix()
    except ValueError as error:
        raise ValueError(f"Artifact path must be inside project root: {path}") from error


def _temporary_sibling(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    return Path(name)


def _ensure_outputs_available(paths: Sequence[Path], *, overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        rendered = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Refusing to overwrite existing embedding artifact: {rendered}")


def resolve_pinned_checkpoint(cache_dir: Path | None = None) -> PinnedCheckpoint:
    """Download the pinned HF files and verify the checkpoint bytes.

    ``huggingface_hub`` is deliberately optional at package-import time.
    """

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:  # pragma: no cover - exercised in the GPU environment.
        raise RuntimeError(
            "Frozen CLIP embedding requires the optional huggingface_hub package"
        ) from error

    arguments: dict[str, Any] = {
        "repo_id": HF_REPOSITORY,
        "revision": HF_REVISION,
    }
    if cache_dir is not None:
        arguments["cache_dir"] = str(cache_dir)
    weights = Path(hf_hub_download(filename=WEIGHT_FILENAME, **arguments))
    actual_weight_sha256 = sha256_file(weights)
    if actual_weight_sha256 != WEIGHT_SHA256:
        raise RuntimeError(
            f"Pinned CLIP weight checksum mismatch: {actual_weight_sha256}; "
            f"expected {WEIGHT_SHA256}"
        )
    config = Path(hf_hub_download(filename=CONFIG_FILENAME, **arguments))
    return PinnedCheckpoint(
        weights=weights,
        config=config,
        config_sha256=sha256_file(config),
    )


def generate_frozen_clip_embeddings(
    *,
    source_manifest: Path,
    raw_root: Path,
    embeddings_path: Path,
    project_root: Path,
    device: str,
    inference_precision: str,
    batch_size: int,
    num_workers: int,
    checkpoint_cache: Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Generate an image-ID-ordered, unit-normalized float32 embedding cache.

    The model is always frozen and evaluated under ``torch.inference_mode``.  The
    caller must choose inference precision explicitly because it is part of the
    scientific contract; cached vectors are float32 in either mode.
    """

    if inference_precision not in {"fp32", "amp_fp16"}:
        raise ValueError("inference_precision must be 'fp32' or 'amp_fp16'")
    if batch_size <= 0 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers non-negative")

    paths = embedding_artifact_paths(embeddings_path)
    _ensure_outputs_available(
        [paths.embeddings, paths.image_ids, paths.metadata], overwrite=overwrite
    )
    source_manifest = Path(source_manifest)
    raw_root = Path(raw_root)
    project_root = Path(project_root)
    source_sha256 = sha256_file(source_manifest)
    records = canonical_training_records(
        read_manifest(source_manifest), raw_root=raw_root, check_files=True
    )

    try:
        import numpy as np
        from PIL import Image
        import open_clip
        import torch
    except ImportError as error:  # pragma: no cover - exercised in the GPU environment.
        raise RuntimeError(
            "Frozen CLIP embedding requires numpy, Pillow, torch, and open_clip"
        ) from error

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested CUDA device is unavailable: {device}")
    pinned = resolve_pinned_checkpoint(checkpoint_cache)
    model, _, eval_preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME,
        pretrained=str(pinned.weights),
        precision="fp32",
        device=device,
    )
    model.eval()
    model.requires_grad_(False)
    assert_frozen_eval_model(model)

    class ManifestImageDataset(torch.utils.data.Dataset):
        def __len__(self) -> int:
            return len(records)

        def __getitem__(self, index: int):
            record = records[index]
            image_path = raw_root / record.relative_path
            try:
                with Image.open(image_path) as image:
                    pixels = eval_preprocess(image.convert("RGB"))
            except Exception as error:
                raise RuntimeError(
                    f"Failed to decode image {record.image_id} at {image_path}"
                ) from error
            return record.image_id, pixels

    loader = torch.utils.data.DataLoader(
        ManifestImageDataset(),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.startswith("cuda"),
        persistent_workers=num_workers > 0,
        drop_last=False,
    )

    temporary_embeddings = _temporary_sibling(paths.embeddings)
    temporary_ids = _temporary_sibling(paths.image_ids)
    embeddings_array = None
    completed = 0
    try:
        embeddings_array = np.lib.format.open_memmap(
            temporary_embeddings,
            mode="w+",
            dtype=np.float32,
            shape=(len(records), EXPECTED_EMBEDDING_DIMENSION),
        )
        image_ids = np.empty(len(records), dtype=np.int64)
        with torch.inference_mode():
            for batch_ids, pixels in loader:
                pixels = pixels.to(device, non_blocking=device.startswith("cuda"))
                if inference_precision == "amp_fp16":
                    if not device.startswith("cuda"):
                        raise ValueError("amp_fp16 inference requires a CUDA device")
                    context = torch.autocast(device_type="cuda", dtype=torch.float16)
                else:
                    context = nullcontext()
                with context:
                    features = model.encode_image(pixels, normalize=False)
                features = features.float()
                if features.ndim != 2 or features.shape[1] != EXPECTED_EMBEDDING_DIMENSION:
                    raise RuntimeError(
                        f"Unexpected CLIP feature shape {tuple(features.shape)}; expected "
                        f"[batch, {EXPECTED_EMBEDDING_DIMENSION}]"
                    )
                if not bool(torch.isfinite(features).all()):
                    raise RuntimeError("CLIP produced a non-finite image feature")
                norms = torch.linalg.vector_norm(features, dim=1, keepdim=True)
                if not bool((norms > 0).all()):
                    raise RuntimeError("CLIP produced a zero-norm image feature")
                features = features / norms
                batch_ids_array = batch_ids.detach().cpu().numpy().astype(np.int64, copy=False)
                expected_ids = np.asarray(
                    [
                        record.image_id
                        for record in records[
                            completed : completed + len(batch_ids_array)
                        ]
                    ],
                    dtype=np.int64,
                )
                if not np.array_equal(batch_ids_array, expected_ids):
                    raise RuntimeError("Data loader changed the canonical image-ID order")
                end = completed + len(batch_ids_array)
                embeddings_array[completed:end] = features.cpu().numpy().astype(
                    np.float32, copy=False
                )
                image_ids[completed:end] = batch_ids_array
                completed = end
        if completed != len(records):
            raise RuntimeError(f"Embedded {completed} images; expected {len(records)}")
        embeddings_array.flush()
        del embeddings_array
        embeddings_array = None
        with temporary_ids.open("wb") as handle:
            np.save(handle, image_ids, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        if sha256_file(source_manifest) != source_sha256:
            raise RuntimeError("Source manifest changed while embeddings were being generated")

        embedding_sha256 = sha256_file(temporary_embeddings)
        image_ids_file_sha256 = sha256_file(temporary_ids)
        metadata: dict[str, Any] = {
            "schema_version": 1,
            "kind": "frozen_clip_image_embeddings",
            "algorithm_version": EMBEDDING_ALGORITHM_VERSION,
            "source": {
                "manifest": _project_relative(source_manifest, project_root),
                "manifest_sha256": source_sha256,
                "raw_root": _project_relative(raw_root, project_root),
                "split": "train2017",
                "record_count": len(records),
                "image_ids_sha256": image_ids_sha256(image_ids.tolist()),
            },
            "checkpoint": {
                **checkpoint_descriptor(),
                "config_filename": CONFIG_FILENAME,
                "config_sha256": pinned.config_sha256,
                "frozen": True,
                "eval_mode": True,
            },
            "embedding": {
                "dimension": EXPECTED_EMBEDDING_DIMENSION,
                "dtype": "float32",
                "l2_normalized": True,
                "feature": "projected_encode_image",
                "order": "ascending_coco_image_id",
                "decoder": "Pillow RGB",
                "preprocess": "open_clip checkpoint-native eval transform",
                "eval_transform_repr": repr(eval_preprocess),
            },
            "runtime": {
                "device": device,
                "inference_precision": inference_precision,
                "batch_size": batch_size,
                "num_workers": num_workers,
                "torch_version": str(torch.__version__),
                "open_clip_version": str(getattr(open_clip, "__version__", "unknown")),
                "numpy_version": str(np.__version__),
                "gpu_name": (
                    torch.cuda.get_device_name(torch.device(device))
                    if device.startswith("cuda")
                    else None
                ),
            },
            "artifacts": {
                "embeddings": {
                    "file": paths.embeddings.name,
                    "sha256": embedding_sha256,
                },
                "image_ids": {
                    "file": paths.image_ids.name,
                    "sha256": image_ids_file_sha256,
                },
            },
        }
        os.replace(temporary_embeddings, paths.embeddings)
        os.replace(temporary_ids, paths.image_ids)
        write_json_atomic(paths.metadata, metadata)
        return metadata
    finally:
        if embeddings_array is not None:
            embeddings_array.flush()
        for temporary in (temporary_embeddings, temporary_ids):
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def load_embedding_artifact(
    metadata_path: Path,
    *,
    source_manifest: Path | None = None,
    mmap_mode: str | None = "r",
):
    """Validate and load an embedding artifact, failing closed on drift."""

    try:
        import numpy as np
    except ImportError as error:  # pragma: no cover - exercised in the GPU environment.
        raise RuntimeError("Loading CLIP embeddings requires numpy") from error

    metadata_path = Path(metadata_path)
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if metadata.get("kind") != "frozen_clip_image_embeddings":
        raise RuntimeError(f"Unexpected embedding artifact kind in {metadata_path}")
    if metadata.get("algorithm_version") != EMBEDDING_ALGORITHM_VERSION:
        raise RuntimeError("Unsupported embedding algorithm version")
    expected_checkpoint = checkpoint_descriptor()
    recorded_checkpoint = metadata.get("checkpoint", {})
    for key, expected in expected_checkpoint.items():
        if recorded_checkpoint.get(key) != expected:
            raise RuntimeError(f"Embedding checkpoint mismatch for {key}")
    if (
        recorded_checkpoint.get("frozen") is not True
        or recorded_checkpoint.get("eval_mode") is not True
    ):
        raise RuntimeError("Embedding artifact was not recorded as frozen eval inference")

    artifact_entries = metadata.get("artifacts", {})
    try:
        embeddings_path = metadata_path.parent / artifact_entries["embeddings"]["file"]
        image_ids_path = metadata_path.parent / artifact_entries["image_ids"]["file"]
    except (KeyError, TypeError) as error:
        raise RuntimeError("Embedding artifact metadata is incomplete") from error
    for path, entry in (
        (embeddings_path, artifact_entries["embeddings"]),
        (image_ids_path, artifact_entries["image_ids"]),
    ):
        if path.parent.resolve() != metadata_path.parent.resolve():
            raise RuntimeError(f"Artifact path escapes metadata directory: {path}")
        if not path.is_file() or sha256_file(path) != entry.get("sha256"):
            raise RuntimeError(f"Embedding artifact checksum mismatch: {path}")
    if source_manifest is not None:
        actual_source_sha256 = sha256_file(Path(source_manifest))
        if actual_source_sha256 != metadata.get("source", {}).get("manifest_sha256"):
            raise RuntimeError("Embedding source manifest hash does not match current manifest")

    embeddings = np.load(embeddings_path, mmap_mode=mmap_mode, allow_pickle=False)
    image_ids = np.load(image_ids_path, mmap_mode=mmap_mode, allow_pickle=False)
    expected_count = int(metadata.get("source", {}).get("record_count", -1))
    expected_dimension = int(metadata.get("embedding", {}).get("dimension", -1))
    if embeddings.dtype != np.float32 or embeddings.shape != (expected_count, expected_dimension):
        raise RuntimeError(
            f"Unexpected embedding array contract: {embeddings.shape}, {embeddings.dtype}"
        )
    if image_ids.dtype != np.int64 or image_ids.shape != (expected_count,):
        raise RuntimeError(
            f"Unexpected image-ID array contract: {image_ids.shape}, {image_ids.dtype}"
        )
    if expected_dimension != EXPECTED_EMBEDDING_DIMENSION:
        raise RuntimeError(f"Unexpected embedding dimension: {expected_dimension}")
    if expected_count and not bool(np.all(image_ids[1:] > image_ids[:-1])):
        raise RuntimeError("Embedding image IDs are not strictly increasing")
    if image_ids_sha256(image_ids.tolist()) != metadata.get("source", {}).get("image_ids_sha256"):
        raise RuntimeError("Embedding image-ID hash does not match metadata")
    for start in range(0, expected_count, 8192):
        chunk = np.asarray(embeddings[start : start + 8192], dtype=np.float32)
        if not bool(np.isfinite(chunk).all()):
            raise RuntimeError("Embedding artifact contains non-finite values")
        norms = np.linalg.norm(chunk, axis=1)
        if not bool(np.all(np.abs(norms - 1.0) <= 1e-4)):
            raise RuntimeError("Embedding artifact contains non-unit vectors")
    return metadata, embeddings, image_ids
