"""Frozen MPNet caption means and reusable, hash-verified artifacts."""

from pathlib import Path
import json
import os
import tempfile

from vlm_data.io import sha256_file, write_json_atomic
from vlm_data.manifest import read_manifest
from vlm_data.visual_embeddings import assert_frozen_eval_model, image_ids_sha256

MODEL = "sentence-transformers/all-mpnet-base-v2"
REVISION = "e8c3b32edf5434bc2275fc9bab85f82640a19130"
VERSION = "frozen-mpnet-normalized-five-caption-mean-v1"


def aggregate_captions(vectors):
    import numpy as np

    x = np.asarray(vectors, dtype=np.float32)
    if x.ndim != 3 or x.shape[1] != 5 or not np.isfinite(x).all():
        raise ValueError("Expected finite [images, 5, dimensions] caption vectors")
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Zero caption norm")
    x = x / norms
    means = x.mean(axis=1)
    lengths = np.linalg.norm(means, axis=-1, keepdims=True)
    if np.any(lengths == 0):
        raise ValueError("Zero aggregate norm")
    # Mean pairwise cosine distance among the ten distinct caption pairs.
    dispersion = (25 - 25 * lengths[:, 0] ** 2) / 20
    return means / lengths, dispersion


def save_array(path, array):
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            np.save(handle, array, allow_pickle=False)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def generate_embeddings(source_manifest, output_dir, model_cache, batch_size=128):
    import numpy as np
    import torch
    import sentence_transformers
    from sentence_transformers import SentenceTransformer
    from huggingface_hub import snapshot_download

    output_dir = Path(output_dir)
    meta_path = output_dir / "mpnet.meta.json"
    if meta_path.exists():
        return load_embeddings(meta_path, source_manifest=source_manifest)[0]
    source_hash = sha256_file(source_manifest)
    records = sorted(read_manifest(source_manifest), key=lambda r: r.image_id)
    if len({r.image_id for r in records}) != len(records) or any(
        r.split != "train2017" or len(r.captions) != 5 for r in records
    ):
        raise ValueError("Expected unique train images with five captions")
    snapshot = Path(
        snapshot_download(
            MODEL,
            revision=REVISION,
            cache_dir=str(model_cache),
            allow_patterns=["*.json", "*.txt", "*.safetensors", "1_Pooling/*"],
        )
    )
    model = SentenceTransformer(str(snapshot), device="cuda", trust_remote_code=False)
    model.eval().requires_grad_(False)
    assert_frozen_eval_model(model)
    texts = [cap.text for r in records for cap in sorted(r.captions, key=lambda x: x.caption_id)]
    if any(not text.strip() for text in texts):
        raise ValueError("Empty caption")
    with torch.inference_mode():
        encoded = model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True,
        ).astype(np.float32)
    grouped = encoded.reshape(len(records), 5, 768)
    means, dispersion = aggregate_captions(grouped)
    ids = np.array([r.image_id for r in records], dtype=np.int64)
    truncated = sum(len(model.tokenizer.encode(t)) > model.max_seq_length for t in texts)
    arrays = {"embeddings": means, "image_ids": ids, "dispersion": dispersion, "captions": grouped}
    artifacts = {}
    for name, array in arrays.items():
        path = output_dir / f"mpnet.{name}.npy"
        save_array(path, array)
        artifacts[name] = {"file": path.name, "sha256": sha256_file(path)}
    if sha256_file(source_manifest) != source_hash:
        raise RuntimeError("Source changed during encoding")
    metadata = {
        "kind": "caption_semantic_embeddings",
        "algorithm_version": VERSION,
        "checkpoint": {
            "model": MODEL,
            "revision": REVISION,
            "frozen": True,
            "files": {
                p.relative_to(snapshot).as_posix(): sha256_file(p)
                for p in snapshot.rglob("*")
                if p.is_file()
            },
        },
        "source": {
            "manifest_sha256": source_hash,
            "record_count": len(records),
            "image_ids_sha256": image_ids_sha256(ids),
        },
        "runtime": {
            "torch": torch.__version__,
            "sentence_transformers": sentence_transformers.__version__,
            "precision": "fp32",
            "batch_size": batch_size,
            "max_seq_length": model.max_seq_length,
            "truncated_caption_count": truncated,
        },
        "artifacts": artifacts,
    }
    write_json_atomic(meta_path, metadata)
    return metadata


def load_embeddings(metadata_path, source_manifest=None, mmap_mode="r"):
    import numpy as np

    metadata_path = Path(metadata_path)
    metadata = json.loads(metadata_path.read_text())
    if metadata["algorithm_version"] != VERSION or metadata["checkpoint"]["revision"] != REVISION:
        raise ValueError("Unsupported caption embedding contract")
    if (
        metadata["checkpoint"].get("model") != MODEL
        or metadata["checkpoint"].get("frozen") is not True
    ):
        raise ValueError("Caption model must be the approved frozen checkpoint")
    if source_manifest and sha256_file(source_manifest) != metadata["source"]["manifest_sha256"]:
        raise ValueError("Caption source hash mismatch")
    arrays = {}
    for name, artifact in metadata["artifacts"].items():
        path = metadata_path.parent / artifact["file"]
        if (
            path.parent.resolve() != metadata_path.parent.resolve()
            or sha256_file(path) != artifact["sha256"]
        ):
            raise ValueError("Caption artifact hash/path mismatch")
        arrays[name] = np.load(path, mmap_mode=mmap_mode, allow_pickle=False)
    x, ids = arrays["embeddings"], arrays["image_ids"]
    count = metadata["source"]["record_count"]
    if ids.shape != (count,) or ids.dtype != np.int64:
        raise ValueError("Invalid caption image-ID array")
    if arrays["captions"].shape != (count, 5, 768) or arrays["dispersion"].shape != (count,):
        raise ValueError("Invalid per-caption/dispersion shape")
    if x.shape != (metadata["source"]["record_count"], 768) or x.dtype != np.float32:
        raise ValueError("Invalid caption embedding shape/dtype")
    if not np.isfinite(x).all() or not np.allclose(np.linalg.norm(x, axis=1), 1, atol=1e-5):
        raise ValueError("Invalid caption embedding norms")
    if (
        not np.all(ids[1:] > ids[:-1])
        or image_ids_sha256(ids) != metadata["source"]["image_ids_sha256"]
    ):
        raise ValueError("Invalid caption ID order")
    return metadata, x, ids
