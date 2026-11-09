"""Exact, deterministic GPU neighbor artifacts for frozen CLIP embeddings."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Sequence

from vlm_data.io import sha256_file, write_json_atomic
from vlm_data.visual_embeddings import image_ids_sha256, load_embedding_artifact


NEIGHBOR_ALGORITHM_VERSION = "faiss-gpu-indexflatip-v1"


@dataclass(frozen=True)
class NeighborArtifactPaths:
    """Committed files belonging to one maximum-k neighbor graph."""

    similarities: Path
    image_ids: Path
    metadata: Path


@dataclass(frozen=True)
class CanonicalNeighborRow:
    """One deterministic row, with self first and non-self ties by image ID."""

    image_ids: tuple[int, ...]
    similarities: tuple[float, ...]
    positions: tuple[int, ...]


def neighbor_artifact_paths(similarities_path: Path) -> NeighborArtifactPaths:
    """Return deterministic companion paths for an ``.npy`` similarity matrix."""

    similarities_path = Path(similarities_path)
    if similarities_path.suffix != ".npy":
        raise ValueError("The neighbor similarity output path must end in .npy")
    stem = similarities_path.name[: -len(similarities_path.suffix)]
    return NeighborArtifactPaths(
        similarities=similarities_path,
        image_ids=similarities_path.with_name(f"{stem}.image_ids.npy"),
        metadata=similarities_path.with_name(f"{stem}.meta.json"),
    )


def canonicalize_neighbor_candidates(
    *,
    query_position: int,
    candidate_positions: Iterable[int],
    candidate_similarities: Iterable[float],
    image_ids: Sequence[int],
    k: int,
    include_self: bool = True,
) -> CanonicalNeighborRow:
    """Canonicalize one FAISS result row.

    Self is removed from backend output and, when requested, inserted exactly
    once at similarity 1.  Remaining candidates are sorted by descending
    similarity and then ascending COCO image ID.  This makes self handling
    independent of FAISS's behavior for identical vectors.
    """

    if k <= 0:
        raise ValueError("k must be positive")
    if not 0 <= query_position < len(image_ids):
        raise IndexError(f"Query position is out of range: {query_position}")
    positions = list(candidate_positions)
    similarities = list(candidate_similarities)
    if len(positions) != len(similarities):
        raise ValueError("Candidate positions and similarities have different lengths")

    best_by_position: dict[int, float] = {}
    for position_value, similarity_value in zip(positions, similarities, strict=True):
        position = int(position_value)
        similarity = float(similarity_value)
        if position < 0:
            continue
        if position >= len(image_ids):
            raise IndexError(f"Neighbor position is out of range: {position}")
        if not math.isfinite(similarity):
            raise ValueError("Neighbor similarity must be finite")
        if position == query_position:
            continue
        prior = best_by_position.get(position)
        if prior is None or similarity > prior:
            best_by_position[position] = similarity

    ordered = sorted(
        (
            (similarity, int(image_ids[position]), position)
            for position, similarity in best_by_position.items()
        ),
        key=lambda item: (-item[0], item[1]),
    )
    required_nonself = k - 1 if include_self else k
    if len(ordered) < required_nonself:
        raise RuntimeError(
            f"Only {len(ordered)} distinct non-self neighbors are available; "
            f"need {required_nonself}"
        )
    selected = ordered[:required_nonself]
    if include_self:
        selected.insert(0, (1.0, int(image_ids[query_position]), query_position))
    return CanonicalNeighborRow(
        image_ids=tuple(item[1] for item in selected),
        similarities=tuple(item[0] for item in selected),
        positions=tuple(item[2] for item in selected),
    )


def _temporary_sibling(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    return Path(name)


def _project_relative(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError as error:
        raise ValueError(f"Artifact path must be inside project root: {path}") from error


def _ensure_outputs_available(paths: Sequence[Path], *, overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        rendered = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Refusing to overwrite existing neighbor artifact: {rendered}")


def _stable_exact_row(embeddings: Any, image_ids: Any, query_position: int, k: int):
    """Resolve a rare FAISS cutoff tie against every vector on CPU."""

    import numpy as np

    similarities = np.asarray(embeddings) @ np.asarray(embeddings[query_position])
    similarities = similarities.astype(np.float32, copy=False)
    similarities[query_position] = np.float32(1.0)
    positions = np.arange(len(image_ids), dtype=np.int64)
    nonself = positions != query_position
    nonself_positions = positions[nonself]
    order = np.lexsort((image_ids[nonself_positions], -similarities[nonself_positions]))
    chosen = nonself_positions[order[: k - 1]]
    row_positions = np.concatenate(
        (np.asarray([query_position], dtype=np.int64), chosen.astype(np.int64, copy=False))
    )
    return row_positions, similarities[row_positions]


def build_exact_gpu_neighbor_artifact(
    *,
    embedding_metadata_path: Path,
    source_manifest: Path,
    similarities_path: Path,
    project_root: Path,
    max_k: int,
    query_batch_size: int,
    gpu_device: int = 0,
    overwrite: bool = False,
    embedding_loader=None,
) -> dict[str, Any]:
    """Run one exact max-k FAISS search and atomically commit its graph.

    Every smaller candidate k is a leading-column view of this artifact, so no
    second neighbor search or approximate index can change the calibration.
    """

    if max_k < 2:
        raise ValueError("max_k must be at least 2 because each row includes self")
    if query_batch_size <= 0:
        raise ValueError("query_batch_size must be positive")
    paths = neighbor_artifact_paths(similarities_path)
    _ensure_outputs_available(
        [paths.similarities, paths.image_ids, paths.metadata], overwrite=overwrite
    )
    try:
        import faiss
        import numpy as np
    except ImportError as error:  # pragma: no cover - exercised in the GPU environment.
        raise RuntimeError("Exact GPU neighbor search requires numpy and faiss-gpu") from error

    loader = load_embedding_artifact if embedding_loader is None else embedding_loader
    embedding_metadata, embeddings, source_image_ids = loader(
        embedding_metadata_path,
        source_manifest=source_manifest,
        mmap_mode="r",
    )
    count, dimension = embeddings.shape
    if max_k > count:
        raise ValueError(f"max_k {max_k} exceeds embedding count {count}")
    if gpu_device < 0 or gpu_device >= int(faiss.get_num_gpus()):
        raise RuntimeError(
            f"FAISS GPU device {gpu_device} is unavailable; found {faiss.get_num_gpus()} GPU(s)"
        )
    if not hasattr(faiss, "StandardGpuResources") or not hasattr(faiss, "index_cpu_to_gpu"):
        raise RuntimeError("Installed FAISS build does not provide GPU IndexFlat support")

    contiguous_embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
    cpu_index = faiss.IndexFlatIP(dimension)
    resources = faiss.StandardGpuResources()
    index = faiss.index_cpu_to_gpu(resources, gpu_device, cpu_index)
    index.add(contiguous_embeddings)
    if int(index.ntotal) != count:
        raise RuntimeError(f"FAISS index contains {index.ntotal} vectors; expected {count}")

    temporary_similarities = _temporary_sibling(paths.similarities)
    temporary_ids = _temporary_sibling(paths.image_ids)
    similarity_array = None
    neighbor_id_array = None
    tie_fallback_rows = 0
    # Two extra results expose an exact tie immediately outside the retained
    # boundary. A boundary tie is resolved against all vectors deterministically.
    backend_k = min(count, max_k + 2)
    try:
        similarity_array = np.lib.format.open_memmap(
            temporary_similarities,
            mode="w+",
            dtype=np.float32,
            shape=(count, max_k),
        )
        neighbor_id_array = np.lib.format.open_memmap(
            temporary_ids,
            mode="w+",
            dtype=np.int64,
            shape=(count, max_k),
        )
        for start in range(0, count, query_batch_size):
            end = min(count, start + query_batch_size)
            raw_similarities, raw_positions = index.search(
                contiguous_embeddings[start:end], backend_k
            )
            for local_index, query_position in enumerate(range(start, end)):
                full_row = canonicalize_neighbor_candidates(
                    query_position=query_position,
                    candidate_positions=raw_positions[local_index],
                    candidate_similarities=raw_similarities[local_index],
                    image_ids=source_image_ids,
                    k=max_k + 1 if max_k < count else max_k,
                    include_self=True,
                )
                boundary_tied = (
                    max_k < len(full_row.similarities)
                    and abs(
                        full_row.similarities[max_k - 1]
                        - full_row.similarities[max_k]
                    )
                    <= 1e-7
                )
                if boundary_tied:
                    row_positions, row_similarities = _stable_exact_row(
                        contiguous_embeddings, source_image_ids, query_position, max_k
                    )
                    tie_fallback_rows += 1
                    row_ids = source_image_ids[row_positions]
                else:
                    row_positions = np.asarray(full_row.positions[:max_k], dtype=np.int64)
                    row_similarities = np.asarray(
                        full_row.similarities[:max_k], dtype=np.float32
                    )
                    row_ids = source_image_ids[row_positions]
                if int(row_ids[0]) != int(source_image_ids[query_position]):
                    raise RuntimeError("Canonical neighbor row does not begin with self")
                if not bool(np.isfinite(row_similarities).all()):
                    raise RuntimeError("FAISS returned a non-finite similarity")
                similarity_array[query_position] = row_similarities
                neighbor_id_array[query_position] = row_ids
        similarity_array.flush()
        neighbor_id_array.flush()
        del similarity_array
        del neighbor_id_array
        similarity_array = None
        neighbor_id_array = None

        embedding_metadata_sha256 = sha256_file(Path(embedding_metadata_path))
        metadata: dict[str, Any] = {
            "schema_version": 1,
            "kind": (
                "exact_clip_neighbor_graph" if embedding_loader is None
                else "exact_embedding_neighbor_graph"
            ),
            "algorithm_version": NEIGHBOR_ALGORITHM_VERSION,
            "source": {
                "embedding_metadata": _project_relative(
                    Path(embedding_metadata_path), Path(project_root)
                ),
                "embedding_metadata_sha256": embedding_metadata_sha256,
                "manifest": _project_relative(Path(source_manifest), Path(project_root)),
                "manifest_sha256": embedding_metadata["source"]["manifest_sha256"],
                "record_count": count,
                "image_ids_sha256": image_ids_sha256(source_image_ids.tolist()),
            },
            "search": {
                "backend": "faiss-gpu",
                "index": "IndexFlatIP",
                "metric": "inner_product_on_l2_normalized_vectors",
                "exact": True,
                "max_k": max_k,
                "includes_self": True,
                "self_policy": "insert_once_at_column_zero_with_similarity_one",
                "nonself_order": "descending_similarity_then_ascending_coco_image_id",
                "smaller_k_policy": "leading_columns_of_max_k_graph",
                "query_batch_size": query_batch_size,
                "gpu_device": gpu_device,
                "boundary_tie_fallback_rows": tie_fallback_rows,
            },
            "runtime": {
                "faiss_version": str(getattr(faiss, "__version__", "unknown")),
                "numpy_version": str(np.__version__),
                "gpu_count": int(faiss.get_num_gpus()),
            },
            "artifacts": {
                "similarities": {
                    "file": paths.similarities.name,
                    "sha256": sha256_file(temporary_similarities),
                    "dtype": "float32",
                    "shape": [count, max_k],
                },
                "neighbor_image_ids": {
                    "file": paths.image_ids.name,
                    "sha256": sha256_file(temporary_ids),
                    "dtype": "int64",
                    "shape": [count, max_k],
                },
            },
        }
        os.replace(temporary_similarities, paths.similarities)
        os.replace(temporary_ids, paths.image_ids)
        write_json_atomic(paths.metadata, metadata)
        return metadata
    finally:
        if similarity_array is not None:
            similarity_array.flush()
        if neighbor_id_array is not None:
            neighbor_id_array.flush()
        for temporary in (temporary_similarities, temporary_ids):
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def load_neighbor_artifact(
    metadata_path: Path,
    *,
    embedding_metadata_path: Path | None = None,
    k: int | None = None,
    mmap_mode: str | None = "r",
):
    """Validate a graph and return an optional leading-column k view."""

    try:
        import numpy as np
    except ImportError as error:  # pragma: no cover - exercised in the GPU environment.
        raise RuntimeError("Loading neighbor artifacts requires numpy") from error

    metadata_path = Path(metadata_path)
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if metadata.get("kind") not in {"exact_clip_neighbor_graph", "exact_embedding_neighbor_graph"}:
        raise RuntimeError(f"Unexpected neighbor artifact kind in {metadata_path}")
    if metadata.get("algorithm_version") != NEIGHBOR_ALGORITHM_VERSION:
        raise RuntimeError("Unsupported neighbor algorithm version")
    if embedding_metadata_path is not None:
        actual = sha256_file(Path(embedding_metadata_path))
        if actual != metadata.get("source", {}).get("embedding_metadata_sha256"):
            raise RuntimeError("Neighbor graph embedding metadata hash mismatch")
    search = metadata.get("search", {})
    max_k = int(search.get("max_k", -1))
    requested_k = max_k if k is None else int(k)
    if not 1 <= requested_k <= max_k:
        raise ValueError(f"Requested k must be between 1 and {max_k}")

    entries = metadata.get("artifacts", {})
    try:
        similarity_path = metadata_path.parent / entries["similarities"]["file"]
        image_ids_path = metadata_path.parent / entries["neighbor_image_ids"]["file"]
    except (KeyError, TypeError) as error:
        raise RuntimeError("Neighbor artifact metadata is incomplete") from error
    for path, entry in (
        (similarity_path, entries["similarities"]),
        (image_ids_path, entries["neighbor_image_ids"]),
    ):
        if path.parent.resolve() != metadata_path.parent.resolve():
            raise RuntimeError(f"Artifact path escapes metadata directory: {path}")
        if not path.is_file() or sha256_file(path) != entry.get("sha256"):
            raise RuntimeError(f"Neighbor artifact checksum mismatch: {path}")
    similarities = np.load(similarity_path, mmap_mode=mmap_mode, allow_pickle=False)
    neighbor_ids = np.load(image_ids_path, mmap_mode=mmap_mode, allow_pickle=False)
    count = int(metadata.get("source", {}).get("record_count", -1))
    expected_shape = (count, max_k)
    if similarities.dtype != np.float32 or similarities.shape != expected_shape:
        raise RuntimeError("Neighbor similarity array violates its metadata contract")
    if neighbor_ids.dtype != np.int64 or neighbor_ids.shape != expected_shape:
        raise RuntimeError("Neighbor ID array violates its metadata contract")
    if not bool(np.isfinite(similarities).all()):
        raise RuntimeError("Neighbor artifact contains non-finite similarities")
    source_ids = neighbor_ids[:, 0]
    if count and not bool(np.all(source_ids[1:] > source_ids[:-1])):
        raise RuntimeError("Neighbor rows are not in ascending source image-ID order")
    if image_ids_sha256(source_ids.tolist()) != metadata.get("source", {}).get(
        "image_ids_sha256"
    ):
        raise RuntimeError("Neighbor source image-ID hash does not match metadata")
    if not bool(np.all(np.abs(similarities[:, 0] - 1.0) <= 1e-6)):
        raise RuntimeError("Neighbor artifact does not contain explicit self similarity")
    for start in range(0, count, 4096):
        end = min(count, start + 4096)
        row_ids = neighbor_ids[start:end]
        row_similarities = similarities[start:end]
        if requested_k > 1:
            nonself_ids = row_ids[:, 1:requested_k]
            if bool(np.any(nonself_ids == source_ids[start:end, None])):
                raise RuntimeError("Neighbor artifact repeats self in a non-self column")
            left_similarity = row_similarities[:, 1 : requested_k - 1]
            right_similarity = row_similarities[:, 2:requested_k]
            if left_similarity.size:
                increasing = right_similarity > left_similarity + 1e-7
                if bool(np.any(increasing)):
                    raise RuntimeError("Neighbor similarities are not descending")
                # Canonical sorting uses exact stored scores. Nearby unequal
                # scores retain similarity priority over the image-ID tie rule.
                tied = left_similarity == right_similarity
                wrong_tie_order = tied & (
                    nonself_ids[:, 1:] < nonself_ids[:, :-1]
                )
                if bool(np.any(wrong_tie_order)):
                    raise RuntimeError("Neighbor similarity ties are not ordered by image ID")
    return metadata, similarities[:, :requested_k], neighbor_ids[:, :requested_k]
