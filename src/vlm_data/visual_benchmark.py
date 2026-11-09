"""Real-data precision and batch-size benchmark for frozen CLIP embeddings.

The module keeps heavyweight numerical and model imports inside runtime functions so the
standard-library data foundation remains importable without the visual environment.
"""

from __future__ import annotations

import gc
import hashlib
import math
import platform
import time
from collections.abc import Iterable, Sequence
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from vlm_data.io import sha256_file
from vlm_data.manifest import ImageRecord, read_manifest
from vlm_data.visual_embeddings import (
    EXPECTED_EMBEDDING_DIMENSION,
    MODEL_NAME,
    assert_frozen_eval_model,
    canonical_training_records,
    checkpoint_descriptor,
    image_ids_sha256,
    resolve_pinned_checkpoint,
)

BENCHMARK_ALGORITHM_VERSION = "frozen-openclip-benchmark-v1"
MAX_MEAN_COSINE_DRIFT = 1e-4
MIN_MEAN_TOP50_NEIGHBOR_OVERLAP = 0.99
NEIGHBOR_COUNT = 50
UNIT_NORM_TOLERANCE = 1e-4


def deterministic_sample_records(
    records: Iterable[ImageRecord], *, sample_size: int, seed: int
) -> list[ImageRecord]:
    """Select a stable seed-dependent sample and return it in ascending COCO-ID order."""

    materialized = list(records)
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    if sample_size > len(materialized):
        raise ValueError(f"sample_size {sample_size} exceeds record count {len(materialized)}")
    image_ids = [int(record.image_id) for record in materialized]
    if len(set(image_ids)) != len(image_ids):
        raise ValueError("records must have unique image IDs")

    def rank(record: ImageRecord) -> tuple[bytes, int]:
        payload = f"{int(seed)}\0{int(record.image_id)}".encode("ascii")
        return hashlib.sha256(payload).digest(), int(record.image_id)

    sampled = sorted(materialized, key=rank)[:sample_size]
    return sorted(sampled, key=lambda record: int(record.image_id))


def _validated_rows(
    embeddings: Sequence[Sequence[float]], image_ids: Sequence[int], *, label: str
) -> list[list[float]]:
    rows = [[float(value) for value in row] for row in embeddings]
    if len(rows) != len(image_ids):
        raise ValueError(f"{label} row count does not match image IDs")
    if not rows or not rows[0]:
        raise ValueError(f"{label} embeddings must be a nonempty matrix")
    dimension = len(rows[0])
    for row in rows:
        if len(row) != dimension:
            raise ValueError(f"{label} embeddings are ragged")
        if not all(math.isfinite(value) for value in row):
            raise ValueError(f"{label} embeddings contain non-finite values")
        norm = math.sqrt(sum(value * value for value in row))
        if abs(norm - 1.0) > UNIT_NORM_TOLERANCE:
            raise ValueError(f"{label} embeddings are not unit normalized")
    return rows


def _compare_embeddings_stdlib(
    reference: Sequence[Sequence[float]],
    candidate: Sequence[Sequence[float]],
    image_ids: Sequence[int],
    *,
    neighbor_k: int,
) -> dict[str, Any]:
    reference_rows = _validated_rows(reference, image_ids, label="reference")
    candidate_rows = _validated_rows(candidate, image_ids, label="candidate")
    if len(reference_rows[0]) != len(candidate_rows[0]):
        raise ValueError("reference and candidate dimensions differ")

    paired_cosines = [
        sum(left * right for left, right in zip(reference_row, candidate_row))
        for reference_row, candidate_row in zip(reference_rows, candidate_rows)
    ]
    mean_drift = sum(max(0.0, 1.0 - min(1.0, cosine)) for cosine in paired_cosines) / len(
        paired_cosines
    )

    def neighbor_sets(rows: list[list[float]]) -> list[set[int]]:
        result: list[set[int]] = []
        for query_position, query in enumerate(rows):
            scored: list[tuple[float, int]] = []
            for position, row in enumerate(rows):
                if position == query_position:
                    continue
                similarity = sum(left * right for left, right in zip(query, row))
                scored.append((similarity, int(image_ids[position])))
            scored.sort(key=lambda item: (-item[0], item[1]))
            result.append({image_id for _, image_id in scored[:neighbor_k]})
        return result

    reference_neighbors = neighbor_sets(reference_rows)
    candidate_neighbors = neighbor_sets(candidate_rows)
    overlaps = [
        len(reference_set & candidate_set) / neighbor_k
        for reference_set, candidate_set in zip(reference_neighbors, candidate_neighbors)
    ]
    return {
        "sample_count": len(reference_rows),
        "neighbor_k": neighbor_k,
        "mean_cosine_drift": float(mean_drift),
        "mean_top_k_neighbor_overlap": float(sum(overlaps) / len(overlaps)),
    }


def compare_embedding_outputs(
    reference: Sequence[Sequence[float]],
    candidate: Sequence[Sequence[float]],
    image_ids: Sequence[int],
    *,
    neighbor_k: int = NEIGHBOR_COUNT,
) -> dict[str, Any]:
    """Compare unit embeddings without side effects, using COCO IDs for neighbor ties."""

    normalized_ids = [int(image_id) for image_id in image_ids]
    if len(set(normalized_ids)) != len(normalized_ids):
        raise ValueError("image_ids must be unique")
    if len(normalized_ids) < 2:
        raise ValueError("at least two embeddings are required")
    if neighbor_k <= 0:
        raise ValueError("neighbor_k must be positive")
    effective_k = min(int(neighbor_k), len(normalized_ids) - 1)

    try:
        import numpy as np
    except ImportError:  # pragma: no cover - standard-library fallback is environment-specific.
        return _compare_embeddings_stdlib(
            reference,
            candidate,
            normalized_ids,
            neighbor_k=effective_k,
        )

    reference_array = np.asarray(reference, dtype=np.float32)
    candidate_array = np.asarray(candidate, dtype=np.float32)
    if reference_array.ndim != 2 or candidate_array.ndim != 2:
        raise ValueError("embeddings must be two-dimensional")
    if reference_array.shape != candidate_array.shape:
        raise ValueError("reference and candidate shapes differ")
    if reference_array.shape[0] != len(normalized_ids) or reference_array.shape[1] == 0:
        raise ValueError("embedding shape does not match image IDs")
    if not bool(np.isfinite(reference_array).all()) or not bool(np.isfinite(candidate_array).all()):
        raise ValueError("embeddings contain non-finite values")
    for label, values in (("reference", reference_array), ("candidate", candidate_array)):
        norms = np.linalg.norm(values, axis=1)
        if not bool(np.all(np.abs(norms - 1.0) <= UNIT_NORM_TOLERANCE)):
            raise ValueError(f"{label} embeddings are not unit normalized")

    paired_cosines = np.einsum(
        "ij,ij->i",
        reference_array.astype(np.float64),
        candidate_array.astype(np.float64),
    )
    mean_drift = float(np.maximum(0.0, 1.0 - np.minimum(1.0, paired_cosines)).mean())
    ids_array = np.asarray(normalized_ids, dtype=np.int64)

    def neighbor_sets(values: Any) -> list[set[int]]:
        similarities = values @ values.T
        result: list[set[int]] = []
        for query_position in range(len(normalized_ids)):
            order = np.lexsort((ids_array, -similarities[query_position]))
            positions = [int(position) for position in order if int(position) != query_position][
                :effective_k
            ]
            result.append({normalized_ids[position] for position in positions})
        return result

    reference_neighbors = neighbor_sets(reference_array)
    candidate_neighbors = neighbor_sets(candidate_array)
    overlaps = [
        len(reference_set & candidate_set) / effective_k
        for reference_set, candidate_set in zip(reference_neighbors, candidate_neighbors)
    ]
    return {
        "sample_count": int(reference_array.shape[0]),
        "neighbor_k": effective_k,
        "mean_cosine_drift": mean_drift,
        "mean_top_k_neighbor_overlap": float(sum(overlaps) / len(overlaps)),
    }


def case_passes_guardrails(
    run: dict[str, Any],
    *,
    max_mean_cosine_drift: float = MAX_MEAN_COSINE_DRIFT,
    min_mean_neighbor_overlap: float = MIN_MEAN_TOP50_NEIGHBOR_OVERLAP,
) -> bool:
    """Return whether one successful benchmark run satisfies the project guardrails."""

    if run.get("status") != "ok":
        return False
    comparison = run.get("comparison_to_fp32_reference")
    if not isinstance(comparison, dict):
        return False
    try:
        drift = float(comparison["mean_cosine_drift"])
        overlap = float(comparison["mean_top_k_neighbor_overlap"])
    except (KeyError, TypeError, ValueError):
        return False
    return (
        math.isfinite(drift)
        and math.isfinite(overlap)
        and drift <= max_mean_cosine_drift
        and overlap >= min_mean_neighbor_overlap
    )


def recommend_benchmark_case(
    runs: Sequence[dict[str, Any]],
    *,
    max_mean_cosine_drift: float = MAX_MEAN_COSINE_DRIFT,
    min_mean_neighbor_overlap: float = MIN_MEAN_TOP50_NEIGHBOR_OVERLAP,
) -> dict[str, Any] | None:
    """Choose the fastest passing run, with stable conservative secondary ties."""

    passing: list[dict[str, Any]] = []
    for run in runs:
        if not case_passes_guardrails(
            run,
            max_mean_cosine_drift=max_mean_cosine_drift,
            min_mean_neighbor_overlap=min_mean_neighbor_overlap,
        ):
            continue
        try:
            throughput = float(run["throughput_images_per_second"])
            peak_memory = int(run["peak_cuda_memory_bytes"])
            batch_size = int(run["batch_size"])
        except (KeyError, TypeError, ValueError):
            continue
        if not math.isfinite(throughput) or throughput <= 0 or peak_memory < 0:
            continue
        candidate = dict(run)
        candidate["throughput_images_per_second"] = throughput
        candidate["peak_cuda_memory_bytes"] = peak_memory
        candidate["batch_size"] = batch_size
        passing.append(candidate)
    if not passing:
        return None

    precision_rank = {"fp32": 0, "amp_fp16": 1}
    passing.sort(
        key=lambda run: (
            -float(run["throughput_images_per_second"]),
            int(run["peak_cuda_memory_bytes"]),
            int(run["batch_size"]),
            precision_rank.get(str(run.get("precision")), 2),
            str(run.get("case_id", "")),
        )
    )
    selected = passing[0]
    return {
        "case_id": str(selected.get("case_id", "")),
        "precision": str(selected["precision"]),
        "batch_size": int(selected["batch_size"]),
        "throughput_images_per_second": float(selected["throughput_images_per_second"]),
        "peak_cuda_memory_bytes": int(selected["peak_cuda_memory_bytes"]),
        "comparison_to_fp32_reference": dict(selected["comparison_to_fp32_reference"]),
        "reason": "fastest successful case satisfying both precision guardrails",
    }


def benchmark_frozen_clip_embeddings(
    *,
    source_manifest: Path,
    raw_root: Path,
    sample_size: int,
    seed: int,
    batch_sizes: Sequence[int],
    device: str = "cuda:0",
    checkpoint_cache: Path | None = None,
) -> dict[str, Any]:
    """Benchmark FP32 and AMP-FP16 frozen inference on a deterministic real sample."""

    if sample_size <= NEIGHBOR_COUNT:
        raise ValueError(
            f"sample_size must exceed {NEIGHBOR_COUNT} for top-{NEIGHBOR_COUNT} comparison"
        )
    normalized_batch_sizes = sorted({int(batch_size) for batch_size in batch_sizes})
    if not normalized_batch_sizes or normalized_batch_sizes[0] <= 0:
        raise ValueError("batch_sizes must contain positive integers")
    if not device.startswith("cuda"):
        raise ValueError("the visual benchmark requires a CUDA device")

    try:
        import numpy as np
        import open_clip
        import PIL
        import torch
        import torchvision
        from PIL import Image
    except ImportError as error:  # pragma: no cover - exercised in the visual environment.
        raise RuntimeError(
            "Visual benchmarking requires numpy, Pillow, torch, torchvision, and open_clip"
        ) from error

    if not torch.cuda.is_available():
        raise RuntimeError(f"Requested CUDA device is unavailable: {device}")
    device_object = torch.device(device)
    source_manifest = Path(source_manifest)
    raw_root = Path(raw_root)
    records = canonical_training_records(
        read_manifest(source_manifest), raw_root=raw_root, check_files=True
    )
    sampled_records = deterministic_sample_records(records, sample_size=sample_size, seed=seed)
    sample_ids = [record.image_id for record in sampled_records]

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

    preprocess_started = time.perf_counter()
    prepared_images = []
    for record in sampled_records:
        image_path = raw_root / record.relative_path
        try:
            with Image.open(image_path) as image:
                prepared_images.append(eval_preprocess(image.convert("RGB")))
        except Exception as error:
            raise RuntimeError(
                f"Failed to decode benchmark image {record.image_id} at {image_path}"
            ) from error
    pixels = torch.stack(prepared_images)
    del prepared_images
    preprocess_seconds = time.perf_counter() - preprocess_started
    if pixels.ndim != 4 or pixels.shape[0] != sample_size:
        raise RuntimeError(f"Unexpected preprocessed tensor shape: {tuple(pixels.shape)}")

    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    outputs_by_case: dict[str, Any] = {}
    runs: list[dict[str, Any]] = []

    def encode_case(precision: str, batch_size: int) -> tuple[dict[str, Any], Any | None]:
        case_id = f"{precision}_bs{batch_size}"
        run: dict[str, Any] = {
            "case_id": case_id,
            "precision": precision,
            "batch_size": batch_size,
            "sample_count": sample_size,
        }
        autocast_factory = (
            (lambda: torch.autocast(device_type="cuda", dtype=torch.float16))
            if precision == "amp_fp16"
            else nullcontext
        )
        try:
            torch.cuda.empty_cache()
            warmup = pixels[: min(batch_size, sample_size)].to(device_object)
            with torch.inference_mode(), autocast_factory():
                warmup_output = model.encode_image(warmup, normalize=False)
                warmup_output = warmup_output.float()
                warmup_output /= torch.linalg.vector_norm(warmup_output, dim=1, keepdim=True)
            torch.cuda.synchronize(device_object)
            del warmup, warmup_output

            torch.cuda.reset_peak_memory_stats(device_object)
            encoded_batches = []
            started = time.perf_counter()
            with torch.inference_mode():
                for start in range(0, sample_size, batch_size):
                    batch = pixels[start : start + batch_size].to(device_object)
                    with autocast_factory():
                        features = model.encode_image(batch, normalize=False)
                    features = features.float()
                    if features.ndim != 2 or features.shape[1] != EXPECTED_EMBEDDING_DIMENSION:
                        raise RuntimeError(f"Unexpected feature shape: {tuple(features.shape)}")
                    if not bool(torch.isfinite(features).all()):
                        raise RuntimeError("CLIP produced non-finite benchmark features")
                    norms = torch.linalg.vector_norm(features, dim=1, keepdim=True)
                    if not bool((norms > 0).all()):
                        raise RuntimeError("CLIP produced zero-norm benchmark features")
                    encoded_batches.append((features / norms).cpu())
                    del batch, features, norms
            torch.cuda.synchronize(device_object)
            elapsed = time.perf_counter() - started
            output = torch.cat(encoded_batches, dim=0).numpy().astype(np.float32, copy=False)
            del encoded_batches
            run.update(
                {
                    "status": "ok",
                    "elapsed_seconds": float(elapsed),
                    "throughput_images_per_second": float(sample_size / elapsed),
                    "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device_object)),
                    "output_dtype": "float32",
                    "l2_normalized": True,
                }
            )
            return run, output
        except Exception as error:  # noqa: BLE001 - every benchmark case is reported.
            message = str(error)
            is_oom = error.__class__.__name__ == "OutOfMemoryError" or (
                "out of memory" in message.lower()
            )
            run.update(
                {
                    "status": "oom" if is_oom else "failed",
                    "error_type": error.__class__.__name__,
                    "error": message,
                    "throughput_images_per_second": None,
                    "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device_object)),
                }
            )
            gc.collect()
            torch.cuda.empty_cache()
            return run, None

    for precision in ("fp32", "amp_fp16"):
        for batch_size in normalized_batch_sizes:
            run, output = encode_case(precision, batch_size)
            runs.append(run)
            if output is not None:
                outputs_by_case[run["case_id"]] = output

    fp32_runs = [run for run in runs if run["precision"] == "fp32" and run["status"] == "ok"]
    reference_case_id = fp32_runs[0]["case_id"] if fp32_runs else None
    if reference_case_id is not None:
        reference_output = outputs_by_case[reference_case_id]
        for run in runs:
            output = outputs_by_case.get(run["case_id"])
            if output is None:
                run["comparison_to_fp32_reference"] = None
                run["guardrail_pass"] = False
                continue
            comparison = compare_embedding_outputs(
                reference_output,
                output,
                sample_ids,
                neighbor_k=NEIGHBOR_COUNT,
            )
            run["comparison_to_fp32_reference"] = comparison
            run["guardrail_pass"] = case_passes_guardrails(run)
    else:
        for run in runs:
            run["comparison_to_fp32_reference"] = None
            run["guardrail_pass"] = False

    recommendation = recommend_benchmark_case(runs)
    properties = torch.cuda.get_device_properties(device_object)
    failures = [
        {
            "case_id": run["case_id"],
            "status": run["status"],
            "error_type": run.get("error_type"),
            "error": run.get("error"),
        }
        for run in runs
        if run["status"] != "ok"
    ]
    report: dict[str, Any] = {
        "schema_version": 1,
        "kind": "frozen_clip_embedding_benchmark",
        "algorithm_version": BENCHMARK_ALGORITHM_VERSION,
        "source": {
            "manifest": source_manifest.as_posix(),
            "manifest_sha256": sha256_file(source_manifest),
            "raw_root": raw_root.as_posix(),
            "split": "train2017",
        },
        "sample": {
            "method": "sha256(seed, coco_image_id) rank; ascending-ID execution",
            "seed": int(seed),
            "size": int(sample_size),
            "image_ids": [int(image_id) for image_id in sample_ids],
            "image_ids_sha256": image_ids_sha256(sample_ids),
        },
        "checkpoint": {
            **checkpoint_descriptor(),
            "config_sha256": pinned.config_sha256,
            "frozen": True,
            "eval_mode": True,
        },
        "embedding": {
            "dimension": EXPECTED_EMBEDDING_DIMENSION,
            "output_dtype": "float32",
            "l2_normalized": True,
            "feature": "projected_encode_image",
            "preprocess": "open_clip checkpoint-native eval transform",
            "eval_transform_repr": repr(eval_preprocess),
        },
        "guardrails": {
            "provenance": "project guardrails, not literature-derived thresholds",
            "max_mean_cosine_drift": MAX_MEAN_COSINE_DRIFT,
            "min_mean_top50_neighbor_overlap": MIN_MEAN_TOP50_NEIGHBOR_OVERLAP,
            "neighbor_count": NEIGHBOR_COUNT,
            "neighbor_tie_break": "ascending COCO image ID",
        },
        "versions": {
            "python": platform.python_version(),
            "numpy": str(np.__version__),
            "pillow": str(PIL.__version__),
            "torch": str(torch.__version__),
            "torchvision": str(torchvision.__version__),
            "open_clip": str(getattr(open_clip, "__version__", "unknown")),
            "cuda_runtime": str(torch.version.cuda),
        },
        "hardware": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "device": str(device_object),
            "gpu_name": str(properties.name),
            "gpu_compute_capability": [int(properties.major), int(properties.minor)],
            "gpu_total_memory_bytes": int(properties.total_memory),
            "cuda_device_count": int(torch.cuda.device_count()),
        },
        "preprocessing": {
            "seconds": float(preprocess_seconds),
            "tensor_shape": [int(value) for value in pixels.shape],
            "tensor_dtype": str(pixels.dtype),
        },
        "fp32_reference_case_id": reference_case_id,
        "runs": runs,
        "failures": failures,
        "recommendation": recommendation,
    }
    return report
