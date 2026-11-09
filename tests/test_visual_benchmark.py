import json
import math
import unittest

from vlm_data.manifest import ImageRecord
from vlm_data.visual_benchmark import (
    case_passes_guardrails,
    compare_embedding_outputs,
    deterministic_sample_records,
    recommend_benchmark_case,
)


def _record(image_id: int) -> ImageRecord:
    file_name = f"{image_id:012d}.jpg"
    return ImageRecord(
        image_id=image_id,
        split="train2017",
        file_name=file_name,
        relative_path=f"train2017/{file_name}",
        width=10,
        height=10,
        captions=(),
        category_ids=(),
        object_count=0,
    )


def _run(
    case_id: str,
    *,
    throughput: float,
    drift: float,
    overlap: float,
    precision: str = "amp_fp16",
    batch_size: int = 32,
    peak_memory: int = 100,
    status: str = "ok",
) -> dict:
    return {
        "case_id": case_id,
        "status": status,
        "precision": precision,
        "batch_size": batch_size,
        "throughput_images_per_second": throughput,
        "peak_cuda_memory_bytes": peak_memory,
        "comparison_to_fp32_reference": {
            "mean_cosine_drift": drift,
            "mean_top_k_neighbor_overlap": overlap,
        },
    }


class VisualBenchmarkPureTests(unittest.TestCase):
    def test_sample_is_seeded_order_independent_and_id_sorted(self) -> None:
        forward = [_record(image_id) for image_id in range(1, 21)]
        reverse = list(reversed(forward))
        first = deterministic_sample_records(forward, sample_size=7, seed=5942)
        second = deterministic_sample_records(reverse, sample_size=7, seed=5942)
        other_seed = deterministic_sample_records(forward, sample_size=7, seed=5943)

        first_ids = [record.image_id for record in first]
        self.assertEqual(first_ids, [record.image_id for record in second])
        self.assertEqual(first_ids, sorted(first_ids))
        self.assertNotEqual(first_ids, [record.image_id for record in other_seed])

    def test_sampling_rejects_invalid_size_and_duplicate_ids(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive"):
            deterministic_sample_records([_record(1)], sample_size=0, seed=1)
        with self.assertRaisesRegex(ValueError, "exceeds"):
            deterministic_sample_records([_record(1)], sample_size=2, seed=1)
        with self.assertRaisesRegex(ValueError, "unique"):
            deterministic_sample_records([_record(1), _record(1)], sample_size=1, seed=1)

    def test_identical_embeddings_have_zero_drift_and_full_overlap(self) -> None:
        root_two = math.sqrt(0.5)
        embeddings = [
            [1.0, 0.0],
            [0.0, 1.0],
            [root_two, root_two],
            [-1.0, 0.0],
        ]
        comparison = compare_embedding_outputs(
            embeddings, embeddings, [40, 10, 30, 20], neighbor_k=2
        )
        self.assertAlmostEqual(comparison["mean_cosine_drift"], 0.0, places=7)
        self.assertEqual(comparison["mean_top_k_neighbor_overlap"], 1.0)
        self.assertEqual(comparison["neighbor_k"], 2)

    def test_neighbor_similarity_ties_use_smaller_coco_id(self) -> None:
        reference = [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, -1.0],
        ]
        candidate = [
            [1.0, 0.0],
            [0.1, math.sqrt(0.99)],
            [0.0, -1.0],
        ]
        comparison = compare_embedding_outputs(
            reference,
            candidate,
            [100, 20, 10],
            neighbor_k=1,
        )

        # Image 100's FP32 tie is resolved to ID 10, while the perturbed candidate
        # favors ID 20. The other two rows retain their nearest neighbor.
        self.assertAlmostEqual(
            comparison["mean_top_k_neighbor_overlap"],
            2.0 / 3.0,
        )

    def test_comparison_rejects_nonunit_and_duplicate_ids(self) -> None:
        with self.assertRaisesRegex(ValueError, "unit normalized"):
            compare_embedding_outputs(
                [[2.0, 0.0], [0.0, 1.0]],
                [[1.0, 0.0], [0.0, 1.0]],
                [1, 2],
                neighbor_k=1,
            )
        with self.assertRaisesRegex(ValueError, "unique"):
            compare_embedding_outputs(
                [[1.0, 0.0], [0.0, 1.0]],
                [[1.0, 0.0], [0.0, 1.0]],
                [1, 1],
                neighbor_k=1,
            )

    def test_guardrails_are_inclusive(self) -> None:
        run = _run(
            "boundary",
            throughput=100.0,
            drift=1e-4,
            overlap=0.99,
        )
        self.assertTrue(case_passes_guardrails(run))
        run["comparison_to_fp32_reference"]["mean_cosine_drift"] = 1.0001e-4
        self.assertFalse(case_passes_guardrails(run))

    def test_recommendation_uses_fastest_passing_case(self) -> None:
        runs = [
            _run(
                "fp32_bs16",
                throughput=80.0,
                drift=0.0,
                overlap=1.0,
                precision="fp32",
                batch_size=16,
            ),
            _run(
                "amp_fast_but_drifted",
                throughput=250.0,
                drift=2e-4,
                overlap=1.0,
            ),
            _run(
                "amp_bs32",
                throughput=180.0,
                drift=5e-5,
                overlap=0.995,
            ),
            _run(
                "amp_oom",
                throughput=999.0,
                drift=0.0,
                overlap=1.0,
                status="oom",
            ),
        ]
        recommendation = recommend_benchmark_case(runs)
        self.assertIsNotNone(recommendation)
        self.assertEqual(recommendation["case_id"], "amp_bs32")
        json.dumps(recommendation)

    def test_recommendation_ties_prefer_memory_then_smaller_batch(self) -> None:
        runs = [
            _run(
                "large",
                throughput=100.0,
                drift=0.0,
                overlap=1.0,
                batch_size=64,
                peak_memory=90,
            ),
            _run(
                "small",
                throughput=100.0,
                drift=0.0,
                overlap=1.0,
                batch_size=16,
                peak_memory=90,
            ),
            _run(
                "low-memory",
                throughput=100.0,
                drift=0.0,
                overlap=1.0,
                batch_size=128,
                peak_memory=80,
            ),
        ]
        recommendation = recommend_benchmark_case(runs)
        self.assertEqual(recommendation["case_id"], "low-memory")


if __name__ == "__main__":
    unittest.main()
