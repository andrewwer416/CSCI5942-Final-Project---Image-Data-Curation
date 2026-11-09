import importlib.util
from pathlib import Path
import tempfile
import unittest

from vlm_data.visual_analysis import (
    build_duplicate_groups,
    compute_visual_density,
    derive_sparse_csr,
    enforce_random_difference,
    map_neighbor_ids_to_positions,
    mark_low_density_outliers,
    recommend_neighbor_k,
    render_calibration_markdown,
    write_calibration_report,
)


NUMPY_AVAILABLE = importlib.util.find_spec("numpy") is not None


class VisualAnalysisDecisionTests(unittest.TestCase):
    def test_random_difference_gate(self) -> None:
        comparison = enforce_random_difference(
            [1, 2, 5, 6],
            [1, 2, 3, 4],
            expected_count=4,
            minimum_replacements=2,
        )
        self.assertEqual(comparison["overlap_count"], 2)
        self.assertEqual(comparison["replacement_count"], 2)
        self.assertEqual(comparison["replacement_fraction"], 0.5)
        with self.assertRaisesRegex(ValueError, "at least 3"):
            enforce_random_difference(
                [1, 2, 5, 6],
                [1, 2, 3, 4],
                expected_count=4,
                minimum_replacements=3,
            )

    def test_k_recommendation_uses_all_tolerances(self) -> None:
        recommendation = recommend_neighbor_k(
            {
                50: {
                    "mean_coverage": 0.990,
                    "p95_distance": 0.22,
                    "p99_distance": 0.34,
                },
                100: {
                    "mean_coverage": 0.9995,
                    "p95_distance": 0.204,
                    "p99_distance": 0.309,
                },
                200: {
                    "mean_coverage": 1.0,
                    "p95_distance": 0.200,
                    "p99_distance": 0.300,
                },
                400: {
                    "mean_coverage": 1.0,
                    "p95_distance": 0.200,
                    "p99_distance": 0.300,
                    "hard_requirements_met": False,
                },
            }
        )
        self.assertEqual(recommendation["recommended_k"], 100)
        rows = {row["k"]: row for row in recommendation["candidates"]}
        self.assertFalse(rows[50]["eligible"])
        self.assertTrue(rows[100]["eligible"])
        self.assertFalse(rows[400]["eligible"])

    def test_report_helpers_record_recommendation(self) -> None:
        recommendation = recommend_neighbor_k(
            {
                50: {
                    "mean_coverage": 0.9,
                    "p95_distance": 0.2,
                    "p99_distance": 0.3,
                }
            }
        )
        markdown = render_calibration_markdown(recommendation)
        self.assertIn("Recommended neighbor count: `50`", markdown)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_calibration_report(
                report={"k_recommendation": recommendation},
                json_path=root / "report.json",
                markdown_path=root / "report.md",
            )
            self.assertIn('"recommended_k": 50', (root / "report.json").read_text())
            self.assertEqual((root / "report.md").read_text(), markdown)


@unittest.skipUnless(NUMPY_AVAILABLE, "numpy is an optional visual-analysis dependency")
class VisualAnalysisNumericalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import numpy as np

        cls.np = np

    def test_neighbor_id_mapping_fails_on_unknown_id(self) -> None:
        np = self.np
        source = np.asarray([10, 20, 30], dtype=np.int64)
        neighbors = np.asarray([[10, 20], [20, 30], [30, 10]], dtype=np.int64)
        positions = map_neighbor_ids_to_positions(source, neighbors)
        np.testing.assert_array_equal(positions, [[0, 1], [1, 2], [2, 0]])
        neighbors[1, 1] = 99
        with self.assertRaisesRegex(ValueError, "unknown image IDs: 99"):
            map_neighbor_ids_to_positions(source, neighbors)

    def test_reverse_csr_matches_forward_edges(self) -> None:
        np = self.np
        positions = np.asarray([[0, 1], [1, 2], [2, 0]], dtype=np.int64)
        similarities = np.asarray(
            [[1.0, 0.8], [1.0, 0.7], [1.0, 0.4]], dtype=np.float32
        )
        graph = derive_sparse_csr(positions, similarities, k=2)
        np.testing.assert_array_equal(graph.forward_indptr, [0, 2, 4, 6])
        np.testing.assert_array_equal(graph.forward_indices, [0, 1, 1, 2, 2, 0])
        np.testing.assert_array_equal(graph.reverse_indptr, [0, 2, 4, 6])
        np.testing.assert_array_equal(graph.reverse_indices, [0, 2, 0, 1, 1, 2])
        np.testing.assert_allclose(
            graph.reverse_similarities,
            [1.0, 0.4, 0.8, 1.0, 0.7, 1.0],
        )

    def test_density_and_bottom_fraction_are_deterministic(self) -> None:
        np = self.np
        source = np.asarray([10, 20, 30, 40, 50], dtype=np.int64)
        neighbors = np.asarray(
            [
                [10, 20, 30],
                [20, 10, 30],
                [30, 20, 40],
                [40, 30, 50],
                [50, 30, 40],
            ],
            dtype=np.int64,
        )
        similarities = np.asarray(
            [
                [1.0, 0.2, 0.2],
                [1.0, 0.2, 0.2],
                [1.0, 0.5, 0.5],
                [1.0, 0.8, 0.8],
                [1.0, 0.9, 0.9],
            ],
            dtype=np.float32,
        )
        density = compute_visual_density(
            source, neighbors, similarities, nonself_count=2
        )
        np.testing.assert_allclose(density, [0.2, 0.2, 0.5, 0.8, 0.9])
        outliers = mark_low_density_outliers(source, density, fraction=0.4)
        self.assertEqual(outliers.image_ids, (10, 20))
        np.testing.assert_array_equal(outliers.mask, [True, True, False, False, False])

    def test_representative_groups_do_not_transitively_chain(self) -> None:
        np = self.np
        image_ids = np.asarray([10, 20, 30, 40], dtype=np.int64)
        angles = np.asarray([-0.3, -0.1, 0.1, 0.3], dtype=np.float32)
        embeddings = np.stack((np.cos(angles), np.sin(angles)), axis=1).astype(np.float32)
        # Direct high-similarity chain: 10--20--30--40. Centers 20 and 30
        # tie on degree, so image ID 20 owns the first group {10,20,30}.
        neighbor_ids = np.asarray(
            [
                [10, 20, 30, 40],
                [20, 10, 30, 40],
                [30, 20, 40, 10],
                [40, 30, 20, 10],
            ],
            dtype=np.int64,
        )
        similarities = np.asarray(
            [
                [1.0, 0.99, 0.90, 0.80],
                [1.0, 0.99, 0.99, 0.90],
                [1.0, 0.99, 0.99, 0.90],
                [1.0, 0.99, 0.90, 0.80],
            ],
            dtype=np.float32,
        )
        result = build_duplicate_groups(
            embeddings=embeddings,
            source_image_ids=image_ids,
            neighbor_image_ids=neighbor_ids,
            neighbor_similarities=similarities,
            threshold=0.98,
        )
        self.assertEqual(len(result.groups), 1)
        self.assertEqual(result.groups[0]["center_image_id"], 20)
        self.assertEqual(result.groups[0]["member_image_ids"], [10, 20, 30])
        self.assertEqual(result.groups[0]["medoid_image_id"], 20)
        self.assertEqual(int(result.group_ids[3]), -1)
        np.testing.assert_array_equal(result.candidate_allowed, [False, True, False, True])


if __name__ == "__main__":
    unittest.main()
