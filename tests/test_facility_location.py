import math
import random
import unittest

from vlm_data.facility_location import (
    SparseFacilityGraph,
    facility_location_value,
    select_facility_location_lazy,
    select_facility_location_naive,
    select_facility_location_numpy,
)


def _graph(image_ids: list[int], rows: list[list[tuple[int, float]]]) -> SparseFacilityGraph:
    indptr = [0]
    indices: list[int] = []
    similarities: list[float] = []
    for row in rows:
        for candidate, similarity in row:
            indices.append(candidate)
            similarities.append(similarity)
        indptr.append(len(indices))
    return SparseFacilityGraph.from_forward_csr(
        image_ids=image_ids,
        indptr=indptr,
        indices=indices,
        similarities=similarities,
    )


class FacilityLocationTests(unittest.TestCase):
    def test_lazy_matches_naive_on_sparse_graph_with_constraints(self) -> None:
        generator = random.Random(5942)
        image_ids = [90, 10, 70, 20, 60, 30, 50, 40]
        rows: list[list[tuple[int, float]]] = []
        for demand in range(len(image_ids)):
            neighbors = {demand}
            while len(neighbors) < 4:
                neighbors.add(generator.randrange(len(image_ids)))
            rows.append(
                [
                    (candidate, 1.0 if candidate == demand else generator.uniform(0.1, 0.9))
                    for candidate in sorted(neighbors)
                ]
            )
        graph = _graph(image_ids, rows)
        options = {
            "budget": 5,
            "duplicate_group_by_image_id": {10: "near-duplicate", 20: "near-duplicate"},
            "outlier_image_ids": {30, 40, 50},
            "max_outliers": 1,
        }

        naive = select_facility_location_naive(graph, **options)
        lazy = select_facility_location_lazy(graph, **options)

        self.assertEqual(lazy.selected_image_ids, naive.selected_image_ids)
        self.assertEqual(lazy.marginal_gains, naive.marginal_gains)
        self.assertEqual(lazy.best_similarities, naive.best_similarities)
        self.assertEqual(lazy.objective_value, naive.objective_value)
        self.assertEqual(lazy.outlier_count, 1)
        self.assertLessEqual(
            len({10, 20} & set(lazy.selected_image_ids)),
            1,
        )
        self.assertAlmostEqual(
            lazy.objective_value,
            facility_location_value(graph, lazy.selected_image_ids),
        )

    def test_exact_gain_ties_use_smallest_image_id(self) -> None:
        graph = _graph(
            [50, 10, 30],
            [[(0, 1.0)], [(1, 1.0)], [(2, 1.0)]],
        )
        result = select_facility_location_lazy(graph, budget=3)
        self.assertEqual(result.selected_image_ids, (10, 30, 50))
        self.assertEqual(result.marginal_gains, (1.0, 1.0, 1.0))

    def test_zero_budget_is_valid(self) -> None:
        graph = _graph([1], [[(0, 1.0)]])
        result = select_facility_location_lazy(graph, budget=0)
        self.assertEqual(result.selected_image_ids, ())
        self.assertEqual(result.objective_value, 0.0)

    def test_constraints_fail_before_an_infeasible_selection(self) -> None:
        graph = _graph(
            [1, 2, 3],
            [[(0, 1.0)], [(1, 1.0)], [(2, 1.0)]],
        )
        with self.assertRaisesRegex(ValueError, "constraints"):
            select_facility_location_lazy(
                graph,
                budget=2,
                duplicate_group_by_image_id={1: "same", 2: "same", 3: "same"},
            )
        with self.assertRaisesRegex(ValueError, "constraints"):
            select_facility_location_lazy(
                graph,
                budget=2,
                outlier_image_ids={1, 2, 3},
                max_outliers=1,
            )

    def test_unknown_constraint_ids_and_invalid_caps_fail(self) -> None:
        graph = _graph([1], [[(0, 1.0)]])
        with self.assertRaisesRegex(ValueError, "unknown image ID 2"):
            select_facility_location_lazy(
                graph,
                budget=1,
                duplicate_group_by_image_id={2: "missing"},
            )
        with self.assertRaisesRegex(ValueError, "unknown image ID 2"):
            select_facility_location_lazy(
                graph,
                budget=1,
                outlier_image_ids={2},
            )
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            select_facility_location_lazy(graph, budget=1, max_outliers=-1)
        with self.assertRaisesRegex(ValueError, "budget must be nonnegative"):
            select_facility_location_lazy(graph, budget=-1)

    def test_malformed_forward_csr_fails(self) -> None:
        cases = [
            {
                "image_ids": [1, 2],
                "indptr": [0, 1],
                "indices": [0],
                "similarities": [1.0],
            },
            {
                "image_ids": [1, 2],
                "indptr": [0, 1, 1],
                "indices": [2],
                "similarities": [1.0],
            },
            {
                "image_ids": [1],
                "indptr": [0, 2],
                "indices": [0, 0],
                "similarities": [1.0, 0.9],
            },
            {
                "image_ids": [1],
                "indptr": [0, 1],
                "indices": [0],
                "similarities": [math.nan],
            },
            {
                "image_ids": [1],
                "indptr": [0, 1],
                "indices": [0],
                "similarities": [-0.1],
            },
        ]
        for case in cases:
            with self.subTest(case=case), self.assertRaises(ValueError):
                SparseFacilityGraph.from_forward_csr(**case)

    def test_duplicate_image_ids_and_forward_reverse_edge_mismatch_fail(self) -> None:
        with self.assertRaisesRegex(ValueError, "image_ids must be unique"):
            _graph([1, 1], [[(0, 1.0)], [(1, 1.0)]])

        with self.assertRaisesRegex(ValueError, "same edge count"):
            SparseFacilityGraph(
                image_ids=[1],
                forward_indptr=[0, 1],
                forward_indices=[0],
                forward_similarities=[1.0],
                reverse_indptr=[0, 0],
                reverse_indices=[],
                reverse_similarities=[],
            )

    def test_facility_value_rejects_unknown_or_repeated_ids(self) -> None:
        graph = _graph([1, 2], [[(0, 1.0)], [(1, 1.0)]])
        with self.assertRaisesRegex(ValueError, "unique"):
            facility_location_value(graph, [1, 1])
        with self.assertRaisesRegex(ValueError, "unknown"):
            facility_location_value(graph, [3])

    def test_numpy_selector_matches_reference_with_constraints_and_mask(self) -> None:
        try:
            import numpy as np
        except ImportError:
            self.skipTest("NumPy is an optional visual-selection dependency")

        graph = _graph(
            [90, 10, 70, 20, 60, 30],
            [
                [(0, 1.0), (1, 0.8), (4, 0.4)],
                [(0, 0.8), (1, 1.0), (2, 0.5)],
                [(1, 0.5), (2, 1.0), (3, 0.7)],
                [(2, 0.7), (3, 1.0), (5, 0.4)],
                [(0, 0.4), (4, 1.0), (5, 0.9)],
                [(3, 0.4), (4, 0.9), (5, 1.0)],
            ],
        )
        allowed = np.array([False, True, True, True, True, True])
        groups = np.array([-1, 4, -1, 4, -1, -1], dtype=np.int32)
        outliers = np.array([False, False, True, False, True, False])
        accelerated = select_facility_location_numpy(
            image_ids=np.asarray(graph.image_ids, dtype=np.int64),
            reverse_indptr=np.asarray(graph.reverse_indptr, dtype=np.int64),
            reverse_indices=np.asarray(graph.reverse_indices, dtype=np.int32),
            reverse_similarities=np.asarray(graph.reverse_similarities, dtype=np.float32),
            budget=3,
            allowed_candidate_mask=allowed,
            duplicate_groups=groups,
            outlier_mask=outliers,
            max_outliers=1,
        )
        reference = select_facility_location_naive(
            graph,
            budget=3,
            duplicate_group_by_image_id={10: 4, 20: 4},
            outlier_image_ids={70, 60},
            max_outliers=1,
        )

        # Candidate 90 is disallowed. It is not selected by this fixture's
        # unconstrained reference either, so both paths exercise the same pool.
        self.assertNotIn(90, reference.selected_image_ids)
        self.assertEqual(accelerated.selected_image_ids, reference.selected_image_ids)
        self.assertEqual(accelerated.outlier_count, reference.outlier_count)
        self.assertAlmostEqual(accelerated.objective_value, reference.objective_value, places=6)
        for accelerated_gain, reference_gain in zip(
            accelerated.marginal_gains, reference.marginal_gains
        ):
            self.assertAlmostEqual(accelerated_gain, reference_gain, places=6)

    def test_numpy_selector_uses_image_id_ties_and_allowed_mask(self) -> None:
        try:
            import numpy as np
        except ImportError:
            self.skipTest("NumPy is an optional visual-selection dependency")

        graph = _graph([50, 10, 30], [[(0, 1.0)], [(1, 1.0)], [(2, 1.0)]])
        result = select_facility_location_numpy(
            image_ids=np.asarray(graph.image_ids),
            reverse_indptr=np.asarray(graph.reverse_indptr),
            reverse_indices=np.asarray(graph.reverse_indices),
            reverse_similarities=np.asarray(graph.reverse_similarities),
            budget=2,
            allowed_candidate_mask=np.array([True, True, False]),
        )
        self.assertEqual(result.selected_image_ids, (10, 50))

    def test_numpy_selector_vectorially_validates_inputs(self) -> None:
        try:
            import numpy as np
        except ImportError:
            self.skipTest("NumPy is an optional visual-selection dependency")

        valid = {
            "image_ids": np.array([1, 2]),
            "reverse_indptr": np.array([0, 1, 2]),
            "reverse_indices": np.array([0, 1]),
            "reverse_similarities": np.array([1.0, 1.0]),
            "budget": 1,
        }
        invalid_overrides = [
            {"image_ids": np.array([1.0, 2.0])},
            {"reverse_indptr": np.array([0, 2, 1])},
            {"reverse_indices": np.array([0, 2])},
            {"reverse_similarities": np.array([1.0, np.nan])},
            {"allowed_candidate_mask": np.array([1, 0])},
            {"duplicate_groups": np.array([0.0, -1.0])},
            {"outlier_mask": np.array([False])},
        ]
        for override in invalid_overrides:
            with self.subTest(override=override):
                arguments = valid | override
                with self.assertRaises(ValueError):
                    select_facility_location_numpy(**arguments)

    def test_numpy_selector_rejects_infeasible_masked_constraints(self) -> None:
        try:
            import numpy as np
        except ImportError:
            self.skipTest("NumPy is an optional visual-selection dependency")

        with self.assertRaisesRegex(ValueError, "constraints"):
            select_facility_location_numpy(
                image_ids=np.array([1, 2, 3]),
                reverse_indptr=np.array([0, 1, 2, 3]),
                reverse_indices=np.array([0, 1, 2]),
                reverse_similarities=np.ones(3),
                budget=2,
                allowed_candidate_mask=np.array([True, True, False]),
                duplicate_groups=np.array([8, 8, -1]),
            )


if __name__ == "__main__":
    unittest.main()
