from pathlib import Path
import unittest

from vlm_data.visual_neighbors import (
    canonicalize_neighbor_candidates,
    neighbor_artifact_paths,
)


class VisualNeighborContractTests(unittest.TestCase):
    def test_artifact_paths_are_unambiguous(self) -> None:
        paths = neighbor_artifact_paths(Path("cache/clip_neighbors_k400.npy"))
        self.assertEqual(
            paths.image_ids,
            Path("cache/clip_neighbors_k400.image_ids.npy"),
        )
        self.assertEqual(paths.metadata, Path("cache/clip_neighbors_k400.meta.json"))
        with self.assertRaises(ValueError):
            neighbor_artifact_paths(Path("cache/neighbors.bin"))

    def test_self_is_explicit_and_nonself_ties_use_image_id(self) -> None:
        row = canonicalize_neighbor_candidates(
            query_position=0,
            candidate_positions=[0, 3, 2, 1],
            candidate_similarities=[0.9999999, 0.8, 0.8, 0.9],
            image_ids=[10, 40, 20, 30],
            k=4,
        )
        self.assertEqual(row.image_ids, (10, 40, 20, 30))
        self.assertEqual(row.positions, (0, 1, 2, 3))
        self.assertEqual(row.similarities[0], 1.0)

    def test_backend_self_and_duplicate_positions_are_removed(self) -> None:
        row = canonicalize_neighbor_candidates(
            query_position=1,
            candidate_positions=[1, 1, 2, 2, 0],
            candidate_similarities=[1.0, 0.99, 0.7, 0.8, 0.6],
            image_ids=[10, 20, 30],
            k=3,
        )
        self.assertEqual(row.image_ids, (20, 30, 10))
        self.assertEqual(row.similarities, (1.0, 0.8, 0.6))

    def test_invalid_or_short_rows_fail_closed(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "need 2"):
            canonicalize_neighbor_candidates(
                query_position=0,
                candidate_positions=[0, 1],
                candidate_similarities=[1.0, 0.5],
                image_ids=[10, 20, 30],
                k=3,
            )
        with self.assertRaisesRegex(ValueError, "finite"):
            canonicalize_neighbor_candidates(
                query_position=0,
                candidate_positions=[1],
                candidate_similarities=[float("nan")],
                image_ids=[10, 20],
                k=1,
                include_self=False,
            )


if __name__ == "__main__":
    unittest.main()
