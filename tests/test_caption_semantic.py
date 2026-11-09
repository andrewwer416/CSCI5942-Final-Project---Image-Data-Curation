import unittest

from vlm_data.caption_diagnostics import phrase_features, summarize_features


class CaptionDiagnosticTests(unittest.TestCase):
    def test_cache_rejects_corrupted_bytes_and_source_drift(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("Optional NumPy environment")
        import tempfile
        from pathlib import Path
        from vlm_data.caption_semantic import MODEL, REVISION, VERSION, load_embeddings, save_array
        from vlm_data.io import sha256_file, write_json_atomic
        from vlm_data.visual_embeddings import image_ids_sha256

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.write_text("source")
            vectors = np.zeros((2, 768), dtype=np.float32)
            vectors[:, 0] = 1
            ids = np.array([1, 2], dtype=np.int64)
            arrays = {
                "embeddings": vectors,
                "image_ids": ids,
                "captions": np.repeat(vectors[:, None, :], 5, axis=1),
                "dispersion": np.zeros(2),
            }
            artifacts = {}
            for name, array in arrays.items():
                path = root / f"{name}.npy"
                save_array(path, array)
                artifacts[name] = {"file": path.name, "sha256": sha256_file(path)}
            meta = {
                "algorithm_version": VERSION,
                "checkpoint": {"model": MODEL, "revision": REVISION, "frozen": True},
                "source": {
                    "record_count": 2,
                    "manifest_sha256": sha256_file(source),
                    "image_ids_sha256": image_ids_sha256(ids),
                },
                "artifacts": artifacts,
            }
            write_json_atomic(root / "meta.json", meta)
            load_embeddings(root / "meta.json", source_manifest=source)
            source.write_text("changed")
            with self.assertRaisesRegex(ValueError, "source hash"):
                load_embeddings(root / "meta.json", source_manifest=source)
            (root / "embeddings.npy").write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "artifact hash"):
                load_embeddings(root / "meta.json")

    def test_phrase_roles_and_incidence_coverage(self):
        row = phrase_features(
            [
                ("two", "CD"),
                ("dogs", "NNS"),
                ("running", "VBG"),
                ("on", "IN"),
                ("green", "JJ"),
                ("grass", "NN"),
            ]
        )
        self.assertEqual(row["verbs"], {"running"})
        self.assertEqual(row["quantities"], {"two dogs"})
        self.assertEqual(row["relations"], {"on green grass"})
        self.assertEqual(row["scenes"], {"grass"})
        summary = summarize_features([{"a", "b"}, {"a"}, {"c"}], [0])
        self.assertEqual(summary["source_incidence_mass_covered"], 0.75)

    def test_aggregation_is_normalized_and_dispersion_matches_pairs(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("Optional NumPy environment")
        from vlm_data.caption_semantic import aggregate_captions

        x = np.array([[[2, 0], [1, 0], [0, 3], [0, 1], [1, 1]]], dtype=np.float32)
        mean, dispersion = aggregate_captions(x)
        unit = x / np.linalg.norm(x, axis=-1, keepdims=True)
        expected = np.mean([1 - unit[0, i] @ unit[0, j] for i in range(5) for j in range(i)])
        self.assertAlmostEqual(float(dispersion[0]), float(expected), places=5)
        self.assertAlmostEqual(float(np.linalg.norm(mean)), 1, places=5)
        with self.assertRaises(ValueError):
            aggregate_captions(np.zeros((1, 5, 2)))
