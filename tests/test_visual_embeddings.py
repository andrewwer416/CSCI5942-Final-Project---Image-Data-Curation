import hashlib
from pathlib import Path
import tempfile
import unittest

from vlm_data.manifest import ImageRecord
from vlm_data.visual_embeddings import (
    assert_frozen_eval_model,
    canonical_training_records,
    checkpoint_descriptor,
    embedding_artifact_paths,
    image_ids_sha256,
)


def _record(image_id: int, *, split: str = "train2017") -> ImageRecord:
    file_name = f"{image_id:012d}.jpg"
    return ImageRecord(
        image_id=image_id,
        split=split,
        file_name=file_name,
        relative_path=f"{split}/{file_name}",
        width=20,
        height=10,
        captions=(),
        category_ids=(),
        object_count=0,
    )


class _Parameter:
    def __init__(self, requires_grad: bool) -> None:
        self.requires_grad = requires_grad


class _Model:
    def __init__(self, *, training: bool, trainable: bool) -> None:
        self.training = training
        self._parameters = [_Parameter(trainable)]

    def parameters(self):
        return iter(self._parameters)


class VisualEmbeddingContractTests(unittest.TestCase):
    def test_artifact_paths_are_unambiguous(self) -> None:
        paths = embedding_artifact_paths(Path("cache/train_clip.npy"))
        self.assertEqual(paths.image_ids, Path("cache/train_clip.image_ids.npy"))
        self.assertEqual(paths.metadata, Path("cache/train_clip.meta.json"))
        with self.assertRaises(ValueError):
            embedding_artifact_paths(Path("cache/train_clip.bin"))

    def test_image_id_hash_preserves_order(self) -> None:
        expected = hashlib.sha256(b"3\n7\n11\n").hexdigest()
        self.assertEqual(image_ids_sha256([3, 7, 11]), expected)
        self.assertNotEqual(image_ids_sha256([11, 7, 3]), expected)

    def test_checkpoint_descriptor_is_pinned(self) -> None:
        descriptor = checkpoint_descriptor()
        self.assertEqual(descriptor["model_name"], "ViT-B-16")
        self.assertEqual(descriptor["pretrained_tag"], "datacomp_xl_s13b_b90k")
        self.assertEqual(len(descriptor["hf_revision"]), 40)
        self.assertEqual(len(descriptor["weight_sha256"]), 64)

    def test_model_must_be_eval_and_frozen(self) -> None:
        assert_frozen_eval_model(_Model(training=False, trainable=False))
        with self.assertRaisesRegex(RuntimeError, "eval mode"):
            assert_frozen_eval_model(_Model(training=True, trainable=False))
        with self.assertRaisesRegex(RuntimeError, "requires_grad"):
            assert_frozen_eval_model(_Model(training=False, trainable=True))

    def test_records_are_sorted_and_files_checked(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            raw_root = Path(directory)
            train = raw_root / "train2017"
            train.mkdir()
            for image_id in (7, 3):
                (train / f"{image_id:012d}.jpg").write_bytes(b"synthetic")
            records = canonical_training_records(
                [_record(7), _record(3)], raw_root=raw_root
            )
            self.assertEqual([record.image_id for record in records], [3, 7])

    def test_records_fail_closed_on_duplicates_split_and_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            raw_root = Path(directory)
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                canonical_training_records(
                    [_record(1), _record(1)], raw_root=raw_root, check_files=False
                )
            with self.assertRaisesRegex(ValueError, "expected 'train2017'"):
                canonical_training_records(
                    [_record(1, split="val2017")], raw_root=raw_root, check_files=False
                )
            with self.assertRaises(FileNotFoundError):
                canonical_training_records([_record(1)], raw_root=raw_root)


if __name__ == "__main__":
    unittest.main()
