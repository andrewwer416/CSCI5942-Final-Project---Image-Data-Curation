import json
from pathlib import Path
import tempfile
import unittest

from vlm_data.coco import build_coco_manifest
from vlm_data.io import sha256_file
from vlm_data.manifest import read_manifest
from vlm_data.selection import select_category_balanced, select_random, write_selection
from vlm_data.validation import validate_manifest


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


class ManifestPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.captions_path = self.root / "captions.json"
        self.instances_path = self.root / "instances.json"
        images = [
            {"id": image_id, "file_name": f"{image_id:012d}.jpg", "width": 20, "height": 10}
            for image_id in range(1, 7)
        ]
        captions = [
            {
                "id": image_id * 10 + index,
                "image_id": image_id,
                "caption": f"image {image_id} caption {index}",
            }
            for image_id in range(1, 7)
            for index in range(5)
        ]
        instances = [
            {"id": image_id, "image_id": image_id, "category_id": 1 if image_id < 6 else 2}
            for image_id in range(1, 7)
        ]
        _write_json(self.captions_path, {"images": images, "annotations": captions})
        _write_json(
            self.instances_path,
            {
                "images": images,
                "annotations": instances,
                "categories": [
                    {"id": 1, "name": "common", "supercategory": "test"},
                    {"id": 2, "name": "rare", "supercategory": "test"},
                ],
            },
        )
        self.manifest_path = self.root / "train2017.jsonl"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _build(self) -> list:
        metadata = build_coco_manifest(
            split="train2017",
            image_directory_name="train2017",
            captions_path=self.captions_path,
            instances_path=self.instances_path,
            output_path=self.manifest_path,
        )
        self.assertEqual(metadata["record_count"], 6)
        return list(read_manifest(self.manifest_path))

    def test_builds_deterministic_canonical_manifest(self) -> None:
        records = self._build()
        first_bytes = self.manifest_path.read_bytes()
        build_coco_manifest(
            split="train2017",
            image_directory_name="train2017",
            captions_path=self.captions_path,
            instances_path=self.instances_path,
            output_path=self.manifest_path,
        )
        self.assertEqual(first_bytes, self.manifest_path.read_bytes())
        self.assertEqual([record.image_id for record in records], [1, 2, 3, 4, 5, 6])
        self.assertTrue(all(len(record.captions) == 5 for record in records))

    def test_random_selection_is_exact_and_stable(self) -> None:
        records = self._build()
        first = select_random(records, budget=3, seed=5942)
        second = select_random(reversed(records), budget=3, seed=5942)
        self.assertEqual([item.image_id for item in first], [item.image_id for item in second])
        self.assertEqual(len({item.image_id for item in first}), 3)

    def test_category_balanced_includes_rare_category(self) -> None:
        records = self._build()
        selected = select_category_balanced(records, budget=2, seed=5942)
        categories = {category for record in selected for category in record.category_ids}
        self.assertEqual(categories, {1, 2})

    def test_written_selection_validates_against_source(self) -> None:
        records = self._build()
        output = self.root / "random.jsonl"
        metadata = write_selection(
            records=records,
            output_path=output,
            source_manifest=self.manifest_path,
            method="random",
            seed=5942,
            budget=3,
        )
        report, _ = validate_manifest(
            output,
            expected_split="train2017",
            expected_captions=5,
            expected_count=3,
            source_records={record.image_id: record for record in records},
            source_manifest_sha256=sha256_file(self.manifest_path),
        )
        self.assertEqual(metadata["record_count"], 3)
        self.assertTrue(report["valid"], report["errors"])

    def test_extra_captions_are_trimmed_deterministically(self) -> None:
        document = json.loads(self.captions_path.read_text(encoding="utf-8"))
        document["annotations"].append(
            {"id": 999, "image_id": 1, "caption": "extra caption"}
        )
        _write_json(self.captions_path, document)
        metadata = build_coco_manifest(
            split="train2017",
            image_directory_name="train2017",
            captions_path=self.captions_path,
            instances_path=self.instances_path,
            output_path=self.manifest_path,
        )
        first_record = next(read_manifest(self.manifest_path))
        self.assertEqual(len(first_record.captions), 5)
        self.assertEqual(metadata["raw_caption_count_distribution"], {"5": 5, "6": 1})
        self.assertNotIn(999, [caption.caption_id for caption in first_record.captions])

    def test_selection_record_must_match_canonical_source(self) -> None:
        records = self._build()
        output = self.root / "random.jsonl"
        write_selection(
            records=records,
            output_path=output,
            source_manifest=self.manifest_path,
            method="random",
            seed=5942,
            budget=3,
        )
        rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
        rows[0]["width"] += 1
        output.write_text(
            "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
            encoding="utf-8",
        )
        meta_path = output.with_suffix(".meta.json")
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        metadata["manifest_sha256"] = sha256_file(output)
        meta_path.write_text(json.dumps(metadata), encoding="utf-8")

        report, _ = validate_manifest(
            output,
            expected_split="train2017",
            expected_captions=5,
            expected_count=3,
            source_records={record.image_id: record for record in records},
            source_manifest_sha256=sha256_file(self.manifest_path),
        )
        self.assertFalse(report["valid"])
        self.assertIn("source_record_mismatch", {error["code"] for error in report["errors"]})


if __name__ == "__main__":
    unittest.main()
