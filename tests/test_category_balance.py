import json
from pathlib import Path
import tempfile
import unittest

from vlm_data.category_balance import (
    add_selection_simulation,
    analyze_category_balance,
    compute_category_targets,
    render_category_balance_markdown,
    write_category_balance_analysis,
)
from vlm_data.manifest import Caption, ImageRecord
from vlm_data.selection import (
    CATEGORY_BALANCED_ALGORITHM_VERSION,
    RANDOM_ALGORITHM_VERSION,
    select_category_balanced,
    select_random,
    write_selection,
)


def _record(image_id: int, categories: tuple[int, ...]) -> ImageRecord:
    return ImageRecord(
        image_id=image_id,
        split="train2017",
        file_name=f"{image_id:012d}.jpg",
        relative_path=f"train2017/{image_id:012d}.jpg",
        width=20,
        height=10,
        captions=tuple(Caption(index, f"caption {index}") for index in range(5)),
        category_ids=categories,
        object_count=len(categories),
    )


class CategoryBalanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.manifest = self.root / "train2017.jsonl"
        self.manifest.write_text("synthetic manifest\n", encoding="utf-8")
        self.manifest.with_suffix(".meta.json").write_text(
            json.dumps(
                {
                    "categories": {
                        "1": {"name": "common"},
                        "2": {"name": "rare"},
                    }
                }
            ),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_target_formula_reports_cap_activity(self) -> None:
        targets = compute_category_targets(
            {1: 10, 2: 1},
            total_images=10,
            budget=5,
            frequency_threshold=0.9,
            max_boost=2.0,
        )
        self.assertEqual(targets[1].effective_boost, 1.0)
        self.assertAlmostEqual(targets[2].raw_boost, 3.0)
        self.assertEqual(targets[2].effective_boost, 2.0)
        self.assertTrue(targets[2].affected)
        self.assertTrue(targets[2].capped)
        self.assertEqual(targets[2].target, 1)

    def test_analysis_rule_can_recover_proposed_values(self) -> None:
        records = [
            _record(image_id, (1, 2) if image_id <= 4 else (1,))
            for image_id in range(1, 101)
        ]
        report = analyze_category_balance(
            records,
            source_manifest=self.manifest,
            budget=50,
            proposed_threshold=0.16,
            proposed_max_boost=2.0,
            threshold_candidates=(0.01, 0.04, 0.16),
            max_boost_candidates=(2.0, 3.0),
            max_extra_target_fraction=0.05,
        )
        self.assertEqual(report["recommendation"]["frequency_threshold"], 0.16)
        self.assertEqual(report["recommendation"]["max_boost"], 2.0)
        self.assertTrue(report["recommendation"]["matches_proposal"])
        self.assertEqual(report["recommended_scenario"]["affected_category_count"], 1)
        self.assertIn("Selection simulation was skipped", render_category_balance_markdown(report))
        rare = next(
            row
            for row in report["recommended_scenario"]["categories"]
            if row["category_id"] == 2
        )
        self.assertEqual(rare["name"], "rare")

    def test_multilabel_image_has_largest_deficit_gain(self) -> None:
        records = [_record(1, (1, 2)), _record(2, (1,)), _record(3, (2,))]
        selected = select_category_balanced(
            records,
            budget=1,
            seed=5942,
            frequency_threshold=0.9,
            max_boost=3.0,
        )
        self.assertEqual([record.image_id for record in selected], [1])

    def test_selector_is_input_order_independent_and_fills_after_targets(self) -> None:
        records = [
            _record(1, (1,)),
            _record(2, ()),
            _record(3, ()),
            _record(4, ()),
        ]
        first = select_category_balanced(
            records,
            budget=3,
            seed=5942,
            frequency_threshold=0.5,
            max_boost=3.0,
        )
        second = select_category_balanced(
            reversed(records),
            budget=3,
            seed=5942,
            frequency_threshold=0.5,
            max_boost=3.0,
        )
        self.assertEqual(
            [record.image_id for record in first],
            [record.image_id for record in second],
        )
        self.assertIn(1, {record.image_id for record in first})
        self.assertEqual(len(first), 3)

    def test_no_affected_categories_returns_exact_random_baseline(self) -> None:
        records = [_record(image_id, (1,)) for image_id in range(1, 11)]
        balanced = select_category_balanced(
            records,
            budget=5,
            seed=5942,
            frequency_threshold=0.01,
            max_boost=3.0,
        )
        baseline = select_random(records, budget=5, seed=5942)
        self.assertEqual(balanced, baseline)

    def test_report_includes_real_selection_comparison_and_is_deterministic(self) -> None:
        records = [
            _record(image_id, (1, 2) if image_id <= 4 else (1,))
            for image_id in range(1, 101)
        ]
        report = analyze_category_balance(
            records,
            source_manifest=self.manifest,
            budget=50,
            proposed_threshold=0.16,
            proposed_max_boost=2.0,
            threshold_candidates=(0.04, 0.16),
            max_boost_candidates=(2.0, 3.0),
            max_extra_target_fraction=0.05,
        )
        selected = select_category_balanced(
            records,
            budget=50,
            seed=5942,
            frequency_threshold=0.16,
            max_boost=2.0,
        )
        baseline = select_random(records, budget=50, seed=5942)
        add_selection_simulation(
            report,
            selected=selected,
            random_baseline=baseline,
            seed=5942,
            balanced_algorithm_version=CATEGORY_BALANCED_ALGORITHM_VERSION,
            baseline_algorithm_version=RANDOM_ALGORITHM_VERSION,
        )
        self.assertEqual(report["selection_simulation"]["seed"], 5942)
        self.assertEqual(
            report["selection_simulation"]["balanced_algorithm_version"],
            "sqrt_frequency_baseline_repair_v1",
        )
        self.assertEqual(len(report["selection_simulation"]["selected_image_ids_sha256"]), 64)
        markdown = render_category_balance_markdown(report)
        self.assertIn("Real-selection simulation", markdown)
        self.assertIn("rare", markdown)

        json_output = self.root / "analysis.json"
        markdown_output = self.root / "analysis.md"
        write_category_balance_analysis(
            report,
            json_output=json_output,
            markdown_output=markdown_output,
        )
        first_json = json_output.read_bytes()
        first_markdown = markdown_output.read_bytes()
        write_category_balance_analysis(
            report,
            json_output=json_output,
            markdown_output=markdown_output,
        )
        self.assertEqual(first_json, json_output.read_bytes())
        self.assertEqual(first_markdown, markdown_output.read_bytes())

    def test_written_selection_records_method_parameters_and_targets(self) -> None:
        records = [_record(1, (1, 2)), _record(2, (1,)), _record(3, (2,))]
        source = self.root / "source.jsonl"
        source.write_text(
            "".join(json.dumps(record.to_dict()) + "\n" for record in records),
            encoding="utf-8",
        )
        output = self.root / "selection.jsonl"
        metadata = write_selection(
            records=records,
            output_path=output,
            source_manifest=source,
            method="category-balanced",
            seed=5942,
            budget=2,
            category_frequency_threshold=0.5,
            category_max_boost=2.0,
            category_minimum_changed_fraction=0.0,
        )
        self.assertEqual(metadata["algorithm_version"], "sqrt_frequency_baseline_repair_v1")
        self.assertEqual(metadata["method_parameters"]["frequency_threshold"], 0.5)
        self.assertIn("selected_coverage", metadata["category_targets"]["1"])
        self.assertIn("changed_fraction", metadata["random_baseline_comparison"])

    def test_written_selection_enforces_minimum_changed_fraction(self) -> None:
        records = [_record(image_id, (1,)) for image_id in range(1, 11)]
        source = self.root / "source.jsonl"
        source.write_text(
            "".join(json.dumps(record.to_dict()) + "\n" for record in records),
            encoding="utf-8",
        )
        output = self.root / "selection.jsonl"
        with self.assertRaisesRegex(ValueError, "below the required"):
            write_selection(
                records=records,
                output_path=output,
                source_manifest=source,
                method="category-balanced",
                seed=5942,
                budget=5,
                category_frequency_threshold=0.01,
                category_max_boost=3.0,
                category_minimum_changed_fraction=0.20,
            )
        self.assertFalse(output.exists())

    def test_invalid_analysis_parameters_fail_early(self) -> None:
        with self.assertRaisesRegex(ValueError, "frequency threshold"):
            compute_category_targets(
                {1: 1},
                total_images=1,
                budget=1,
                frequency_threshold=0.0,
                max_boost=3.0,
            )
        with self.assertRaisesRegex(ValueError, "maximum boost"):
            compute_category_targets(
                {1: 1},
                total_images=1,
                budget=1,
                frequency_threshold=0.1,
                max_boost=0.5,
            )


if __name__ == "__main__":
    unittest.main()
