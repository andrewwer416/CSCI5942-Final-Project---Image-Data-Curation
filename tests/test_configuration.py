from contextlib import redirect_stderr
import io
from pathlib import Path
import tomllib
import unittest
from unittest.mock import patch

from vlm_data.cli import main
from vlm_data.config import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ConfigurationTests(unittest.TestCase):
    def test_data_config_resolves_from_project_root(self) -> None:
        config = load_config(PROJECT_ROOT / "configs/data.toml")
        self.assertEqual(config.project_root, PROJECT_ROOT)
        self.assertEqual(config.train_image_budget, 15000)
        self.assertEqual(config.val_images_dir, PROJECT_ROOT / "data/raw/coco/val2017")
        self.assertEqual(config.category_balance.frequency_threshold, 0.025)
        self.assertEqual(config.category_balance.max_boost, 4.0)
        self.assertIn(0.001, config.category_balance.threshold_candidates)
        self.assertIn(0.0225, config.category_balance.threshold_candidates)
        self.assertIn(0.025, config.category_balance.threshold_candidates)
        self.assertIn(0.03, config.category_balance.threshold_candidates)
        self.assertIn(0.05, config.category_balance.threshold_candidates)
        self.assertIn(6.0, config.category_balance.max_boost_candidates)
        self.assertEqual(config.category_balance.max_extra_target_fraction, 0.14)
        self.assertEqual(config.category_balance.minimum_changed_fraction, 0.05)

    def test_initial_matrix_has_one_shared_baseline_and_nine_runs(self) -> None:
        with (PROJECT_ROOT / "configs/experiment_matrix.toml").open("rb") as handle:
            matrix = tomllib.load(handle)
        runs = matrix["initial_runs"]
        self.assertEqual(len(runs), 9)
        baselines = [run for run in runs if run.get("shared_baseline")]
        self.assertEqual(len(baselines), 1)
        self.assertEqual(baselines[0]["run_id"], "random__standard_rgb")

    def test_train_download_requires_explicit_confirmation(self) -> None:
        with patch("vlm_data.cli.download_components") as download:
            with redirect_stderr(io.StringIO()):
                result = main(
                    [
                        "--config",
                        str(PROJECT_ROOT / "configs/data.toml"),
                        "download",
                        "train",
                    ]
                )
        self.assertEqual(result, 2)
        download.assert_not_called()


if __name__ == "__main__":
    unittest.main()
