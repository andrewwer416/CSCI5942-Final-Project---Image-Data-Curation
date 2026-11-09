from pathlib import Path
import tempfile
import unittest

try:
    from vlm_train.train import _parser, train
except ImportError as error:
    raise unittest.SkipTest("Optional training environment unavailable") from error


class TrainingRunContractTests(unittest.TestCase):
    def test_existing_run_is_rejected_before_loading_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            record = output / "run_config.json"
            record.write_text("existing result")
            args = _parser().parse_args(
                [
                    "--selection",
                    "missing.jsonl",
                    "--output-dir",
                    str(output),
                    "--device",
                    "cpu",
                    "--model-seed",
                    "5943",
                ]
            )
            with self.assertRaisesRegex(FileExistsError, "overwrite"):
                train(args)
            self.assertEqual(record.read_text(), "existing result")
