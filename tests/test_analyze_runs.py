"""Synthetic checks for scripts/analyze_runs.py (no real runs needed)."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "analyze_runs.py"
_spec = importlib.util.spec_from_file_location("analyze_runs", SCRIPT)
analyze_runs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(analyze_runs)


def _row(epoch: int, recall: float, **extra: float) -> dict:
    row = {name: recall for name in analyze_runs.RECALLS}
    row.update({"epoch": epoch, "train_loss": 1.0, "t2i_recall@10_color": recall, **extra})
    return row


def _write_run(root: Path, name: str, recalls: list[float], epochs: int, **final_extra: float) -> None:
    run = root / name
    run.mkdir()
    (run / "run_config.json").write_text(json.dumps({"epochs": epochs}))
    rows = [_row(i, r) for i, r in enumerate(recalls)]
    rows[-1].update(final_extra)
    (run / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


class AnalyzeRunsTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.runs = self.root / "runs"
        self.runs.mkdir()
        _write_run(self.runs, "random__standard_rgb__seed5942", [0.1] * 5 + [0.2], 6)
        _write_run(self.runs, "random__standard_rgb__seed5943", [0.1] * 5 + [0.4], 6)
        _write_run(self.runs, "visual_diversity__standard_rgb__seed5942", [0.1] * 5 + [0.5], 6)
        _write_run(
            self.runs,
            "random__rgb_plus_clip_features__seed5942",
            [0.3] * 6,
            6,
            **{f"zeroed_clip/{m}": 0.05 for m in analyze_runs.RECALLS},
        )
        _write_run(self.runs, "category_balanced__standard_rgb__seed5942", [0.9] * 2, 6)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_labels_and_rsum(self) -> None:
        run = analyze_runs.load_run(self.runs / "visual_diversity__standard_rgb__seed5942")
        self.assertEqual((run["arm"], run["seed"]), ("D3xP1", 5942))
        self.assertAlmostEqual(run["final"]["rsum"], 6 * 0.5)
        self.assertAlmostEqual(run["last_mean"]["t2i_recall@10"], (4 * 0.1 + 0.5) / 5)

    def test_comparison_uses_baseline_seed_range(self) -> None:
        runs = [analyze_runs.load_run(d) for d in sorted(self.runs.iterdir())]
        runs = [r for r in runs if r["complete"]]
        base, rows = analyze_runs.compare(runs, "final")
        self.assertAlmostEqual(base["t2i_recall@10"]["mean"], 0.3)
        by_arm = {row["run"]["arm"]: row["metrics"]["t2i_recall@10"] for row in rows}
        self.assertFalse(by_arm["D3xP1"]["inside"])  # 0.5 is above the 0.2-0.4 seed range
        self.assertAlmostEqual(by_arm["D3xP1"]["delta"], 0.2)
        self.assertTrue(by_arm["D1xP5"]["inside"])

    def test_main_writes_reports_and_skips_incomplete(self) -> None:
        out = self.root / "out"
        self.assertEqual(analyze_runs.main(["--runs", str(self.runs), "--out", str(out)]), 0)
        report = (out / "results.md").read_text()
        self.assertIn("D2xP1 seed 5942 (2/6 epochs)", report)
        self.assertIn("zeroed_clip", report)
        for name in ("results.csv", "curves.csv", "t2i_recall10.svg", "train_loss.svg"):
            self.assertTrue((out / name).stat().st_size > 0, name)


if __name__ == "__main__":
    unittest.main()
