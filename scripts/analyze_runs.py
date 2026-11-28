"""Summarize finished study runs: final-epoch Recall@K, baseline seed spread, and curves.

Reads every ``<runs>/<run_id>__seed<seed>/{run_config.json,metrics.jsonl}`` and writes, to
``reports/results/`` (ignored by Git):

* ``results.md``: per-run tables and each arm compared with the D1xP1 seed spread;
* ``results.csv``: one row per run and metric (final epoch and mean of the last 5 epochs);
* ``curves.csv``: every logged epoch of every run;
* ``t2i_recall10.svg`` and ``train_loss.svg``: per-epoch curves.

Reporting rules (fixed before comparing arms, see docs/study-methods.md):

* The headline number is the **final epoch** (epoch 40). The best epoch is never reported,
  because choosing it would use val2017 to make a decision.
* The mean of the last 5 epochs is a robustness check, reported next to the final epoch.
* The baseline is D1xP1. Its seeds differ only in weight initialization, so their spread is the
  run-to-run noise. An arm is "inside" when its value lies within the seeds' min-max range.
  With three seeds this is descriptive, not a significance test.

Uses only the standard library.

    python3 scripts/analyze_runs.py --runs artifacts/runs
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

SELECTIONS = {
    "random": "D1",
    "category_balanced": "D2",
    "visual_diversity": "D3",
    "caption_semantic_diversity": "D4",
    "quality_aware_diversity": "D5",
}
PREPROCESSING = {
    "standard_rgb": "P1",
    "aspect_ratio_preserving_rgb": "P2",
    "random_resized_crop": "P3",
    "color_contrast_normalization": "P4",
    "rgb_plus_clip_features": "P5",
}
RECALLS = [f"{d}_recall@{k}" for d in ("t2i", "i2t") for k in (1, 5, 10)]
METRICS = RECALLS + ["rsum", "t2i_recall@10_color", "train_loss"]
HEADLINE = "t2i_recall@10"
BASELINE = ("D1", "P1")
LAST_N = 5
DIAGNOSTIC_MODES = ("zeroed_clip", "shuffled_clip")


def label(run_dir: Path) -> tuple[str, str, int]:
    """``random__standard_rgb__seed5942`` -> ("D1", "P1", 5942)."""

    selection, preprocessing, seed = run_dir.name.split("__")
    return SELECTIONS[selection], PREPROCESSING[preprocessing], int(seed.removeprefix("seed"))


def with_rsum(row: dict) -> dict:
    """RSUM: the sum of the six Recall@K values, a common one-number retrieval summary."""

    row = dict(row)
    if all(name in row for name in RECALLS):
        row["rsum"] = sum(row[name] for name in RECALLS)
    return row


def load_run(run_dir: Path) -> dict:
    config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    rows = [
        with_rsum(json.loads(line))
        for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    d, p, seed = label(run_dir)
    run = {
        "name": run_dir.name,
        "d": d,
        "p": p,
        "seed": seed,
        "arm": f"{d}x{p}",
        "epochs": config["epochs"],
        "rows": rows,
        "complete": len(rows) == config["epochs"],
    }
    if rows:
        last = rows[-LAST_N:]
        run["final"] = {m: rows[-1][m] for m in METRICS if m in rows[-1]}
        run["last_mean"] = {
            m: statistics.fmean(r[m] for r in last) for m in METRICS if all(m in r for r in last)
        }
        run["diagnostics"] = {
            mode: with_rsum(
                {k.split("/", 1)[1]: v for k, v in rows[-1].items() if k.startswith(mode + "/")}
            )
            for mode in DIAGNOSTIC_MODES
            if any(k.startswith(mode + "/") for k in rows[-1])
        }
    return run


def spread(values: list[float]) -> dict:
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def compare(runs: list[dict], key: str) -> tuple[dict, list[dict]]:
    """Baseline seed spread per metric, and every other arm's offset from the baseline mean."""

    base = [r for r in runs if (r["d"], r["p"]) == BASELINE]
    if not base:
        raise SystemExit("No complete D1xP1 baseline run found")
    base_spread = {m: spread([r[key][m] for r in base]) for m in base[0][key]}
    rows = []
    for run in runs:
        if (run["d"], run["p"]) == BASELINE:
            continue
        row = {"run": run, "metrics": {}}
        for m, s in base_spread.items():
            value = run[key][m]
            row["metrics"][m] = {
                "value": value,
                "delta": value - s["mean"],
                "inside": s["min"] <= value <= s["max"],
            }
        rows.append(row)
    return base_spread, rows


def fmt(value: float) -> str:
    return f"{value:.4f}"


def markdown(runs: list[dict], incomplete: list[dict]) -> str:
    lines = ["# Study results", ""]
    lines += [
        "Final epoch is the headline; the last-5-epoch mean is a robustness check. The baseline",
        "spread comes from the D1xP1 seeds (weight initialization only). \"Inside\" means within",
        "the seeds' min-max range. Three seeds give a descriptive range, not a significance test.",
        "",
    ]
    if incomplete:
        lines += ["Not yet complete (excluded): " + ", ".join(
            f"{r['arm']} seed {r['seed']} ({len(r['rows'])}/{r['epochs']} epochs)" for r in incomplete
        ), ""]

    cols = RECALLS + ["rsum", "t2i_recall@10_color", "train_loss"]
    short = [c.replace("_recall", "") for c in cols]
    for key, title in (("final", "Final epoch"), ("last_mean", f"Mean of the last {LAST_N} epochs")):
        lines += [f"## {title}", "", "| Arm | Seed | " + " | ".join(short) + " |",
                  "|---|---:|" + "---:|" * len(cols)]
        for r in runs:
            lines.append(f"| {r['arm']} | {r['seed']} | " + " | ".join(fmt(r[key][c]) for c in cols) + " |")
        lines.append("")

    for key, title in (("final", "final epoch"), ("last_mean", f"last-{LAST_N} mean")):
        base, rows = compare(runs, key)
        b = base[HEADLINE]
        lines += [
            f"## Compared with D1xP1 ({title})", "",
            f"D1xP1 {HEADLINE}: mean {fmt(b['mean'])}, sd {fmt(b['sd'])}, range "
            f"{fmt(b['min'])}-{fmt(b['max'])} over {b['n']} seeds. RSUM: mean "
            f"{base['rsum']['mean']:.4f}, range {base['rsum']['min']:.4f}-{base['rsum']['max']:.4f}.",
            "",
            "| Arm | Seed | t2i@10 | Δ vs D1xP1 | Inside | RSUM | Δ RSUM | Inside |",
            "|---|---:|---:|---:|:-:|---:|---:|:-:|",
        ]
        for row in sorted(rows, key=lambda x: -x["metrics"][HEADLINE]["value"]):
            h, s = row["metrics"][HEADLINE], row["metrics"]["rsum"]
            lines.append(
                f"| {row['run']['arm']} | {row['run']['seed']} | {fmt(h['value'])} | {h['delta']:+.4f} | "
                f"{'yes' if h['inside'] else '**no**'} | {s['value']:.4f} | {s['delta']:+.4f} | "
                f"{'yes' if s['inside'] else '**no**'} |"
            )
        lines.append("")

    diag = [r for r in runs if r.get("diagnostics")]
    if diag:
        lines += ["## P5 diagnostics (final epoch)", "",
                  "If the model uses the CLIP vector, zeroing or shuffling it should lower recall.", "",
                  "| Run | Eval input | t2i@10 | RSUM |", "|---|---|---:|---:|"]
        for r in diag:
            lines.append(f"| {r['arm']} s{r['seed']} | real CLIP | {fmt(r['final'][HEADLINE])} | {r['final']['rsum']:.4f} |")
            for mode, m in r["diagnostics"].items():
                lines.append(f"| | {mode} | {fmt(m[HEADLINE])} | {m['rsum']:.4f} |")
        lines.append("")
    return "\n".join(lines)


PALETTE = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2", "#be185d", "#4d7c0f"]


def svg_curves(runs: list[dict], metric: str, title: str) -> str:
    """A plain line chart: one line per run, D1xP1 seeds in gray."""

    width, height, left, right, top, bottom = 760, 420, 60, 190, 36, 44
    series = [(r, [(row["epoch"] + 1, row[metric]) for row in r["rows"]]) for r in runs]
    xs = [x for _, pts in series for x, _ in pts]
    ys = [y for _, pts in series for _, y in pts]
    x0, x1 = 1, max(xs)
    y0, y1 = min(0.0, min(ys)), max(ys) * 1.05
    sx = lambda x: left + (x - x0) / (x1 - x0) * (width - left - right)  # noqa: E731
    sy = lambda y: top + (1 - (y - y0) / (y1 - y0)) * (height - top - bottom)  # noqa: E731
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'font-family="sans-serif" font-size="12">',
        f'<rect width="{width}" height="{height}" fill="white"/>',
        f'<text x="{left}" y="22" font-size="14" font-weight="bold">{title}</text>',
    ]
    for i in range(6):
        y = y0 + (y1 - y0) * i / 5
        out.append(f'<line x1="{left}" x2="{width - right}" y1="{sy(y):.1f}" y2="{sy(y):.1f}" stroke="#e5e7eb"/>')
        out.append(f'<text x="{left - 6}" y="{sy(y) + 4:.1f}" text-anchor="end">{y:.3f}</text>')
    for x in range(0, x1 + 1, 5):
        if x >= x0:
            out.append(f'<text x="{sx(x):.1f}" y="{height - bottom + 16}" text-anchor="middle">{x}</text>')
    out.append(f'<text x="{(left + width - right) / 2}" y="{height - 8}" text-anchor="middle">epoch</text>')
    color = iter(PALETTE)
    legend_y = top
    for r, pts in series:
        is_base = (r["d"], r["p"]) == BASELINE
        stroke = "#9ca3af" if is_base else next(color, "#111827")
        path = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in pts)
        out.append(f'<polyline points="{path}" fill="none" stroke="{stroke}" stroke-width="{1.5 if is_base else 2}"/>')
        out.append(f'<line x1="{width - right + 12}" x2="{width - right + 32}" y1="{legend_y}" y2="{legend_y}" stroke="{stroke}" stroke-width="2"/>')
        out.append(f'<text x="{width - right + 38}" y="{legend_y + 4}">{r["arm"]} s{r["seed"]}</text>')
        legend_y += 18
    out.append("</svg>")
    return "\n".join(out)


def write_csvs(runs: list[dict], out: Path) -> None:
    with (out / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["run", "arm", "seed", "metric", "final_epoch", f"last{LAST_N}_mean"])
        for r in runs:
            for m in r["final"]:
                writer.writerow([r["name"], r["arm"], r["seed"], m, r["final"][m], r["last_mean"].get(m, "")])
    with (out / "curves.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["run", "arm", "seed", "epoch", *METRICS])
        for r in runs:
            for row in r["rows"]:
                writer.writerow([r["name"], r["arm"], r["seed"], row["epoch"] + 1, *(row.get(m, "") for m in METRICS)])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=Path, default=Path("artifacts/runs"))
    parser.add_argument("--out", type=Path, default=Path("reports/results"))
    args = parser.parse_args(argv)

    loaded = [load_run(d) for d in sorted(args.runs.iterdir()) if (d / "metrics.jsonl").is_file()]
    order = lambda r: (r["p"], r["d"], r["seed"])  # noqa: E731
    runs = sorted((r for r in loaded if r["complete"]), key=order)
    incomplete = sorted((r for r in loaded if not r["complete"]), key=order)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.md").write_text(markdown(runs, incomplete), encoding="utf-8")
    write_csvs(runs, args.out)
    (args.out / "t2i_recall10.svg").write_text(
        svg_curves(runs, HEADLINE, "Text-to-image Recall@10 on val2017"), encoding="utf-8"
    )
    # P5 sits about 5x higher and flattens every other line; draw the rest on their own scale too.
    pixel_only = [r for r in runs if r["p"] != "P5"]
    if len(pixel_only) < len(runs):
        (args.out / "t2i_recall10_without_p5.svg").write_text(
            svg_curves(pixel_only, HEADLINE, "Text-to-image Recall@10 on val2017 (without P5)"),
            encoding="utf-8",
        )
    (args.out / "train_loss.svg").write_text(
        svg_curves(runs, "train_loss", "Training loss"), encoding="utf-8"
    )
    print(f"{len(runs)} complete runs, {len(incomplete)} incomplete -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
