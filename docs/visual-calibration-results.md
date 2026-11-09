# Frozen visual selection: train-only calibration

The accepted configuration is OpenCLIP `ViT-B-16` / `datacomp_xl_s13b_b90k`,
permanently frozen, with exact GPU FAISS neighbors, sparse facility lazy greedy,
**k = 50**, and **duplicate cutoff = 0.99**.

| Metric | Random | Visual k=50 |
|---|---:|---:|
| Images | 15,000 | 15,000 |
| Mean exact cosine coverage | 0.799252 | 0.821705 |
| Mean nearest-selected cosine distance | 0.200748 | 0.178295 |
| p95 distance | 0.358660 | 0.315450 |
| p99 distance | 0.425660 | 0.349715 |

The visual set retains 1,927 random images and replaces **13,073 (87.153%)**.
The minimum requirement is 750 replacements. The mean distance falls by about
11.2%; p95 falls by about 12.0%. These are representation-space improvements;
retrieval performance remains unmeasured until training/evaluation.

| k | Mean exact coverage | p95 distance | p99 distance |
|---|---:|---:|---:|
| 50 | 0.821705 | 0.315450 | 0.349715 |
| 100 | 0.821968 | 0.314737 | 0.348307 |
| 200 | 0.822067 | 0.314142 | 0.347738 |
| 400 | 0.822104 | 0.314058 | 0.347301 |

All candidates pass the hard gates. k=50 is the smallest candidate within 0.1%
relative mean coverage of the best, within 0.005 of the best p95, and within
0.010 of the best p99. Its sparse graph leaves no image uncovered.

The bottom-1% local-density group contributes exactly 300 selected images, so
the 2% cap is active. This is a substantive part of the selector, not a dormant
safety limit. Duplicate grouping excludes only 38 candidates; the coverage
objective produces most of the selection difference.

Reproduce with the locked Pixi environment:

```bash
PYTHONPATH=src pixi run python scripts/audit_visual_pairs.py
PYTHONPATH=src pixi run python scripts/calibrate_visual.py
PYTHONPATH=src pixi run python -m vlm_data validate
```

The calibration script uses the recorded 0.99 audit decision. It writes
`reports/data/visual-k{k}.json`, `visual-calibration.json`, and
`visual-calibration.md`, then the accepted
`data/processed/selections/visual_diversity_k50_dup0p99_seed5942_n15000.jsonl`
and metadata. Reports include selected IDs, exact coverage, replacement counts,
outlier count, sparse uncovered fraction, category incidence, object density,
gain evaluations, timings, and artifact/config/implementation hashes.

The approved selection manifest and sidecar are tracked in Git for collaborator handoff.
Image data, weights, vectors, graphs, and reports remain ignored. The tracked implementation
makes them regenerable. No validation images or captions chose the selection parameters.
