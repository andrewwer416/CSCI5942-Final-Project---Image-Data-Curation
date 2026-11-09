# Train-only visual duplicate audit

The cutoff is **0.99**, selected from the approved `{0.98, 0.99, 0.995}` sweep after
reviewing every proposed center/member pair in the frozen CLIP top-400 graph.

The reviewer was the project assistant. This is a descriptive visual audit, not an
independent human annotation study or a confidence-bound guarantee.

The criterion is the same photographic capture, including resizing, recompression,
small crops, or minor rendering differences. Different poses, exposures, or scene
states are conservatively retained as separate images.

| Cutoff | Groups | Excluded images/pairs | Accepted pairs | Observed precision |
|---|---:|---:|---:|---:|
| 0.98 | 57 | 59 | 56 | 94.915% |
| 0.99 | 36 | 38 | 38 | 100% |
| 0.995 | 26 | 28 | 28 | 100% |

At 0.98 the rejected zero-based pair indices in the audit JSON are 29 (cow pose),
30 (bear pose), and 42 (classroom occupants). They share subjects or composition,
but preserve different visible information. The lowest passing cutoff is 0.99.

Reproduce contact sheets using:

```bash
PYTHONPATH=src pixi run python scripts/audit_visual_pairs.py
```

All pairs, IDs, scores, and page paths are saved locally in
`reports/data/duplicate-audit/pairs.json`. The five 0.98 contact sheets cover the
full 59-pair superset. The higher-cutoff sheets record their corresponding subsets.
Raw images and generated sheets remain ignored by Git.

Duplicate removal is a small redundancy guard here: 38 images are excluded from
the candidate pool. Facility coverage, rather than duplicate removal alone, must
produce the required 750-image minimum replacement relative to random.
