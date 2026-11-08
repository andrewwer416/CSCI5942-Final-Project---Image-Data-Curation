# `vlm_data`

Standard-library package for COCO acquisition, canonical manifest construction, deterministic
subset selection, per-epoch caption sampling, and integrity validation. The public entry point is
the `vlm-data` command implemented in `cli.py`. Category-balance calibration is implemented in
`category_balance.py` and is available through `vlm-data analyze-category-balance`.

The operational selection contract—including the category-balanced baseline-repair algorithm, its
hard 5% minimum-change requirement, current real-data measurements, and placeholders for unresolved
methods—is documented in `data/processed/selections/README.md`.
