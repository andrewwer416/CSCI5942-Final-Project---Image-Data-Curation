# Parameter Ownership

`configs/data.toml` is authoritative for fixed budgets, seeds, external COCO integrity counts,
paths, archive URLs, and the calibrated category-balance method. `configs/experiment_matrix.toml`
is authoritative only for the initial run list. Do not copy editable parameter values into new
machine-readable files.

Category balancing uses a `0.025` frequency threshold and `4.0` maximum boost. These are project
choices selected by `vlm-data analyze-category-balance`, not literature constants. The analysis
grid, 14%-of-budget intervention guardrail, and 5% minimum-change requirement also live in
`configs/data.toml`.

The following are intentionally unresolved and have no authoritative value yet:

- CLIP checkpoint and revision;
- embedding aggregation and diversity algorithm;
- quality metrics and weights;
- random-resized-crop ranges;
- resize interpolation and padding policy;
- color/contrast normalization formula; and
- auxiliary-feature fusion design.

Approve each choice, put its machine value in a focused config file, and record a method/version in
derived-artifact metadata.
