# Parameter Ownership

`configs/data.toml` is authoritative for fixed budgets, seeds, external COCO integrity counts,
paths, and archive URLs. `configs/experiment_matrix.toml` is authoritative only for the initial
run list. Do not copy editable parameter values into new machine-readable files.

The following are intentionally unresolved and have no authoritative value yet:

- CLIP checkpoint and revision;
- embedding aggregation and diversity algorithm;
- category-balancing objective;
- quality metrics and weights;
- random-resized-crop ranges;
- resize interpolation and padding policy;
- color/contrast normalization formula; and
- auxiliary-feature fusion design.

Approve each choice, put its machine value in a focused config file, and record a method/version in
derived-artifact metadata.
