# Configuration

- `data.toml` is the single machine-readable source for paths, fixed assignment budgets, seeds, and
  official archive locations.
- `experiment_matrix.toml` enumerates the nine initial comparisons, counting the shared baseline
  exactly once.

Method-specific hyperparameters belong here only after their scientific decision and calibration
rule are recorded. The category-balance values meet that requirement in
`docs/background-research.md`: a 2.5% frequency threshold, 4× safety cap, 14%-of-budget target-mass
guardrail, and hard 5% minimum change from the random baseline. The remaining methods retain
placeholders in `data/processed/selections/README.md` until their decisions are approved.
