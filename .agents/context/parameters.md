# Parameter Ownership

`configs/data.toml` is authoritative for fixed budgets, seeds, external COCO integrity counts,
paths, archive URLs, and the calibrated category-balance method. `configs/experiment_matrix.toml`
is authoritative only for the initial run list. Do not copy editable parameter values into new
machine-readable files.

Category balancing uses a `0.025` frequency threshold and `4.0` maximum boost. These are project
choices selected by `vlm-data analyze-category-balance`, not literature constants. The analysis
grid, 14%-of-budget intervention guardrail, and 5% minimum-change requirement also live in
`configs/data.toml`.

Approved method parameters are versioned constants in code and recorded in every artifact's or
run's metadata:

| Method | Where the values live |
|---|---|
| D3 visual diversity (CLIP checkpoint, k, duplicate cutoff, outlier cap) | `src/vlm_data/visual_embeddings.py`, `configs/data.toml` |
| D4 caption-semantic diversity | `src/vlm_data/` caption modules, `configs/data.toml` |
| D5 quality-aware diversity (TOPIQ filter, λ rule, gates) | `scripts/calibrate_quality.py` constants |
| P2 letterbox and padding mask | `src/vlm_train/data.py` (`letterbox_geometry`, `padding_patch_mask`) |
| P3 crop ranges | `src/vlm_train/data.py` (`CROP_*`) |
| P4 appearance normalization | `src/vlm_train/data.py` (`SHADES_OF_GRAY_P`, `*_GAIN_CAP`, `CONTRAST_PERCENTILES`) |
| P5 fusion | `src/vlm_train/model.py` (`ModelConfig.clip_dim`, `clip_projection`) |
| Training defaults | `src/vlm_train/train.py` CLI defaults (batch size still to be measured) |

Changing any of these is a new method version: update the version constant and its document, and
re-run the relevant audit or calibration.
