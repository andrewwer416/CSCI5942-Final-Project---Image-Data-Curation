# Data

Downloaded data and intermediate artifacts live here and are not committed. The four approved
study selection manifests and their metadata are tracked so collaborators use the exact same sets.

| Directory | Policy |
|---|---|
| `raw/coco/` | Immutable official archives and extracted files |
| `interim/manifests/` | Rebuildable canonical JSONL and metadata |
| `processed/selections/` | Four tracked approved selections and sidecars; experimental outputs ignored |

Use `vlm-data download`, `vlm-data prepare`, and `vlm-data select` rather than placing files by hand.
Run `vlm-data validate --check-files` after downloading image archives. See
`docs/data-foundation.md` for the data contract and leakage rules. See
`processed/selections/README.md` for the implemented random and category-balanced methods, the 5%
minimum-change requirement, and placeholders for future selection methods.
