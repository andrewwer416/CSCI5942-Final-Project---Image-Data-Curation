# Glossary

| Term | Meaning in this project |
|---|---|
| Canonical manifest | One deterministic row per COCO image, before experiment selection |
| Selection manifest | Exact 15,000-image training set plus algorithm provenance |
| Shared baseline | Random selection with standard RGB preprocessing, trained once |
| Initial run | One of nine selection-only or preprocessing-only comparisons |
| Follow-up run | Optional pairing chosen after initial retrieval results |
| Stable hash rank | Order-independent deterministic substitute for in-process random sampling |
| Baseline repair | Category balancing that swaps the minimum needed images into the paired random set |
| Added target mass | Extra category-image incidences requested by a balance scenario; not distinct images |
| Metadata sidecar | `.meta.json` file containing counts, hashes, seed, and algorithm version |
| Leakage | Any use of `val2017` information to construct or fit a training input |
