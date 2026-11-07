# VLM Data Selection Study

Reproducible COCO 2017 data preparation for a CSCI 5942 image–text retrieval experiment.

## Commands

```bash
make test                 # Run synthetic unit tests; no data or GPU required
make fetch-annotations    # Download official captions and instances annotations
make prepare              # Build canonical train2017 and val2017 manifests
make baseline             # Generate the shared deterministic 15k subset
make validate             # Validate counts, hashes, captions, and split isolation
```

For file checks after image download:

```bash
PYTHONPATH=src python3 -m vlm_data validate --check-files
```

## Stack

- Python 3.11+
- Standard library for the complete data-foundation path
- TOML configuration
- JSONL manifests with JSON metadata sidecars
- `unittest` synthetic fixtures

## Sources of truth

| Concern | Location |
|---|---|
| Assignment design | `CSCI_5942_MegaAssignment_Draft (1).pdf` |
| Machine parameters and paths | `configs/data.toml` |
| Initial nine-run matrix | `configs/experiment_matrix.toml` |
| Data contract and leakage policy | `docs/data-foundation.md` |
| Current implementation status | `STATUS.md` |
| Session continuity | `.agents/HANDOFF.md` |

## Project structure

```text
configs/                 Fixed parameters and experiment matrix
data/                    Local raw, interim, and processed data (ignored)
docs/                    Architecture and scientific-method notes
reports/                 Regenerable validation output (ignored)
src/vlm_data/            Data CLI and library
tests/                   Synthetic, offline test suite
.agents/context/         Glossary, parameter ownership, troubleshooting
```

## Invariants

- Train only on `train2017`; evaluate on the held-out `val2017` split.
- Every training selection has exactly 15,000 unique images.
- The shared random selection is reused for every preprocessing-only comparison.
- Each canonical image record has five captions.
- Training caption selection is deterministic in seed, epoch, and image ID.
- Raw COCO files are immutable.
- Every generated manifest has a metadata sidecar and SHA-256 provenance.
- Validation data never informs subset construction or fitted preprocessing statistics.

## Always do

- Run `make test` after changing the data contract or algorithms.
- Run `vlm-data validate` after regenerating manifests.
- Version selection and feature-generation algorithms in metadata.
- Keep paths relative to the project or data root in persisted artifacts.
- Label proposal gaps as decisions instead of selecting silent defaults.

## Ask first

- Downloading the large train/validation image archives (the CLI requires explicit confirmation).
- Adding third-party packages or pretrained model dependencies.
- Changing the 15,000-image, 40-epoch, or 640×640 assignment constants.
- Choosing CLIP checkpoints, diversity objectives, or quality-score formulas.
- Changing the validation protocol or creating a new split.

## Never do

- Commit COCO images, generated embeddings, checkpoints, or secrets.
- Modify anything under `data/raw/` during preparation.
- Fit statistics or choose examples using `val2017`.
- Regenerate a selection without recording its seed and source hash.
- Treat the provisional category-balancing algorithm as an approved study method.
- overwrite a user's generated artifact without an atomic write.

## Notes

The proposal does not fully specify several advanced selection and preprocessing methods. Their
open choices are listed in `docs/data-foundation.md`. Resolve and pin them before generating
expensive embeddings or starting training runs.
