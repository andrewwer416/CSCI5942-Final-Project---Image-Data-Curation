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
- The category-balanced selection must replace at least 750 baseline images (5%); fail generation
  rather than adding category-unrelated swaps to reach that floor.
- Each canonical image record has five captions.
- Training caption selection is deterministic in seed, epoch, and image ID.
- Raw COCO files are immutable.
- Every generated manifest has a metadata sidecar and SHA-256 provenance.
- Validation data never informs subset construction or fitted preprocessing statistics.

## Always do

- Run `make test` after changing the data contract or algorithms.
- Run `vlm-data validate` after regenerating manifests.
- Version selection and feature-generation algorithms in metadata.
- Re-run `make analyze-category-balance` before changing its threshold, cap, source, or budget.
- Keep unresolved selection methods as documented placeholders until their decisions are approved.
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
- Treat an uncalibrated category-balance override as an approved study method.
- Overwrite a user's generated artifact without an atomic write.

## Training on CURC Alpine

Study runs are Slurm jobs in `slurm/alpine/` (one `run_*.sbatch` per run; how-to in
`slurm/alpine/README.md`). `slurm/delta/` was used only to measure the batch size.

| Location | Quota | Backed up | Use it for |
|---|---|---|---|
| `/home/$USER` | 2 GB | Yes | Nothing for this project; too small |
| `/projects/$USER` | 250 GB | Yes, never purged | The Python environment (`software/mega-env`), pip/conda caches, result backups |
| `/scratch/alpine/$USER` | 10 TB | No; purged 90 days after creation | The repository clone, COCO images, logs, and training runs |

- CURC terminates jobs that do heavy I/O on `/home` or `/projects`, and may suspend the account.
  Submit every job from the clone under `/scratch/alpine/$USER/mega`. `slurm/alpine/env.sh`
  refuses to run a job submitted from `/home` or `/projects`.
- Copy finished runs and logs to `/projects/$USER/mega-results/` with `rsync`, because scratch is
  neither backed up nor permanent.
- Account `ucb857_peak1`. QoS names are `gpu-normal` and `cpu-normal`, not `normal`. GPU partitions
  are `aa100` (default for runs) and `al40` (48 GB, the fallback if batch 64 does not fit).
- Every arm uses batch 64. It was measured on a Delta A40: 36.2 GB peak and 1.70 s/step, while
  batch 96 ran out of memory. A 40-epoch run takes about 5–6 h on an A40.
- Logging in needs a password and Duo. Claude can reach Alpine only through `ssh alpine`, an
  SSH ControlMaster alias, after the user has logged in with it once.
- Ask before submitting full training runs: they spend a shared allocation.

## Notes

The proposal does not fully specify several advanced selection and preprocessing methods. Category
balancing is resolved; the remaining open choices are listed in `docs/data-foundation.md`. Resolve
and pin them before generating expensive embeddings or starting training runs.
