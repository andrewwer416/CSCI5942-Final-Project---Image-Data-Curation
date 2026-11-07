# Session Handoff

> **Purpose**: Read-only dashboard for session continuity. Update only at session end.

## Last session

**Date**: 2026-11-07
**Objective**: Turn the assignment proposal into a reproducible data foundation and prepare the
first usable data artifacts.

### What was done

- Extracted and traced the four-page assignment proposal.
- Created the Python project, data CLI, configuration, docs, status, and offline tests.
- Downloaded COCO 2017 annotations only; verified the pinned byte count and SHA-256.
- Built canonical manifests for 118,287 train and 5,000 validation images.
- Generated the seed-5942 random baseline with exactly 15,000 unique training images.
- Validated counts, five-caption policy, hashes, source identity, and split isolation with no errors.

### Decisions made

| Decision | Rationale | Reversible? |
|---|---|---|
| Python 3.11 standard-library data path | Reproducible setup with no dependency gate | Yes |
| Stable hash ranking for random baseline | Independent of row order and Python RNG changes | Yes, versioned |
| Retain lowest five caption IDs | Real COCO has a few images with 6–7 captions; proposal requires five | Yes, versioned |
| HTTP downloads require pinned size and SHA-256 | Official COCO page publishes HTTP archive links | Yes |
| Defer large image archives | Not needed to establish manifest and selection integrity | Yes |
| Keep category balancing provisional | Proposal does not define the balancing objective | Yes |

### Main files added

- `README.md`, `CLAUDE.md`, `STATUS.md` — entry points and current state
- `configs/` — data parameters and exact nine-run matrix
- `src/vlm_data/` — acquisition, manifest, selection, caption sampling, and validation
- `tests/` — synthetic offline regression suite
- `docs/data-foundation.md` — contract, leakage boundary, and open decisions

## Current state

### Ready

- `make test`: 16 tests passing.
- `make validate`: real manifests and baseline valid.
- Generated data and reports are local and ignored by Git.

### Blockers / deliberate deferrals

- `train2017` and `val2017` images are not downloaded.
- CLIP checkpoint, diversity algorithms, quality formula, and transform details need team decisions.

### Environment

- Git branch: `main`.
- Remote: `https://github.com/andrewwer416/CSCI5942-Final-Project---Image-Data-Curation.git`.
- Running processes: none expected.
- Large local artifact: checksum-verified COCO annotations archive and extracted JSON files.

## Next session should

### Immediate

1. Review and resolve the open scientific choices in `docs/data-foundation.md`.
2. Check disk capacity before running `vlm-data download train val --confirm-large-download`.
3. After image acquisition, run `vlm-data validate --check-files`.

### Then

- Implement the dataset loader and lazy preprocessing transforms.
- Pin the frozen CLIP model/revision before generating visual or caption embeddings.
- Approve or replace the provisional category-balancing objective before using it in a study run.

## Session log

### Session 1 — 2026-11-07

- Objective: lay out the project foundation and prepare data.
- Outcome: annotation-backed manifests and shared 15k random baseline prepared and validated.
- Commit: initial project commit on `main`.

### Session 2 — 2026-11-07

- Objective: publish the completed foundation to the user-provided GitHub repository.
- Outcome: repository initialized on `main`; source, configuration, tests, PDF, and documentation
  committed while downloaded/generated COCO artifacts remained ignored.
- Remote: `andrewwer416/CSCI5942-Final-Project---Image-Data-Curation`.
