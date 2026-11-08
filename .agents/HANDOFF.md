# Session Handoff

> **Purpose**: Read-only dashboard for session continuity. Update only at session end.

## Last session

**Date**: 2026-11-07
**Objective**: Ground category balancing in reproducible analysis of the real COCO training
manifest and generate the approved selection.

### What was done

- Extracted and traced the four-page assignment proposal.
- Created the Python project, data CLI, configuration, docs, status, and offline tests.
- Downloaded COCO 2017 annotations only; verified the pinned byte count and SHA-256.
- Built canonical manifests for 118,287 train and 5,000 validation images.
- Generated the seed-5942 random baseline with exactly 15,000 unique training images.
- Validated counts, five-caption policy, hashes, source identity, and split isolation with no errors.
- Added a count-sensitivity command that recommends the 2.5% threshold and nonbinding 4× cap.
- Generated a 15,000-image category-balanced set by swapping 948 random-baseline images (6.32%),
  retaining 93.68% of the paired baseline and meeting 36 of 36 targets.

### Decisions made

| Decision | Rationale | Reversible? |
|---|---|---|
| Python 3.11 standard-library data path | Reproducible setup with no dependency gate | Yes |
| Stable hash ranking for random baseline | Independent of row order and Python RNG changes | Yes, versioned |
| Retain lowest five caption IDs | Real COCO has a few images with 6–7 captions; proposal requires five | Yes, versioned |
| HTTP downloads require pinned size and SHA-256 | Official COCO page publishes HTTP archive links | Yes |
| Defer large image archives | Not needed to establish manifest and selection integrity | Yes |
| Calibrate category balance from train counts | Avoid presenting 2.5% / 4× as literature constants | Yes, versioned |
| Repair the paired random baseline | Isolate rare coverage and avoid an object-density confound | Yes, versioned |

### Main files added

- `README.md`, `CLAUDE.md`, `STATUS.md` — entry points and current state
- `configs/` — data parameters and exact nine-run matrix
- `src/vlm_data/` — acquisition, manifest, category analysis, selection, caption sampling, and validation
- `tests/` — synthetic offline regression suite
- `docs/data-foundation.md` and `docs/background-research.md` — contract, evidence, and decisions

## Current state

### Ready

- `make test`: 25 tests passing.
- `make validate`: canonical, random, and category-balanced manifests valid.
- `make analyze-category-balance`: real train-only report recommends 2.5% / 4× and verifies at
  least 5% of the paired baseline changes.
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
- Use the SHA-pinned category-balance report and selection metadata in the study audit trail.

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

### Session 3 — 2026-11-07

- Objective: ground the proposed category-balance constants in real-data analysis.
- Outcome: added a train-only sensitivity command, selected the 2.5% threshold and nonbinding 4×
  cap, repaired 948 baseline images (6.32%), met 36 of 36 targets, generated the real 15k
  selection, and passed validation.
