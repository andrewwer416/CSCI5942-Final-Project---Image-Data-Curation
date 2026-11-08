# Project Status

**Last Updated**: 2026-11-07

## Current focus

Resolve and implement the train-only data-selection methods before model implementation.

## Active work

| Task | Status | Notes |
|---|---|---|
| Annotation acquisition | Complete | Official 2017 archive checksum verified |
| Image acquisition | Deferred | Large image downloads intentionally not started |
| Canonical manifests | Complete | 118,287 train and 5,000 validation records |
| Random baseline selection | Complete | 15,000 records, seed 5942, stable-hash algorithm |
| Category-balance analysis | Complete | Real train-count grid recommends 2.5% threshold / 4× cap |
| Category-balanced selection | Complete | 15,000 unique images; 948 swaps (6.32%), 93.68% retained |
| Advanced selection methods | Pending | Checkpoint/algorithm decisions required |

## Foundation progress

- [x] Extract assignment data requirements.
- [x] Define project layout and machine-readable experiment matrix.
- [x] Implement safe acquisition, manifests, provenance, and validation.
- [x] Implement deterministic random selection and caption sampling.
- [x] Add synthetic offline tests.
- [x] Download COCO annotations and build canonical manifests.
- [x] Generate and validate the shared 15,000-image baseline.
- [x] Calibrate, generate, and validate the category-balanced 15,000-image selection.
- [ ] Resolve advanced selection and preprocessing decisions.

## Key metrics

| Metric | Current | Target |
|---|---:|---:|
| Initial run definitions | 9 | 9 |
| Canonical training images | 118,287 | 118,287 |
| Shared baseline images | 15,000 unique | 15,000 unique |
| Category-balanced images | 15,000 unique | 15,000 unique |
| Validation images | 5,000 | 5,000 |
| Captions retained per image | 5 | 5 |
| Synthetic tests | 25 passing | All passing |

## Blockers

- COCO train/validation image archives are not downloaded locally.
- CLIP checkpoint and diversity/quality algorithms are not specified by the proposal.

## Next steps

1. Review the open decisions in `docs/data-foundation.md` before advanced feature generation.
2. Confirm sufficient disk space, then download train/validation images when training work begins.
3. Run `vlm-data validate --check-files` after image acquisition.

## Recent completions

### 2026-11-07

- Downloaded and checksum-verified the official COCO 2017 annotation archive.
- Built and validated canonical train/validation manifests and the shared random baseline.
- Added 25 offline tests covering provenance, determinism, category analysis, safe extraction, and
  tamper checks.
- Added train-only category-balance sensitivity analysis and selected 2.5% / 4× under a recorded
  14%-of-budget intervention-size rule plus a hard 5% minimum-change check.
- Replaced the provisional equalizing selector with versioned baseline repair, generated the real
  selection, and validated all 15,000 records.
