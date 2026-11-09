# Project Status

**Last Updated**: 2026-11-08

## Current focus

Resolve and implement the train-only data-selection methods before model implementation.

## Active work

| Task | Status | Notes |
|---|---|---|
| Annotation acquisition | Complete | Official 2017 archive checksum verified |
| Image acquisition | Train complete | 118,287 train JPEGs; validation images remain deferred |
| Canonical manifests | Complete | 118,287 train and 5,000 validation records |
| Random baseline selection | Complete | 15,000 records, seed 5942, stable-hash algorithm |
| Category-balance analysis | Complete | Real train-count grid recommends 2.5% threshold / 4× cap |
| Category-balanced selection | Complete | 15,000 unique images; 948 swaps (6.32%), 93.68% retained |
| Visual-diversity selection | Complete | k=50, duplicate cutoff 0.99; 13,073 replacements (87.153%) |
| Caption-semantic selection | Complete | Frozen MPNet, k=50; 13,116 replacements (87.44%) |
| Quality-aware selection | Pending | Design remains a placeholder |

## Foundation progress

- [x] Extract assignment data requirements.
- [x] Define project layout and machine-readable experiment matrix.
- [x] Implement safe acquisition, manifests, provenance, and validation.
- [x] Implement deterministic random selection and caption sampling.
- [x] Add synthetic offline tests.
- [x] Download COCO annotations and build canonical manifests.
- [x] Generate and validate the shared 15,000-image baseline.
- [x] Calibrate, generate, and validate the category-balanced 15,000-image selection.
- [x] Download and checksum-verify all 118,287 `train2017` images.
- [x] Pin and smoke-test the frozen OpenCLIP + GPU FAISS environment.
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
| Synthetic tests | 65 passing in the locked GPU environment | All passing |

## Blockers

- COCO validation images remain deferred; they are not needed for train-only selection calibration.
- Quality-aware selection remains an unresolved proposal gap.

## Next steps

1. Review the recorded visual and caption-semantic calibration and diagnostic results.
2. Resolve quality-aware selection next.
3. Download validation images when model evaluation begins.

## Recent completions

### 2026-11-08

- Pinned OpenCLIP `ViT-B-16` / `datacomp_xl_s13b_b90k` and its immutable weight provenance for
  visual selection.
- Chose global sparse facility-location selection, exact GPU FAISS neighbors, deterministic lazy
  greedy optimization, duplicate grouping, bounded outlier enrichment, and a 5% difference floor.
- Audited the local RTX 2080 / 30 GiB RAM machine and recorded the real-data calibration plan;
  empirical `k` and duplicate cutoff remain under real-data calibration.
- Downloaded the 19,336,861,798-byte `train2017` archive, verified its pinned SHA-256, extracted
  exactly 118,287 JPEGs, and retained download provenance.
- Locked and installed a CUDA 12.4 environment with frozen OpenCLIP and exact GPU FAISS; GPU and
  full offline test smoke checks pass.
- Completed all 118,287 frozen image embeddings and exact top-400 GPU neighbors with verified
  artifact hashes. The train-only benchmark selected AMP-FP16 / batch 256: 543.97 images/s,
  mean cosine drift 0.000002479, and top-50 overlap 99.785% against FP32.
- Completed frozen MPNet encoding of 591,435 captions without truncation, calibrated k=50,
  generated the caption-semantic 15k set, and passed all acceptance and canonical validation gates.
- Added full category allocation, language-proxy, caption agreement, redundancy, semantic-outlier,
  and visual cross-view diagnostics. The observed extreme-tail share is 7.32%, with no quota.
- Recorded the real caption sweep and tradeoffs in `docs/caption-semantic-results.md`.

### 2026-11-07

- Downloaded and checksum-verified the official COCO 2017 annotation archive.
- Built and validated canonical train/validation manifests and the shared random baseline.
- Added 25 offline tests covering provenance, determinism, category analysis, safe extraction, and
  tamper checks.
- Added train-only category-balance sensitivity analysis and selected 2.5% / 4× under a recorded
  14%-of-budget intervention-size rule plus a hard 5% minimum-change check.
- Replaced the provisional equalizing selector with versioned baseline repair, generated the real
  selection, and validated all 15,000 records.
