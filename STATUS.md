# Project Status

**Last Updated**: 2026-11-08

## Current focus

Move the smoke-tested training pipeline to Alpine and find a workable 640×640 batch size, while
resolving the remaining selection and preprocessing methods.

## Active work

| Task | Status | Notes |
|---|---|---|
| Annotation acquisition | Complete | Official 2017 archive checksum verified |
| Image acquisition | Partial | Andrew's machine: all 118,287 train JPEGs, validation deferred. Bill's laptop: smoke sample only (512 train / 128 val). Alpine: not downloaded |
| Canonical manifests | Complete | 118,287 train and 5,000 validation records |
| Random baseline selection | Complete | 15,000 records, seed 5942, stable-hash algorithm |
| Category-balance analysis | Complete | Real train-count grid recommends 2.5% threshold / 4× cap |
| Category-balanced selection | Complete | 15,000 unique images; 948 swaps (6.32%), 93.68% retained |
| Visual-diversity selection | Complete | k=50, duplicate cutoff 0.99; 13,073 replacements (87.153%) |
| Caption-semantic selection | Complete | Frozen MPNet, k=50; 13,116 replacements (87.44%) |
| Quality-aware selection | Pending | Design remains a placeholder |
| Recall@K evaluation | Complete | Both retrieval directions; hand-checked in unit tests |
| Standard RGB preprocessing (P1) | Complete | Direct resize, fixed scale to [-1, 1] |
| Other preprocessing (P2–P5) | Not started | P5 can reuse the pinned OpenCLIP checkpoint |
| VLM model and training loop | Smoke-tested | Runs end to end on 4 GB; no full-length run yet |
| Alpine job scripts | Not started | — |

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
- [x] Implement the model, training loop, P1 preprocessing, and Recall@K evaluation.
- [x] Smoke-test training on a real image sample.
- [ ] Resolve advanced selection and preprocessing decisions.
- [ ] Run the baseline (D1×P1) at full size on Alpine.

## Key metrics

| Metric | Current | Target |
|---|---:|---:|
| Initial run definitions | 9 | 9 |
| Initial runs completed | 0 | 9 |
| Canonical training images | 118,287 | 118,287 |
| Shared baseline images | 15,000 unique | 15,000 unique |
| Category-balanced images | 15,000 unique | 15,000 unique |
| Validation images | 5,000 | 5,000 |
| Captions retained per image | 5 | 5 |
| Synthetic tests | 84 passing in the locked environment | All passing |

## Training choices to review

See `docs/pr-1-consolidation.md` for proposed answers to PR #1, the distinction between
confirmed selector decisions and pending training choices, and the artifact handoff.
The local consolidation adds explicit model seeds, source hashes, selection validation,
and protection against overwriting completed runs. No full study training was launched.

The proposal does not specify these. They affect every run, so please raise any disagreement before
the first full-length run.

| Choice | Value | Reason |
|---|---|---|
| Loss | Symmetric contrastive (as in CLIP), learnable temperature | Standard for image-text retrieval |
| Tokenizer | Own vocabulary from all train2017 captions, words seen at least 5 times (10,186 entries) | Needs no pretrained resources, never sees val2017, and is identical across selections |
| Optimizer | AdamW, lr 1e-4, weight decay 0.01 | Starting point; not tuned yet |
| Pixel scaling for P1 | Fixed scale to [-1, 1] | Not fitted to data, so P4 remains a distinct configuration |
| Text pooling | Output at the `<eos>` position | Same as CLIP |
| Image encoder width | 768 dimensions, 12 heads | Standard ViT-B size |
| Mixed precision | On for GPU runs (`--no-amp` disables it) | Reduces GPU memory use |
| Batch size | Not decided; `--batch-size`, default 32 | To be measured at 640×640 on Alpine |

Proposal values are used as written: ViT-B/16 at 640×640, 12-layer text Transformer with d=512,
d_ff=2048, L=77, 40 epochs.

## Blockers

- Full COCO train and validation images are not on Alpine; validation images are needed for model
  evaluation.
- The visual and caption-semantic selections exist on Andrew's machine and must be copied to the training
  machine.
- Alpine job scripts do not exist yet.
- Quality-aware selection and P2–P5 details remain unapproved design choices.

## Next steps

1. Review the training choices above and the recorded visual calibration and duplicate-audit results.
2. Download the full COCO images on Alpine and run `vlm-data validate --check-files`.
3. Run a short Alpine job to find a workable batch size at 640×640, then train the D1×P1 baseline.
4. Copy visual and caption-semantic selections and metadata to Alpine with checksums.
5. Implement P2–P4 preprocessing.
6. Resolve the quality-aware method and P5; review caption diagnostics.

## Recent completions

### 2026-11-08 (Bill)

Added the training pipeline on top of the data layer. Nothing in `src/vlm_data/` or `configs/`
changed, and the data pipeline reproduced the recorded numbers on a second machine (948 swaps,
93.68% overlap, `validate` clean).

New files:

| File | Purpose |
|---|---|
| `src/vlm_train/metrics.py` | Recall@K in both directions; image→text counts the best of an image's five captions |
| `src/vlm_train/tokenizer.py` | Word-level vocabulary and caption encoding |
| `src/vlm_train/data.py` | P1 preprocessing and the train/eval datasets, using the existing `sample_caption` |
| `src/vlm_train/model.py` | ViT-B/16 image encoder, 12-layer text Transformer, contrastive loss |
| `src/vlm_train/train.py` | Training loop, per-epoch val2017 evaluation, run logging |
| `tests/test_metrics.py`, `tests/test_training_components.py` | 18 synthetic tests; skipped when PyTorch is not installed |

Changed files:

- `pyproject.toml`: added a `train` optional dependency group (PyTorch, Pillow, NumPy). The base
  install stays standard-library only.
- `Makefile`: added a `smoke-train` target.
- `README.md`: added install, training, and smoke-test instructions, and updated the status table.

Smoke-test results on an RTX A1000 Laptop GPU (4 GB). These only show that the code runs and can
learn. They are not experimental results.

| Check | Setup | Result |
|---|---|---|
| Full path, one epoch | 512 train / 128 val images, 224 px, batch 8 | Completes; 0.24 s/step, 2.5 GB peak; loss and Recall at chance level, as expected after 64 steps |
| Overfitting | 64 train images, 224 px, batch 8, 60 epochs | Mean training loss falls from 2.66 to 0.60 |
| Full resolution | 640 px, batch 2 | Fits, 2.9 GB peak |
| Full resolution | 640 px, batch 4 | Out of memory on this GPU |

### 2026-11-08 (Andrew)

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
