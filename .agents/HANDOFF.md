# Session Handoff

> **Purpose**: Read-only dashboard for session continuity. Update only at session end.

## Last session

**Date**: 2026-11-25
**Objective**: Finish every selection (D1–D5) and preprocessing arm (P1–P5), so that the first
round of training can start on Alpine.

### What was done

- **D5 quality-aware diversity.**
  - Removed images with a short side under 192 px, then the worst 0.25% by TOPIQ-NR.
  - Ran D3's facility-location selection plus a TOPIQ bonus (λ = 0.1099).
  - Result: 6,012 images differ from D3, and it keeps 90.06% of D3's coverage gain.
  - The full record of the nine quality measures tried is in `docs/quality-aware-selection.md`.
- **P2 aspect-ratio-preserving RGB:** centered letterbox on mid-gray, with padding-only patches
  masked in attention. P2 measures letterboxing plus masking.
- **P3 random resized crop:** aspect ratio 0.8–1.25× the image's own, area 50–100%, stateless per
  `(seed, epoch, image_id)`, evaluated with P1. 95.5% of crops keep the captioned objects in the
  train-only audit (`scripts/audit_crop_consistency.py`).
- **P4 color and contrast normalization:**
  - Per-image Shades-of-Gray white balance (p=6) plus a brightness stretch between the 1st and 99th
    percentiles, both capped at 2×. Unfitted, and applied at evaluation too.
  - Known side effect: clear blue skies turn white or gray (strong corrections on 4.1% of images).
- **P5 RGB + CLIP features:**
  - Frozen OpenCLIP vector added as `projection(LayerNorm(CLS + W·clip))`, with no gate and no
    dropout, and initialization paired with P1.
  - Zeroed and shuffled CLIP diagnostics at the final epoch.
- **Evaluation reporting:** every arm now also reports text-to-image recall for captions that name
  a color.
- **val2017:** images downloaded and checksum-verified on Andrew's machine, and the val2017 CLIP
  cache built.
- **End-to-end smoke test:** all five arms trained for one epoch and evaluated on real val2017
  images. 128 unit tests pass.

### Decisions made

All were approved by Andrew. Rationale is in each method document.

| Decision | Where recorded |
|---|---|
| D5: TOPIQ-NR alone, global ranking, filter worst 0.25%, λ = 0.1099 | `docs/quality-aware-selection.md` |
| P2: letterbox, mid-gray padding, mask padding-only patches, upscale small images | `docs/preprocessing-p2.md` |
| P3: image-relative ratio ×0.8–1.25, area 50–100%, P1 evaluation, no flips | `docs/preprocessing-p3.md` |
| P4: per-image appearance normalization, 2× caps, color-caption reporting | `docs/preprocessing-p4.md` |
| P5: late fusion, no gate, no dropout, cached CLIP inputs, P1 as control | `docs/preprocessing-p5.md` |
| PIL bilinear for every arm | `docs/preprocessing-p2.md` |

## Current state

### Ready

- **Selections:** all five (D1–D5) are tracked in `data/processed/selections/`.
- **Preprocessing:** all five arms are selectable with
  `python -m vlm_train.train --preprocessing {standard_rgb, aspect_ratio_preserving_rgb,
  random_resized_crop, color_contrast_normalization, rgb_plus_clip_features}`.
- **Handoff bundles** (ignored by Git, under `artifacts/handoff/`):
  - `selection-manifests.zip`: all five selections with checksums, from
    `scripts/package_selections.py`.
  - `clip-features.zip` (253 MB, SHA-256 `087c4406…a5f5`): train2017 and val2017 CLIP caches with
    checksums, from `scripts/package_clip_features.py`. Unpack into `data/interim/visual/` for P5.
    It is also attached to a temporary, public GitHub prerelease, since embeddings are not committed:
    https://github.com/andrewwer416/CSCI5942-Final-Project---Image-Data-Curation/releases/tag/handoff-clip-features-2026-11-25. Delete the release (`gh release delete handoff-clip-features-2026-11-25 --cleanup-tag`)
    once Alpine has a verified copy.
- **Training defaults** are approved (see `STATUS.md`). Batch size is still to be measured on Alpine.

### Blockers and deferrals

- **Nothing is on Alpine yet:** not the images, the selections, the CLIP caches or the job scripts.
- **Unconfirmed work split:** the proposal was Bill on P2–P4 and the Alpine scripts, Andrew on P5.
  All five arms are now implemented.
- **Pending review:** all work is in PR #2 (branch `andrew/quality-aware-selection`), awaiting
  Bill's review.

### Environment

- **Branch:** `andrew/quality-aware-selection` (PR #2). Remote:
  `andrewwer416/CSCI5942-Final-Project---Image-Data-Curation`.
- **Commands:** run with `export PATH=$PWD/.pixi-home/bin:$PATH` and `PYTHONPATH=src`. CLIP
  commands also need `HF_HUB_OFFLINE=1`.
- **Large local artifacts (ignored):** train2017 and val2017 JPEGs, CLIP caches (train and val),
  MPNet embeddings, quality scores, and pinned weights in `data/raw/model-cache/`.

## Next session

1. Get PR #2 reviewed and merged.
2. On Alpine:
   - fetch the selections from Git and the CLIP bundle from the release, and check its SHA-256;
   - download the train and val images;
   - run `vlm-data validate --check-files`.
3. Measure the batch size at 640×640 on Alpine, using the most memory-hungry arm, and hold it fixed
   across all arms. Masked P2 measured the same as unmasked on the RTX 2080.
4. Train the D1×P1 baseline with model seeds 5942, 5943 and 5944, then the other eight first-round
   runs.
5. **Optional:** move the scratchpad joint and CLIP quality-test scripts into `scripts/`. Their
   reports are recorded, but the repo cannot regenerate them.

Interpretation notes to carry into the results:
- **P2** is letterboxing plus masking; an unmasked letterbox is a candidate follow-up run.
- **P4** could lower color-caption recall through its blue-sky side effect; a weaker white-balance
  cap is the follow-up variant.
- **P5** may benefit from CLIP's possible pretraining exposure to COCO. Use the zeroed and shuffled
  diagnostics when interpreting it.

## Session log

### Sessions 1–3 — 2026-11-07

Built the data foundation, manifests and random baseline, and published them to GitHub. Calibrated
and generated the category-balanced selection (948 swaps, 36/36 targets).

### 2026-11-08 (Andrew and Bill)

- Andrew: pinned OpenCLIP, generated D3 visual diversity (k=50, 0.99 duplicates) and D4
  caption-semantic diversity (MPNet, k=50).
- Bill: added the training pipeline in PR #1, smoke-tested on a 4 GB GPU.
- Approved the training defaults and the extra baseline seeds.

### 2026-11-21 to 11-22 (Andrew)

D5 quality-aware diversity: decisions, measure audits, TOPIQ choice and generation.

### 2026-11-22 to 11-25 (Andrew)

Decided and implemented P2–P5. Added the crop audit, the color-caption recall breakdown, the
val2017 download, the val2017 CLIP cache and the CLIP handoff bundle. Ran the first end-to-end smoke
test of every arm on real val2017 images.
