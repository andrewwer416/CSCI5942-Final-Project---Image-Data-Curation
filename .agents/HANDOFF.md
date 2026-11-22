# Session Handoff

> **Purpose**: Read-only dashboard for session continuity. Update only at session end.

## Last session

**Date**: 2026-11-22
**Objective**: Finalize D5 quality-aware diversity so that all five selections (D1–D5) exist, then
move on to the preprocessing decisions (P2–P5).

### What was done

- Scored all 118,287 train images with these measures, all at native resolution:
  - classical: NIQE, Laplacian blur, clipped exposure, and object-box variants of blur and exposure;
  - CLIP zero-shot quality prompts;
  - learned models trained on human ratings: TOPIQ-NR and MUSIQ (KonIQ-10k).
- Audited each measure for category bias and defect precision. Only TOPIQ came close to the 95%
  precision bar: about 88% clear defects and 98% clear or borderline in its worst 0.25%.
- Andrew accepted TOPIQ with a broader definition of defect that includes content-obscuring blur
  and heavy filters.
- Added an optional per-image modular bonus (`modular_gains`) to the facility-location selector,
  with a brute-force greedy test (85 tests pass).
- Generated D5: `quality_aware_k50_dup0p99_topiq_lam0p1099_seed5942_n15000.jsonl`. All gates
  passed and `vlm-data validate` is clean.
- Rewrote `docs/quality-aware-selection.md` as the full record: the final method, results, the λ
  sweep, the nine approaches tried and why each was rejected, and the original decisions with
  their outcomes.

### Decisions made

| Decision | Rationale | Reversible? |
|---|---|---|
| TOPIQ-NR alone, global ranking | Only measure near 95% precision. Relative ranking cancels semantic quality signals | Yes, versioned |
| Defect includes content-obscuring blur and filters | Such images hide what the caption describes. Approved by Andrew | Yes |
| Filter: < 192 px short side, then TOPIQ worst 0.25% | Precision drops quickly past 0.25% | Yes, versioned |
| λ = 0.1099 (largest keeping ≥ 90% of D3's coverage gain) | Approved sweep rule from the original decisions | Yes, versioned |
| pyiqa 0.1.15.post2 in an isolated `quality` pixi environment | Default locked environment unchanged | Yes |

## Current state

### Ready

- All five selections (D1–D5) are tracked in `data/processed/selections/`, each with a metadata
  sidecar. `scripts/package_selections.py` bundles them with checksums for Alpine.
- P1 (direct resize to 640×640, fixed [-1, 1] scaling) is implemented in
  `src/vlm_train/data.py:load_standard_rgb`.
- Training defaults are approved (see `STATUS.md`). Batch size is still to be measured on Alpine.

### Blockers and deferrals

- Nothing is on Alpine yet: neither the images, the selections nor the job scripts.
- val2017 images are not downloaded on Andrew's machine.
- P2–P5 methods are unapproved. Next session works through them.

### Environment

- Branch: `main`. Remote: `andrewwer416/CSCI5942-Final-Project---Image-Data-Curation`.
- Run commands with `export PATH=$PWD/.pixi-home/bin:$PATH` and `PYTHONPATH=src`. CLIP scripts
  also need `HF_HUB_OFFLINE=1`.
- Large local artifacts (ignored): train2017 JPEGs, CLIP and MPNet embeddings, quality scores in
  `data/interim/quality/`, and pinned weights in `data/raw/model-cache/`.

## Next session: P decisions

Each arm differs from P1 in one variable only and trains on D1 (random). Starting proposals are in
`docs/background-research.md` §5–7. The table at its end lists what each needs before approval.
The proposed work split (Bill: P2–P4 and Alpine scripts; Andrew: P5) is still unconfirmed.

| Arm | Matrix name | Open decisions | Proposed starting point | Evidence needed first |
|---|---|---|---|---|
| P2 | `aspect_ratio_preserving_rgb` | Letterbox vs. resize-shorter-side + center crop; padding value; whether padding is masked from attention; interpolation | Letterbox to 640, pad value 0 after [-1, 1] scaling (mid-gray), no mask | Fraction of images with a 640 long side (83.4%) and the aspect-ratio distribution, both already known from the D5 work |
| P3 | `random_resized_crop` | Scale and aspect ranges, interpolation, evaluation transform | Scale [0.70, 1.00], ratio [0.80, 1.25], bicubic + antialias; evaluation uses the P1 path | Train-only audit of how often crops keep the captioned objects, at minimum scales {0.50, 0.70, 0.80}, targeting 90–95% caption-consistent crops |
| P4 | `color_contrast_normalization` | Train-fitted channel mean/std alone, or a deterministic appearance normalization (Shades-of-Gray p=6, 1/99% luminance stretch, 2× gain cap) | The appearance pipeline, followed by the shared standardization | Train-only mean/std; a color-word caption audit, since normalization can erase "red bus" and similar descriptions |
| P5 | `rgb_plus_clip_features` | Fusion mechanism and location, dropout, parameter-matched control, val CLIP feature cache | Frozen OpenCLIP B/16 vector → zero-gated residual projection before the retrieval head. Fusion module instantiated in every run, fed zeros outside P5 | Model hidden width (768); evaluation with real, zeroed and batch-shuffled CLIP vectors |

Cross-cutting decisions to settle alongside them:
- **Interpolation.** P1 uses bilinear, but P3's proposal uses bicubic. Decide whether one choice is
  shared by every arm, so interpolation doesn't confound the comparison.
- **Evaluation transform.** Decide on one deterministic evaluation transform per arm, and record
  whether P2 evaluates with letterboxing.
- **Leakage.** Any fitted statistic (P4 mean/std, P3 audit) comes from train2017 only.

### Then

1. Copy all five selections to Alpine with checksums. Download train and val images there, then
   run `vlm-data validate --check-files`.
2. Measure batch size at 640×640 on Alpine and train the D1×P1 baseline with seeds 5942, 5943 and
   5944.
3. Optionally promote the scratchpad joint and CLIP quality-test scripts into `scripts/`. Their
   reports are recorded, but the repo cannot regenerate them.

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

D5 quality-aware diversity: decisions, measure audits, TOPIQ choice and generation. See "Last
session" above.
