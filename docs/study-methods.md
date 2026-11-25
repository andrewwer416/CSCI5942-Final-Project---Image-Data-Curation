# Study methods: final definitions

Status as of 2026-11-25: **every selection (D1–D5) and preprocessing arm (P1–P5) is implemented**,
approved by Andrew, and covered by tests. Each arm differs from the shared baseline (D1 × P1) in one
variable. This page is the overview; each linked document holds the full decision record,
alternatives considered, and evidence.

## Selections (all train2017, 15,000 unique images, seed 5942)

| ID | Method | Final definition | Key result | Record |
|---|---|---|---|---|
| D1 | Random | Stable keyed-hash ranking of canonical train2017 | Shared baseline | `data/processed/selections/README.md` |
| D2 | Category balanced | Square-root rare-category targets (2.5% threshold, 4× cap) applied as minimal swaps to D1 | 948 swaps (6.32%), 36/36 targets met | `data/processed/selections/README.md`, `docs/background-research.md` §1 |
| D3 | Visual diversity | Sparse facility location over frozen OpenCLIP ViT-B-16 vectors, k = 50, duplicate groups at 0.99, outlier cap 300 | 13,073 differ from D1 | `docs/visual-calibration-results.md`, `docs/visual-duplicate-audit.md` |
| D4 | Caption-semantic diversity | Same selector over the mean of five frozen MPNet caption vectors, k = 50 | 13,116 differ from D1 | `docs/caption-semantic-selection.md`, `docs/caption-semantic-results.md` |
| D5 | Quality-aware diversity | Drop short side < 192 px and the worst 0.25% by TOPIQ-NR, then D3's selector plus a TOPIQ bonus, λ = 0.1099 | 6,012 differ from D3; 90.06% of D3's coverage gain kept; mean TOPIQ +0.063 | `docs/quality-aware-selection.md` |

## Preprocessing arms (all trained on D1)

| ID | Method | Training input | Evaluation input | Record |
|---|---|---|---|---|
| P1 | Standard RGB | Stretch to 640 × 640 (PIL bilinear), fixed [−1, 1] scaling | Same | `STATUS.md` (training choices) |
| P2 | Aspect-ratio-preserving RGB | Long side to 640, centered on mid-gray. Patches made only of padding are masked in attention | Same | `docs/preprocessing-p2.md` |
| P3 | Random resized crop | Crop with aspect ratio 0.8–1.25× the image's own and 50–100% of its area, stateless per `(seed, epoch, image_id)`, then P1 | P1, no crop | `docs/preprocessing-p3.md` |
| P4 | Color and contrast normalization | Per-image Shades-of-Gray white balance (p = 6) and a 1st–99th-percentile brightness stretch, both capped at 2×, then P1 | Same | `docs/preprocessing-p4.md` |
| P5 | RGB + frozen CLIP features | P1 plus the cached OpenCLIP vector, fused as `projection(LayerNorm(CLS + W·clip))` | Same. Final epoch adds zeroed and shuffled CLIP diagnostics | `docs/preprocessing-p5.md` |

## Rules shared by every arm

- **Interpolation:** PIL bilinear everywhere, so interpolation is never a second variable.
- **Nothing fitted from val2017:** P2–P5 fit no statistics at all. The val2017 CLIP cache is a
  per-image model input, not a fitted statistic.
- **Seeds:** captions, data order and P3 crops are keyed on seed 5942, epoch and image ID, so the
  baseline replicate seeds (5942–5944) differ only in weight initialization.
- **Reporting:** every run reports Recall@1/5/10 in both directions, plus text-to-image recall for
  captions that name a color, which mainly serves P4.
- **Batch size:** measured once on Alpine at 640 × 640 and held fixed across arms.

## What each comparison can and cannot claim

- **P2** is letterboxing *plus* padding masking. An unmasked letterbox run would separate the two.
  Shape correction and fewer content patches cannot be separated.
- **P3's crop audit** is a proxy. It checks named COCO objects only, covering 85.6% of captions, and
  not actions or relations.
- **P4** turns clear blue skies white or gray, with strong corrections on 4.1% of images. The color-
  caption recall is the check; a weaker white-balance cap is the follow-up variant.
- **P5's CLIP** may have seen COCO images during pretraining. Interpret P5 with its zeroed and
  shuffled diagnostics.
- **D5's quality filter** reaches about 88% clear defects (98% clear or borderline), not the original
  95% bar. Andrew accepted it with a definition of defect that includes content-obscuring blur.

## Reproduction and handoff

| Need | Command or location |
|---|---|
| Rebuild D5 | `PYTHONPATH=src pixi run python scripts/calibrate_quality.py` (needs the TOPIQ scores from `scripts/score_learned_quality.py`) |
| P3 crop audit | `PYTHONPATH=src pixi run python scripts/audit_crop_consistency.py` |
| CLIP caches | `vlm-data embed-visual --split {train2017,val2017} --inference-precision amp_fp16 --batch-size 256` |
| Selections for Alpine | Tracked in Git under `data/processed/selections/`, or `scripts/package_selections.py` |
| CLIP caches for Alpine | `scripts/package_clip_features.py` → `clip-features.zip`, also attached to the temporary GitHub release `handoff-clip-features-2026-11-25`. Unpack into `data/interim/visual/` |
| Train one arm | `python -m vlm_train.train --selection <manifest> --preprocessing <arm> --output-dir <dir> --batch-size <N>` |
