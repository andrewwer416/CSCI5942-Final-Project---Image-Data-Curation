# P4 color and contrast normalization

Status: **implemented** (2026-11-25). Decisions approved by Andrew on 2026-11-24.

The proposal says only: "apply standardized color and contrast normalization before producing the
RGB input." This document pins down the rest. P4 trains on the shared random selection (D1) and is
compared against P1.

## Method

P4 is **per-image appearance normalization**. Each image's color cast is corrected and its contrast
stretched. Nothing is fitted to data. Steps, on the stored pixels:

1. **White balance (Shades-of-Gray, p = 6).** Scale each channel so its p-norm mean equals the mean
   of the three. This leaves the overall level unchanged. Each channel gain is capped to [½, 2].
2. **Contrast.** Stretch brightness (BT.601 luma) linearly so its 1st and 99th percentiles map to 0
   and 1, with the gain capped at 2×. When the cap binds, the stretch centers on the percentile
   midpoint. The same brightness change is added to all three channels, so hue and saturation
   (Cb, Cr) are preserved except where values clip.
3. **Resize and scale as P1.** Round back to 8-bit, PIL bilinear to 640 × 640, then fixed [−1, 1]
   scaling.

| Decision | Choice |
|---|---|
| Meaning of "normalization" | Per-image appearance normalization, not dataset mean/std standardization |
| Fitted statistics | None, so there is no leakage risk |
| Evaluation | val2017 gets the same per-image normalization |
| Grayscale images (3.3%) | No special case: white-balance gains are exactly 1, and contrast still applies |
| Reporting | Text-to-image Recall@K is also reported for captions that name a basic color, in every arm |
| Computation | On the fly, at native resolution |

Code:
- `src/vlm_train/data.py`: `normalize_appearance` and `load_color_contrast_normalized`.
- `src/vlm_train/metrics.py`: `mentions_color` and `caption_subsets`.
- Select the arm with `python -m vlm_train.train --preprocessing color_contrast_normalization`.
  `run_config.json` records the parameters, and every run's metrics include `t2i_recall@K_color`
  and `color_caption_count`.

## Why these choices

**Per-image, not dataset standardization.** Subtracting a train-set channel mean and dividing by a
channel standard deviation is a fixed per-channel scale and shift. The patch-embedding layer can
absorb it exactly through its weights and bias. It could only matter through optimization, and
AdamW is largely insensitive to input scale, so it would likely measure close to nothing. Per-image
normalization actually changes what the model sees, by removing variation in lighting and camera
color.

**Shades-of-Gray p = 6.** p = 1 is gray-world, which is easily fooled by large single-color areas.
p = ∞ is white-patch, which is easily fooled by one bright highlight. p = 6 is the commonly used
compromise from the color-constancy literature.

**The 2× caps.** These are project safeguards, not literature values. They limit the damage on
images where the assumptions fail, such as an almost-monochrome scene or a nearly flat image.

**Brightness-only stretch.** Stretching each channel separately would act as a second white
balance. Stretching only brightness keeps colors' hue and saturation. A global stretch is used
instead of CLAHE (local, tile-based equalization), which amplifies noise and creates halos.

**Fixed [−1, 1] scaling and P1's resize.** Appearance is then the only thing that differs from P1.

## What it does to COCO

Measured on train-only samples of D1, with seed 5942:

| Measure | Result |
|---|---|
| Average pixel change per image (p10 / p50 / p90 / p99, out of 255) | 2.7 / 9.4 / 24.1 / 51.5 |
| Images changing less than 5/255 on average | 22.8% |
| Strongest white-balance gain above 1.1× / 1.25× / 1.5× | 24.2% / 4.1% / 0.9% |
| Contrast stretch needed (p50 / p90) | 1.11× / 1.40× |
| Images hitting the 2× contrast cap | 1.2% |
| Load time per image, P1 vs. P4 | 4.1 ms vs. 14.5 ms |

The change is mild for most images, since COCO photos are mostly well exposed and close to
neutral. A small P4 effect would therefore be a plausible, legitimate result.

**Known side effect: dominant colors are treated as casts.** The most-changed images are mostly
clear blue skies, which become white or gray. Shades-of-Gray reads a large blue area as a blue cast
and removes it. Other strongly colored scenes shift too, such as an orange wall turning yellow. The
sheet `reports/data/p4-before-after.jpg` shows this.

The effect is limited. Strong corrections (above 1.25×) affect 4.1% of images, and captions with
"blue … sky" appear for 1.3% of images. This is the risk the color-caption breakdown is meant to
measure. If P4 hurts color-caption recall noticeably while overall recall is flat, a lower
white-balance cap is the obvious follow-up variant.

## Caption color words

Captions are flagged if they contain a basic color term as a whole word: red, orange, yellow,
green, blue, purple, pink, brown, black, white, gray/grey, silver, gold/golden, tan, beige. On D1,
17.2% of captions and 48.7% of images have one. Lighting words such as "sunset", "dark" and "night"
appear in only 1.9% of captions, too few for a separate breakdown. The flag only affects reporting,
never training.

## Tests

- **`tests/test_training_components.py`:**
  - cast removal;
  - the white-balance cap, with its exact expected effect;
  - grayscale staying gray;
  - the 2× contrast cap and its centering;
  - percentile mapping;
  - chroma preservation under the brightness stretch;
  - flat and invalid inputs;
  - training and evaluation both using the normalization.
- **`tests/test_metrics.py`:**
  - whole-word color matching;
  - hand-checked subset recall;
  - empty or mis-shaped subsets being rejected.
