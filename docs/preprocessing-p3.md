# P3 random resized crop

Status: **implemented** (2026-11-24). Decisions approved by Andrew on 2026-11-24.

The proposal says only: "apply random resized cropping during training before producing a
640 × 640 input." This document pins down the rest. P3 trains on the shared random selection (D1)
and is compared against P1.

## Method

| Decision | Choice |
|---|---|
| Crop shape | Aspect ratio log-uniform in **0.8–1.25 × the image's own ratio**, not a fixed near-square range |
| Crop area | Drawn uniform in **50–100%** of the image. Accepted crops average 70% (see below) |
| Position | Uniform over valid positions |
| Attempts and fallback | Up to 10 attempts, then the full image. A single attempt fits 79% of the time, so the fallback essentially never happens |
| Resize | Crop the stored pixels first, then PIL bilinear to 640 × 640 and the same fixed [−1, 1] scaling as P1 |
| Randomness | Stateless: each crop is a hash of `(seed, epoch, image_id)` (BLAKE2b, `crop-v1`), as with caption sampling |
| Evaluation | val2017 uses P1 unchanged: the full image stretched to 640 × 640, with no crop |
| Left out | Horizontal flips, color changes, box-aware crop rejection, padding masks |

Code:
- `src/vlm_train/data.py`: `sample_crop_box`, `load_random_resized_crop`, and `EVAL_PREPROCESSING`.
- Select the arm with `python -m vlm_train.train --preprocessing random_resized_crop`.
  `run_config.json` records the crop parameters and the evaluation transform.

## Train-only audit

`scripts/audit_crop_consistency.py` writes `reports/data/crop-audit.json`. It uses train2017 instance
boxes on D1, and no image pixels or val2017 data.

For the image-relative rows, it audits the exact crops training epochs 0–7 will see. A crop is
**caption-consistent** if every COCO category a caption names keeps at least 50% of one of its
boxes. Only the 85.6% of captions that name an annotated category can be checked. A **small object**
is one whose largest box covers under 5% of the image. That covers 31.6% of checks.

| Sampler | Min area | Caption-consistent | Small objects | Fallback | Mean area | Shape gap vs. evaluation (p50 / p90) |
|---|---:|---:|---:|---:|---:|---:|
| Fixed ratio 0.75–1.33 (SimCLR) | 8% | 61.2% | 60.6% | 0.0% | 41% | 1.35× / 1.77× |
| Fixed ratio 0.8–1.25 | 50% | 91.4% | 86.7% | 2.4% | 63% | 1.31× / 1.61× |
| Fixed ratio 0.8–1.25 (old proposal) | 70% | 97.6% | 95.2% | 25.5% | 78% | 1.20× / 1.34× |
| Fixed ratio 0.8–1.25 | 80% | 98.9% | 97.5% | 56.9% | 84% | 1.18× / 1.22× |
| **Image-relative ×0.8–1.25 (chosen)** | **50%** | **95.5%** | **92.3%** | **0.0%** | **70%** | **1.10× / 1.21×** |
| Image-relative ×0.8–1.25 | 70% | 98.7% | 97.1% | 0.0% | 81% | 1.09× / 1.20× |
| Image-relative ×0.8–1.25 | 80% | 99.5% | 98.6% | ≤ 0.2% | 87% | 1.07× / 1.16× |

The "shape gap" is how much more or less stretched an object looks in training than at evaluation,
where P1 stretches the full image.

## Why these choices

**Image-relative ratio.** The usual sampler picks near-square crops regardless of the photo. On
COCO, where 73% of images are landscape at 4:3 or 3:2, that goes wrong in two ways:
- **Frequent fallback.** Large near-square crops often don't fit a wide photo, so a quarter of
  attempts fall back to a center crop at a 70% minimum area.
- **Train/evaluation mismatch.** Training sees mildly stretched crops, but evaluation stretches the
  full image by a median 1.37×.

Sampling relative to each image's ratio keeps P1's stretch, so P3 is "P1 plus random cropping" and
cropping is the variable being tested. The remaining ×0.8–1.25 jitter adds mild shape variation,
with a median 1.10× gap from evaluation.

**50% minimum area.** It is the strongest setting audited that clears 95% caption consistency
(95.5%), and it is in line with the lower scale bounds used by ALBEF and BLIP. A from-scratch ViT on
15,000 images likely benefits from the stronger variation. The cost is weaker small-object
consistency: 92.3%, against 97.1% at a 70% minimum.

**Accepted crops average 70% area, not 75%.** A crop covering nearly the whole image only fits if
its shape stays close to the image's. So large-area draws with a shape change are rejected and
redrawn, which skews accepted crops smaller. torchvision's `RandomResizedCrop` rejects crops the
same way.

**Crop first, then resize.** Cropping the stored pixels uses the most real detail. A 50%-area crop
of a 640 × 480 image is about 453 × 339 px, which is upscaled about 1.41× per side. Bilinear matches
P1 and P2, so interpolation is not a second variable.

**Stateless randomness.** With a hash keyed on `(seed, epoch, image_id)`, crops do not depend on the
number of DataLoader workers, global RNG state or the model seed. Replicate runs with model seeds
5942–5944 therefore see identical crops, as they see identical captions. It also avoids the PyTorch
pitfall where workers duplicate random state.

**P1 evaluation.** Retrieval is scored on the full image in every arm, so P3 is compared with the
baseline on identical val2017 inputs. A crop at evaluation would discard content the captions may
describe.

**What stays out, and why:**
- **Flips:** captions say "left" and "right", so flipping can make a caption false.
- **Color changes:** they belong to P4.
- **Box-aware rejection:** this would be a different method, "caption-consistent cropping", and it
  would use annotations during training.
- **Masks:** a crop has no padding.

## Limits of the audit

The audit is a proxy:
- It checks only the categories a caption names, using names and a short synonym list, so 14.4% of
  captions are not checked at all.
- It cannot tell when a crop removes an action, relation or count the caption describes ("two dogs",
  "next to the table").
- The 50% box-visibility threshold is a project choice, not a measured standard.

## Tests

`tests/test_training_components.py` covers:
- determinism, independent of global RNG state;
- variation across epoch, image and seed;
- bounds and ranges for landscape, portrait, square and wide images;
- the area and log-ratio distributions;
- the full-image fallback;
- invalid ranges being rejected;
- the crop coming from stored pixels;
- training using the epoch's crop while evaluation uses P1;
- crops without a key being rejected.
