# P2 aspect-ratio-preserving RGB

Status: **implemented** (2026-11-22). Decisions approved by Andrew on 2026-11-22.

The proposal says only: "resize the image while preserving its original aspect ratio and pad the
resulting image to 640 × 640." This document pins down the rest. P2 trains on the shared random
selection (D1) and is compared against P1, which stretches every image straight to 640 × 640.

## Method

| Decision | Choice |
|---|---|
| Resize | Scale the long side to exactly 640 px, keeping the aspect ratio. Round the short side to the nearest pixel (minimum 1 px) |
| Small images | Upscale them like P1: 16.6% of train images have a long side under 640 px |
| Placement | Center on the canvas. When the padding splits unevenly, the extra pixel goes to the bottom or right |
| Resampling | PIL bilinear, the same as P1 (see below) |
| Pixel scaling | The same fixed scale as P1, 8-bit RGB / 127.5 − 1 → [−1, 1]. Nothing is fitted |
| Padding value | 0 after scaling (8-bit mid-gray, 127.5) |
| Attention mask | Patches made entirely of padding are hidden from attention in every image-encoder layer. Partially padded border patches stay visible |
| Mask source | The resize geometry, never pixel values |
| Patch-grid snapping | None: content is not rounded to 16-px multiples |
| EXIF orientation | Not applied, as in P1. Pixels are used in their stored orientation |
| Evaluation | val2017 is letterboxed and masked exactly as in training |

Code:
- `src/vlm_train/data.py`: `letterbox_geometry`, `padding_patch_mask`, `load_aspect_preserving_rgb`.
- `src/vlm_train/model.py`: `ImageEncoder.forward(images, patch_mask)`.
- Select the arm with `python -m vlm_train.train --preprocessing aspect_ratio_preserving_rgb`. `run_config.json`
  records `preprocessing` and `padding_masked`.

## What the change looks like on COCO

Computed from image sizes in the train2017 manifest:

| Fact | Value |
|---|---|
| Orientation | 72.7% landscape, 23.6% portrait, 3.7% square |
| Most common ratios | 4:3 (27%), 3:2 (26%), 3:4 (10%), 2:3 (7%) |
| Stretch P1 applies to a 4:3 image | 1.33× along the short side |
| Long side exactly 640 px | 83.4% (no resampling needed under P2) |
| Canvas area that is padding | mean 28%; 10th–90th percentile 23–34% |
| Patches that are padding only (of 1,600) | median 400 (25%). A real-image check gave the same median, maximum 1,040 |
| Ratio beyond 2:1 | 1.0% |

## Why these choices

**Masking (approved after discussion).** The masked encoder is the ordinary encoder with an empty
mask, so P1 is unchanged: unpadded arms pass no mask at all. Without a mask, about a quarter of
P2's image tokens would be uniform gray and still take part in attention. The model would have to
learn to ignore them, and they would dilute the class token's attention.

So P2 measures **letterboxing plus padding masking**, not letterboxing alone. If P2 differs
noticeably from P1, an unmasked letterbox run is a candidate for one of the optional follow-up
runs to separate the two effects.

Masking does not save compute. Padded tokens still pass through every layer and are only excluded
as attention keys. On the RTX 2080, peak training memory at batch 6 was 5,168 MiB without a mask
and 5,170 MiB with one. So the batch size can be measured with any arm and shared by all of them.

**Mid-gray padding.** It is the midpoint of the fixed scale and needs no statistics fitted from
the data, so P4 stays the only arm that uses fitted normalization. A zero input contributes
nothing through the patch-embedding weights, leaving only their constant bias. Because padding-only
patches are masked, the value matters only inside partially padded border patches.

**PIL bilinear.** Three reasons:
1. P1 already uses it, so P1 and P2 differ only in geometry, not in how pixels are resampled.
2. Under P2, the 83.4% of train images with a 640-px long side keep their stored pixels exactly.
   Interpolation only touches the 16.6% that are upscaled.
3. It is a simple, smooth filter for that upscaling. Bicubic is sharper but can overshoot and ring
   at edges. We did not test whether that affects retrieval.

No train2017 image has a side over 640 px, so neither P1 nor P2 ever shrinks a dimension, and
antialiasing plays no role here. Switching filters for one arm would add a second variable, so
every arm keeps bilinear unless the whole study changes together.

**Upscaling small images.** This matches P1, so aspect ratio stays the only difference. The
alternative of keeping small images at native size would also change their apparent object scale.

**Centering.** It is the usual letterbox convention, and the image encoder's position embeddings
are learned, so either placement can be learned. Centering also keeps content in the same region
for landscape and portrait images.

**Masking only fully padded patches.** A border patch holds real pixels, so masking it would hide
image content. The mask is computed from geometry so that gray image regions, such as an overcast
sky, are never mistaken for padding.

## Tests

`tests/test_training_components.py` covers:
- geometry for landscape, portrait, square, upscaled and extreme-ratio images, plus rounding and the
  uneven-split rule;
- mask counts, including partially padded border patches staying visible;
- loader output: centered content, mid-gray padding, and the matching mask;
- both datasets using the selected arm;
- masked pixels being unable to change the image embedding;
- an empty mask matching no mask;
- the mask shape check.
