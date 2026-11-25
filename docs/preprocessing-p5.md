# P5 RGB + frozen CLIP features

Status: **implemented** (2026-11-25). Decisions approved by Andrew on 2026-11-25.

The proposal says: "provide the processed RGB image together with a frozen 512-dimensional CLIP
image embedding as an auxiliary feature." This document pins down the rest. P5 trains on the shared
random selection (D1) with P1's RGB input and is compared against P1.

## Method

| Decision | Choice |
|---|---|
| CLIP model | Frozen OpenCLIP ViT-B-16 `datacomp_xl_s13b_b90k`, revision `767e6d14…`, the same checkpoint as D3 |
| CLIP input | CLIP's own transform of the stored image (224-px bicubic resize, center crop, CLIP normalization), precomputed once |
| CLIP vector | 512-D projected image feature, L2-normalized, float32 (inference in AMP FP16) |
| Fusion | Late: `projection(LayerNorm(CLS + W·clip))`, with `W` a 512→768 linear layer, no bias, Xavier-uniform |
| Gate | None |
| Dropout | 0, the approved project default |
| Control | P1 itself. The module exists only in P5, and its initialization leaves every other weight identical to P1's for the same seed |
| RGB input | P1, unchanged |
| Evaluation | val2017 with real CLIP vectors every epoch. At the final epoch, also with zeroed vectors and with vectors shuffled between images |

Code:
- `src/vlm_train/model.py`: `ModelConfig.clip_dim` and `RetrievalModel.clip_projection`.
- `src/vlm_train/data.py`: `ClipFeatureTable`.
- Select the arm with `python -m vlm_train.train --preprocessing rgb_plus_clip_features`.
  `run_config.json` records the cache metadata hashes. The final epoch's metrics include
  `zeroed_clip/*` and `shuffled_clip/*`.

## CLIP caches

| Split | File stem | Records | Purpose |
|---|---|---:|---|
| train2017 | `openclip_vit_b16_datacomp_xl_s13b_b90k_train2017` | 118,287 | D3 selection and P5 training inputs (the existing cache, reused) |
| val2017 | `openclip_vit_b16_datacomp_xl_s13b_b90k_val2017` | 5,000 | P5 evaluation inputs only |

Both caches live in `data/interim/visual/`. They were made by
`vlm-data embed-visual --split {train2017,val2017} --inference-precision amp_fp16 --batch-size 256`
with the same pinned checkpoint, transform and precision.

The val cache is a model input computed per image, like decoding the val images for evaluation.
Nothing is fitted or chosen from it, which the leakage policy allows.

Embeddings are never committed. `scripts/package_clip_features.py` writes
`artifacts/handoff/clip-features.zip` (253 MB) with a `SHA256.json`. Unpack it into
`data/interim/visual/` on the training machine. A P5 run re-verifies every hash and the source
manifests when it loads the caches.

## Why these choices

**Late fusion.**
- **Clean attribution.** CLIP enters at one point, so zeroing or shuffling it at evaluation cleanly
  removes its contribution. As an extra token in the image transformer, it would mix into all
  12 layers.
- **Matches the proposal.** "Auxiliary feature" suggests a side input to the representation.
- **No head change.** Concatenation would widen the final projection from 768 to 1,280 inputs and
  change the head.
- **Bounded scale.** The add happens before the existing LayerNorm, so the combined vector is
  renormalized whatever scale the CLIP branch reaches.
- **No bias.** A zero CLIP vector gives exactly the encoder-only path.

**No zero-initialized gate.**
- **Nothing to protect.** Gates that start at zero exist to preserve a pretrained model's behavior
  when an input is added. This model is randomly initialized, so starting exactly at the baseline
  has no value.
- **A hidden time constant.** With AdamW at learning rate 1e-4, a gate parameter moves at most
  about 1e-4 per step, and a 40-epoch run at batch 32 is 18,720 steps. How fast the gate opens could
  limit CLIP's use for reasons unrelated to its value.
- **CLIP already starts small.** A unit-length CLIP vector through a Xavier 512→768 layer gives
  values around 0.04 per element against a class-token output of order 1. It grows only if it helps.
- **No weight-decay exception needed** for a gate parameter.

**No dropout.** Zero dropout is the approved setting for every arm, so 0.1 in P5 alone would add a
regularization difference. Reliance on CLIP is what P5 measures, and the diagnostics report it
instead of suppressing it.

**CLIP's own preprocessing, precomputed.**
- **CLIP's input format.** CLIP's features are only reliable on the 224-px format it was trained
  on. A stretched 640-px input would be outside it and would need resized position embeddings.
- **Shared provenance.** The train cache is the one D3 used, with a pinned revision and verified
  hashes.
- **Nothing lost by caching.** P5's input does not change between epochs, so recomputing per step
  would repeat identical work.
- **The cost:** a 4:3 image's CLIP view keeps the central 75% of its width.
- **Caveat for follow-ups.** If P3 and P5 are combined in a follow-up, the random crop and the fixed
  CLIP view would show different regions of the image.

**P1 as the control, with paired initialization.**
- **The zero-input control is just P1.** With no bias, feeding zeros gives exactly P1's outputs and
  gradients, so a "zero-input control" run would be P1 again.
- **Paired initialization.** P5's projection is created last, under a forked random-number state.
  Every other weight therefore starts exactly as in P1 with the same model seed, and a test checks
  this.
- **Baseline seeds unaffected.** P1 needs no change, so the approved baseline seeds 5942–5944 are
  undisturbed.
- **Negligible size.** The projection is 393,216 parameters (129.9M vs. 129.5M total).

**Final-epoch diagnostics.**
- **Shuffled vectors** give each val image another image's vector, under a fixed derangement with
  seed 5942 in which no image keeps its own. A large drop means the model uses image-specific CLIP
  information.
- **Zeroed vectors** show what the trained image encoder does on its own. The model never saw zeros
  in training, so this is a rough lower bound.
- **Reporting only.** Neither result informs any training or method choice.

## Limitation: possible pretraining exposure

CLIP was pretrained on DataComp, collected from Common Crawl. COCO images are hosted on Flickr and
widely copied, so pretraining may have included COCO images or near-copies of them, val2017
included. DataComp's authors report removing near-duplicates of their evaluation sets, but whether
that covers COCO val2017 has not been verified here. Part of any P5 gain could reflect that
exposure, so state this wherever P5 results are presented.

## Verification

- **`tests/test_training_components.py`:**
  - lookup by image ID;
  - a reproducible derangement that moves every vector;
  - P5 datasets returning P1 images with the right vector;
  - CLIP vectors required exactly for P5;
  - all other weights identical to P1 for the same seed;
  - a zero vector reproducing the encoder-only embedding.
- **End-to-end smoke runs on real images.** All five arms trained for one epoch on 64 D1 images and
  evaluated on 200 real val2017 images at 224 px. Every run completed, and recall was at chance as
  expected after 8 steps.
