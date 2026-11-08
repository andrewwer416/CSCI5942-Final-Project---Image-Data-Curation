# Background Research and Initial Design Recommendations

- **Project:** CSCI 5942 — COCO data selection for image–text retrieval
- **Status:** Category balance calibrated; remaining methods are research-backed starting points
- **Last updated:** 2026-11-07

## Purpose and decision rule

This note records the background research behind seven open design questions in the project:
object-category balance, CLIP-based visual diversity, caption-semantic diversity, image-quality
selection, resized crops, color/contrast normalization, and frozen-CLIP feature fusion.

The literature can justify the form of an objective and a sensible search range, but it generally
cannot supply a universally correct constant for this exact COCO subset, model, and retrieval task.
Accordingly, this document separates three things:

- **Literature-backed principles**, which should remain stable.
- **Proposed defaults**, which are concrete enough to implement and preregister.
- **Calibration checks**, which use only `train2017` data and must be completed before freezing the
  experiment configuration. `val2017` must not be used to tune selection or preprocessing.

## Recommended starting configuration

| Design question | Recommended initial choice |
|---|---|
| Category balancing | Custom square-root rare-category targets, calibrated on real COCO counts and applied as minimal swaps to the paired random baseline |
| Visual diversity | Frozen OpenCLIP `ViT-B-16` / `datacomp_xl_s13b_b90k` embeddings with sparse facility-location selection |
| Caption-semantic diversity | Mean of the five normalized `all-mpnet-base-v2` caption vectors, followed by the same facility-location selector |
| Quality-aware diversity | Reject only the bottom 5% by no-reference quality, then use an 80/20 diversity–quality objective |
| Random resized crop | Scale `[0.70, 1.00]`, aspect ratio `[0.80, 1.25]`, calibrated against COCO boxes |
| Color/contrast normalization | Shared train-set channel standardization plus a deterministic white-balance/luminance-normalization experiment arm |
| CLIP feature fusion | Frozen, normalized CLIP vector injected through a zero-gated residual projection |

These choices deliberately isolate one factor per initial experiment arm. For example, category
quotas should not also be added to the first visual-diversity experiment, because that would make it
impossible to attribute a performance change to visual diversity alone.

## 1. Object-category balancing

### What the literature supports

Equal-sized category buckets are usually not the right objective for COCO:

- COCO images are multi-label, so selecting one image can satisfy several category targets.
- Natural frequencies carry information; forcing every class to the same count can severely
  overrepresent a tiny, visually narrow class.
- The number of examples is not the same as the number of useful examples. Long-tail work based on
  the *effective number of samples* models diminishing benefit from repeated, similar examples.
- The LVIS paper's repeat-factor sampling baseline increases exposure for rare categories with a
  smoothed square-root inverse-frequency rule instead of forcing a uniform distribution.
- Class-wise difficulty methods show that frequency alone is incomplete, but difficulty is not an
  immutable property of a category. It changes with model, task, image scale, occlusion, label
  quality, and training stage.

There is therefore no defensible universal list of “hard categories.” A category difficult for
classification may not be the difficult category for image–text retrieval, especially when it is
present visually but absent from the sampled caption.

LVIS is a large-vocabulary instance-segmentation dataset and benchmark built over the COCO image
collection. It is not itself a subset-selection method. In an appendix baseline, the LVIS authors
repeated images containing rare categories during training. Their best threshold on LVIS v0.5 was
`0.001`, or 0.1% image prevalence. They did not construct a fixed-size set of unique images, impose
a 3× cap, or measure visual redundancy. This project borrows only the square-root shape of that
sampling rule.

### Dataset-grounded evidence

The prepared `train2017` manifest has 118,287 images. A 15,000-image uniform sample has inclusion
rate

```text
selection rate = selection budget / available training images
               = 15,000 / 118,287
               = 0.1268
```

Some observed train-set category counts illustrate why modest rare-class support is preferable to
equalization:

| Category | Full train count | Natural 15k expectation | Example proposed target |
|---|---:|---:|---:|
| hair drier | 189 | 24 | 95 |
| toaster | 217 | 28 | 102 |
| parking meter | 705 | 89 | 184 |
| scissors | 947 | 120 | 213 |
| person | 64,115 | 8,130 | natural coverage; no boost |

The current deterministic random baseline happens to contain 22 hair driers, 33 toasters, and 92
parking meters, which is close to the natural expectation and provides a useful comparison point.

### Proposed coverage target

For each category `c`, define:

- `N[c]`: number of training images containing category `c`;
- `frequency[c]`: fraction of all training images containing category `c`;
- `boost[c]`: multiplier applied to the category's natural expected count; and
- `target[c]`: desired number of selected images containing category `c`.

```text
frequency[c] = N[c] / 118,287

boost[c] = min(
    4,
    max(1, sqrt(0.025 / frequency[c]))
)

target[c] = min(
    N[c],
    ceil(selection_rate * N[c] * boost[c])
)
```

Here, `sqrt` means square root, `ceil` rounds up to the next integer, `min` chooses the smaller
argument, and `max` chooses the larger argument.

Interpretation:

- categories at or above 2.5% prevalence retain their natural expected coverage;
- rarer categories receive a smooth square-root boost; and
- no category is boosted by more than 4× its natural expected count.

The 2.5% threshold and 4× cap are **project choices**, not constants established by the
literature. They are retained because the stronger real-data sensitivity below selects them under
an explicit intervention-size rule—not because LVIS used those values.

### Real-data calibration of the values

Run the analysis directly against the canonical `train2017` JSONL; image archives are not required:

```bash
make analyze-category-balance
```

The command writes complete machine-readable and human-readable reports to:

- `reports/data/category-balance-analysis.json`
- `reports/data/category-balance-analysis.md`

Both are generated artifacts and intentionally ignored by Git. The report records the source
manifest SHA-256, simulation seed, selector versions, and SHA-256 hashes of both image-ID sets, so
every numerical claim can be tied to exact inputs and outputs.

The sensitivity grid is:

```text
frequency thresholds: 0.001, 0.0025, 0.005, 0.01, 0.02, 0.0225, 0.025, 0.03, 0.04, 0.05
maximum boosts:        2×, 3×, 4×, 5×, 6×, and uncapped
```

For each scenario, the script computes affected and capped categories, every category target, and
the added target mass:

```text
added target mass = sum, over categories, of:

    natural_expected[category] * (effective_boost[category] - 1)
```

This is a count of additional **category-image incidences**, not necessarily distinct images. A
single multi-label image can satisfy several targets.

The train-only decision rule is deliberately simple and auditable:

1. Select the largest tested threshold that affects at least one category while adding no more than
   14% of the 15,000-image budget in category-image target mass.
2. At that threshold, select the smallest tested integer cap that does not bind any current COCO
   category.
3. Construct the actual multi-label selection and require at least 5% of its images (750 of 15,000)
   to differ from the paired random baseline. The command fails instead of padding the result with
   arbitrary swaps if this requirement is not met.

The 14% target-mass guardrail and 5% actual-change floor are also project choices. The initial 1%
guardrail produced only 103 swaps (0.69% of the set), and the later 10% guardrail produced 740 swaps
(4.93%), just below the required operational separation. Raising the mass guardrail to 14% admits
the next tested threshold, 2.5%, without admitting the much broader 3% case. None of these values is
a literature optimum.

Results on the current 118,287-image manifest are:

| Threshold | Affected categories | Uncapped added mass | Share of 15k budget |
|---:|---:|---:|---:|
| 0.001 | 0 | 0.00 | 0.000% |
| 0.0025 | 2 | 10.62 | 0.071% |
| 0.005 | 2 | 36.34 | 0.242% |
| 0.01 | 6 | 137.34 | 0.916% |
| 0.02 | 28 | 1,021.87 | 6.812% |
| 0.0225 | 33 | 1,476.34 | 9.842% |
| **0.025** | **36** | **1,962.43** | **13.083%** |
| 0.03 | 50 | 3,178.93 | 21.193% |
| 0.04 | 67 | 6,505.19 | 43.368% |
| 0.05 | 71 | 10,160.77 | 67.738% |

At threshold `0.025`, the largest required boost is `3.956×` for hair drier. A 3× cap would alter
two categories, while 4× is the smallest candidate cap that changes none. Thus the rule recovers
the proposed `0.025` threshold and `4×` safety cap. The cap is empirically inactive on current
COCO; its purpose is to bound behavior if the source distribution changes.

The boundary is meaningful. The next grid value, `0.03`, exceeds the 14% intervention guardrail,
affects 50 of 80 categories, and adds 21.19% of the image budget as target mass. At that point the
method begins changing much of the distribution rather than concentrating on the tail. The `0.04`
and `0.05` cases affect 67 and 71 categories and add 43.37% and 67.74% target mass, respectively;
those settings are retained in the sensitivity report but rejected as broad redistribution arms.

### Multi-label selector and real feasibility check

The selector begins with the exact deterministic random 15,000-image baseline. It enforces targets
only for the 36 categories below the 2.5% threshold. While any target is deficient, an unselected
image receives this gain:

```text
gain(image) = sum, over deficient boosted categories in that image, of:

    max(
        0,
        (target[category] - selected_count[category]) / target[category]
    )
```

The highest-gain image is swapped for a baseline image whose removal cannot break an already met
boosted-category target. All ties use stable seeded hashes. This baseline-repair design is important:
an initial from-scratch greedy simulation met the targets but retained only 49.2% of the random set
and strongly favored images with many labels, creating an unintended object-density treatment.

The stronger real-data simulation makes 948 swaps (6.32% of the set), retains 14,052 of 15,000
baseline images (93.68%), and meets all 36 targets. Mean distinct categories per image rises from
`2.919` to `3.021`, a 3.5% increase. This clears the explicit 5% difference requirement,
while the high overlap and small label-density change preserve a useful paired comparison with the
random baseline. It does not guarantee a statistically significant retrieval improvement; that
still requires training the model and comparing outcome variation across seeds. Achieved coverage
for representative categories is:

| Category | Random baseline | Target | Balanced selection |
|---|---:|---:|---:|
| parking meter | 92 | 184 | 184 |
| bear | 127 | 214 | 214 |
| toaster | 33 | 102 | 102 |
| scissors | 137 | 213 | 213 |
| hair drier | 22 | 95 | 95 |
| toothbrush | 133 | 219 | 219 |

Some selected counts can exceed their targets because swap-in images satisfy several categories;
this is expected for multi-label selection. The analysis records exact selected coverage and
residual shortfalls, rather than assuming summed quotas are jointly feasible.

### How to investigate true category difficulty

Use a train-only audit split, for example 90% development / 10% audit, and report category-
conditioned retrieval recall together with:

- category prevalence;
- caption mention rate versus visual presence;
- median object area and fraction of small objects;
- crowding and co-occurrence count; and
- variability across at least two training seeds for the baseline and strongest method.

COCO object labels and caption nouns do not align automatically. A controlled synonym map similar
to the CHAIR evaluation protocol is needed before calling a category “mentioned.” Frequency-only
balancing should remain the initial arm; a difficulty-adaptive selector is better treated as a
follow-up experiment because it introduces model-dependent feedback.

## 2. CLIP checkpoint and visual-diversity objective

### Checkpoint options

| Option | Output width | Strengths | Main trade-off |
|---|---:|---|---|
| OpenCLIP `ViT-B-16`, `datacomp_xl_s13b_b90k` | 512 | Strong, reproducible open checkpoint trained under the DataComp framework; matches the proposed 512-D auxiliary interface | Newer training corpus than original CLIP, so it is a different pretrained prior |
| OpenAI `ViT-B/16` | 512 | Conservative and closest to the original CLIP paper | Older and generally less competitive than strong OpenCLIP checkpoints |
| OpenCLIP `ViT-B-16` trained on LAION-2B | 512 | Widely used open alternative | Different and less controlled source-data provenance |
| `ViT-B/32` family | 512 | Faster and cheaper to embed | Coarser spatial patches may miss small-object distinctions important in COCO |
| `ViT-L/14` family | commonly 768 | Stronger representation | More compute and a projection is required to preserve a 512-D project interface |

The recommended primary checkpoint is OpenCLIP `ViT-B-16` with pretrained tag
`datacomp_xl_s13b_b90k`. If strict adherence to the original proposal matters more than expected
representation quality, OpenAI `ViT-B/16` is the clean conservative alternative.

Whichever checkpoint is selected, pin all of the following in the run metadata:

- model name, pretrained tag, package version, and weight revision or hash;
- the checkpoint's official resize, center-crop, and normalization transform;
- evaluation mode and frozen parameters;
- embedding precision; and
- whether stored vectors are normalized.

Compute embeddings in batches, convert to float32 for storage/scoring, and L2-normalize them. Do not
feed the checkpoint the project model's experimental color-normalized images; frozen CLIP should
receive its own native preprocessing unless CLIP robustness itself is the object of study.

### Recommended diversity objective: facility location

For unit-normalized image embeddings, define nonnegative cosine similarity as follows. Because the
vectors are unit-normalized, their dot product is their cosine similarity.

```text
similarity(i, j) = max(0, dot(embedding[i], embedding[j]))
```

For a candidate selected set `S`, define its coverage of all training images as:

```text
coverage(S) = sum, over every training image i, of:

    max(similarity(i, j)) for every selected image j in S

Goal: maximize coverage(S), subject to size(S) = 15,000
```

This facility-location objective rewards a selected set that covers the whole embedding population.
Unlike pure farthest-point or maximum-distance selection, it does not strongly reward isolated
outliers simply because they are far from everything else. It is monotone and submodular, so greedy
selection has a well-understood approximation guarantee.

A dense 118,287 by 118,287 similarity matrix is unnecessary. Build a cosine nearest-neighbor graph
with `k = 100`, clamp similarities at zero, and run deterministic lazy greedy on this sparse graph.
Use image ID as the final tie breaker. Check sensitivity at `k ∈ {50, 100, 200}` on train-only
coverage statistics.

If facility-location infrastructure proves too expensive, a reasonable fallback is spherical
`k`-means with `K = 15,000`, five seeded restarts, followed by the nearest unique image medoid for
each centroid. This is easier to implement but is more sensitive to initialization and empty or
unbalanced clusters.

### What to report

In addition to retrieval scores, report:

- mean and high-percentile nearest-selected cosine distance over all train images;
- cluster occupancy and duplicate/near-duplicate rates;
- object-category coverage as a diagnostic, not a constraint; and
- visual examples from dense regions, sparse regions, and selected outliers.

## 3. Caption-semantic diversity

Caption-semantic selection is related to category balancing, but it is not equivalent. Category
labels capture the presence of 80 object classes. Captions also encode actions, relationships,
attributes, scenes, quantities, and wording, and may omit visually present objects. Its value is
precisely that it can cover semantic structure not available in the instance labels.

### Proposed representation

Use the pinned `sentence-transformers/all-mpnet-base-v2` checkpoint:

1. Encode each of an image's five captions independently.
2. L2-normalize each caption vector.
3. Compute their arithmetic mean.
4. L2-normalize the resulting image-level caption vector.

Then apply the same sparse facility-location objective and the same `k = 100` neighbor-graph
construction used for visual diversity. Holding the optimizer constant makes the comparison answer
a cleaner question: does diversity in caption space select better training data than diversity in
visual CLIP space?

The CLIP text encoder is a lower-infrastructure alternative, but it couples caption selection to the
same pretrained model family used for visual selection. MPNet provides a more independent semantic
view. A more expensive follow-up could perform group facility location over all five caption vectors
rather than collapsing them to one vector.

### Required diagnostics

Audit more than object nouns:

- noun and verb coverage;
- scene, relationship, attribute, and quantity phrase coverage;
- mean nearest-selected distance in caption-embedding space;
- within-image dispersion among the five captions; and
- category coverage, to reveal when semantic selection indirectly removes rare visual classes.

## 4. Quality-aware diversity

### Why PSNR and SSIM are not the right primary scores

PSNR and SSIM are full-reference metrics: they measure a distorted image against a known reference.
The source COCO image has no pristine reference image, so neither metric directly answers whether a
source image is intrinsically high quality. They can be useful later for auditing how much a resize
or normalization transform changes an image, but they should not drive source-image selection.

### Proposed no-reference quality score

Compute pre-resize measures on every valid train image:

- **NIQE**, where lower is better;
- **normalized sharpness**, using log variance of the Laplacian after resizing the short side to 256;
- **resolution adequacy**, using a capped function of total pixels or short-side length; and
- **exposure adequacy**, penalizing the fraction of near-black and near-white pixels.

Reject decode failures, extreme aspect-ratio errors, and corrupted dimensions before scoring.
Convert each measure to a robust percentile rank, preferably within broad semantic clusters so that
the score does not simply prefer particular types of scenes. Let `exposure_quality` be high when
clipped-pixel fraction is low. The proposed composite is:

```text
quality[image] =
      0.30 * percentile_rank(-NIQE[image])
    + 0.25 * percentile_rank(sharpness[image])
    + 0.25 * percentile_rank(resolution[image])
    + 0.20 * percentile_rank(exposure_quality[image])
```

The minus sign before `NIQE` reverses that measure so a higher percentile always means better
quality.

The weights are an explicit project hypothesis, not a literature-derived optimum.

First remove only the bottom 5% by `quality[image]`. On the remaining candidates, select 15,000
images by maximizing:

```text
normalized_coverage(S) = coverage(S) / number_of_training_images

average_quality(S) =
    sum(quality[image] for every image in S) / size(S)

selection_objective(S) =
      0.80 * normalized_coverage(S)
    + 0.20 * average_quality(S)
```

Both terms lie in `[0, 1]` when similarities and percentile ranks lie in `[0, 1]`. This makes
quality a guard rail rather than the dominant objective. A pure top-quality ranking would tend to
collapse semantic and visual diversity.

BRISQUE is a reasonable alternative to NIQE but is trained on particular synthetic distortions.
MUSIQ is a useful sensitivity check, though learned image-quality models can import aesthetic and
content preferences. Entropy and edge density measure visual complexity, not quality, and should not
be mislabeled as quality scores.

## 5. Random resized crop ranges

### Evidence and risk

The standard SimCLR crop is intentionally aggressive, with scale approximately `[0.08, 1.00]` and
aspect ratio `[0.75, 1.33]`. That is effective for invariance learning but can crop away the object
or relation described by an image–text pair. Vision–language systems such as ALBEF and BLIP use less
aggressive lower scale bounds in relevant training configurations, commonly around `0.5`.

For this retrieval study, the initial recommendation is:

```toml
scale = [0.70, 1.00]
aspect_ratio = [0.80, 1.25]
interpolation = "bicubic"
antialias = true
attempts = 10
```

This range still adds scale and framing variation while reducing the chance that the sampled caption
becomes false for the crop.

### Grounding the range in physical annotations

Before freezing the range, run a train-only Monte Carlo audit for minimum scale values
`{0.50, 0.70, 0.80}`. For each candidate crop:

1. Map caption noun mentions to COCO categories using a documented synonym list.
2. For every mentioned category with an annotated box, compute the fraction of box area retained.
3. Mark the crop caption-consistent if every mentioned category retains at least 50% of at least one
   matching box.
4. Report consistency by object size, number of mentioned categories, and crop scale.

A reasonable project target is 90–95% caption-consistent crops. The 50% visibility and 90–95%
acceptance thresholds are proposed operational values and should be shown in sensitivity analysis.

Use the annotations to choose a fixed crop distribution; do not silently perform box-aware rejection
during the main training run. Rejection sampling would create a different method—a
caption-consistent crop—and should be named and tested separately. Evaluation should always use a
deterministic full-image resize/center-crop path.

## 6. Color and contrast normalization

Two distinct operations are often both called “normalization” and should be kept separate.

### Numerical channel standardization

This is the ordinary tensor transform applied before the randomly initialized vision model:

```text
standardized_pixel[channel] =
    (pixel[channel] - training_mean[channel]) / training_std[channel]
```

Compute `training_mean` and `training_std` once from `train2017` only, pin them in the
configuration, and reuse them for all project-model runs. This is an optimization convention and
should not be changed only for one selection method.

Frozen CLIP is different: it must use the normalization packaged with its checkpoint. The original
OpenAI CLIP transform uses RGB mean
`(0.48145466, 0.45782750, 0.40821073)` and standard deviation
`(0.26862954, 0.26130258, 0.27577711)`, but the implementation should read these values from the
selected checkpoint's official transform rather than assume they are universal.

### Experimental appearance normalization

For the dedicated color/contrast-normalization arm, use a deterministic pipeline:

1. Apply Shades-of-Gray white balance with Minkowski exponent `p = 6`.
2. Convert to a luminance/chroma representation.
3. Stretch luminance between its 1st and 99th image percentiles.
4. Cap total contrast gain at 2×.
5. Preserve chroma, convert back to RGB, clip safely, and then apply the shared tensor
   standardization.

The Shades-of-Gray `p = 6` setting has empirical precedent. The 1st/99th percentiles and 2× gain cap
are proposed conservative safeguards and must be documented as project choices.

This transform can erase meaningful color language—“red bus,” “blue shirt,” or lighting
descriptions—so audit retrieval separately for captions containing color words. CLAHE is a useful
sensitivity alternative, but local histogram equalization can amplify noise and produce unnatural
local contrast. `ColorJitter` is stochastic augmentation, not dataset normalization, and should be
treated as a different variable.

## 7. Using and fusing CLIP features

CLIP can play two scientifically different roles in this project:

1. **Selection only:** frozen CLIP embeddings define which images form the visual-diversity subset.
2. **Representation injection:** frozen CLIP embeddings become an auxiliary input to the project
   model.

The second role should not be described as ordinary preprocessing; it transfers a large pretrained
representation into an otherwise randomly initialized model.

### Recommended fusion mechanism

Let `model_vector` be the project model's pooled image representation, and let `clip_vector` be the
frozen 512-dimensional CLIP image embedding. Fuse them as follows:

```text
clip_vector = stop_gradient(L2_normalize(CLIP(image)))

gate = tanh(alpha)

fused_vector = layer_normalize(
    model_vector
    + gate * dropout(clip_projection(clip_vector))
)

retrieval_vector = L2_normalize(output_projection(fused_vector))
```

Implementation defaults:

- `clip_projection` (the `W_c` matrix): linear projection `512 → d`, no bias, Xavier initialization;
- `alpha`: scalar initialized to zero and excluded from weight decay;
- dropout: `0.10`;
- CLIP: frozen and in evaluation mode, using native 224-pixel preprocessing; and
- fusion location: immediately before the final retrieval projection/head.

The zero-initialized gate means the model starts exactly as the non-CLIP baseline and learns whether
to use the auxiliary representation. This is easier to attribute than replacing the encoder and less
parameter-confounded than concatenating a 512-D vector and widening the head.

For a parameter-matched control, instantiate the fusion module in every run and feed a zero vector
outside the CLIP-fusion arm. At evaluation, compare real, zeroed, and batch-shuffled CLIP vectors.
Large degradation under shuffling is evidence that the model uses image-specific CLIP information
rather than merely benefiting from extra parameters or regularization.

Alternatives include concatenation plus an MLP, FiLM-style affine modulation, and cross-attention
over CLIP patch tokens. Concatenation is simple but changes head width; FiLM is more expressive but
harder to interpret; cross-attention is unnecessarily complex if the project retains only one pooled
CLIP vector.

## Cross-cutting experimental controls

Before running the matrix, freeze and record:

- symmetric image-to-text and text-to-image InfoNCE definitions;
- temperature initialization and whether it is learned;
- batch size, negative construction, tokenizer, and maximum caption length;
- shared nonexperimental image transforms;
- separate seeds for selection, model weights, caption sampling, and data order;
- paired model-weight seeds across compared arms;
- padding color and padding-mask behavior; and
- every derived artifact's source-manifest hash, method version, and checkpoint revision.

One training run per method does not estimate training variance. At minimum, rerun the random
baseline and the strongest selection/preprocessing candidates with a second model seed. Bootstrap
confidence intervals over validation queries quantify query sampling uncertainty, not optimizer or
initialization uncertainty.

Optional follow-up combinations chosen after inspecting `val2017` are adaptive to that validation
set. They must be labeled exploratory, or evaluated on an additional untouched test split.

## Decisions still requiring explicit approval

| Item | Proposed value | Needed before implementation |
|---|---|---|
| Primary CLIP checkpoint | OpenCLIP B/16 DataComp | Pin package and exact weight revision |
| Facility-location graph | cosine kNN, `k = 100` | Benchmark memory/time and `{50,100,200}` stability |
| Caption encoder | `all-mpnet-base-v2` | Pin model revision and license metadata |
| Quality filter/objective | bottom 5%; 80/20 | Validate metric distributions and normalize objective scales |
| Crop range | `[0.70,1.00]`, `[0.80,1.25]` | Complete box/caption-retention audit |
| Appearance normalization | SoG `p=6`, 1/99%, 2× cap | Complete color-word and artifact audit |
| CLIP fusion | zero-gated residual projection | Confirm target model hidden width and control design |

## Primary references

### Long-tail balance and category difficulty

- Cui et al., [Class-Balanced Loss Based on Effective Number of Samples](https://openaccess.thecvf.com/content_CVPR_2019/html/Cui_Class-Balanced_Loss_Based_on_Effective_Number_of_Samples_CVPR_2019_paper.html), CVPR 2019.
- Gupta et al., [LVIS: A Dataset for Large Vocabulary Instance Segmentation](https://arxiv.org/abs/1908.03195), 2019.
- Huang et al., [Class-wise Difficulty-Balanced Loss for Solving Class-Imbalance](https://arxiv.org/abs/2010.01824), 2020.
- Wu et al., [Distribution-Balanced Loss for Multi-Label Classification in Long-Tailed Datasets](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123490154.pdf), ECCV 2020.
- Rohrbach et al., [Object Hallucination in Image Captioning](https://aclanthology.org/D18-1437/), EMNLP 2018 (CHAIR object/synonym mapping).

### CLIP, representation learning, and subset selection

- Radford et al., [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020), 2021.
- OpenAI, [CLIP reference implementation](https://github.com/openai/CLIP).
- Ilharco et al., [OpenCLIP](https://github.com/mlfoundations/open_clip).
- Gadre et al., [DataComp: In Search of the Next Generation of Multimodal Datasets](https://arxiv.org/abs/2304.14108), NeurIPS 2023.
- Sener and Savarese, [Active Learning for Convolutional Neural Networks: A Core-Set Approach](https://openreview.net/forum?id=H1aIuk-RW), ICLR 2018.
- Wei, Iyer, and Bilmes, [Submodularity in Data Subset Selection and Active Learning](https://proceedings.mlr.press/v37/wei15.html), ICML 2015.

### Sentence representations

- Reimers and Gurevych, [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://aclanthology.org/D19-1410/), EMNLP-IJCNLP 2019.
- Song et al., [MPNet: Masked and Permuted Pre-training for Language Understanding](https://arxiv.org/abs/2004.09297), 2020.
- Sentence Transformers, [`all-mpnet-base-v2` model card](https://huggingface.co/sentence-transformers/all-mpnet-base-v2).

### Image quality and transforms

- Wang et al., [Image Quality Assessment: From Error Visibility to Structural Similarity](https://ece.uwaterloo.ca/~z70wang/publications/ssim.pdf), IEEE TIP 2004.
- Mittal et al., [Making a Completely Blind Image Quality Analyzer](https://doi.org/10.1109/LSP.2012.2227726), IEEE SPL 2013 (NIQE).
- Mittal et al., [No-Reference Image Quality Assessment in the Spatial Domain](https://doi.org/10.1109/TIP.2012.2214050), IEEE TIP 2012 (BRISQUE).
- Ke et al., [MUSIQ: Multi-Scale Image Quality Transformer](https://openaccess.thecvf.com/content/ICCV2021/html/Ke_MUSIQ_Multi-Scale_Image_Quality_Transformer_ICCV_2021_paper.html), ICCV 2021.
- Chen et al., [A Simple Framework for Contrastive Learning of Visual Representations](https://proceedings.mlr.press/v119/chen20j.html), ICML 2020 (SimCLR).
- Li et al., [Align before Fuse](https://arxiv.org/abs/2107.07651) and the [official ALBEF implementation](https://github.com/salesforce/ALBEF), NeurIPS 2021.
- Li et al., [BLIP](https://arxiv.org/abs/2201.12086) and the [official implementation](https://github.com/salesforce/BLIP), ICML 2022.
- Finlayson and Trezzi, [Shades of Gray and Colour Constancy](https://doi.org/10.1117/12.511559), 2004.
- Pizer et al., [Adaptive Histogram Equalization and Its Variations](https://doi.org/10.1016/S0734-189X(87)80186-X), 1987.

### Multimodal fusion examples

- Perez et al., [FiLM: Visual Reasoning with a General Conditioning Layer](https://arxiv.org/abs/1709.07871), AAAI 2018.
- Alayrac et al., [Flamingo: a Visual Language Model for Few-Shot Learning](https://arxiv.org/abs/2204.14198), NeurIPS 2022.
- Mokady et al., [ClipCap: CLIP Prefix for Image Captioning](https://arxiv.org/abs/2111.09734), 2021.
