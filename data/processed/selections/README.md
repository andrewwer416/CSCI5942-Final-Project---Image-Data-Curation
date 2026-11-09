# Training-Set Selections

This directory contains the exact 15,000-image `train2017` manifests used by selection experiments.
The generated JSONL manifests and metadata sidecars are ignored by Git; this README is tracked so
the selection contract remains visible even when the local data artifacts are absent.

## Shared contract

Every implemented selection must:

- contain exactly 15,000 unique images from the canonical `train2017` manifest;
- use no `val2017` image, caption, statistic, or evaluation result during construction;
- record its seed, algorithm version, source-manifest SHA-256, and output SHA-256 in a metadata
  sidecar; and
- be generated before training rather than selecting images dynamically in the data loader.

## Method status

| Selection | Status | Local artifact |
|---|---|---|
| Random baseline | Implemented | `random_seed5942_n15000.jsonl` |
| Category balanced | Implemented | `category_balanced_t0p025_cap4_seed5942_n15000.jsonl` |
| Visual diversity | Implemented; train-only calibrated | `visual_diversity_k50_dup0p99_seed5942_n15000.jsonl` |
| Caption-semantic diversity | Implemented; train-only calibrated | `caption_semantic_k50_seed5942_n15000.jsonl` |
| Quality-aware diversity | Placeholder — design pending | Not generated |

## Random baseline

The random baseline uses seed `5942` and ranks every canonical training image by a stable keyed
hash. The first 15,000 ranked images form the baseline. Stable hashing makes the result independent
of source-row order and Python random-number-generator changes.

This same manifest is the control set for every selection-only comparison. It is also the shared
input for preprocessing-only experiments.

Generate it with:

```bash
make baseline
```

## Category-balanced selection

### Objective

The category-balanced method increases coverage of less-prevalent COCO object categories without
equalizing every category and without constructing an unrelated set from scratch. It begins with
the exact random baseline and deterministically swaps images only while an affected category target
is deficient.

For category `c`, the checked-in target is:

```text
frequency[c] = source_images_with_category[c] / 118,287

boost[c] = min(4, max(1, sqrt(0.025 / frequency[c])))

target[c] = min(
    source_images_with_category[c],
    ceil(15,000 * frequency[c] * boost[c])
)
```

The 2.5% threshold and 4× cap are project parameters calibrated from `train2017`; they are not
constants supplied by LVIS or another paper. At these settings, 36 categories receive boosted
targets. The selector scores an unselected image by the normalized deficits it would reduce across
all affected categories it contains, then removes a baseline image only when doing so preserves
targets already met. Seeded hashes resolve ties.

### Minimum-change requirement

The category-balanced treatment must replace at least 5% of the paired random baseline. “Changed”
means selected image IDs that are present in the category-balanced set but absent from the random
set:

```text
changed_count = size(category_balanced_ids - random_baseline_ids)
changed_fraction = changed_count / 15,000
required_changed_count = ceil(0.05 * 15,000) = 750
```

Because both sets have the same size, the same number of baseline images is removed. This measure
counts 948 replaced positions, not the 1,896 IDs in the two-sided symmetric difference.

The generator checks this requirement before writing either the JSONL or its metadata. If fewer
than 750 images change, generation fails. It does not add unrelated swaps merely to reach the
floor; the threshold must produce enough category-driven changes.

The current seed-5942 result is:

| Measurement | Result |
|---|---:|
| Images replaced relative to random | 948 / 15,000 (6.32%) |
| Images retained from random | 14,052 / 15,000 (93.68%) |
| Required minimum | 750 / 15,000 (5.00%) |
| Affected-category targets met | 36 / 36 |
| Mean distinct categories per image | 2.919 random → 3.021 balanced |

Generate and audit it with:

```bash
make analyze-category-balance
make category-balanced
make validate
```

The analysis report is written to `reports/data/category-balance-analysis.md`. The selection
metadata repeats the threshold, cap, minimum-change fraction, actual overlap, changed count, and
whether the requirement was met.

## Future selection placeholders

These methods remain intentionally unresolved. Research notes in `docs/background-research.md` are
starting points, not approved operational specifications. Each method should receive its own
parameter decision, implementation, real-data audit, metadata contract, and tests before its
placeholder is replaced.

### Visual diversity — design record

The accepted manifest uses **k=50** and **duplicate cutoff 0.99**. It contains 15,000 unique
train images, replacing 13,073 random-baseline images (87.153%). All four k candidates passed;
the approved smallest-k tolerance rule selected 50. See
[calibration results](../../../docs/visual-calibration-results.md) and
[duplicate audit](../../../docs/visual-duplicate-audit.md) for the evidence.

#### Locked representation

| Field | Decision |
|---|---|
| Library model name | OpenCLIP `ViT-B-16` |
| Pretrained tag | `datacomp_xl_s13b_b90k` |
| Weight repository | `laion/CLIP-ViT-B-16-DataComp.XL-s13B-b90K` |
| Repository revision | `767e6d14be23c46e4f80af31e6b4e73f323e72f5` |
| Weight SHA-256 | `80905176486d952914c1deb3b753b8e653ab54aa4ffb14208bc5007cf7643a16` |
| Architecture facts | 224-pixel input, 16-pixel patches, 512-dimensional output |

Why this checkpoint:

- It is an open, named OpenCLIP model/tag pair with a public immutable revision and weight hash.
- `ViT-B-16` preserves more local detail than a 32-pixel-patch model, which matters for COCO's
  small objects, while remaining practical on the available 8 GiB GPU.
- Its 512-dimensional vectors are compact: all 118,287 float32 embeddings require about 231 MiB.
- OpenCLIP reports 73.5% ImageNet zero-shot accuracy for the 224-pixel DataComp-1B B/16 model after
  13 billion samples, making it a strong representation without the cost of a large ViT-L tower.
- DataComp was explicitly designed to evaluate multimodal dataset curation across 38 downstream
  test sets, so its training provenance is unusually aligned with this project's selection focus.

This checkpoint defines the geometry in which “visually similar” is measured. It is a pretrained
prior, not ground truth: patterns the checkpoint ignores cannot be recovered by the selector.

Primary references:

- [OpenCLIP pretrained registry](https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/pretrained.py)
- [OpenCLIP model results](https://github.com/mlfoundations/open_clip)
- [Pinned Hugging Face revision and weight hash](https://huggingface.co/laion/CLIP-ViT-B-16-DataComp.XL-s13B-b90K/commit/767e6d14be23c46e4f80af31e6b4e73f323e72f5)
- [Pinned model configuration](https://huggingface.co/laion/CLIP-ViT-B-16-DataComp.XL-s13B-b90K/blob/767e6d14be23c46e4f80af31e6b4e73f323e72f5/open_clip_config.json)
- [DataComp paper](https://arxiv.org/abs/2304.14108)

#### Frozen embedding contract

The CLIP tower is permanently frozen for this method. There is no configuration or CLI option to
fine-tune, adapt, or unfreeze it. The implementation asserts both `model.training == false` and
`requires_grad == false` for every parameter before it accepts an image batch.

| Contract field | Fixed behavior | What it controls |
|---|---|---|
| Source population | Canonical `train2017`, sorted by COCO image ID | Vector-to-record alignment and split isolation |
| Decoder | Pillow decode followed by RGB conversion | Exact pixels and channel convention presented to CLIP |
| Preprocessing | Checkpoint-native OpenCLIP evaluation transform | Resize, center crop, interpolation, and pretrained pixel normalization |
| Model state | `eval()`, frozen parameters, `torch.inference_mode()` | Removes training updates and stochastic training behavior |
| Feature | Projected global output from `encode_image` | The 512-dimensional CLIP image–text geometry used for similarity |
| Normalization | L2 normalization in float32 | Makes inner product equal cosine similarity and removes magnitude as a signal |
| Cache dtype | Float32 `.npy` | Preserves stable neighbor computation even when inference uses AMP |
| Failure policy | Stop on a missing/unreadable image or nonfinite/zero vector | Prevents silent population changes |
| Ordering | Strict ascending image ID plus an ordered-ID SHA-256 | Prevents cached rows from being attached to the wrong images |
| Provenance | Source, checkpoint, config, package, and artifact hashes | Makes the representation auditable and reproducible |

Inference precision and batch size are chosen by a real GPU benchmark, not assumed. The benchmark
compares AMP-FP16 output with an FP32 reference on a deterministic train-only sample, records
throughput and peak memory, and accepts AMP only if mean cosine drift is at most `0.0001` and mean
top-50 neighbor-set overlap is at least `0.99`. These are project reproducibility guardrails rather
than literature-derived constants. The selected precision and batch size are recorded in the
embedding metadata.

The completed seed-5942 benchmark used 512 train images and selected AMP-FP16, batch 256:
543.97 images/second, 1,695,973,376 bytes peak CUDA allocation, mean cosine drift
`0.000002479`, and mean top-50 overlap `0.9978515625`. All 118,287 embeddings and the exact
top-400 graph have been generated and hash validated. The benchmark JSON and cache sidecars are
local artifacts under `reports/data/` and `data/interim/visual/`.

#### Objective choice

The visual-diversity set will be constructed globally rather than by repairing the random set.
The primary objective is sparse facility-location coverage over unit-normalized CLIP embeddings.

For image embeddings `e[i]` and `e[j]`:

```text
similarity(i, j) = max(0, dot(e[i], e[j]))

neighbors_k(i) = the k most similar training images to image i, including i

coverage_k(S) = sum, over every training image i, of:

    max(similarity(i, j)) for selected j in S intersect neighbors_k(i)
```

Each training image contributes the similarity of its best selected representative. A candidate
therefore has high marginal value when it improves representation for many currently poorly
covered images. Dense visual modes receive enough representatives because many images benefit,
while a single isolated image cannot dominate merely because it is far away. Sparse coverage is an
approximation to the dense objective: `k` controls how many possible representatives each image may
use.

Alternatives considered:

| Objective | What it optimizes | Why it is not primary |
|---|---|---|
| Random | No embedding-space objective | Control condition only |
| Farthest-first / k-center | Worst represented image | Strongly attracts isolated or corrupted outliers |
| Spherical k-means / medoids | Average cluster distortion | Useful fallback, but sensitive to initialization and 15,000-cluster edge cases |
| Determinantal point process | Pairwise repulsion or volume | Much more expensive at this scale and can over-reward boundary points |
| Facility location | Population coverage by selected representatives | Selected: balances representativeness and non-redundancy |

#### Neighbor search and optimizer

The approved neighbor backend is exact GPU inner-product search with FAISS `IndexFlatIP`. Because
vectors are L2-normalized, inner product equals cosine similarity. At this scale, the exact index
fits easily in GPU memory, so an approximate IVF/HNSW/CAGRA index would add a recall parameter and
another source of selection variation without solving a real capacity problem.

The sparse graph will be stored as deterministic forward and reverse CSR adjacency. Selection will
use memoized lazy greedy optimization:

1. Cache the best similarity currently covering every training image.
2. Give every unselected candidate a heap entry containing an upper bound on its marginal gain.
3. Recompute a candidate only when it reaches the top of the heap.
4. Select it if its recomputed gain remains largest; otherwise return it to the heap.
5. Break exact gain ties by the smaller COCO image ID.

Facility location is monotone submodular, so marginal gains can only decrease as the selected set
grows. That makes the heap bounds valid and lets lazy greedy return the same choices as ordinary
greedy for the sparse objective while avoiding most full rescoring passes. If profiling shows that
lazy greedy is still too slow, a seeded stochastic-greedy implementation must receive a different
algorithm version rather than silently replacing it.

#### Hardware-informed implementation

The development machine has an NVIDIA RTX 2080 with 8 GiB VRAM, a Ryzen 9 9900X with 12 cores / 24
threads, 30 GiB RAM, and 264 GiB free NVMe storage. This supports exact GPU neighbors and CPU sparse
optimization:

| Artifact | Approximate size |
|---|---:|
| 118,287 × 512 float32 embeddings | 231 MiB |
| Forward graph at `k=50` | 45 MiB |
| Forward graph at `k=100` | 90 MiB |
| Forward graph at `k=200` | 181 MiB |
| Forward graph at `k=400` | 361 MiB |

Reverse CSR roughly doubles graph storage. A dense float32 similarity matrix would require about
52 GiB and will not be materialized. `pixi.toml` and `pixi.lock` pin Python 3.11, NumPy 1.26.4,
PyTorch 2.5.1, torchvision 0.20.1, CUDA 12.4, OpenCLIP 3.3.0, and GPU FAISS 1.12.0. The smoke test
confirms CUDA compute capability 7.5 and a visible FAISS GPU before real work begins.

#### Train-only calibration still required

The actual values must come from the real CLIP embeddings; they must not be guessed from literature
or tuned on `val2017`.

Neighbor count:

1. Generate selections for `k in {50, 100, 200, 400}`.
2. Require every candidate to contain 15,000 unique `train2017` images and replace at least 750
   images from the paired random baseline (5%).
3. Evaluate every candidate with the same batched exact cosine-coverage calculation, rather than
   comparing each candidate only under its own truncated graph.
4. Choose the smallest `k` whose mean exact coverage is within 0.1% of the best candidate and whose
   p95 distance is no more than `0.005` above the best candidate and whose p99 distance is no more
   than `0.010` above the best candidate.

Near duplicates:

1. Inspect the empirical nearest-neighbor cosine tail at candidate cutoffs `{0.98, 0.99, 0.995}`.
2. Audit image pairs around every cutoff and choose the lowest cutoff with at least 95% precision
   for genuine visual near-duplicates.
3. Build stable representative-centered groups to avoid transitive chaining, retain the medoid as
   the selectable representative, and break medoid ties by image ID.
4. Permit at most one selected image from each accepted near-duplicate group.

The real sweep yields 57 groups / 59 excluded images at `0.98`, 36 / 38 at `0.99`, and 26 / 28
at `0.995`. Every pair was visually audited; `0.99` was the lowest cutoff meeting the 95%
observed-precision gate under the recorded conservative same-capture criterion.

Outliers are desirable in moderation. Define visual density as mean cosine similarity to the ten
nearest non-self neighbors. Images in the bottom 1% of density are “extreme outliers”; they remain
eligible, but at most 2% of the final set (300 images) may come from that group. This preserves rare
visual modes without allowing corruptions or isolated anomalies to saturate the selection. The 1%
definition and 2% cap are project guardrails, not values established by the literature; allowing up
to twice the source rate deliberately permits moderate outlier enrichment.

#### Acceptance scorecard

The visual-diversity manifest is accepted only if all hard requirements pass:

- exactly 15,000 unique canonical `train2017` IDs;
- at least 750 replacements relative to the seed-5942 random baseline (5%);
- no more than one selected representative per accepted near-duplicate group;
- no more than 300 extreme-outlier images (2%); and
- higher batched exact facility coverage than the random baseline, with lower mean and p95
  nearest-selected cosine distance; and
- complete checkpoint, embedding, graph, optimizer, source, and selected-ID provenance.

Among candidates that pass, report mean/p50/p95/p99 nearest-selected cosine distance, exact
facility coverage, sparse uncovered rate, random overlap, near-duplicate rate, outlier share,
category coverage, label density, runtime, peak CPU/GPU memory, and visual audits of dense modes,
sparse modes, duplicate groups, and outliers.

The training archive is installed and checksum verified. The real embedding, duplicate audit,
four-candidate calibration, and canonical-manifest validation have passed. Reproduce the audit
with `scripts/audit_visual_pairs.py` and the selection with `scripts/calibrate_visual.py` in
the locked Pixi environment. The outlier cap is active at 300 selected images.

### Caption-semantic diversity

The approved method uses frozen `sentence-transformers/all-mpnet-base-v2` at revision
`e8c3b32edf5434bc2275fc9bab85f82640a19130`. Independently normalize each of five caption
vectors, average them, then normalize the image-level mean. Reuse exact GPU cosine neighbors,
sparse facility coverage, lazy greedy, and the train-only k sweep / acceptance rule.

Text duplicates reduce marginal gain without excluding images. Semantic outliers and category
allocation are descriptive audits without quotas. See the
[full method and diagnostic contract](../../../docs/caption-semantic-selection.md).

Run `make caption-semantic` and `make caption-diagnostics` in the locked Pixi environment.
The accepted manifest is generated only after the common exact coverage, distance, and 5%
replacement gates pass. Per-category coverage is a proxy for representation demand, not a
causal estimate of how many training examples a retrieval model needs.

The real sweep selected **k=50**. All four candidates passed; the accepted set replaces
**13,116 images (87.44%)** from random. Mean exact caption distance improves from `0.120189`
to `0.097451`; p95 improves from `0.281780` to `0.209511`. No source image is uncovered by
the accepted sparse graph. All 591,435 captions were encoded without truncation.
The [results and diagnostic interpretation](../../../docs/caption-semantic-results.md)
record category reallocations, modest visual cross-view gains, mixed lexical-proxy changes,
and the observed 7.32% extreme semantic-outlier share.

### Quality-aware diversity — TODO

- Decide the quality signals, rejection policy, and diversity–quality trade-off.
- Define score normalization, missing-value behavior, and deterministic optimization.
- Audit which categories and image conditions the quality rule removes.
- Do not generate a manifest yet.
