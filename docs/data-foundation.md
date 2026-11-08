# Data Foundation

## Scope

This phase prepares auditable inputs for the study. It does not define the model, implement image
transforms, generate pretrained embeddings, or download large archives automatically during tests.
Official COCO archive links use HTTP; acquisition therefore verifies pinned byte counts and SHA-256
digests before safe extraction.

## Reproducible flow

```text
official COCO archives
        ↓ immutable extraction
captions + instances + image metadata
        ↓ deterministic join
canonical train2017 / val2017 manifests
        ↓ versioned selection method + seed
exact 15,000-image training manifests
        ↓ run manifest + lazy image transform
training and retrieval evaluation
```

The canonical layer separates data discovery from experiment selection. A run should reference a
selection-manifest hash rather than selecting images inside the training process.

## Canonical manifest contract

Each JSONL row represents one image and has these fields:

| Field | Type | Meaning |
|---|---|---|
| `image_id` | integer | Official COCO image identifier |
| `split` | string | `train2017` or `val2017` |
| `file_name` | string | Official image filename |
| `relative_path` | string | Path relative to `data/raw/coco/` |
| `width`, `height` | integer | Original pixel dimensions |
| `captions` | array | Five `{caption_id, text}` records |
| `category_ids` | integer array | Sorted distinct COCO object categories present |
| `object_count` | integer | Number of instance annotations for the image |

Rows are sorted by image ID and serialized with stable JSON key ordering. The metadata sidecar
records input checksums, output checksum, row counts, caption distribution, and the COCO category
dictionary. A small number of official records can contain more than five captions; to preserve the
study's exact-five contract, preparation retains the five lowest caption IDs and records both the raw
count distribution and this policy in metadata. Any image with fewer than five captions fails early.

## Selection manifests

Selection outputs use the same row contract, so training can consume either canonical or selected
manifests with one reader. Their metadata additionally records:

- selection method and algorithm version;
- seed and requested budget;
- canonical-manifest SHA-256;
- selected-manifest SHA-256; and
- per-category image coverage.

The random baseline ranks each image by a stable keyed hash. It does not depend on source row order
or Python's evolving pseudo-random sampling implementation.

The category-balanced manifest is paired with that exact random baseline. At least 5% of its 15,000
selected IDs must be replacements, so at least 750 category-balanced IDs must be absent from the
random set. This is a hard generation requirement: the command fails before writing a manifest if
the target-driven swaps do not clear the floor, and it never pads the result with unrelated swaps.
The current calibrated selection replaces 948 images (6.32%). The complete operational definition
and future-method placeholders live in `data/processed/selections/README.md`.

## Caption sampling

Training chooses one of an image's five captions from a hash of `(seed, epoch, image_id)`. The
choice is reproducible, independent of data-loader worker order, and does not mutate global random
state. Evaluation retains all five captions.

## Validation and leakage boundary

`vlm-data validate` checks schema-level integrity, exact caption counts, manifest hashes, expected
split sizes, selection size, uniqueness, membership in the canonical training source, and train/val
ID disjointness. `--check-files` adds an on-disk existence check after image archives are installed.

No artifact derived from `val2017` may influence training subset selection, normalization statistics,
or early stopping. The proposal uses validation performance to choose optional follow-up pairings;
that makes those follow-ups adaptive to `val2017`. The final report should call this out, or reserve
an additional untouched evaluation split before running combinations.

## Decisions required before advanced data generation

The proposal intentionally leaves several method definitions open. Pin these in configuration and
record model revisions before creating derived artifacts:

1. Frozen CLIP checkpoint, preprocessing, embedding precision, and revision.
2. Visual-diversity objective and scalable selection algorithm.
3. Caption embedding model and how five captions become one image-level representation.
4. Blur, visual-complexity, and resolution metrics plus quality/diversity weighting.
5. Crop scale/aspect ranges, resize interpolation, padding value, and color-normalization formula.
6. How the frozen 512-dimensional CLIP vector is fused into the randomly initialized VLM.
7. Training, caption, data-loader, and weight-initialization seed policy.

Until those choices are approved, generated embeddings and quality scores would look reproducible
while encoding arbitrary scientific decisions. The current foundation therefore implements only the
shared random baseline and the separately calibrated category-balancing method.

See `background-research.md` for the literature review, concrete starting values, and train-only
calibration checks proposed for these decisions.

The category-balance decision is now resolved separately: `vlm-data analyze-category-balance`
evaluates its parameter grid from canonical `train2017`, verifies the real multi-label selection,
and writes SHA-pinned reports under `reports/data/`. The checked-in values are a 2.5% prevalence
threshold and an empirically inactive 4× safety cap, selected under a 14%-of-budget target-mass
guardrail and verified against a 5% minimum-change requirement.
