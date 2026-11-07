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
4. Category-balance objective and treatment of multi-label images.
5. Blur, visual-complexity, and resolution metrics plus quality/diversity weighting.
6. Crop scale/aspect ranges, resize interpolation, padding value, and color-normalization formula.
7. How the frozen 512-dimensional CLIP vector is fused into the randomly initialized VLM.
8. Training, caption, data-loader, and weight-initialization seed policy.

Until those choices are approved, generated embeddings and quality scores would look reproducible
while encoding arbitrary scientific decisions. The current foundation therefore stops at the shared
random baseline and a clearly labeled provisional category-balancing implementation.
