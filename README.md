# COCO Data Selection for Image–Text Retrieval

Foundation for the CSCI 5942 study, **Can Data Selection and Preprocessing Improve
Image–Text Retrieval?** The immediate goal is a reproducible data layer: immutable COCO
2017 inputs, deterministic manifests, an exact 15,000-image training subset, and leakage
checks before model code is added.

The assignment proposal remains the source for the study design. Machine-readable constants
live in `configs/data.toml`, and the nine initial runs are enumerated in
`configs/experiment_matrix.toml`.

## Quick start

The data tools use only the Python 3.11 standard library.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e .

# Smallest useful download: captions and instance annotations.
.venv/bin/vlm-data download annotations

# Build canonical train2017 and val2017 JSONL manifests.
.venv/bin/vlm-data prepare

# Materialize the shared, deterministic 15,000-image baseline.
.venv/bin/vlm-data select --method random

# Validate counts, captions, hashes, split isolation, and selections.
.venv/bin/vlm-data validate
```

Download images only when there is enough disk space. The train archive is large.

```bash
.venv/bin/vlm-data download train val --confirm-large-download
.venv/bin/vlm-data validate --check-files
```

Without an editable install, every command is also available as
`PYTHONPATH=src python3 -m vlm_data ...`. Common commands are wrapped by `make`.

## What is implemented

| Capability | Status | Output |
|---|---|---|
| Official COCO archive acquisition | Ready | `data/raw/coco/` |
| Canonical train/validation manifests | Ready | `data/interim/manifests/` |
| Stable random 15k baseline | Ready | `data/processed/selections/` |
| Category-balanced selection | Provisional algorithm | `data/processed/selections/` |
| Frozen CLIP visual diversity | Design decision pending | — |
| Caption-semantic diversity | Design decision pending | — |
| Quality-aware diversity | Design decision pending | — |
| Five image preprocessing modes | Model phase | — |

“Provisional” is deliberate: the proposal names category balancing but does not define its
objective. The included least-covered-category algorithm is versioned and testable, but it
should not become the experimental method until the team approves it.

## Data layout

```text
data/
├── raw/coco/                    # Untouched downloads and extracted COCO files
│   ├── annotations/
│   ├── train2017/
│   └── val2017/
├── interim/manifests/           # Canonical image/caption/category JSONL
└── processed/selections/        # Exact image sets used by training runs
reports/data/                    # Machine-readable validation reports
```

Generated data, model artifacts, and reports are ignored by Git. Each manifest has a sibling
`.meta.json` file containing source and output SHA-256 hashes, counts, method version, and seed.
COCO's official download page currently serves archive links over HTTP, so every configured archive
is pinned to a known byte count and SHA-256 from the
[TensorFlow Datasets checksum registry](https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/url_checksums/coco_captions.txt)
and is verified before extraction.

## Invariants

- Every training selection contains exactly 15,000 distinct `train2017` images.
- All preprocessing comparisons reuse the same random selection manifest.
- Validation uses every `val2017` image and all five reference captions.
- Training samples one caption per image per epoch with a stateless deterministic function.
- Validation images, captions, and fitted statistics never construct training inputs. Per the
  proposal, aggregate retrieval scores may choose optional follow-up pairings; this adaptive use
  must be disclosed in the final analysis.
- Raw files are never rewritten by preparation code.

See `docs/data-foundation.md` for the schema, flow, open design decisions, and leakage policy.

## Tests

```bash
make test
```

The tests use synthetic COCO annotations; they do not download data or require a GPU.

## Dataset terms

COCO annotations are CC BY 4.0. Image copyrights remain with their owners and image use is
subject to Flickr terms. Review the [official COCO terms](https://cocodataset.org/dataset/termsofuse.htm)
before redistributing any data. Do not commit raw images to this repository.
