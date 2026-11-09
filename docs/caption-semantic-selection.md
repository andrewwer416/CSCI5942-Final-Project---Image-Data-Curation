# Caption-semantic selection

## Approved representation

Use permanently frozen `sentence-transformers/all-mpnet-base-v2`, immutable revision
`e8c3b32edf5434bc2275fc9bab85f82640a19130`. The model produces 768-dimensional vectors.
Encode each of the five canonical captions independently with the checkpoint's
tokenizer and pooling, normalize each vector, average all five, then normalize the
image-level mean. Caption text is passed unchanged. Caption order is ascending caption
ID; image order is ascending COCO image ID. Use FP32 evaluation under inference mode.

The pretrained tokenizer defaults to a 384-token limit; record actual truncation counts.
Missing, empty, nonfinite, and zero-norm data fail generation. Artifacts contain model
file hashes, source/ordered-ID hashes, package versions, normalized caption vectors,
normalized image-level vectors, and within-image dispersion. Data live under
`data/interim/caption_semantic/` and remain ignored by Git.

The official [model card](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)
describes its sentence-similarity use, pooling, dimensionality, and truncation behavior.
These support the representation choice; our aggregation and acceptance thresholds
are project hypotheses, not literature-established optima.

## Selection and acceptance

Reuse exact GPU FAISS `IndexFlatIP`, normalized cosine similarity, sparse facility
coverage, and deterministic lazy greedy. Ties use the smallest image ID. Construct
each set globally. Sweep k in `{50,100,200,400}`, using leading columns of one exact
top-400 graph. Evaluate all candidates with common exact nearest-selected coverage.

Each candidate must contain exactly 15,000 distinct canonical train images, replace
at least 750 seed-5942 random images, improve mean exact coverage, and reduce both
mean and p95 cosine distance versus random. Select the smallest passing k within
0.1% relative mean coverage, 0.005 p95 distance, and 0.010 p99 distance of the best
passing candidate. Generate no accepted manifest if none passes.

Text similarity reduces marginal facility gain; it does not exclude candidates.
Different images can have identical captions. Semantic outliers remain eligible
and are audited through the bottom 1% of top-10-neighbor density, without quotas.
COCO categories impose no quotas either: the objective's allocation is an outcome.

## Explanatory diagnostics

- Per category: source/random/selected counts, selection rate, and source-image
  mean/p95 semantic coverage distance. Categories overlap in multi-label images.
- Language: noun/verb/adjective types, numeric phrases, preposition windows, and
  a fixed scene lexicon; compare selected and random type coverage and covered
  source-incidence mass. Count each feature at most once per image. Preserve raw
  surface forms; spelling and inflection can inflate vocabulary counts.
- Within-image agreement: mean pairwise cosine distance among the ten caption
  pairs. Audit source, random, selected, and dispersion quantiles.
- Redundancy: repeated source caption occurrences and nearest-other selected
  image-level semantic similarity. Exact wording is a descriptive signal.
- Outliers: source/random/selected density distribution, extreme selection rate,
  and the captions of the 50 lowest-density images for inspection.
- Cross-view: exact visual CLIP coverage for caption-selected versus random sets.

NLTK English perceptron POS tags, adjective labels, preposition windows, and the
scene lexicon are noisy reproducible proxies, not verified semantic annotations.
The report records tagger hashes. They explain selection and do not tune k.

The category allocation/coverage table estimates representational demand under
this encoder and objective. It cannot establish a causal sample requirement or
model learning difficulty. Retrieval training and held-out evaluation remain
necessary to assess downstream value. Only train2017 informs these diagnostics.

## Reproduction

```bash
pixi install --locked
PYTHONPATH=src pixi run python scripts/calibrate_caption.py
pixi run python -c "import nltk; nltk.download('averaged_perceptron_tagger_eng', download_dir='data/raw/nltk', raise_on_error=True)"
PYTHONPATH=src pixi run python scripts/diagnose_caption.py
PYTHONPATH=src pixi run python -m vlm_data validate
```

The calibration script caches embeddings and neighbors, writes per-k reports and
selected IDs, then the accepted manifest and sidecar. The diagnostics script
writes `reports/data/caption-diagnostics.json`. Completed caches are hash checked
on reuse. Artifact arrays and metadata use atomic writes; incomplete artifacts
without the metadata commit marker are regenerated.
