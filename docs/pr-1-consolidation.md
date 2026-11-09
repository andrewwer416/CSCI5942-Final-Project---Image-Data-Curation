# PR #1 consolidation and proposed response

PR: https://github.com/andrewwer416/CSCI5942-Final-Project---Image-Data-Curation/pull/1

The local integration branch combines Bill's training work with Andrew's completed
caption-semantic method. Makefile and STATUS conflicts preserve both contributions.
The responses and approvals below accompany the consolidated implementation;
the linked PR records its publication and merge state.

## 1. Training defaults

Andrew approved the PR defaults for the first standardized baseline: symmetric
contrastive loss with learned temperature; from-scratch ViT-B/16 and 12-layer text
Transformer; train-only word vocabulary shared across selections; EOS pooling;
AdamW at 1e-4 with weight decay 0.01; fixed [-1,1] P1 pixel scaling; GPU AMP.
The approval also includes the bidirectional text tower, constant learning rate
without warmup/decay, zero dropout, and the remaining model/tokenizer defaults.
These are approved project settings, not literature-derived optimal values.
The CLI batch-size default of 32 is provisional; measure the study batch size on Alpine.

The proposal fixes 15,000 images, 40 epochs, 640x640, and the text dimensions used
by its compute estimate. Measure a viable batch size on Alpine and hold the actual
contrastive batch size fixed across arms. Gradient accumulation alone does not
provide the same cross-example negatives as a larger contrastive batch.

The text tower here is bidirectional with padding masks and EOS pooling; it is
not an exact architectural reproduction of CLIP's text tower. The proposal does
not require causal masking. Retain this approved architecture consistently
and describing it accurately. A from-scratch word vocabulary is also distinct
from CLIP's pretrained tokenizer.

Integration adds `--model-seed` (default 5942), preserving paired caption/order
seeds; run/source/config hashes; runtime version fields; a study-shape indicator;
and refusal to overwrite a recorded run directory. Selection files are validated
against canonical train records and their hashes before taking a smoke subset.
Existing smoke flags remain
available. These changes do not launch training or change the experiment budget.

## 2. Checkpoints

- D3: approved and implemented frozen OpenCLIP `ViT-B-16`,
  `datacomp_xl_s13b_b90k`, revision `767e6d14be23c46e4f80af31e6b4e73f323e72f5`.
- D4: approved and implemented frozen `sentence-transformers/all-mpnet-base-v2`,
  revision `e8c3b32edf5434bc2275fc9bab85f82640a19130`. Normalize each caption,
  average five vectors, normalize the mean; calibrated k=50. It replaces 87.44%
  of baseline images and passes the recorded semantic-coverage gates.
- P5: the same frozen OpenCLIP checkpoint can provide the required 512-D auxiliary
  vector, but its fusion architecture is still a proposal rather than an approved
  completed arm. D3's train feature cache does not include the val feature cache
  P5 evaluation will also need. CLIP remains frozen in every role.

## 3. Extra seeds

Andrew approved two additional baseline model initializations (5943 and 5944), retaining
the same selection, caption schedule, and data order to isolate initialization
variation. This is an approved robustness addendum:
the proposal says one baseline run in the nine-run primary matrix.

Three baseline runs provide a descriptive spread, not precise uncertainty for
every method. If making a strong method-improvement claim, repeat that method
under the same extra model seeds and compare paired deltas when budget permits.
Do not add replicates implicitly or count them as selection/preprocessing pairings.

## 4. Confirmed decisions and evidence

Confirmed directly with Andrew: category balancing and its real-data minimum
5% replacement gate; frozen CLIP checkpoint, facility objective, exact GPU search,
empirical k sweep and duplicate audit; frozen MPNet and five-caption aggregation;
semantic acceptance plus descriptive category/language/cross-view audits without
category quotas. Quality-aware selection, detailed P2-P4 settings, P5 fusion,
remain unapproved. The training defaults and two extra baseline model seeds are approved.

Primary sources were consulted for the relevant representation and selection
choices, and source links appear in the method documents. Do not claim that every
sentence or every suggested numeric value in background-research.md has received
an independent exhaustive citation audit. The document mixes literature-backed
principles with explicit project hypotheses. In particular category thresholds,
caps, k tolerances, duplicate precision, and outlier rules are project choices or
train-calibrated measurements, not universal constants supported by papers.
P2-P5 and quality proposals still need method-specific review before implementation.

## 5. Proposed division of work

Andrew deferred the work-split question; no new assignment or approval is needed now.
The proposal below remains a suggestion, not a confirmed allocation.

Pending Andrew's confirmation: Bill owns P2-P4 implementation and Alpine job/
throughput/resume scripts; Andrew owns D5 quality-aware selection, P5 design/fusion,
and selection artifact provenance. Both review the training defaults and baseline
results. Bill's P2-P4 implementation should follow jointly frozen crop, padding,
normalization, evaluation-transform, and seed contracts before full study runs.

## Artifact request

The visual manifest and sidecar are available locally, as are D4's manifest and
sidecar. Bundle the four available selections with SHA-256 checksums using
`PYTHONPATH=src pixi run python scripts/package_selections.py`. The generated ZIP
is ignored and can be attached/transferred to the collaborator or Alpine using
the team's chosen channel. It contains manifests and metadata, not COCO images
or model weights. No transfer destination was provided and nothing was uploaded.

## Verification

The combined suite passes 84 tests in the locked environment, including the new
run-overwrite regression. Static checks and whitespace validation pass.
Bill's reported 4 GB GPU smoke results remain credited to Bill; they have not
been independently reproduced here as a full-size training run.
