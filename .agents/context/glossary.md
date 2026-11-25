# Glossary

| Term | Meaning in this project |
|---|---|
| Canonical manifest | One deterministic row per COCO image, before experiment selection |
| Selection manifest | Exact 15,000-image training set plus algorithm provenance |
| Shared baseline | Random selection with standard RGB preprocessing, trained once |
| Initial run | One of nine selection-only or preprocessing-only comparisons |
| Follow-up run | Optional pairing chosen after initial retrieval results |
| Stable hash rank | Order-independent deterministic substitute for in-process random sampling |
| Baseline repair | Category balancing that swaps the minimum needed images into the paired random set |
| Added target mass | Extra category-image incidences requested by a balance scenario; not distinct images |
| Metadata sidecar | `.meta.json` file containing counts, hashes, seed, and algorithm version |
| Leakage | Any use of `val2017` information to construct or fit a training input |
| D1–D5 | The five selections: random, category balanced, visual diversity, caption-semantic diversity, quality-aware diversity |
| P1–P5 | The five preprocessing arms: standard RGB, aspect-ratio-preserving, random resized crop, color/contrast normalization, RGB + CLIP features |
| Facility location | Selection objective rewarding every train image having a close selected neighbor |
| TOPIQ-NR | Learned no-reference image-quality model (KonIQ-10k human ratings) used by D5 |
| Letterbox | Resize keeping the aspect ratio, then pad to a square (P2) |
| Padding mask | Patches made only of padding, hidden from image-encoder attention in P2 |
| Image-relative crop | P3 crop whose aspect ratio is drawn relative to the image's own ratio |
| Shades-of-Gray | Per-image white balance using the Minkowski p-norm of each channel (P4, p = 6) |
| Late fusion | P5's addition of the projected CLIP vector to the image encoder's pooled output |
| Color captions | Captions naming a basic color word; reported as a separate recall subset |
