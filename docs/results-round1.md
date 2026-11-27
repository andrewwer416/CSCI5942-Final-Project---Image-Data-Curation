# Round 1 results (Alpine, 2026-11-26)

Eight of the eleven first-round runs: D1–D5 × P1 (D1 with model seeds 5942–5944) and D1 × P2.
D1 × P3, P4 and P5 are queued. Every run: 40 epochs, 640 × 640, batch 64, AMP, on an A100 PCIe
40 GB (0.78–0.81 s/step; P2 1.04 s/step), evaluated on all of val2017 (5,000 images, 25,000
captions) after every epoch.

The per-epoch metrics and run configs are in `results/round1/<run>/`. The tables below are generated
by `python3 scripts/analyze_runs.py --runs results/round1`. Checkpoints (`final.pt`, 499 MB each)
are not in Git; they are on Alpine in `/scratch/alpine/bihs8849/mega/artifacts/runs/`, backed up to
`/projects/bihs8849/mega-results/`.

## Findings so far

- **D3 and D4 score below random selection.** Both fall under the lowest D1 × P1 seed, at the final
  epoch and on the last-5-epoch mean. D4 is lowest from about epoch 5 onward.
- **D2, D5 and P2 are indistinguishable from the baseline.** Their t2i R@10 is inside the D1 × P1
  seed range. D2's RSUM is slightly above the best seed, which one seed cannot confirm.
- **D4 also fits its own training set worse.** Its final training loss is 0.94; every other run
  ends between 0.51 and 0.64.
- **Every run overfits after about epoch 20.** Training loss keeps falling to about 0.5 while val
  R@10 flattens near 0.07. This may compress the differences between selections.

D2–D5 and P2 have one seed each, so these are preliminary. Single epochs are noisy as well: D3 moved
from 0.0674 to 0.0606 between epochs 39 and 40.

![Text-to-image Recall@10 per epoch](figures/round1_t2i_recall10.svg)

![Training loss per epoch](figures/round1_train_loss.svg)

Final epoch is the headline; the last-5-epoch mean is a robustness check. The baseline
spread comes from the D1xP1 seeds (weight initialization only). "Inside" means within
the seeds' min-max range. Three seeds give a descriptive range, not a significance test.

## Final epoch

| Arm | Seed | t2i@1 | t2i@5 | t2i@10 | i2t@1 | i2t@5 | i2t@10 | rsum | t2i@10_color | train_loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| D1xP1 | 5942 | 0.0099 | 0.0393 | 0.0702 | 0.0062 | 0.0330 | 0.0590 | 0.2176 | 0.0949 | 0.5357 |
| D1xP1 | 5943 | 0.0101 | 0.0434 | 0.0768 | 0.0098 | 0.0398 | 0.0698 | 0.2497 | 0.1140 | 0.5373 |
| D1xP1 | 5944 | 0.0112 | 0.0442 | 0.0758 | 0.0108 | 0.0404 | 0.0708 | 0.2533 | 0.1003 | 0.5098 |
| D2xP1 | 5942 | 0.0118 | 0.0450 | 0.0766 | 0.0086 | 0.0448 | 0.0718 | 0.2587 | 0.1006 | 0.5095 |
| D3xP1 | 5942 | 0.0094 | 0.0359 | 0.0606 | 0.0074 | 0.0324 | 0.0540 | 0.1998 | 0.0848 | 0.5472 |
| D4xP1 | 5942 | 0.0069 | 0.0313 | 0.0572 | 0.0064 | 0.0248 | 0.0466 | 0.1732 | 0.0648 | 0.9375 |
| D5xP1 | 5942 | 0.0095 | 0.0408 | 0.0718 | 0.0110 | 0.0394 | 0.0650 | 0.2376 | 0.0999 | 0.5743 |
| D1xP2 | 5942 | 0.0102 | 0.0402 | 0.0719 | 0.0072 | 0.0390 | 0.0644 | 0.2330 | 0.0928 | 0.6401 |

## Mean of the last 5 epochs

| Arm | Seed | t2i@1 | t2i@5 | t2i@10 | i2t@1 | i2t@5 | i2t@10 | rsum | t2i@10_color | train_loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| D1xP1 | 5942 | 0.0097 | 0.0400 | 0.0699 | 0.0072 | 0.0342 | 0.0600 | 0.2211 | 0.0955 | 0.5857 |
| D1xP1 | 5943 | 0.0103 | 0.0414 | 0.0743 | 0.0094 | 0.0387 | 0.0677 | 0.2418 | 0.1085 | 0.5722 |
| D1xP1 | 5944 | 0.0106 | 0.0430 | 0.0750 | 0.0107 | 0.0396 | 0.0688 | 0.2478 | 0.0975 | 0.5491 |
| D2xP1 | 5942 | 0.0113 | 0.0437 | 0.0754 | 0.0102 | 0.0416 | 0.0704 | 0.2526 | 0.0994 | 0.5647 |
| D3xP1 | 5942 | 0.0088 | 0.0364 | 0.0644 | 0.0080 | 0.0334 | 0.0568 | 0.2078 | 0.0933 | 0.6273 |
| D4xP1 | 5942 | 0.0074 | 0.0315 | 0.0565 | 0.0066 | 0.0257 | 0.0463 | 0.1740 | 0.0661 | 1.0362 |
| D5xP1 | 5942 | 0.0095 | 0.0402 | 0.0714 | 0.0093 | 0.0362 | 0.0633 | 0.2298 | 0.1008 | 0.6391 |
| D1xP2 | 5942 | 0.0097 | 0.0395 | 0.0705 | 0.0086 | 0.0376 | 0.0650 | 0.2309 | 0.0952 | 0.6932 |

## Compared with D1xP1 (final epoch)

D1xP1 t2i_recall@10: mean 0.0743, sd 0.0036, range 0.0702-0.0768 over 3 seeds. RSUM: mean 0.2402, range 0.2176-0.2533.

| Arm | Seed | t2i@10 | Δ vs D1xP1 | Inside | RSUM | Δ RSUM | Inside |
|---|---:|---:|---:|:-:|---:|---:|:-:|
| D2xP1 | 5942 | 0.0766 | +0.0024 | yes | 0.2587 | +0.0185 | **no** |
| D1xP2 | 5942 | 0.0719 | -0.0024 | yes | 0.2330 | -0.0072 | yes |
| D5xP1 | 5942 | 0.0718 | -0.0024 | yes | 0.2376 | -0.0026 | yes |
| D3xP1 | 5942 | 0.0606 | -0.0136 | **no** | 0.1998 | -0.0404 | **no** |
| D4xP1 | 5942 | 0.0572 | -0.0171 | **no** | 0.1732 | -0.0670 | **no** |

## Compared with D1xP1 (last-5 mean)

D1xP1 t2i_recall@10: mean 0.0731, sd 0.0028, range 0.0699-0.0750 over 3 seeds. RSUM: mean 0.2369, range 0.2211-0.2478.

| Arm | Seed | t2i@10 | Δ vs D1xP1 | Inside | RSUM | Δ RSUM | Inside |
|---|---:|---:|---:|:-:|---:|---:|:-:|
| D2xP1 | 5942 | 0.0754 | +0.0024 | **no** | 0.2526 | +0.0157 | **no** |
| D5xP1 | 5942 | 0.0714 | -0.0016 | yes | 0.2298 | -0.0071 | yes |
| D1xP2 | 5942 | 0.0705 | -0.0026 | yes | 0.2309 | -0.0060 | yes |
| D3xP1 | 5942 | 0.0644 | -0.0087 | **no** | 0.2078 | -0.0291 | **no** |
| D4xP1 | 5942 | 0.0565 | -0.0166 | **no** | 0.1740 | -0.0629 | **no** |
