#!/bin/bash
#SBATCH --job-name=mega-D1P5
#SBATCH --account=ucb857_peak1
#SBATCH --partition=aa100
#SBATCH --qos=gpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100-40gb:1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --output=artifacts/logs/%x-%j.out
# ---------------------------------------------------------------------
# Study run 11: random__rgb_plus_clip_features, model seed 5942
# D1 x P5: P1 images plus the frozen OpenCLIP ViT-B-16 vector, fused into the image encoder.
# The last epoch also evaluates with zeroed and with shuffled CLIP vectors.
# Needs the CLIP caches in data/interim/visual/ (clip-features.zip from the
# handoff-clip-features-2026-11-25 release); the run re-verifies their hashes.
# See docs/preprocessing-p5.md.
# 40 epochs at 640x640, batch 64; runs 01-08 took 2.5-3 h on an Alpine A100, so 12 h leaves margin.
# Writes artifacts/runs/random__rgb_plus_clip_features__seed5942/{run_config.json,metrics.jsonl,final.pt}.
# Refuses to start if that folder already holds a run; delete it to retry.
#
#   sbatch slurm/alpine/run_11_random__rgb_plus_clip_features__seed5942.sbatch
# ---------------------------------------------------------------------
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source slurm/alpine/env.sh
# Fail before staging (and before spending GPU time) if the caches are missing.
for split in train2017 val2017; do
  meta=data/interim/visual/openclip_vit_b16_datacomp_xl_s13b_b90k_$split.meta.json
  [ -f "$meta" ] || { echo "missing $meta: unpack clip-features.zip first" >&2; exit 1; }
done
run_training \
  --selection data/processed/selections/random_seed5942_n15000.jsonl \
  --preprocessing rgb_plus_clip_features \
  --model-seed 5942 \
  --output-dir artifacts/runs/random__rgb_plus_clip_features__seed5942
