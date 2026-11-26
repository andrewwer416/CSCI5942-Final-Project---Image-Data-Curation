#!/bin/bash
#SBATCH --job-name=mega-D1P4
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
# Study run 10: random__color_contrast_normalization, model seed 5942
# D1 x P4: per-image Shades-of-Gray white balance (p = 6) and a 1-99% luma stretch, both capped
# at 2x, then P1; the same transform at evaluation. More CPU work per image than P1, so a step
# may be slower. See docs/preprocessing-p4.md.
# 40 epochs at 640x640, batch 64; runs 01-08 took 2.5-3 h on an Alpine A100, so 12 h leaves margin.
# Writes artifacts/runs/random__color_contrast_normalization__seed5942/{run_config.json,metrics.jsonl,final.pt}.
# Refuses to start if that folder already holds a run; delete it to retry.
#
#   sbatch slurm/alpine/run_10_random__color_contrast_normalization__seed5942.sbatch
# ---------------------------------------------------------------------
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source slurm/alpine/env.sh
run_training \
  --selection data/processed/selections/random_seed5942_n15000.jsonl \
  --preprocessing color_contrast_normalization \
  --model-seed 5942 \
  --output-dir artifacts/runs/random__color_contrast_normalization__seed5942
