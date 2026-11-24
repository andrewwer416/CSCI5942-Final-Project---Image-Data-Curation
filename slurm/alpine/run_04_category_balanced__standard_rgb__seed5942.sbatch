#!/bin/bash
#SBATCH --job-name=mega-D2P1
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
# Study run 04: category_balanced__standard_rgb, model seed 5942
# D2 x P1: category-balanced selection.
# 40 epochs at 640x640, batch 64; about 5-6 h on an A40, so 12 h leaves margin.
# Writes artifacts/runs/category_balanced__standard_rgb__seed5942/{run_config.json,metrics.jsonl,final.pt}.
# Refuses to start if that folder already holds a run; delete it to retry.
#
#   sbatch slurm/alpine/run_04_category_balanced__standard_rgb__seed5942.sbatch
# ---------------------------------------------------------------------
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source slurm/alpine/env.sh
run_training \
  --selection data/processed/selections/category_balanced_t0p025_cap4_seed5942_n15000.jsonl \
  --preprocessing standard_rgb \
  --model-seed 5942 \
  --output-dir artifacts/runs/category_balanced__standard_rgb__seed5942
