#!/bin/bash
#SBATCH --job-name=mega-D5P1
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
# Study run 07: quality_aware_diversity__standard_rgb, model seed 5942
# D5 x P1: quality-aware (TOPIQ) diversity selection.
# 40 epochs at 640x640, batch 64; about 5-6 h on an A40, so 12 h leaves margin.
# Writes artifacts/runs/quality_aware_diversity__standard_rgb__seed5942/{run_config.json,metrics.jsonl,final.pt}.
# Refuses to start if that folder already holds a run; delete it to retry.
#
#   sbatch slurm/alpine/run_07_quality_aware_diversity__standard_rgb__seed5942.sbatch
# ---------------------------------------------------------------------
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source slurm/alpine/env.sh
run_training \
  --selection data/processed/selections/quality_aware_k50_dup0p99_topiq_lam0p1099_seed5942_n15000.jsonl \
  --preprocessing standard_rgb \
  --model-seed 5942 \
  --output-dir artifacts/runs/quality_aware_diversity__standard_rgb__seed5942
