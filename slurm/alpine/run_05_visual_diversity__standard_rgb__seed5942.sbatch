#!/bin/bash
#SBATCH --job-name=mega-D3P1
#SBATCH --partition=aa100
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --output=artifacts/logs/%x-%j.out
# ---------------------------------------------------------------------
# Study run 05: visual_diversity__standard_rgb, model seed 5942
# D3 x P1: frozen-CLIP visual diversity selection.
# 40 epochs at 640x640, batch 64; about 5-6 h on an A40, so 12 h leaves margin.
# Writes artifacts/runs/visual_diversity__standard_rgb__seed5942/{run_config.json,metrics.jsonl,final.pt}.
# Refuses to start if that folder already holds a run; delete it to retry.
#
#   sbatch slurm/alpine/run_05_visual_diversity__standard_rgb__seed5942.sbatch
# ---------------------------------------------------------------------
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source slurm/alpine/env.sh
run_training \
  --selection data/processed/selections/visual_diversity_k50_dup0p99_seed5942_n15000.jsonl \
  --preprocessing standard_rgb \
  --model-seed 5942 \
  --output-dir artifacts/runs/visual_diversity__standard_rgb__seed5942
