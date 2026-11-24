#!/bin/bash
#SBATCH --job-name=mega-D1P1-s5944
#SBATCH --partition=aa100
#SBATCH --qos=gpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --output=artifacts/logs/%x-%j.out
# ---------------------------------------------------------------------
# Study run 03: random__standard_rgb, model seed 5944
# D1 x P1 replicate: same data and order, only weight initialization differs.
# 40 epochs at 640x640, batch 64; about 5-6 h on an A40, so 12 h leaves margin.
# Writes artifacts/runs/random__standard_rgb__seed5944/{run_config.json,metrics.jsonl,final.pt}.
# Refuses to start if that folder already holds a run; delete it to retry.
#
#   sbatch slurm/alpine/run_03_random__standard_rgb__seed5944.sbatch
# ---------------------------------------------------------------------
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source slurm/alpine/env.sh
run_training \
  --selection data/processed/selections/random_seed5942_n15000.jsonl \
  --preprocessing standard_rgb \
  --model-seed 5944 \
  --output-dir artifacts/runs/random__standard_rgb__seed5944
