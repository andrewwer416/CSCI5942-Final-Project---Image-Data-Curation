#!/bin/bash
#SBATCH --job-name=mega-D1P3-s5944
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
# Follow-up run 13: random__random_resized_crop, model seed 5944
# D1 x P3 replicate: same data and order as the seed-5942 run, only weight initialization differs.
# Exploratory: chosen after inspecting the round-1 val2017 results.
# 40 epochs at 640x640, batch 64; round-1 runs took 2.5-3 h on an Alpine A100, so 12 h leaves margin.
# Writes artifacts/runs/random__random_resized_crop__seed5944/{run_config.json,metrics.jsonl,final.pt}.
# Refuses to start if that folder already holds a run; delete it to retry.
#
#   sbatch slurm/alpine/run_13_random__random_resized_crop__seed5944.sbatch
# ---------------------------------------------------------------------
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source slurm/alpine/env.sh
run_training \
  --selection data/processed/selections/random_seed5942_n15000.jsonl \
  --preprocessing random_resized_crop \
  --model-seed 5944 \
  --output-dir artifacts/runs/random__random_resized_crop__seed5944
