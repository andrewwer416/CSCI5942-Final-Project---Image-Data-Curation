#!/bin/bash
#SBATCH --job-name=mega-D2P3
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
# Follow-up run 14: category_balanced__random_resized_crop, model seed 5942
# D2 x P3: category-balanced selection with random resized crop (docs/preprocessing-p3.md).
# Exploratory: chosen after inspecting the round-1 val2017 results.
# 40 epochs at 640x640, batch 64; round-1 runs took 2.5-3 h on an Alpine A100, so 12 h leaves margin.
# Writes artifacts/runs/category_balanced__random_resized_crop__seed5942/{run_config.json,metrics.jsonl,final.pt}.
# Refuses to start if that folder already holds a run; delete it to retry.
#
#   sbatch slurm/alpine/run_14_category_balanced__random_resized_crop__seed5942.sbatch
# ---------------------------------------------------------------------
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source slurm/alpine/env.sh
run_training \
  --selection data/processed/selections/category_balanced_t0p025_cap4_seed5942_n15000.jsonl \
  --preprocessing random_resized_crop \
  --model-seed 5942 \
  --output-dir artifacts/runs/category_balanced__random_resized_crop__seed5942
