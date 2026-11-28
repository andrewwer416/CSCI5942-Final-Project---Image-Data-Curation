#!/bin/bash
#SBATCH --job-name=mega-D3P1-s5943
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
# Follow-up run 18: visual_diversity__standard_rgb, model seed 5943
# D3 x P1 replicate: same data and order as the seed-5942 run, only weight initialization differs.
# Exploratory: chosen after inspecting the round-1 val2017 results.
# 40 epochs at 640x640, batch 64; round-1 runs took 2.5-3 h on an Alpine A100, so 12 h leaves margin.
# Writes artifacts/runs/visual_diversity__standard_rgb__seed5943/{run_config.json,metrics.jsonl,final.pt}.
# Refuses to start if that folder already holds a run; delete it to retry.
#
#   sbatch slurm/alpine/run_18_visual_diversity__standard_rgb__seed5943.sbatch
# ---------------------------------------------------------------------
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source slurm/alpine/env.sh
run_training \
  --selection data/processed/selections/visual_diversity_k50_dup0p99_seed5942_n15000.jsonl \
  --preprocessing standard_rgb \
  --model-seed 5943 \
  --output-dir artifacts/runs/visual_diversity__standard_rgb__seed5943
