#!/bin/bash
#SBATCH --job-name=mega-probe
#SBATCH --partition=aa100
#SBATCH --qos=gpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --output=artifacts/logs/%x-%j.out
# ---------------------------------------------------------------------
# Check, before the long runs, that the study batch size (64) fits on this
# GPU for both P1 and P2 (P2 adds an attention mask), and measure speed.
# Each arm runs 20 steps and evaluates on 256 validation images.
#
#   sbatch slurm/alpine/01_probe.sbatch
#   sbatch --partition=al40 slurm/alpine/01_probe.sbatch   # L40 (48 GB) instead
# ---------------------------------------------------------------------
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source slurm/alpine/env.sh
print_job_environment

for arm in standard_rgb aspect_ratio_preserving_rgb; do
  echo "=== ${arm} batch ${STUDY_BATCH_SIZE}"
  srun "$PYTHON" -m vlm_train.train \
    --selection data/processed/selections/random_seed5942_n15000.jsonl \
    --preprocessing "$arm" \
    --limit $((STUDY_BATCH_SIZE * 20)) --val-limit 256 --epochs 1 \
    --batch-size "$STUDY_BATCH_SIZE" --num-workers "$STUDY_NUM_WORKERS" \
    --output-dir "artifacts/probe/${arm}-bs${STUDY_BATCH_SIZE}-${SLURM_JOB_ID}"
done
echo "=== probe OK"
