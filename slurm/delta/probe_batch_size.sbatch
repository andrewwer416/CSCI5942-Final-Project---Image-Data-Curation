#!/bin/bash
#SBATCH --job-name=mega-probe
#SBATCH --account=bidb-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60g
#SBATCH --time=00:45:00
#SBATCH --output=artifacts/logs/%x-%j.out
# ---------------------------------------------------------------------
# Measure peak GPU memory and seconds per step at the study shape
# (640 px) for several contrastive batch sizes. Each size runs 30 steps
# on the random baseline and evaluates on 512 validation images.
#
#   mkdir -p artifacts/logs
#   sbatch slurm/delta/probe_batch_size.sbatch            # 32 64 96 128
#   sbatch slurm/delta/probe_batch_size.sbatch 48 80      # custom sizes
#
# Stops at the first size that fails, which is usually out of memory.
# ---------------------------------------------------------------------
set -uo pipefail
module load pytorch-conda/2.12
conda activate base
export PYTHONPATH=src

SIZES=("$@")
if [ ${#SIZES[@]} -eq 0 ]; then
  SIZES=(32 64 96 128)
fi
nvidia-smi --query-gpu=name,memory.total --format=csv

for bs in "${SIZES[@]}"; do
  echo "=== batch size ${bs}"
  if ! srun python -m vlm_train.train \
      --selection data/processed/selections/random_seed5942_n15000.jsonl \
      --limit $((bs * 30)) --val-limit 512 --epochs 1 --batch-size "${bs}" \
      --num-workers 12 --output-dir "artifacts/probe/bs${bs}-${SLURM_JOB_ID}"; then
    echo "=== batch size ${bs} failed; stopping"
    break
  fi
done
