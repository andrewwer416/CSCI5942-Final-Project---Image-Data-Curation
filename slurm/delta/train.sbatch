#!/bin/bash
#SBATCH --job-name=mega-train
#SBATCH --account=bidb-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60g
#SBATCH --time=08:00:00
#SBATCH --output=artifacts/logs/%x-%j.out
# ---------------------------------------------------------------------
# One full study run on one A40. Arguments are passed to vlm_train.train.
# Probe (job 24034477, A40): batch 64 = 1.70 s/step, 36 GB peak; batch 96 OOM.
# A 40-epoch run is about 5-6 h, so 8 h leaves margin; the partition limit is 48 h,
# and training has no resume yet, so a run that hits the limit is lost.
#
#   mkdir -p artifacts/logs
#   sbatch slurm/delta/train.sbatch \
#     --selection data/processed/selections/random_seed5942_n15000.jsonl \
#     --output-dir artifacts/runs/random__standard_rgb__seed5942 \
#     --batch-size 64 --model-seed 5942 --save-checkpoint
# ---------------------------------------------------------------------
set -euo pipefail
module load pytorch-conda/2.12
conda activate base
export PYTHONPATH=src
nvidia-smi --query-gpu=name,memory.total --format=csv
srun python -m vlm_train.train --num-workers 12 "$@"
