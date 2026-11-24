# Shared settings for every Alpine job. Each sbatch script sources this file
# after changing into the repository root ($SLURM_SUBMIT_DIR).
#
# Alpine home directories hold only 2 GB, so the Python environment and the
# pip/conda caches live in /projects/$USER instead.

if [ ! -f configs/data.toml ]; then
  echo "Submit jobs from the repository root (configs/data.toml not found in $PWD)" >&2
  exit 1
fi

export MEGA_ENV="${MEGA_ENV:-/projects/$USER/software/mega-env}"
export PIP_CACHE_DIR="/projects/$USER/.cache/pip"
export CONDA_PKGS_DIRS="/projects/$USER/.cache/conda-pkgs"
export PYTHONPATH=src
export PYTHON="$MEGA_ENV/bin/python"

# Measured on Delta (A40, job 24034477): batch 64 = 36 GB peak, 1.70 s/step;
# batch 96 ran out of memory. Every arm uses the same batch size.
export STUDY_BATCH_SIZE=64
export STUDY_NUM_WORKERS=12

print_job_environment() {
  echo "host=$(hostname) job=${SLURM_JOB_ID:-none} start=$(date -Is)"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
  "$PYTHON" - <<'EOF'
import torch
print(f"torch={torch.__version__} cuda={torch.version.cuda} available={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available; check the torch build against the driver above")
EOF
  echo "commit=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
}

# One full 40-epoch study run. Arguments go straight to vlm_train.train; the
# run scripts pass --selection, --preprocessing, --model-seed and --output-dir.
run_training() {
  print_job_environment
  srun "$PYTHON" -m vlm_train.train \
    --batch-size "$STUDY_BATCH_SIZE" \
    --num-workers "$STUDY_NUM_WORKERS" \
    --save-checkpoint \
    "$@"
  echo "finished=$(date -Is)"
}
