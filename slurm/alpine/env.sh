# Shared settings for every Alpine job. Each sbatch script sources this file
# after changing into the repository root ($SLURM_SUBMIT_DIR).
#
# Alpine home directories hold only 2 GB, so the Python environment and the
# pip/conda caches live in /projects/$USER (software installs belong there).

if [ ! -f configs/data.toml ]; then
  echo "Submit jobs from the repository root (configs/data.toml not found in $PWD)" >&2
  exit 1
fi

# CURC terminates jobs doing heavy I/O on /home or /projects, so the repository,
# COCO images, and run outputs must live under /scratch/alpine/$USER.
if [ "${SLURM_CLUSTER_NAME:-}" = alpine ]; then
  case "$(pwd -P)" in
    /home/*|/projects/*)
      echo "Submit from a clone under /scratch/alpine/$USER, not $(pwd -P)" >&2
      exit 1
      ;;
  esac
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

# Opening many small files on /scratch (NFS) made the probe input-bound (job
# 36287488: 7.2 s/step, while an I/O test read only ~20 images/s). So each run
# first extracts its selection's train images and all val images from the COCO
# zips onto the node's local disk ($SLURM_SCRATCH, removed when the job ends)
# and reads them from there. Prints the image root on its last line.
stage_images() {
  local selection=$1 dest="${SLURM_SCRATCH:?SLURM_SCRATCH is not set}/coco"
  echo "stage start=$(date -Is)" >&2
  "$PYTHON" slurm/alpine/stage_images.py \
    --selection "$selection" --downloads data/raw/coco/downloads --dest "$dest" >&2
  echo "$dest"
}

# One full 40-epoch study run. Arguments go straight to vlm_train.train; the
# run scripts pass --selection, --preprocessing, --model-seed and --output-dir.
run_training() {
  print_job_environment
  local selection="" previous="" image_root
  for arg in "$@"; do
    [ "$previous" = --selection ] && selection=$arg
    previous=$arg
  done
  image_root=$(stage_images "${selection:?run_training needs --selection}")
  srun "$PYTHON" -m vlm_train.train \
    --batch-size "$STUDY_BATCH_SIZE" \
    --num-workers "$STUDY_NUM_WORKERS" \
    --save-checkpoint \
    --image-root "$image_root" \
    "$@"
  echo "finished=$(date -Is)"
}
