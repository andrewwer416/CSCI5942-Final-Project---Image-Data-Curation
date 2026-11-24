#!/bin/bash
#SBATCH --job-name=mega-setup
#SBATCH --account=ucb857_peak1
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --output=artifacts/logs/%x-%j.out
# ---------------------------------------------------------------------
# One-time CPU job: build the Python environment, download COCO 2017
# (about 20 GB of archives), build manifests, and check every image file.
# Safe to resubmit: an existing environment and finished downloads are reused.
#
#   mkdir -p artifacts/logs
#   sbatch slurm/alpine/00_setup.sbatch
# ---------------------------------------------------------------------
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source slurm/alpine/env.sh
mkdir -p "$PIP_CACHE_DIR" "$CONDA_PKGS_DIRS" "$(dirname "$MEGA_ENV")"

if [ ! -x "$PYTHON" ]; then
  echo "== create environment at $MEGA_ENV"
  module load miniforge
  conda create -y -p "$MEGA_ENV" python=3.12 pip
fi

echo "== install packages"
"$PYTHON" -m pip install --upgrade pip
# The cu128 build runs on any recent driver; the probe job checks the GPU side.
"$PYTHON" -m pip install torch --index-url https://download.pytorch.org/whl/cu128
"$PYTHON" -m pip install "numpy>=1.26" "pillow>=10.0"
"$PYTHON" -c "import torch, numpy, PIL; print('torch', torch.__version__, 'numpy', numpy.__version__, 'pillow', PIL.__version__)"

echo "== unit tests"
"$PYTHON" -m unittest discover -s tests 2>&1 | tail -3

echo "== download $(date -Is)"
"$PYTHON" -m vlm_data download all --confirm-large-download
echo "== prepare $(date -Is)"
"$PYTHON" -m vlm_data prepare | grep -E '"(split|record_count)"'
echo "== validate --check-files $(date -Is)"
"$PYTHON" -m vlm_data validate --check-files | grep -E '"(valid|record_count|code)"'
echo "== DONE $(date -Is)"
