#!/bin/bash
# Submit every study run script in slurm/alpine/ (run_*.sbatch), from the repo root.
# Pass sbatch options first if needed, e.g.:  bash slurm/alpine/submit_runs.sh --partition=al40
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p artifacts/logs
for script in slurm/alpine/run_*.sbatch; do
  printf '%-75s ' "$script"
  sbatch --parsable "$@" "$script"
done
