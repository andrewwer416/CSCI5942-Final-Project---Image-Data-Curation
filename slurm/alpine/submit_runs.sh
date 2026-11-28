#!/bin/bash
# Submit every study run script in slurm/alpine/ (run_*.sbatch), from the repo root.
# Skips a script whose output folder already holds a run (train.py would refuse it anyway,
# but only after staging the images on a GPU node).
# Pass sbatch options first if needed, e.g.:  bash slurm/alpine/submit_runs.sh --partition=al40
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p artifacts/logs
for script in slurm/alpine/run_*.sbatch; do
  printf '%-75s ' "$script"
  output_dir=$(sed -n 's/.*--output-dir \([^ ]*\).*/\1/p' "$script")
  if [ -e "$output_dir/metrics.jsonl" ] || [ -e "$output_dir/run_config.json" ]; then
    echo "skip (already has a run in $output_dir)"
    continue
  fi
  sbatch --parsable "$@" "$script"
done
