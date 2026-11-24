# Alpine job scripts

These scripts run the study on CURC Alpine, one job per training run. Submit every job from the
repository root. The Slurm log path `artifacts/logs/` is relative to that folder.

## What runs

Every run trains for 40 epochs at 640×640 with batch 64, saves `final.pt`, and evaluates full
val2017 after each epoch. The batch size was measured on a Delta A40 (job 24034477): batch 64
peaked at 36 GB and took 1.70 s per step, while batch 96 ran out of memory. A run should take
about 5–6 hours on an A40, and each script asks for 12 hours (QoS `gpu-normal`).

| Script | Selection | Preprocessing | Model seed |
|---|---|---|---|
| `run_01_…seed5942` | D1 random | P1 standard RGB | 5942 |
| `run_02_…seed5943` | D1 random | P1 | 5943 |
| `run_03_…seed5944` | D1 random | P1 | 5944 |
| `run_04_category_balanced…` | D2 category-balanced | P1 | 5942 |
| `run_05_visual_diversity…` | D3 visual diversity | P1 | 5942 |
| `run_06_caption_semantic_diversity…` | D4 caption-semantic | P1 | 5942 |
| `run_07_quality_aware_diversity…` | D5 quality-aware | P1 | 5942 |
| `run_08_random__aspect_ratio_preserving_rgb…` | D1 random | P2 letterbox + mask | 5942 |

P3–P5 are not implemented yet, so they have no scripts. A new arm needs one more `run_*.sbatch`
file in the same format.

Results go to `artifacts/runs/<run_id>__seed<seed>/`. A job refuses to start if that folder
already holds a run, so a failed run must be deleted before it is resubmitted.

## Steps

CURC storage rules decide where things go:

| Location | Size | Backed up | Used here for |
|---|---|---|---|
| `/home/$USER` | 2 GB | Yes | Nothing; it is too small |
| `/projects/$USER` | 250 GB | Yes | The Python environment and a backup of finished results |
| `/scratch/alpine/$USER` | 10 TB | **No; files are deleted 90 days after creation** | The repository, COCO images, logs, and training runs |

CURC terminates jobs that do heavy I/O on `/home` or `/projects`. Reading 15,000 images for 40
epochs is heavy I/O, so the repository and data live in scratch. `env.sh` stops a job that is
submitted from `/home` or `/projects`.

```bash
# 1. On a login node, clone into scratch.
mkdir -p /scratch/alpine/$USER && cd /scratch/alpine/$USER
git clone https://github.com/andrewwer416/CSCI5942-Final-Project---Image-Data-Curation.git mega
cd mega
git switch bill/alpine-jobs
mkdir -p artifacts/logs

# 2. One-time setup on a CPU node: Python environment, COCO download, manifests, file check.
#    Takes a few hours. The log should end with "== DONE" and every "valid": true.
sbatch slurm/alpine/00_setup.sbatch

# 3. Short GPU check: batch 64 must fit for both P1 and P2. The log should end with "probe OK".
sbatch slurm/alpine/01_probe.sbatch

# 4. Submit all eight runs, or submit them one at a time with sbatch.
bash slurm/alpine/submit_runs.sh

# Monitor
squeue -u $USER
tail -f artifacts/logs/mega-D1P1-s5942-<jobid>.out

# 5. When runs finish, copy results to /projects, which is backed up and never purged.
rsync -av artifacts/runs artifacts/logs /projects/$USER/mega-results/
```

## Settings to check

- **Account.** Every script charges `ucb857_peak1`. To charge another account, override it on
  the command line, for example `sbatch --account=ucb-general slurm/alpine/01_probe.sbatch` or
  `bash slurm/alpine/submit_runs.sh --account=ucb-general`. List your accounts with
  `sacctmgr -p show associations user=$USER`.
- **CPU partition.** The setup job names no partition, so Slurm uses the cluster default (marked
  `*` in `sinfo -s`). Add `--partition=<name>` if that default is a GPU partition.
- **GPU partition.** The scripts use `aa100`. If the probe runs out of memory at batch 64, use the
  48 GB L40 nodes for every run instead (`--partition=al40`), so that all arms keep the same batch
  size. Command-line `sbatch` options override the `#SBATCH` lines.
- **Internet on compute nodes.** The CURC docs do not say whether compute nodes can reach the
  internet. If the download step of the setup job fails, run
  `PYTHONPATH=src /projects/$USER/software/mega-env/bin/python -m vlm_data download all --confirm-large-download`
  from the repository on a login node, then resubmit the setup job. CURC's recommended route for
  large transfers is its data transfer nodes (`dtn.rc.colorado.edu`) or Globus.
- **Python environment.** It is in `/projects/$USER/software/mega-env`, a conda Python 3.12 with
  the PyTorch cu128 wheel. The probe prints the torch build and fails early if CUDA is not usable.
