"""Copy the images one training run reads onto the compute node's local disk.

Opening thousands of small files on /scratch/alpine (served over NFS) costs about
50 ms each, which made the Alpine probe input-bound (7 s/step instead of ~1.7 s).
Reading the two COCO archives is a few large sequential reads instead, so each job
extracts just its selection's train images plus all val images into local disk:

    python slurm/alpine/stage_images.py --selection <manifest.jsonl> \
        --downloads data/raw/coco/downloads --dest "$SLURM_SCRATCH/coco"

The result has the same layout as data/raw/coco (train2017/, val2017/), so training
takes it as --image-root. The archives are the checksum-verified downloads that the
setup job extracted, so the staged files are byte-identical to data/raw/coco.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import zipfile
from pathlib import Path


def wanted_paths(selection: Path) -> set[str]:
    with selection.open() as handle:
        return {json.loads(line)["relative_path"] for line in handle if line.strip()}


def extract(archive: Path, dest: Path, wanted: set[str] | None) -> int:
    with zipfile.ZipFile(archive) as bundle:
        members = [
            info
            for info in bundle.infolist()
            if not info.is_dir() and (wanted is None or info.filename in wanted)
        ]
        bundle.extractall(dest, members)
    return len(members)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--downloads", type=Path, required=True)
    parser.add_argument("--dest", type=Path, required=True)
    args = parser.parse_args()

    started = time.perf_counter()
    wanted = wanted_paths(args.selection)
    train = extract(args.downloads / "train2017.zip", args.dest, wanted)
    if train != len(wanted):
        print(f"stage: {len(wanted) - train} selected images missing from train2017.zip", file=sys.stderr)
        return 1
    val = extract(args.downloads / "val2017.zip", args.dest, None)
    print(f"stage: {train} train + {val} val images -> {args.dest} in {time.perf_counter() - started:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
