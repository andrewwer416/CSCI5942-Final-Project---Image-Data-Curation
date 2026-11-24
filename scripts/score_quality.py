"""Score every canonical train2017 image with no-reference quality measures (D5).

Train-only. Run in the isolated ``quality`` pixi environment:

    PYTHONPATH=src pixi run -e quality python scripts/score_quality.py

Measures are computed on the decoded native-resolution RGB image (no EXIF
rotation, matching the CLIP and P1 decoders), not on the P1 640x640 resize.
A 2,000-image pilot showed P1-resized NIQE correlates 0.41 (Spearman) with
the aspect-ratio stretch P1 applies, versus 0.02 at native resolution.

Raw measures are stored; no thresholds, ranks, or weights are applied here.
"""

from __future__ import annotations

import argparse
import platform
import time
from pathlib import Path

import cv2
import numpy as np
import pyiqa
import torch
from PIL import Image
from pyiqa.archs.niqe_arch import NIQE
from torch.utils.data import DataLoader, Dataset

from vlm_data.config import load_config
from vlm_data.io import sha256_file, write_json_atomic, write_jsonl_atomic
from vlm_data.manifest import read_manifest
from vlm_data.visual_embeddings import image_ids_sha256

ALGORITHM_VERSION = "native-niqe-laplacian-exposure-v1"
NIQE_PARAMS = Path("data/raw/model-cache/pyiqa/niqe_modelparameters.mat")
NIQE_PARAMS_SHA256 = "fa2aa4d12dd9595c3fb6bf8b4e1a15c79d5d1469238d0901c88dead195be578d"
NIQE_PARAMS_SOURCE = (
    "https://huggingface.co/chaofengc/IQA-PyTorch-Weights/resolve/main/niqe_modelparameters.mat"
)
DARK_LEVEL = 4  # gray <= 4 of 255 counts as near-black
BRIGHT_LEVEL = 251  # gray >= 251 of 255 counts as near-white
NIQE_MIN_SIDE = 96  # pyiqa NIQE block size; no blocks exist below this
SHORT_SIDE = 256  # scale-normalized sharpness: short side resized to 256 (INTER_AREA)


class Images(Dataset):
    def __init__(self, root: Path, image_ids: list[int]):
        self.root, self.image_ids = root, image_ids

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index: int):
        image_id = self.image_ids[index]
        with Image.open(self.root / f"{image_id:012d}.jpg") as image:
            rgb = np.asarray(image.convert("RGB"))
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        height, width = gray.shape
        scale = SHORT_SIDE / min(height, width)
        small = cv2.resize(
            gray, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA
        )
        measures = {
            "image_id": image_id,
            "width": width,
            "height": height,
            "log_laplacian_var_native": float(
                np.log(cv2.Laplacian(gray, cv2.CV_64F).var() + 1e-6)
            ),
            "log_laplacian_var_s256": float(np.log(cv2.Laplacian(small, cv2.CV_64F).var() + 1e-6)),
            "dark_fraction": float(np.mean(gray <= DARK_LEVEL)),
            "bright_fraction": float(np.mean(gray >= BRIGHT_LEVEL)),
            "mean_luma": float(gray.mean()),
            "std_luma": float(gray.std()),
        }
        return torch.from_numpy(rgb.copy()), measures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config = load_config()
    root = config.project_root
    params = root / NIQE_PARAMS
    if sha256_file(params) != NIQE_PARAMS_SHA256:
        raise SystemExit(f"NIQE parameter file hash mismatch: {params}")
    image_ids = [record.image_id for record in read_manifest(config.train_manifest)]
    if args.limit:
        image_ids = image_ids[: args.limit]
    out_dir = root / "data/interim/quality"
    suffix = f"_limit{args.limit}" if args.limit else ""
    output = out_dir / f"quality_scores_train2017{suffix}.jsonl"
    if output.exists() and not args.overwrite:
        raise SystemExit(f"{output} exists; pass --overwrite to replace it atomically")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    niqe = NIQE(pretrained_model_path=str(params)).to(device).eval()
    loader = DataLoader(
        Images(root / config.train_images_dir.relative_to(root), image_ids),
        batch_size=None,
        num_workers=args.num_workers,
        prefetch_factor=8 if args.num_workers else None,
    )
    rows = []
    started = time.perf_counter()
    with torch.no_grad():
        for index, (rgb, measures) in enumerate(loader):
            tensor = rgb.to(device).permute(2, 0, 1)[None].float().div(255)
            # NIQE needs at least one 96x96 block; smaller images get no score
            # and are left to the separately decided small-image floor.
            if min(tensor.shape[-2:]) < NIQE_MIN_SIDE:
                measures["niqe"] = None
            else:
                measures["niqe"] = float(niqe(tensor))
            if measures["niqe"] is not None and not np.isfinite(measures["niqe"]):
                raise SystemExit(f"Non-finite NIQE for image {measures['image_id']}")
            rows.append(measures)
            if (index + 1) % 5000 == 0:
                rate = (index + 1) / (time.perf_counter() - started)
                print(f"{index + 1}/{len(image_ids)} images, {rate:.1f} img/s", flush=True)

    if [row["image_id"] for row in rows] != image_ids:
        raise SystemExit("Scored image order does not match the canonical manifest")
    count, digest = write_jsonl_atomic(output, rows)
    write_json_atomic(
        output.with_suffix(".meta.json"),
        {
            "schema_version": 1,
            "kind": "quality_scores",
            "split": "train2017",
            "algorithm_version": ALGORITHM_VERSION,
            "record_count": count,
            "scores_sha256": digest,
            "source_manifest": config.train_manifest.name,
            "source_manifest_sha256": sha256_file(config.train_manifest),
            "image_ids_sha256": image_ids_sha256(image_ids),
            "decode": "PIL convert('RGB'), no EXIF transpose, native resolution",
            "niqe": {
                "implementation": f"pyiqa {pyiqa.__version__} archs.niqe_arch.NIQE (Y channel)",
                "parameters_file": str(NIQE_PARAMS),
                "parameters_sha256": NIQE_PARAMS_SHA256,
                "parameters_source": NIQE_PARAMS_SOURCE,
                "lower_is_better": True,
                "device": device,
                "min_side": NIQE_MIN_SIDE,
                "unscored_count": sum(row["niqe"] is None for row in rows),
            },
            "sharpness": "log(var(Laplacian(gray)) + 1e-6); native and short-side-256 INTER_AREA",
            "exposure": {"dark_level": DARK_LEVEL, "bright_level": BRIGHT_LEVEL},
            "versions": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "opencv": cv2.__version__,
                "numpy": np.__version__,
            },
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "seconds": round(time.perf_counter() - started, 1),
        },
    )
    print({"output": str(output.relative_to(root)), "records": count}, flush=True)


if __name__ == "__main__":
    main()
