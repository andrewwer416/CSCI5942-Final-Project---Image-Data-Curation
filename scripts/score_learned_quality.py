"""Score train2017 with learned no-reference IQA models trained on human ratings (D5).

Train-only. Run in the isolated ``quality`` pixi environment:

    PYTHONPATH=src pixi run -e quality python scripts/score_learned_quality.py

Models (pyiqa, KonIQ-10k human mean-opinion-score training, higher = better quality):

* TOPIQ-NR (CFANet, ResNet-50), ``topiq_nr``;
* MUSIQ (multi-scale transformer), ``musiq``.

Weights are pinned copies in ``data/raw/model-cache/pyiqa`` verified by SHA-256.
Images are scored at native resolution after PIL ``convert('RGB')`` (no EXIF
transpose), matching ``scripts/score_quality.py``. Raw scores only; no thresholds.
"""

from __future__ import annotations

import argparse
import platform
import time
from pathlib import Path

import numpy as np
import pyiqa
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from vlm_data.config import load_config
from vlm_data.io import sha256_file, write_json_atomic, write_jsonl_atomic
from vlm_data.manifest import read_manifest
from vlm_data.visual_embeddings import image_ids_sha256

ALGORITHM_VERSION = "native-topiq-musiq-koniq-v1"
WEIGHTS_DIR = Path("data/raw/model-cache/pyiqa")
WEIGHT_SOURCE = "https://huggingface.co/chaofengc/IQA-PyTorch-Weights/resolve/main/"
MODELS = {
    "topiq_nr": (
        "cfanet_nr_koniq_res50-9a73138b.pth",
        "9a73138beb3748dd6a83365610ac88fa1cd53225342c77fc9ed44a3be461354c",
    ),
    "musiq": (
        "musiq_koniq_ckpt-e95806b9.pth",
        "e95806b9eae5f3814c410f574ba8e552362bd5bc63d758ed5b97860f5d6185aa",
    ),
}
MIN_SIDE = 32  # below this the backbones have no valid feature map; left unscored


class Images(Dataset):
    def __init__(self, root: Path, image_ids: list[int]):
        self.root, self.image_ids = root, image_ids

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index: int):
        image_id = self.image_ids[index]
        with Image.open(self.root / f"{image_id:012d}.jpg") as image:
            rgb = np.asarray(image.convert("RGB"))
        return image_id, torch.from_numpy(rgb.copy())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config = load_config()
    root = config.project_root
    device = "cuda" if torch.cuda.is_available() else "cpu"
    metrics = {}
    for name, (filename, digest) in MODELS.items():
        path = root / WEIGHTS_DIR / filename
        if sha256_file(path) != digest:
            raise SystemExit(f"Weight hash mismatch: {path}")
        metric = pyiqa.create_metric(name, device=device, pretrained_model_path=str(path))
        metric.eval()
        if metric.lower_better:
            raise SystemExit(f"{name} unexpectedly reports lower_better")
        metrics[name] = metric

    image_ids = [record.image_id for record in read_manifest(config.train_manifest)]
    if args.limit:
        image_ids = image_ids[: args.limit]
    suffix = f"_limit{args.limit}" if args.limit else ""
    output = root / f"data/interim/quality/learned_quality_train2017{suffix}.jsonl"
    if output.exists() and not args.overwrite:
        raise SystemExit(f"{output} exists; pass --overwrite to replace it atomically")

    loader = DataLoader(
        Images(config.train_images_dir, image_ids),
        batch_size=None,
        num_workers=args.num_workers,
        prefetch_factor=8 if args.num_workers else None,
    )
    rows = []
    started = time.perf_counter()
    with torch.no_grad():
        for index, (image_id, rgb) in enumerate(loader):
            tensor = rgb.to(device).permute(2, 0, 1)[None].float().div(255)
            row = {"image_id": int(image_id)}
            for name, metric in metrics.items():
                if min(tensor.shape[-2:]) < MIN_SIDE:
                    row[name] = None
                    continue
                value = float(metric(tensor))
                if not np.isfinite(value):
                    raise SystemExit(f"Non-finite {name} for image {image_id}")
                row[name] = value
            rows.append(row)
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
            "kind": "learned_quality_scores",
            "split": "train2017",
            "algorithm_version": ALGORITHM_VERSION,
            "record_count": count,
            "scores_sha256": digest,
            "source_manifest": config.train_manifest.name,
            "source_manifest_sha256": sha256_file(config.train_manifest),
            "image_ids_sha256": image_ids_sha256(image_ids),
            "decode": "PIL convert('RGB'), no EXIF transpose, native resolution",
            "models": {
                name: {
                    "pyiqa_name": name,
                    "training_data": "KonIQ-10k",
                    "weights_file": str(WEIGHTS_DIR / filename),
                    "weights_sha256": sha,
                    "weights_source": WEIGHT_SOURCE + filename,
                    "higher_is_better": True,
                }
                for name, (filename, sha) in MODELS.items()
            },
            "min_side": MIN_SIDE,
            "unscored_count": sum(row["topiq_nr"] is None for row in rows),
            "device": device,
            "versions": {
                "python": platform.python_version(),
                "pyiqa": pyiqa.__version__,
                "torch": torch.__version__,
                "numpy": np.__version__,
            },
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "seconds": round(time.perf_counter() - started, 1),
        },
    )
    print({"output": str(output.relative_to(root)), "records": count}, flush=True)


if __name__ == "__main__":
    main()
