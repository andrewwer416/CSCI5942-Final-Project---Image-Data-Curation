"""Train one selection/preprocessing configuration and evaluate retrieval on val2017."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader

from vlm_data.config import load_config
from vlm_data.io import sha256_file, write_json_atomic
from vlm_data.manifest import read_manifest
from vlm_data.validation import validate_manifest
from vlm_train.data import (
    APPEARANCE_ALGORITHM_VERSION,
    CLIP_PREPROCESSING,
    ClipFeatureTable,
    COLOR_CONTRAST_NORMALIZATION,
    CONTRAST_GAIN_CAP,
    CONTRAST_PERCENTILES,
    SHADES_OF_GRAY_P,
    WHITE_BALANCE_GAIN_CAP,
    CROP_ALGORITHM_VERSION,
    CROP_ATTEMPTS,
    CROP_MIN_SCALE,
    CROP_RATIO_JITTER,
    EVAL_PREPROCESSING,
    MASKED_PREPROCESSING,
    RANDOM_RESIZED_CROP,
    PREPROCESSING_CHOICES,
    STANDARD_RGB,
    EvalRetrievalDataset,
    TrainRetrievalDataset,
)
from vlm_train.metrics import mentions_color, retrieval_metrics
from vlm_train.model import ModelConfig, RetrievalModel, contrastive_loss
from vlm_train.tokenizer import Vocabulary


LOG_EVERY_STEPS = 20


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/data.toml"))
    parser.add_argument("--selection", type=Path, required=True, help="Selection JSONL to train on")
    parser.add_argument(
        "--preprocessing",
        choices=PREPROCESSING_CHOICES,
        default=STANDARD_RGB,
        help="Image preprocessing arm; evaluation uses the same deterministic transform",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--image-root", type=Path, help="Folder holding train2017/ and val2017/ (default: config)"
    )
    parser.add_argument("--epochs", type=int, help="Default: config value")
    parser.add_argument("--image-size", type=int, help="Default: config value")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int, help="Use only the first N training records")
    parser.add_argument("--val-limit", type=int, help="Use only the first N validation records")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--min-word-frequency", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision")
    parser.add_argument("--save-checkpoint", action="store_true", help="Save final weights")
    parser.add_argument(
        "--model-seed",
        type=int,
        help="Weight initialization seed; caption/order seeds stay paired from config",
    )
    parser.add_argument(
        "--clip-train-metadata",
        type=Path,
        help="P5 train2017 CLIP cache metadata (default: config visual artifact dir)",
    )
    parser.add_argument(
        "--clip-val-metadata",
        type=Path,
        help="P5 val2017 CLIP cache metadata (default: config visual artifact dir)",
    )
    return parser


CLIP_CACHE_STEM = "openclip_vit_b16_datacomp_xl_s13b_b90k"
# P5 final-epoch diagnostics: evaluate with zeroed and with deranged CLIP vectors.
CLIP_DIAGNOSTIC_MODES = ("zeroed_clip", "shuffled_clip")


@torch.no_grad()
def evaluate(
    model: RetrievalModel,
    loader: DataLoader,
    *,
    device: torch.device,
    use_amp: bool,
    masked: bool,
    color_captions: torch.Tensor | None = None,
    use_clip: bool = False,
    zero_clip: bool = False,
) -> dict[str, float]:
    """Embed every image and caption, then score all pairs on the CPU.

    With ``use_clip`` the model receives each image's CLIP vector from the loader;
    ``zero_clip`` replaces those vectors with zeros (a P5 diagnostic).
    """

    model.eval()
    image_batches: list[torch.Tensor] = []
    text_batches: list[torch.Tensor] = []
    captions_per_image = 0
    for images, patch_mask, clip_vectors, tokens in loader:
        captions_per_image = tokens.size(1)
        flat_tokens = tokens.flatten(0, 1).to(device, non_blocking=True)
        mask = patch_mask.to(device, non_blocking=True) if masked else None
        clip = clip_vectors.to(device, non_blocking=True) if use_clip else None
        if clip is not None and zero_clip:
            clip = torch.zeros_like(clip)
        with torch.autocast(device.type, enabled=use_amp):
            image_embeddings = model.encode_image(images.to(device, non_blocking=True), mask, clip)
            text_embeddings = model.encode_text(flat_tokens)
        image_batches.append(image_embeddings.float().cpu())
        text_batches.append(text_embeddings.float().cpu())
    image_embeddings = torch.cat(image_batches)
    text_embeddings = torch.cat(text_batches)
    caption_to_image = torch.arange(image_embeddings.size(0)).repeat_interleave(captions_per_image)
    subsets = {"color": color_captions} if color_captions is not None else None
    return retrieval_metrics(
        image_embeddings @ text_embeddings.t(), caption_to_image, caption_subsets=subsets
    )


def train(args: argparse.Namespace) -> dict[str, float]:
    config = load_config(args.config)
    image_root = args.image_root or config.raw_root
    epochs = config.epochs if args.epochs is None else args.epochs
    image_size = config.image_size if args.image_size is None else args.image_size
    device = torch.device(args.device)
    use_amp = device.type == "cuda" and not args.no_amp
    model_seed = config.seed if args.model_seed is None else args.model_seed
    # Unpadded arms pass no mask, so their attention path is exactly P1's.
    masked = args.preprocessing in MASKED_PREPROCESSING
    use_clip = args.preprocessing in CLIP_PREPROCESSING
    torch.manual_seed(model_seed)
    if (args.output_dir / "run_config.json").exists() or (
        args.output_dir / "metrics.jsonl"
    ).exists():
        raise FileExistsError(f"Refusing to overwrite an existing training run: {args.output_dir}")

    source_records = {record.image_id: record for record in read_manifest(config.train_manifest)}
    selection_report, train_records = validate_manifest(
        args.selection,
        expected_split="train2017",
        expected_captions=config.captions_per_image,
        expected_count=config.train_image_budget,
        source_records=source_records,
        source_manifest_sha256=sha256_file(config.train_manifest),
    )
    if not selection_report["valid"]:
        raise ValueError(f"Invalid training selection: {selection_report['errors']}")
    train_records = train_records[: args.limit]
    val_records = list(read_manifest(config.val_manifest))[: args.val_limit]
    # Built from all of train2017, never val2017 or one selection, so that every run in
    # the study shares the same vocabulary and only its training images differ.
    vocabulary = Vocabulary.from_texts(
        (
            caption.text
            for record in read_manifest(config.train_manifest)
            for caption in record.captions
        ),
        min_frequency=args.min_word_frequency,
    )

    train_clip = val_clip = shuffled_val_clip = None
    clip_artifacts = None
    if use_clip:
        train_clip_metadata = args.clip_train_metadata or (
            config.visual_artifact_dir / f"{CLIP_CACHE_STEM}_train2017.meta.json"
        )
        val_clip_metadata = args.clip_val_metadata or (
            config.visual_artifact_dir / f"{CLIP_CACHE_STEM}_val2017.meta.json"
        )
        train_clip = ClipFeatureTable.from_artifact(
            train_clip_metadata, source_manifest=config.train_manifest
        )
        val_clip = ClipFeatureTable.from_artifact(
            val_clip_metadata, source_manifest=config.val_manifest
        )
        shuffled_val_clip = val_clip.deranged(
            [record.image_id for record in val_records], seed=config.seed
        )
        clip_artifacts = {
            "train_metadata": str(train_clip_metadata),
            "train_metadata_sha256": sha256_file(train_clip_metadata),
            "val_metadata": str(val_clip_metadata),
            "val_metadata_sha256": sha256_file(val_clip_metadata),
            "fusion": "projection(LayerNorm(CLS + W clip)), W 512->768, no bias, Xavier",
            "diagnostics": list(CLIP_DIAGNOSTIC_MODES),
            "shuffle_seed": config.seed,
        }

    train_dataset = TrainRetrievalDataset(
        train_records,
        image_root=image_root,
        vocabulary=vocabulary,
        seed=config.seed,
        image_size=image_size,
        preprocessing=args.preprocessing,
        clip_features=train_clip,
    )
    # Caption order matches EvalRetrievalDataset: each record's captions, records in order.
    color_captions = torch.tensor(
        [mentions_color(caption.text) for record in val_records for caption in record.captions]
    )
    val_dataset = EvalRetrievalDataset(
        val_records,
        image_root=image_root,
        vocabulary=vocabulary,
        image_size=image_size,
        preprocessing=args.preprocessing,
        clip_features=val_clip,
    )
    shuffle_generator = torch.Generator()
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=shuffle_generator,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,  # A contrastive batch must keep its size for comparable losses.
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    shuffled_val_loader = None
    if use_clip:
        shuffled_val_loader = DataLoader(
            EvalRetrievalDataset(
                val_records,
                image_root=image_root,
                vocabulary=vocabulary,
                image_size=image_size,
                preprocessing=args.preprocessing,
                clip_features=shuffled_val_clip,
            ),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
    if len(train_loader) == 0:
        raise ValueError("batch size is larger than the number of training records")

    model_config = ModelConfig(
        vocab_size=len(vocabulary),
        pad_id=vocabulary.pad_id,
        eos_id=vocabulary.eos_id,
        image_size=image_size,
        clip_dim=train_clip.dimension if train_clip is not None else 0,
    )
    model = RetrievalModel(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(
        args.output_dir / "run_config.json",
        {
            "selection": str(args.selection),
            "preprocessing": args.preprocessing,
            "eval_preprocessing": EVAL_PREPROCESSING.get(args.preprocessing, args.preprocessing),
            "padding_masked": masked,
            "clip_features": clip_artifacts,
            "crop": {
                "algorithm_version": CROP_ALGORITHM_VERSION,
                "min_scale": CROP_MIN_SCALE,
                "ratio_jitter_relative_to_image": list(CROP_RATIO_JITTER),
                "attempts": CROP_ATTEMPTS,
                "seed": config.seed,
            }
            if args.preprocessing == RANDOM_RESIZED_CROP
            else None,
            "appearance": {
                "algorithm_version": APPEARANCE_ALGORITHM_VERSION,
                "shades_of_gray_p": SHADES_OF_GRAY_P,
                "white_balance_gain_cap": WHITE_BALANCE_GAIN_CAP,
                "contrast_percentiles": list(CONTRAST_PERCENTILES),
                "contrast_gain_cap": CONTRAST_GAIN_CAP,
            }
            if args.preprocessing == COLOR_CONTRAST_NORMALIZATION
            else None,
            "seed": config.seed,
            "model_seed": model_seed,
            "caption_seed": config.seed,
            "data_order_seed": config.seed,
            "selection_sha256": sha256_file(args.selection),
            "train_manifest_sha256": sha256_file(config.train_manifest),
            "val_manifest_sha256": sha256_file(config.val_manifest),
            "data_config_sha256": sha256_file(args.config),
            "torch_version": str(torch.__version__),
            "cuda_version": torch.version.cuda,
            "device": str(device),
            "study_shape": len(train_records) == config.train_image_budget
            and len(val_records) == config.expected_val_images
            and epochs == config.epochs
            and image_size == config.image_size,
            "epochs": epochs,
            "batch_size": args.batch_size,
            "train_records": len(train_records),
            "val_records": len(val_records),
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "min_word_frequency": args.min_word_frequency,
            "use_amp": use_amp,
            "model": asdict(model_config),
        },
    )
    write_json_atomic(args.output_dir / "vocabulary.json", {"words": vocabulary.words})
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"device={device} amp={use_amp} images={len(train_records)} vocab={len(vocabulary)} "
        f"parameters={parameter_count / 1e6:.1f}M steps_per_epoch={len(train_loader)}",
        flush=True,
    )

    result: dict[str, float] = {}
    log_path = args.output_dir / "metrics.jsonl"
    log_path.write_text("", encoding="utf-8")
    for epoch in range(epochs):
        model.train()
        train_dataset.set_epoch(epoch)
        shuffle_generator.manual_seed(config.seed + epoch)
        started = time.perf_counter()
        total_loss = 0.0
        for step, (images, patch_mask, clip_vectors, tokens) in enumerate(train_loader, start=1):
            images = images.to(device, non_blocking=True)
            tokens = tokens.to(device, non_blocking=True)
            mask = patch_mask.to(device, non_blocking=True) if masked else None
            clip = clip_vectors.to(device, non_blocking=True) if use_clip else None
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=use_amp):
                loss = contrastive_loss(model(images, tokens, mask, clip))
            if not math.isfinite(loss.item()):
                raise RuntimeError(f"Loss became {loss.item()} at epoch {epoch} step {step}")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            model.clamp_logit_scale()
            total_loss += loss.item()
            if step % LOG_EVERY_STEPS == 0 or step == len(train_loader):
                print(
                    f"epoch {epoch} step {step}/{len(train_loader)} loss {loss.item():.4f}",
                    flush=True,
                )
        train_seconds = time.perf_counter() - started

        evaluation = dict(
            device=device,
            use_amp=use_amp,
            masked=masked,
            color_captions=color_captions if color_captions.any() else None,
            use_clip=use_clip,
        )
        result = {
            "epoch": epoch,
            "train_loss": total_loss / len(train_loader),
            "train_seconds": train_seconds,
            "seconds_per_step": train_seconds / len(train_loader),
            **evaluate(model, val_loader, **evaluation),
        }
        if use_clip and epoch == epochs - 1:
            # Reporting only: these never feed back into any training or method choice.
            diagnostics = {
                "zeroed_clip": evaluate(model, val_loader, zero_clip=True, **evaluation),
                "shuffled_clip": evaluate(model, shuffled_val_loader, **evaluation),
            }
            for mode, metrics in diagnostics.items():
                result.update({f"{mode}/{name}": value for name, value in metrics.items()})
        if device.type == "cuda":
            result["peak_gpu_memory_mib"] = torch.cuda.max_memory_allocated(device) / 2**20
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result, sort_keys=True) + "\n")
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)

    if args.save_checkpoint:
        torch.save(
            {"model": model.state_dict(), "model_config": asdict(model_config)},
            args.output_dir / "final.pt",
        )
    return result


def main(argv: list[str] | None = None) -> int:
    train(_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
