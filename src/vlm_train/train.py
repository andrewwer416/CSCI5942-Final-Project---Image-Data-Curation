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
from vlm_data.io import write_json_atomic
from vlm_data.manifest import read_manifest
from vlm_train.data import EvalRetrievalDataset, TrainRetrievalDataset
from vlm_train.metrics import retrieval_metrics
from vlm_train.model import ModelConfig, RetrievalModel, contrastive_loss
from vlm_train.tokenizer import Vocabulary


LOG_EVERY_STEPS = 20


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/data.toml"))
    parser.add_argument("--selection", type=Path, required=True, help="Selection JSONL to train on")
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
    return parser


@torch.no_grad()
def evaluate(
    model: RetrievalModel,
    loader: DataLoader,
    *,
    device: torch.device,
    use_amp: bool,
) -> dict[str, float]:
    """Embed every image and caption, then score all pairs on the CPU."""

    model.eval()
    image_batches: list[torch.Tensor] = []
    text_batches: list[torch.Tensor] = []
    captions_per_image = 0
    for images, tokens in loader:
        captions_per_image = tokens.size(1)
        flat_tokens = tokens.flatten(0, 1).to(device, non_blocking=True)
        with torch.autocast(device.type, enabled=use_amp):
            image_embeddings = model.encode_image(images.to(device, non_blocking=True))
            text_embeddings = model.encode_text(flat_tokens)
        image_batches.append(image_embeddings.float().cpu())
        text_batches.append(text_embeddings.float().cpu())
    image_embeddings = torch.cat(image_batches)
    text_embeddings = torch.cat(text_batches)
    caption_to_image = torch.arange(image_embeddings.size(0)).repeat_interleave(captions_per_image)
    return retrieval_metrics(image_embeddings @ text_embeddings.t(), caption_to_image)


def train(args: argparse.Namespace) -> dict[str, float]:
    config = load_config(args.config)
    image_root = args.image_root or config.raw_root
    epochs = config.epochs if args.epochs is None else args.epochs
    image_size = config.image_size if args.image_size is None else args.image_size
    device = torch.device(args.device)
    use_amp = device.type == "cuda" and not args.no_amp
    torch.manual_seed(config.seed)

    train_records = list(read_manifest(args.selection))[: args.limit]
    val_records = list(read_manifest(config.val_manifest))[: args.val_limit]
    # Built from all of train2017, never val2017 or one selection, so that every run in
    # the study shares the same vocabulary and only its training images differ.
    vocabulary = Vocabulary.from_texts(
        (caption.text for record in read_manifest(config.train_manifest) for caption in record.captions),
        min_frequency=args.min_word_frequency,
    )

    train_dataset = TrainRetrievalDataset(
        train_records,
        image_root=image_root,
        vocabulary=vocabulary,
        seed=config.seed,
        image_size=image_size,
    )
    val_dataset = EvalRetrievalDataset(
        val_records, image_root=image_root, vocabulary=vocabulary, image_size=image_size
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
    if len(train_loader) == 0:
        raise ValueError("batch size is larger than the number of training records")

    model_config = ModelConfig(
        vocab_size=len(vocabulary),
        pad_id=vocabulary.pad_id,
        eos_id=vocabulary.eos_id,
        image_size=image_size,
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
            "seed": config.seed,
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
        for step, (images, tokens) in enumerate(train_loader, start=1):
            images = images.to(device, non_blocking=True)
            tokens = tokens.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=use_amp):
                loss = contrastive_loss(model(images, tokens))
            if not math.isfinite(loss.item()):
                raise RuntimeError(f"Loss became {loss.item()} at epoch {epoch} step {step}")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            model.clamp_logit_scale()
            total_loss += loss.item()
            if step % LOG_EVERY_STEPS == 0 or step == len(train_loader):
                print(f"epoch {epoch} step {step}/{len(train_loader)} loss {loss.item():.4f}", flush=True)
        train_seconds = time.perf_counter() - started

        result = {
            "epoch": epoch,
            "train_loss": total_loss / len(train_loader),
            "train_seconds": train_seconds,
            "seconds_per_step": train_seconds / len(train_loader),
            **evaluate(model, val_loader, device=device, use_amp=use_amp),
        }
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
