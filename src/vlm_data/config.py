"""Load the project's TOML configuration without third-party dependencies."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib


@dataclass(frozen=True)
class DownloadArtifact:
    name: str
    url: str
    archive: str
    expected_directory: str
    expected_bytes: int
    sha256: str


@dataclass(frozen=True)
class DataConfig:
    project_root: Path
    schema_version: int
    seed: int
    train_image_budget: int
    epochs: int
    image_size: int
    captions_per_image: int
    expected_train_images: int
    expected_val_images: int
    raw_root: Path
    download_cache: Path
    annotations_dir: Path
    train_images_dir: Path
    val_images_dir: Path
    manifest_dir: Path
    selection_dir: Path
    report_dir: Path
    train_captions: Path
    train_instances: Path
    val_captions: Path
    val_instances: Path
    downloads: dict[str, DownloadArtifact]

    @property
    def train_manifest(self) -> Path:
        return self.manifest_dir / "train2017.jsonl"

    @property
    def val_manifest(self) -> Path:
        return self.manifest_dir / "val2017.jsonl"

    def baseline_manifest(self) -> Path:
        return self.selection_dir / (
            f"random_seed{self.seed}_n{self.train_image_budget}.jsonl"
        )


def _resolve(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def load_config(path: str | Path = "configs/data.toml") -> DataConfig:
    config_path = Path(path).resolve()
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)

    # The checked-in config lives in <project>/configs. Keeping path resolution
    # anchored here makes commands independent of the caller's working directory.
    project_root = config_path.parent.parent
    experiment = raw["experiment"]
    paths = raw["paths"]
    annotations = raw["annotations"]

    annotations_dir = _resolve(project_root, paths["annotations_dir"])
    downloads = {
        name: DownloadArtifact(
            name=name,
            url=values["url"],
            archive=values["archive"],
            expected_directory=values["expected_directory"],
            expected_bytes=int(values["expected_bytes"]),
            sha256=str(values["sha256"]),
        )
        for name, values in raw["downloads"].items()
    }

    return DataConfig(
        project_root=project_root,
        schema_version=int(raw["schema_version"]),
        seed=int(experiment["seed"]),
        train_image_budget=int(experiment["train_image_budget"]),
        epochs=int(experiment["epochs"]),
        image_size=int(experiment["image_size"]),
        captions_per_image=int(experiment["captions_per_image"]),
        expected_train_images=int(experiment["expected_train_images"]),
        expected_val_images=int(experiment["expected_val_images"]),
        raw_root=_resolve(project_root, paths["raw_root"]),
        download_cache=_resolve(project_root, paths["download_cache"]),
        annotations_dir=annotations_dir,
        train_images_dir=_resolve(project_root, paths["train_images_dir"]),
        val_images_dir=_resolve(project_root, paths["val_images_dir"]),
        manifest_dir=_resolve(project_root, paths["manifest_dir"]),
        selection_dir=_resolve(project_root, paths["selection_dir"]),
        report_dir=_resolve(project_root, paths["report_dir"]),
        train_captions=annotations_dir / annotations["train_captions"],
        train_instances=annotations_dir / annotations["train_instances"],
        val_captions=annotations_dir / annotations["val_captions"],
        val_instances=annotations_dir / annotations["val_instances"],
        downloads=downloads,
    )
