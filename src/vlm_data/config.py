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
class CategoryBalanceConfig:
    frequency_threshold: float
    max_boost: float
    threshold_candidates: tuple[float, ...]
    max_boost_candidates: tuple[float, ...]
    max_extra_target_fraction: float
    minimum_changed_fraction: float


@dataclass(frozen=True)
class VisualDiversityConfig:
    model_name: str
    pretrained_tag: str
    weight_repository: str
    weight_revision: str
    weight_filename: str
    weight_sha256: str
    input_size: int
    embedding_dimension: int
    neighbor_k_candidates: tuple[int, ...]
    duplicate_similarity_candidates: tuple[float, ...]
    duplicate_precision_target: float
    density_neighbor_count: int
    extreme_outlier_fraction: float
    max_selected_outlier_fraction: float
    minimum_changed_fraction: float
    mean_coverage_relative_tolerance: float
    p95_distance_tolerance: float
    p99_distance_tolerance: float


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
    visual_artifact_dir: Path
    model_cache_dir: Path
    train_captions: Path
    train_instances: Path
    val_captions: Path
    val_instances: Path
    category_balance: CategoryBalanceConfig
    visual_diversity: VisualDiversityConfig
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


def _validate_visual_diversity(config: VisualDiversityConfig) -> None:
    if config.input_size <= 0 or config.embedding_dimension <= 0:
        raise ValueError("visual input size and embedding dimension must be positive")
    if (
        not config.neighbor_k_candidates
        or tuple(sorted(set(config.neighbor_k_candidates)))
        != config.neighbor_k_candidates
        or config.neighbor_k_candidates[0] < 2
    ):
        raise ValueError("visual neighbor k candidates must be unique, sorted, and at least 2")
    if (
        not config.duplicate_similarity_candidates
        or tuple(sorted(set(config.duplicate_similarity_candidates)))
        != config.duplicate_similarity_candidates
        or not all(0.0 < value <= 1.0 for value in config.duplicate_similarity_candidates)
    ):
        raise ValueError(
            "visual duplicate similarity candidates must be unique, sorted, and in (0, 1]"
        )
    fractions = {
        "duplicate precision target": config.duplicate_precision_target,
        "extreme outlier fraction": config.extreme_outlier_fraction,
        "maximum selected outlier fraction": config.max_selected_outlier_fraction,
        "minimum changed fraction": config.minimum_changed_fraction,
    }
    for label, value in fractions.items():
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"visual {label} must be in [0, 1]")
    if config.density_neighbor_count <= 0:
        raise ValueError("visual density neighbor count must be positive")
    if config.mean_coverage_relative_tolerance < 0.0:
        raise ValueError("visual mean coverage tolerance must be nonnegative")
    if config.p95_distance_tolerance < 0.0 or config.p99_distance_tolerance < 0.0:
        raise ValueError("visual distance tolerances must be nonnegative")
    if len(config.weight_sha256) != 64:
        raise ValueError("visual weight SHA-256 must contain 64 hexadecimal characters")
    try:
        int(config.weight_sha256, 16)
    except ValueError as error:
        raise ValueError("visual weight SHA-256 must be hexadecimal") from error


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
    category_balance = raw["category_balance"]
    visual_diversity = raw["visual_diversity"]

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

    visual_config = VisualDiversityConfig(
        model_name=str(visual_diversity["model_name"]),
        pretrained_tag=str(visual_diversity["pretrained_tag"]),
        weight_repository=str(visual_diversity["weight_repository"]),
        weight_revision=str(visual_diversity["weight_revision"]),
        weight_filename=str(visual_diversity["weight_filename"]),
        weight_sha256=str(visual_diversity["weight_sha256"]),
        input_size=int(visual_diversity["input_size"]),
        embedding_dimension=int(visual_diversity["embedding_dimension"]),
        neighbor_k_candidates=tuple(
            int(value) for value in visual_diversity["neighbor_k_candidates"]
        ),
        duplicate_similarity_candidates=tuple(
            float(value) for value in visual_diversity["duplicate_similarity_candidates"]
        ),
        duplicate_precision_target=float(visual_diversity["duplicate_precision_target"]),
        density_neighbor_count=int(visual_diversity["density_neighbor_count"]),
        extreme_outlier_fraction=float(visual_diversity["extreme_outlier_fraction"]),
        max_selected_outlier_fraction=float(
            visual_diversity["max_selected_outlier_fraction"]
        ),
        minimum_changed_fraction=float(visual_diversity["minimum_changed_fraction"]),
        mean_coverage_relative_tolerance=float(
            visual_diversity["mean_coverage_relative_tolerance"]
        ),
        p95_distance_tolerance=float(visual_diversity["p95_distance_tolerance"]),
        p99_distance_tolerance=float(visual_diversity["p99_distance_tolerance"]),
    )
    _validate_visual_diversity(visual_config)

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
        visual_artifact_dir=_resolve(project_root, paths["visual_artifact_dir"]),
        model_cache_dir=_resolve(project_root, paths["model_cache_dir"]),
        train_captions=annotations_dir / annotations["train_captions"],
        train_instances=annotations_dir / annotations["train_instances"],
        val_captions=annotations_dir / annotations["val_captions"],
        val_instances=annotations_dir / annotations["val_instances"],
        category_balance=CategoryBalanceConfig(
            frequency_threshold=float(category_balance["frequency_threshold"]),
            max_boost=float(category_balance["max_boost"]),
            threshold_candidates=tuple(
                float(value) for value in category_balance["threshold_candidates"]
            ),
            max_boost_candidates=tuple(
                float(value) for value in category_balance["max_boost_candidates"]
            ),
            max_extra_target_fraction=float(
                category_balance["max_extra_target_fraction"]
            ),
            minimum_changed_fraction=float(
                category_balance["minimum_changed_fraction"]
            ),
        ),
        visual_diversity=visual_config,
        downloads=downloads,
    )
