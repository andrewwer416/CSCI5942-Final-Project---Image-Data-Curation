"""Download and safely extract the official COCO 2017 archives."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import urllib.request
import zipfile
from typing import Any

from vlm_data.config import DataConfig, DownloadArtifact
from vlm_data.io import sha256_file, write_json_atomic


def _safe_members(archive: zipfile.ZipFile, destination: Path) -> list[zipfile.ZipInfo]:
    root = destination.resolve()
    safe: list[zipfile.ZipInfo] = []
    for member in archive.infolist():
        target = (destination / member.filename).resolve()
        try:
            target.relative_to(root)
        except ValueError as error:
            raise ValueError(f"Unsafe archive path: {member.filename}") from error
        unix_mode = member.external_attr >> 16
        if unix_mode and stat.S_ISLNK(unix_mode):
            raise ValueError(f"Archive contains a symbolic link: {member.filename}")
        safe.append(member)
    return safe


def extract_zip_safely(archive_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        members = _safe_members(archive, destination)
        archive.extractall(destination, members=members)


def _verify_archive(path: Path, *, expected_bytes: int, expected_sha256: str) -> str:
    actual_bytes = path.stat().st_size
    if actual_bytes != expected_bytes:
        raise RuntimeError(
            f"Archive size mismatch for {path.name}: {actual_bytes} bytes; "
            f"expected {expected_bytes}"
        )
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"Archive checksum mismatch for {path.name}: {actual_sha256}; "
            f"expected {expected_sha256}"
        )
    return actual_sha256


def _download(
    url: str,
    destination: Path,
    *,
    expected_bytes: int,
    expected_sha256: str,
) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        print(f"Using existing archive: {destination}")
        return _verify_archive(
            destination,
            expected_bytes=expected_bytes,
            expected_sha256=expected_sha256,
        )

    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "vlm-data-selection/0.1"})
    digest = hashlib.sha256()
    downloaded = 0
    with urllib.request.urlopen(request) as response, temporary.open("wb") as output:
        total_header = response.headers.get("Content-Length")
        total = int(total_header) if total_header else None
        while chunk := response.read(1024 * 1024):
            output.write(chunk)
            digest.update(chunk)
            downloaded += len(chunk)
            if total:
                percent = 100 * downloaded / total
                print(
                    f"\rDownloading {destination.name}: {percent:5.1f}%",
                    end="",
                    file=sys.stderr,
                    flush=True,
                )
    if total:
        print(file=sys.stderr)
        if downloaded != total:
            raise RuntimeError(
                f"Incomplete download for {destination.name}: received {downloaded} of {total} bytes"
            )
    if downloaded != expected_bytes:
        raise RuntimeError(
            f"Archive size mismatch for {destination.name}: received {downloaded} bytes; "
            f"expected {expected_bytes}"
        )
    actual_sha256 = digest.hexdigest()
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"Archive checksum mismatch for {destination.name}: {actual_sha256}; "
            f"expected {expected_sha256}"
        )
    os.replace(temporary, destination)
    return actual_sha256


def download_artifact(config: DataConfig, artifact: DownloadArtifact) -> dict[str, Any]:
    archive_path = config.download_cache / artifact.archive
    expected_path = config.raw_root / artifact.expected_directory
    archive_sha256 = _download(
        artifact.url,
        archive_path,
        expected_bytes=artifact.expected_bytes,
        expected_sha256=artifact.sha256,
    )
    marker_path = config.raw_root / ".extraction-complete" / f"{artifact.name}.json"
    marker_matches = False
    if marker_path.is_file() and expected_path.exists():
        try:
            with marker_path.open("r", encoding="utf-8") as handle:
                marker_matches = json.load(handle).get("archive_sha256") == archive_sha256
        except (OSError, ValueError, TypeError):
            marker_matches = False
    if not marker_matches:
        print(f"Extracting {archive_path.name} into {config.raw_root}")
        extract_zip_safely(archive_path, config.raw_root)
    if not expected_path.exists():
        raise RuntimeError(
            f"Archive extraction completed but expected path is missing: {expected_path}"
        )
    write_json_atomic(
        marker_path,
        {
            "artifact": artifact.name,
            "archive": artifact.archive,
            "archive_sha256": archive_sha256,
        },
    )
    try:
        extracted_path = str(expected_path.relative_to(config.project_root))
    except ValueError:
        extracted_path = str(expected_path)
    return {
        "name": artifact.name,
        "url": artifact.url,
        "archive": archive_path.name,
        "archive_sha256": archive_sha256,
        "extracted_path": extracted_path,
    }


def download_components(config: DataConfig, components: list[str]) -> dict[str, Any]:
    expanded = list(config.downloads) if "all" in components else components
    unknown = sorted(set(expanded) - set(config.downloads))
    if unknown:
        raise ValueError(f"Unknown download components: {', '.join(unknown)}")
    results = [download_artifact(config, config.downloads[name]) for name in dict.fromkeys(expanded)]
    provenance_path = config.raw_root / "download-provenance.json"
    prior_by_name: dict[str, dict[str, Any]] = {}
    if provenance_path.is_file():
        try:
            with provenance_path.open("r", encoding="utf-8") as handle:
                prior = json.load(handle)
            prior_by_name = {item["name"]: item for item in prior.get("artifacts", [])}
        except (OSError, ValueError, TypeError, KeyError):
            prior_by_name = {}
    prior_by_name.update({item["name"]: item for item in results})
    provenance = {
        "schema_version": config.schema_version,
        "artifacts": [prior_by_name[name] for name in sorted(prior_by_name)],
    }
    write_json_atomic(provenance_path, provenance)
    return provenance
