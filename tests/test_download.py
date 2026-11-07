from contextlib import redirect_stderr, redirect_stdout
import io
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
import zipfile

from vlm_data.config import DownloadArtifact
from vlm_data.download import download_artifact, extract_zip_safely
from vlm_data.io import sha256_file


class SafeExtractionTests(unittest.TestCase):
    def test_extracts_normal_archive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "safe.zip"
            destination = root / "output"
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr("annotations/example.json", "{}")
            extract_zip_safely(archive, destination)
            self.assertEqual((destination / "annotations/example.json").read_text(), "{}")

    def test_rejects_parent_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "unsafe.zip"
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr("../outside.txt", "not allowed")
            with self.assertRaises(ValueError):
                extract_zip_safely(archive, root / "output")

    def test_existing_partial_directory_is_reextracted_without_marker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw"
            cache = raw / "downloads"
            cache.mkdir(parents=True)
            (raw / "annotations").mkdir()
            archive = cache / "annotations.zip"
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr("annotations/complete.json", "{}")
            config = SimpleNamespace(
                raw_root=raw,
                download_cache=cache,
                project_root=root,
            )
            artifact = DownloadArtifact(
                name="annotations",
                url="https://invalid.example/unused",
                archive=archive.name,
                expected_directory="annotations",
                expected_bytes=archive.stat().st_size,
                sha256=sha256_file(archive),
            )
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                download_artifact(config, artifact)
            self.assertTrue((raw / "annotations/complete.json").is_file())
            self.assertTrue((raw / ".extraction-complete/annotations.json").is_file())

    def test_existing_archive_must_match_pinned_checksum(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw"
            cache = raw / "downloads"
            cache.mkdir(parents=True)
            archive = cache / "annotations.zip"
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr("annotations/example.json", "{}")
            config = SimpleNamespace(
                raw_root=raw,
                download_cache=cache,
                project_root=root,
            )
            artifact = DownloadArtifact(
                name="annotations",
                url="https://invalid.example/unused",
                archive=archive.name,
                expected_directory="annotations",
                expected_bytes=archive.stat().st_size,
                sha256="0" * 64,
            )
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, "checksum mismatch"):
                    download_artifact(config, artifact)


if __name__ == "__main__":
    unittest.main()
