"""Package validated selection manifests and metadata for collaborator handoff."""

import json
import zipfile

from vlm_data.config import load_config
from vlm_data.io import sha256_file, write_json_atomic
from vlm_data.manifest import metadata_path
from vlm_data.validation import validate_project


def main():
    c = load_config()
    if not validate_project(c)["valid"]:
        raise RuntimeError("Selection validation failed")
    output = c.project_root / "artifacts/handoff"
    output.mkdir(parents=True, exist_ok=True)
    files = []
    for path in sorted(c.selection_dir.glob("*.jsonl")):
        files.extend([path, metadata_path(path)])
    checksums = {path.name: sha256_file(path) for path in files}
    archive = output / "selection-manifests.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        for path in files:
            handle.write(path, path.name)
        handle.writestr("SHA256.json", json.dumps(checksums, indent=2, sort_keys=True) + "\n")
    write_json_atomic(
        output / "selection-manifests.meta.json",
        {
            "archive_sha256": sha256_file(archive),
            "files": checksums,
            "source_manifest_sha256": sha256_file(c.train_manifest),
        },
    )
    print(
        {
            "archive": str(archive),
            "bytes": archive.stat().st_size,
            "files": len(files),
            "sha256": sha256_file(archive),
        }
    )


if __name__ == "__main__":
    main()
