"""Package the frozen CLIP train2017 and val2017 caches for P5 runs on another machine.

Embeddings are never committed to Git, so this writes a checksummed bundle instead:

    PYTHONPATH=src pixi run python scripts/package_clip_features.py

Unpack ``artifacts/handoff/clip-features.zip`` into ``data/interim/visual/`` on the training
machine. ``vlm_train.train`` then re-verifies every file hash and the source manifests when a
P5 run loads the caches.
"""

import json
import zipfile

from vlm_data.config import load_config
from vlm_data.io import sha256_file, write_json_atomic
from vlm_data.visual_embeddings import load_embedding_artifact

STEM = "openclip_vit_b16_datacomp_xl_s13b_b90k"


def main():
    c = load_config()
    files = []
    sources = {}
    for split, manifest in (("train2017", c.train_manifest), ("val2017", c.val_manifest)):
        metadata_file = c.visual_artifact_dir / f"{STEM}_{split}.meta.json"
        metadata, _embeddings, _ids = load_embedding_artifact(
            metadata_file, source_manifest=manifest
        )
        if metadata["source"]["split"] != split:
            raise RuntimeError(f"{metadata_file} records split {metadata['source']['split']}")
        sources[split] = {
            "manifest_sha256": metadata["source"]["manifest_sha256"],
            "record_count": metadata["source"]["record_count"],
        }
        files.append(metadata_file)
        for entry in metadata["artifacts"].values():
            files.append(c.visual_artifact_dir / entry["file"])
    checksums = {path.name: sha256_file(path) for path in files}
    output = c.project_root / "artifacts/handoff"
    output.mkdir(parents=True, exist_ok=True)
    archive = output / "clip-features.zip"
    # Float arrays barely compress, so store them as-is.
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as handle:
        for path in files:
            handle.write(path, path.name)
        handle.writestr("SHA256.json", json.dumps(checksums, indent=2, sort_keys=True) + "\n")
    write_json_atomic(
        output / "clip-features.meta.json",
        {
            "archive_sha256": sha256_file(archive),
            "files": checksums,
            "sources": sources,
            "unpack_to": "data/interim/visual/",
        },
    )
    print(
        {
            "archive": str(archive.relative_to(c.project_root)),
            "bytes": archive.stat().st_size,
            "files": len(files),
            "sha256": sha256_file(archive),
        }
    )


if __name__ == "__main__":
    main()
