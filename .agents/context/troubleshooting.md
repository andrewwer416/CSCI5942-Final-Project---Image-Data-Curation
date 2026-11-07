# Troubleshooting

## `Required COCO annotation file is missing`

Run `make fetch-annotations`. If files were downloaded manually, the four JSON files must be under
`data/raw/coco/annotations/` with their official COCO 2017 names.

## `Canonical training manifest is missing`

Run `make prepare` before `make baseline`.

## Validation reports missing images

`--check-files` requires extracted `train2017/` and `val2017/` directories. Download them with
`vlm-data download train val --confirm-large-download`; these archives require substantial disk
space.

## Manifest hash mismatch

Treat the manifest as modified or corrupt. Regenerate it from the immutable annotations rather than
editing JSONL rows or their metadata sidecars manually.

## Import error without installation

Either install the package in a virtual environment with `python -m pip install -e .` or prefix the
command with `PYTHONPATH=src`.
