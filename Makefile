PYTHON ?= python3
CONFIG ?= configs/data.toml

.PHONY: test fetch-annotations prepare analyze-category-balance baseline category-balanced validate smoke-train

test:
	PYTHONPATH=src $(PYTHON) -m unittest discover -s tests -v

fetch-annotations:
	PYTHONPATH=src $(PYTHON) -m vlm_data --config $(CONFIG) download annotations

prepare:
	PYTHONPATH=src $(PYTHON) -m vlm_data --config $(CONFIG) prepare

analyze-category-balance:
	PYTHONPATH=src $(PYTHON) -m vlm_data --config $(CONFIG) analyze-category-balance

baseline:
	PYTHONPATH=src $(PYTHON) -m vlm_data --config $(CONFIG) select --method random

category-balanced:
	PYTHONPATH=src $(PYTHON) -m vlm_data --config $(CONFIG) select --method category-balanced

validate:
	PYTHONPATH=src $(PYTHON) -m vlm_data --config $(CONFIG) validate

smoke-train:
	PYTHONPATH=src $(PYTHON) -m vlm_train.train --config $(CONFIG) \
		--selection data/processed/selections/random_seed5942_n15000.jsonl \
		--image-root data/raw/coco_smoke --limit 512 --val-limit 128 \
		--epochs 1 --image-size 224 --batch-size 8 --output-dir artifacts/smoke
