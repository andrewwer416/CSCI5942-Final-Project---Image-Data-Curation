PYTHON ?= python3
CONFIG ?= configs/data.toml

.PHONY: test fetch-annotations prepare analyze-category-balance baseline category-balanced validate

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
