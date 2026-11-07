PYTHON ?= python3
CONFIG ?= configs/data.toml

.PHONY: test fetch-annotations prepare baseline validate

test:
	PYTHONPATH=src $(PYTHON) -m unittest discover -s tests -v

fetch-annotations:
	PYTHONPATH=src $(PYTHON) -m vlm_data --config $(CONFIG) download annotations

prepare:
	PYTHONPATH=src $(PYTHON) -m vlm_data --config $(CONFIG) prepare

baseline:
	PYTHONPATH=src $(PYTHON) -m vlm_data --config $(CONFIG) select --method random

validate:
	PYTHONPATH=src $(PYTHON) -m vlm_data --config $(CONFIG) validate
