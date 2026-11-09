PYTHON ?= python3
CONFIG ?= configs/data.toml
PIXI ?= pixi
VISUAL_PRECISION ?= amp_fp16
VISUAL_BATCH_SIZE ?= 256
VISUAL_WORKERS ?= 8

.PHONY: test fetch-annotations prepare analyze-category-balance baseline category-balanced \
	visual-stack-smoke visual-benchmark visual-embeddings visual-neighbors \
	caption-semantic caption-diagnostics validate

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

visual-stack-smoke:
	$(PIXI) run visual-stack-smoke

visual-benchmark:
	PYTHONPATH=src $(PIXI) run python -m vlm_data --config $(CONFIG) benchmark-visual

visual-embeddings:
	PYTHONPATH=src $(PIXI) run python -m vlm_data --config $(CONFIG) embed-visual \
		--inference-precision $(VISUAL_PRECISION) --batch-size $(VISUAL_BATCH_SIZE) \
		--num-workers $(VISUAL_WORKERS)

visual-neighbors:
	PYTHONPATH=src $(PIXI) run python -m vlm_data --config $(CONFIG) build-visual-neighbors

caption-semantic:
	PYTHONPATH=src $(PIXI) run python scripts/calibrate_caption.py

caption-diagnostics:
	PYTHONPATH=src $(PIXI) run python scripts/diagnose_caption.py

validate:
	PYTHONPATH=src $(PYTHON) -m vlm_data --config $(CONFIG) validate
