.PHONY: install test lint pipeline serve docker-build docker-run

install:
	python -m pip install -e ".[dev]"

test:
	python -m pytest

lint:
	python -m ruff check src tests

pipeline:
	python -m apexrisk pipeline \
		--config configs/simulation.yaml \
		--features-config configs/features.yaml \
		--output data/generated

serve:
	python -m uvicorn apexrisk.api.main:app --reload

docker-build:
	docker build -t apexrisk:local .

docker-run:
	docker run --rm -p 8000:8000 apexrisk:local

