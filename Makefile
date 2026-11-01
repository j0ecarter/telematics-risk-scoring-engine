.PHONY: install test lint pipeline serve docker-build docker-run

DOCKER_PORT ?= 8003

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
		--claims-config configs/claims.yaml \
		--output data/generated

serve:
	python -m uvicorn apexrisk.api.main:app --reload

docker-build:
	docker build -t apexrisk:local .

docker-run:
	docker run --rm -p 127.0.0.1:$(DOCKER_PORT):8000 apexrisk:local
