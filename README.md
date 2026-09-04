# ApexRisk: Synthetic-First Telematics Risk Scoring Engine

ApexRisk is an early-stage portfolio project that demonstrates how vehicle journey data can
flow through a reproducible risk-engineering pipeline. The current thin slice generates a
synthetic population, synthetic vehicles and context-aware synthetic trips; stores the
results as Parquet; builds exposure-normalised driver-month features; and exposes a minimal
FastAPI health endpoint.

> **Important:** every dataset produced by this repository is synthetic. The project is a
> methodological and software-engineering demonstration. It is not evidence of real-world
> predictive accuracy, an actuarially adequate rating model, or a production insurance quote.

## What works today

```text
Simulation configuration
        ↓
Synthetic drivers + private latent traits
        ↓
Synthetic vehicles and context-aware trips
        ↓
Trip behavioural features
        ↓
Driver-month behavioural features
        ↓
Parquet tables + synthetic-data manifest

FastAPI /health endpoint (separate deployment check)
```

The generator uses unobserved traits such as aggressiveness and attentiveness to create
coherent relationships between people, journey context and driving behaviour. Those private
traits are deliberately excluded from the published Parquet tables so future models cannot
learn directly from the simulation's hidden ground truth.

Initial features include:

- hard braking, hard acceleration and harsh cornering events per 100 km;
- time-weighted night-driving fraction;
- time-weighted speeding fraction; and
- mean speeding excess while speeding.

## Repository layout

```text
configs/                    Simulation, feature, model and pricing assumptions
src/apexrisk/
  api/                      FastAPI application
  features/                 Exposure-normalised behavioural features
  models/                   Placeholder for actuarial models
  pricing/                  Placeholder for technical-premium logic
  simulation/               Driver, vehicle and journey generators
  cli.py                    Command-line interface
  pipeline.py               Thin end-to-end synthetic pipeline
tests/                      Unit and integration tests
Dockerfile                 Reproducible API container
```

There is intentionally no Airflow integration. The core workflow is ordinary Python and can
later be scheduled by any suitable platform without moving business logic into an orchestrator.
A `docker-compose.yml` is also intentionally omitted: this version has a single API service, so
Compose would not simplify its operation.

## Quick start

Python 3.11 or later is required.

```bash
python -m venv .venv
source .venv/bin/activate
make install
make test
make pipeline
```

The pipeline writes the following ignored local artefacts to `data/generated/`:

```text
drivers.parquet
vehicles.parquet
trips.parquet
trip_features.parquet
driver_month_features.parquet
manifest.json
```

You can also run it directly with explicit paths:

```bash
apexrisk pipeline \
  --config configs/simulation.yaml \
  --features-config configs/features.yaml \
  --output data/generated
```

## API and Docker

Run the API locally:

```bash
make serve
curl http://localhost:8000/health
```

Or build and run the container:

```bash
make docker-build
make docker-run
```

The health response explicitly identifies the service as synthetic and not production pricing:

```json
{
  "status": "ok",
  "service": "apexrisk",
  "version": "0.1.0",
  "data_mode": "synthetic",
  "production_pricing": false
}
```

## Methodological boundaries

The current simulator is suitable for verifying data contracts, feature calculations,
referential integrity, deterministic runs and deployment mechanics. It must not be used to
claim that a fitted model predicts genuine accidents, claims or premiums. Model performance
on data generated from known rules is primarily a test of whether the implementation can
recover those rules.

Future validation should compare selected feature distributions with credible external
benchmarks or appropriately licensed real data. Assumptions should remain versioned and
reviewable in `configs/`.

## Next milestones

1. Add second-by-second telemetry generation and robust event detection.
2. Generate policy exposure, accidents, claims and heavy-tailed claim severity separately.
3. Train exposure-aware Poisson frequency and Gamma severity baselines.
4. Compare the two-part model with a Tweedie pure-premium model.
5. Add grouped and out-of-time validation, calibration and simulation-ground-truth recovery.
6. Implement credibility weighting, uncertainty and an indicative technical-premium endpoint.
7. Add model cards, fairness checks, lineage and drift scenarios.

