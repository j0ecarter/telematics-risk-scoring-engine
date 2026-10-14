# ApexRisk: Synthetic-First Telematics Risk Scoring Engine

ApexRisk is an early-stage portfolio project that demonstrates how vehicle journey data can
flow through a reproducible risk-engineering pipeline. The current thin slice generates a
synthetic population, synthetic vehicles and context-aware synthetic trips; builds
exposure-normalised driver-month features; simulates policy exposure and individual claims;
stores the results as Parquet; and exposes a minimal FastAPI health endpoint.

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
Policy-month exposure + synthetic claims
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

The outcome simulator creates one record for every insured driver-month, including months
with no claims. Claim frequency varies stochastically with exposure, behavioural signals,
driver traits and vehicle safety. Ordinary claim costs use a lognormal distribution, while
a small Pareto-mixture component creates occasional heavy-tailed large losses. These are
explicitly configurable simulation assumptions, not empirical insurance estimates.

## Repository layout

```text
configs/                    Simulation, feature, model and pricing assumptions
src/apexrisk/
  api/                      FastAPI application
  features/                 Exposure-normalised behavioural features
  models/                   Poisson frequency baseline and held-out evaluation
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
policy_exposure.parquet
claims.parquet
manifest.json
```

You can also run it directly with explicit paths:

```bash
apexrisk pipeline \
  --config configs/simulation.yaml \
  --features-config configs/features.yaml \
  --claims-config configs/claims.yaml \
  --output data/generated
```

## Frequency baseline and evaluation

After installing the project and generating the data with `make pipeline`, run:

```bash
python -m apexrisk.models.frequency
python -m apexrisk.models.evaluation
```

The frequency model uses previous-month driving features to predict claim counts,
with insured exposure included in a Poisson GLM. It trains on earlier usable months
and holds out the latest month. Evaluation compares its predictions with a
constant annual claim-rate benchmark using portfolio totals, Poisson deviance and
predicted-risk-group calibration. Drivers may appear in both periods; a separate
held-out-driver evaluation is still planned.

`configs/model.yaml` records implementation status only; these scripts do not read
it to configure training. Gamma severity and Tweedie pure-premium models remain
planned. The small default synthetic dataset is a smoke check, not evidence of
reliable predictive performance.

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

## Telemetry assumptions

Version 1 detector settings and planned GPS noise/dropout settings are recorded
in `configs/telemetry.yaml`, with sources and rationale in
[Telemetry assumptions](docs/telemetry_assumptions.md). The detector settings
load through `EventDetectionConfig`; the separate simulation settings are
reserved for the future trace generator and do not yet affect the pipeline.

At one reading per second, brief events can be missed and filtering can reduce
peaks. Detected rates are specific to this sampling and algorithm, not directly
comparable to higher-frequency accelerometer results. The planned speeding
measure counts speed above the limit plus a tolerance; it is an estimate that
omits smaller excesses. Thresholds and GPS errors remain assumptions requiring
later sensitivity checks, not validated real-world safety or pricing measures.

## Next milestones

1. Add second-by-second telemetry generation and robust event detection.
2. Strengthen the existing Poisson frequency baseline and add Gamma severity.
3. Compare the two-part model with a Tweedie pure-premium model.
4. Add grouped and out-of-time validation, calibration and simulation-ground-truth recovery.
5. Implement credibility weighting, uncertainty and an indicative technical-premium endpoint.
6. Add model cards, fairness checks, lineage and drift scenarios.
