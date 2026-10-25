# ApexRisk: Synthetic-First Telematics Risk Scoring Engine

ApexRisk is an early-stage portfolio project that demonstrates how vehicle journey data can
flow through a reproducible risk-engineering pipeline. The current thin slice generates a
synthetic population, synthetic vehicles and context-aware synthetic trips; builds
exposure-normalised driver-month features; simulates policy exposure and individual claims;
stores the results as Parquet; and provides a Poisson claim-frequency baseline with
held-out evaluation. A separate, tested telemetry detector counts driving events from
one-second speed and heading samples. A minimal FastAPI health endpoint supports
service checks.

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

Separate telemetry detector (not yet connected to the pipeline):
1 Hz speed + heading + GPS validity → preprocessing → event counts + speeding seconds

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

The pipeline currently simulates trip event counts and speeding summaries directly.
It does not yet generate second-by-second telemetry or call the telemetry detector.

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
  telemetry/                1 Hz data validation, preprocessing and event detection
  cli.py                    Command-line interface
  pipeline.py               Thin end-to-end synthetic pipeline
tests/                      Unit and integration tests
docs/                       Telemetry contract and assumption rationale
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
held-out-driver evaluation is a possible future extension.

`configs/model.yaml` records implementation status only; these scripts do not read
it to configure training. Gamma severity and Tweedie pure-premium models are outside
the current completion scope. The small default synthetic dataset is a smoke check,
not evidence of reliable predictive performance.

## Telemetry detection and tests

`apexrisk.telemetry.detect_events` accepts speed and heading arrays, a GPS validity
mask, a speed limit and an `EventDetectionConfig`. It returns braking, acceleration
and cornering counts in mutually exclusive moderate, harsh and severe bands, plus
speeding seconds. Each event belongs only to its highest severity band.

Preprocessing handles smoothing, heading wrap and segment-edge trimming. GPS gaps
split the trace into independent valid segments, so events cannot bridge missing
data. Duration and separation rules suppress isolated spikes and merge nearby
qualifying runs; a minimum-speed gate applies to cornering.

See the [telemetry contract](docs/telemetry_contract.md) for units, validation and
exposure rules. Run the full suite or the detector checks alone with:

```bash
make test
python -m pytest tests/test_detection.py
```

As verified on 25 October 2026, all 113 project tests pass, including 57 detector
checks and 10 demo-generator checks. Coverage includes severity boundaries, duration and separation rules,
GPS gaps, speeding tolerance, heading wrap, invalid inputs and deterministic
behaviour. CI runs the full test suite and Ruff.

`apexrisk.telemetry.demo.build_demo_scenarios()` now creates six deterministic,
three-minute traces: steady driving, braking, acceleration, cornering, speeding
and GPS dropout. Each scenario includes its name, description, speed limit and a
validated telemetry table. The generator does not write files or run detection.
Expected-result checks, saved demo outputs and a CLI command remain to be added
before the telemetry demonstration is complete.

## API and Docker

Run the API locally:

```bash
make serve
```

With the server running, check it from a second terminal:

```bash
curl http://localhost:8000/health
```

Or build and run the container:

```bash
make docker-build
make docker-run
```

Use the same `curl` command from a second terminal to check the container.
The Dockerfile and commands exist; end-to-end Docker verification remains a
completion task. `/health` reports service liveness, and `/docs` provides generated
API documentation. There is no journey-analysis or risk-scoring endpoint.

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
comparable to higher-frequency accelerometer results. The implemented speeding
measure counts valid, smoothed speed samples strictly above the limit plus a
tolerance; it is an estimate that omits smaller excesses. Thresholds and GPS errors
remain assumptions requiring
later sensitivity checks, not validated real-world safety or pricing measures.

## Completion roadmap

The current finish line is a reproducible portfolio demonstration with tested
event detection, a verified API container, and clear results and limitations.

- [x] **01 — Event counting:** implement banded braking, acceleration and cornering
  counts, plus speeding seconds, with GPS-gap handling.
- [x] **02 — Detector tests:** verify normal, boundary and regression cases and
  include all detector tests in CI.
- [ ] **03 — Deterministic telemetry demonstration:** generate a steady-driving
  baseline and controlled braking, acceleration, cornering, speeding and GPS-dropout
  scenarios; compare expected and detected results; provide a reproducible command
  and saved traces/results. Use a fixed seed wherever randomness is introduced.
- [ ] **04 — Docker verification:** build and run the container, confirm `/health`
  and `/docs` respond, and record the reproducible checks.
- [ ] **05 — Results and limitations:** document demonstrated behaviour, model
  results, synthetic-data assumptions and sampling/detection limitations.
- [ ] **06 — Repository polish:** review setup commands, documentation and examples
  against the finished implementation, and confirm tests and lint pass.

A steady-driving baseline is a controlled reference, not proof of real-world
safety. No detected events does not establish safe following distance, attention,
or speed appropriate to weather and road conditions.

Full telemetry integration into the population pipeline, additional insurance
models, credibility weighting, pricing endpoints, fairness/drift analysis and
cloud deployment are possible future extensions, not requirements for this finish line.
