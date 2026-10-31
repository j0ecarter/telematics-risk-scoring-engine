# ApexRisk: Synthetic Telematics Demonstrator

ApexRisk is a Python portfolio project demonstrating **1 Hz driving-event
detection**, a reproducible synthetic-data pipeline and an exposure-aware Poisson
claim-frequency baseline. Its main demo runs controlled journeys through the
detector, compares expected and actual results, and saves the evidence.

All data is synthetic. Results demonstrate the implementation under documented
assumptions; they do not establish real-world accident prediction or insurance
pricing accuracy.

## Current scope and status

The finish line is a reproducible telemetry demonstration, a verified API
container, and clear results and limitations. See [Project scope](docs/scope.md)
for completion criteria and deferred work.

| Component | Current behaviour |
|---|---|
| Telemetry detector | Processes 1 Hz speed, heading and GPS validity; counts braking, acceleration and cornering by severity, plus speeding seconds |
| Deterministic demo | Six controlled scenarios, fixed expected results, Parquet traces and a JSON comparison summary |
| Synthetic population pipeline | Generates drivers, vehicles, trip summaries, behavioural features, policy exposure and claims |
| Frequency baseline | Poisson GLM using previous-month behaviour and insured exposure; latest-month evaluation against a constant-rate benchmark |
| API | `/health` liveness check and generated `/docs`; no journey-analysis or scoring endpoint |
| Docker | Single-service Dockerfile and commands; end-to-end container verification remains open |

The demo and population pipeline are separate workflows. The population pipeline
simulates event counts directly; it does **not** run the telemetry detector.
Pricing, additional insurance models, population-wide telemetry integration and
cloud deployment are optional future extensions.

## Quick start

Use Python 3.11 or later, from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
make install
make lint
make test
python -m apexrisk telemetry-demo
```

The demo prints:

```text
PASS: steady
PASS: braking
PASS: acceleration
PASS: cornering
PASS: speeding
PASS: gps_dropout
6/6 scenarios passed.
Results saved to: data/generated/telemetry_demo/summary.json
```

It writes six observed-telemetry Parquet files and `summary.json`. The summary
records settings, expected and actual counts, mismatches and valid GPS exposure.
Repeated runs with the same settings produce the same traces and summary.

To choose paths explicitly:

```bash
python -m apexrisk telemetry-demo \
  --config configs/telemetry.yaml \
  --output data/generated/telemetry_demo
```

The installed `apexrisk telemetry-demo` command works too. Exit codes are `0` for
all comparisons passing, `1` for mismatches and `2` for usage, configuration or
file errors. See [Demo usage, results and limitations](docs/telemetry_demo.md)
for the Python interface, saved outputs and failure examples.

## Synthetic pipeline and frequency baseline

```bash
make pipeline
python -m apexrisk.models.frequency
python -m apexrisk.models.evaluation
```

The pipeline writes these ignored local outputs to `data/generated/`:

```text
drivers.parquet              vehicles.parquet
trips.parquet                trip_features.parquet
driver_month_features.parquet
policy_exposure.parquet      claims.parquet
manifest.json
```

The manifest records simulation assumptions, row counts and synthetic-data status.
Private latent driver traits influence generation but are excluded from the
published tables. Features include events per 100 km, night-driving fraction,
speeding-time fraction and mean speeding excess while speeding.

The baseline predicts synthetic claim counts from previous-month behaviour,
using insured exposure in a Poisson GLM. It trains on earlier usable months and
holds out the latest month. Evaluation reports portfolio totals, Poisson deviance
and calibration by predicted risk group against a constant annual claim-rate
benchmark. Drivers can appear in both periods; this is not a held-out-driver test.

The default dataset is a small smoke check. Recovering synthetic relationships
does not demonstrate real-world predictive performance. `configs/model.yaml`
records status only; the model scripts do not load it as training configuration.

## API and Docker

Start the local API:

```bash
make serve
```

From another terminal:

```bash
curl --fail http://localhost:8000/health
curl --fail http://localhost:8000/docs
```

The health response is:

```json
{
  "status": "ok",
  "service": "apexrisk",
  "version": "0.1.0",
  "data_mode": "synthetic",
  "production_pricing": false
}
```

To build and run the same API in Docker:

```bash
make docker-build
make docker-run
```

Run the two `curl` checks again from another terminal. The Dockerfile is present;
a successful build and running-container check still need to be recorded before
the Docker milestone is complete. A healthy service does not mean the telemetry
detector or model is exposed through the API.

## Repository layout

```text
configs/                    Active assumptions and labelled future design settings
src/apexrisk/
  api/                      Health-check API
  telemetry/                Schema, preprocessing, event detection and fixed demo
  simulation/               Synthetic drivers, vehicles, trip summaries and claims
  features/                 Trip and driver-month behavioural features
  models/                   Poisson frequency baseline and evaluation
  pricing/                  Unimplemented future-extension placeholder
  cli.py                    Pipeline and telemetry-demo commands
  pipeline.py               Population simulation and Parquet export
tests/                      Unit, integration and CLI checks
docs/                       Scope, demo, telemetry contract and assumptions
Dockerfile                  API container definition
```

There is one API service. Airflow, Compose and cloud infrastructure are outside
the current scope.

## Assumptions and limits

Detector settings live in `configs/telemetry.yaml`. The
[assumption rationale](docs/telemetry_assumptions.md) distinguishes reference
magnitudes from project choices. Each manoeuvre is counted once in its highest
severity band. GPS gaps split valid segments before processing or merging events.
Speeding counts valid, smoothed samples strictly above the limit plus a tolerance;
that tolerance is an engineering setting, not a legal allowance.

At 1 Hz, brief events can be missed and filtering can reduce peaks. Thresholds,
synthetic claims and GPS-error design settings are assumptions, not validated
safety or pricing measures. No detected events does not establish safe driving.
The fixed demo has no random noise; detector tests cover noise and boundary cases
separately.

The [telemetry contract](docs/telemetry_contract.md) defines existing schema and
detector rules and also retains the design for a deferred population trace
generator. Its full-population integration and sample-cohort sections describe
future work, not current pipeline behaviour. Neither the demo nor the population
pipeline uses the reserved `telemetry_simulation` settings.

CI runs the full test suite and Ruff on pushes to `main` and on pull requests.
To run focused demo checks:

```bash
python -m pytest tests/test_detection.py tests/test_telemetry_demo.py tests/test_cli.py
```
