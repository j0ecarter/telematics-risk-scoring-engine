# ApexRisk: Synthetic Telematics Demonstrator

ApexRisk is a Python portfolio project demonstrating **1 Hz driving-event
detection**, a reproducible synthetic-data pipeline and an exposure-aware Poisson
claim-frequency baseline. Its main demo runs controlled journeys through the
detector, compares expected and actual results, and saves the evidence.

All data is synthetic. Results demonstrate the implementation under documented
assumptions; they do not establish real-world accident prediction or insurance
pricing accuracy.

## Current scope and status

The agreed portfolio demonstration is complete: a reproducible telemetry demo,
a validated journey-analysis API, a verified container, and documented results
and limitations. See [Project scope](docs/scope.md)
for completion criteria and deferred work.

| Component | Current behaviour |
|---|---|
| Telemetry detector | Processes 1 Hz speed, heading and GPS validity; counts braking, acceleration and cornering by severity, plus speeding seconds |
| Deterministic demo | Six controlled scenarios, fixed expected results, Parquet traces and a JSON comparison summary |
| Synthetic population pipeline | Generates drivers, vehicles, trip summaries, behavioural features, policy exposure and claims |
| Frequency baseline | Poisson GLM using previous-month behaviour and insured exposure; latest-month evaluation against a constant-rate benchmark |
| API | `POST /journeys/analyse` validates one small 1 Hz trace and returns detector counts; `/health` and interactive `/docs` included |
| Docker | Image build, health/docs, valid and invalid journey requests, six-scenario detector agreement and container shutdown verified on 1 November 2026 |

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

See [Results and limitations](docs/results.md) for the dated evidence, held-out
benchmark comparison and calibration. On the default run, the Poisson model does
not beat the constant-rate benchmark; the held-out month contains only one claim.

## API and Docker

Run commands from the repository root. Start the local API with the project's
environment active:

```bash
make serve
```

Open [interactive documentation](http://localhost:8000/docs). Expand
`POST /journeys/analyse`, click **Try it out**, then **Execute** to send the supplied
ten-second speeding example. The health check is `GET /health`.

The same request is saved in [examples/journey_speeding.json](examples/journey_speeding.json):

```bash
curl --fail-with-body http://localhost:8000/journeys/analyse \
  -H 'Content-Type: application/json' \
  --data-binary @examples/journey_speeding.json
```

Expected response:

```json
{
  "brake_events_moderate": 0,
  "brake_events_harsh": 0,
  "brake_events_severe": 0,
  "accel_events_moderate": 0,
  "accel_events_harsh": 0,
  "accel_events_severe": 0,
  "corner_events_moderate": 0,
  "corner_events_harsh": 0,
  "corner_events_severe": 0,
  "speeding_seconds": 10
}
```

The request contains `speed_limit_kmh`, `speed_kmh`, `heading_degrees` and
`is_gps_valid`. Arrays must have equal lengths of 1–600 samples, with each position
representing one second. Numbers must be finite; speed and speed limit must be
non-negative, and headings must be in `[0, 360)`. A missing GPS sample stays in
the arrays with `false` validity and `null` speed and heading. Unexpected fields
and invalid types are rejected. Invalid requests return HTTP 422 with error
locations and messages.

The endpoint uses the same detector and `configs/telemetry.yaml` settings as the
Python demo. It returns event counts by severity and speeding seconds, without
event timestamps, claim predictions, premiums or a real-world safety rating.

Build and start the packaged API:

```bash
make docker-build
make docker-run
```

Docker publishes to your computer's loopback address on port **8003**, while the
application listens on port 8000 inside the container. Open
[container documentation](http://localhost:8003/docs), or check from another terminal:

```bash
curl --fail http://localhost:8003/health
curl --fail-with-body http://localhost:8003/journeys/analyse \
  -H 'Content-Type: application/json' \
  --data-binary @examples/journey_speeding.json
```

Change an occupied host port with `make docker-run DOCKER_PORT=8004` and use that
port in the URLs. Stop the foreground container with **Ctrl+C**; `--rm` removes it.
For local development on another port, use
`python -m uvicorn apexrisk.api.main:app --reload --port 8002`.

The final image, API requests and detector agreement were checked on
**1 November 2026**. See [recorded verification](docs/results.md#docker-verification)
and [machine-readable completion evidence](docs/evidence/completion-2026-11-01.json).

## Repository layout

```text
configs/                    Active assumptions and labelled future design settings
src/apexrisk/
  api/                      Validated single-journey analysis and health API
  telemetry/                Schema, preprocessing, event detection and fixed demo
  simulation/               Synthetic drivers, vehicles, trip summaries and claims
  features/                 Trip and driver-month behavioural features
  models/                   Poisson frequency baseline and evaluation
  pricing/                  Unimplemented future-extension placeholder
  cli.py                    Pipeline and telemetry-demo commands
  pipeline.py               Population simulation and Parquet export
tests/                      Unit, integration and CLI checks
docs/                       Scope, results, saved evidence and detector assumptions
examples/                   Runnable journey-analysis JSON request
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

## Potential improvements

After the [current completion milestones](docs/scope.md) are finished, the following
targets could extend the project. They are optional directions, ordered from
smaller improvements towards larger extensions, rather than additional completion
requirements.

| Area | Improvement target | Evidence of completion |
|---|---|---|
| Detector robustness | Vary smoothing, thresholds, noise and GPS dropout on controlled traces | A reproducible sensitivity report showing false detections, missed events and merged events against known expectations |
| Journey visualisation | Plot speed, heading, GPS gaps and detected manoeuvre intervals | A generated report that makes each demo result inspectable alongside its input trace |
| Container checks | Automate image build and service checks in CI | A clean CI run starts the container and verifies `/health` and `/docs` |
| Pipeline integration | Generate observed telemetry per trip and derive population trip summaries through the detector | Seeded runs reconcile distance and duration, use valid GPS exposure, and process traces without retaining the full population's telemetry |
| Model validation | Evaluate larger synthetic samples across multiple seeds and add a held-out-driver split | Report Poisson deviance, calibration and uncertainty against the constant-rate benchmark for both time and driver holdouts |
| External validation | Compare with appropriately licensed real journeys or compatible published benchmarks | Document data provenance, sampling and feature compatibility, and where synthetic assumptions disagree with observed data |
| API exposure reporting | Return valid GPS time, observed distance and coverage alongside event counts | Response tests reconcile the exposure fields with `summarise_valid_exposure` |

Further research could explore Gamma severity and Tweedie pure-premium models,
credibility weighting, fairness and drift. These would need their own evaluation
criteria before becoming project commitments. Pricing or deployment extensions
would not, by themselves, establish real-world predictive accuracy or production
readiness.

## Licence

ApexRisk is distributed under the [MIT licence](LICENSE).
