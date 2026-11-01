# Project scope and completion criteria

This is the current completion scope for ApexRisk. Earlier insurance-engine and
population-telemetry plans are retained as extension ideas, not requirements to
finish this portfolio demonstration.

## Purpose

Demonstrate a small, inspectable workflow from observed synthetic journey readings
to detected driving events, with repeatable checks and saved evidence. Retain the
existing synthetic population pipeline and Poisson claim-frequency baseline as a
separate modelling demonstration.

Neither workflow establishes real-world safety, accident risk or pricing accuracy.

## Completion milestones

| Milestone | Status | Completion evidence |
|---|---|---|
| 01 — Event counting | Implemented | Banded braking, acceleration and cornering counts; speeding seconds; GPS-gap handling |
| 02 — Detector tests | Implemented | Normal, boundary, noise and regression cases included in the full CI suite |
| 03 — Deterministic telemetry demo | Implemented | Six fixed scenarios; expected/actual comparisons; one command; saved Parquet traces and JSON results |
| 04 — Docker verification | Verified | Updated API image, health/docs, successful and invalid journey requests, six-scenario detector agreement and shutdown checked on 1 November 2026; [evidence](results.md#docker-verification) |
| 05 — Results and limitations | Complete | [Dated results](results.md), saved evidence, held-out benchmark comparison, calibration and synthetic-data/sampling limits |
| 06 — Repository polish | Complete | Setup and Docker commands aligned; runnable request and MIT licence included; local links, 196 tests and Ruff checked; [completion evidence](evidence/completion-2026-11-01.json) |
| Journey-analysis API | Complete | One 1 Hz trace, bounded request validation, counts from the existing detector, HTTP success/error and parity tests, executable `/docs` example |

The agreed synthetic portfolio demonstration is complete. Completion records
observed verification; it does not establish production insurance suitability.

## Boundaries between workflows

```text
Fixed 1 Hz demo traces → schema validation → detector → comparison summary + traces

Synthetic population → simulated trip summaries → behavioural features
                     → policy exposure + claims → Poisson model + evaluation

FastAPI POST /journeys/analyse → validated arrays → existing detector → event counts
FastAPI /health + /docs → liveness and executable API documentation
```

The population pipeline generates trip counts and speeding summaries directly.
It does not call the detector. The separate journey-analysis API now calls the
existing detector for one request containing 1–600 consecutive 1 Hz sample slots.
It returns severity-banded event counts and speeding seconds. It does not return
claim predictions, premiums, event timestamps or GPS-exposure fields.

The Poisson baseline uses a latest-month holdout. It does not establish generalisation
to unseen drivers, real claims or a different generator. Gamma severity, Tweedie
pure premium and grouped-driver validation are optional future extensions.

## Configuration ownership

| File | Current use |
|---|---|
| `configs/simulation.yaml` | Population, time range and random seed for the synthetic pipeline |
| `configs/features.yaml` | Minimum monthly distance for behavioural rates |
| `configs/claims.yaml` | Synthetic claim-frequency and severity assumptions |
| `configs/telemetry.yaml`: `telemetry` | Detector settings used by the demo and journey-analysis API |
| `configs/telemetry.yaml`: `telemetry_simulation` | Reserved design assumptions; unused by the demo and pipeline |
| `configs/model.yaml` | Status metadata; model scripts do not read it |
| `configs/pricing.yaml` | Unused future pricing design assumptions |

The frozen [telemetry contract](telemetry_contract.md) still governs the implemented
schema and detector. Its population trace-generation, sample-cohort and pipeline
integration design remains deferred; this scope update does not change that contract.

## Optional future extensions

- Generate noisy population traces and derive trip summaries through the detector.
- Add sensitivity studies, compatible real-data benchmarks and held-out-driver checks.
- Extend insurance modelling with severity or pure-premium baselines.
- Explore credibility, uncertainty, fairness and drift analysis.
- Add GPS-exposure reporting or an indicative-premium API if separately justified.
- Add orchestration or cloud deployment if a concrete need arises.

These are possibilities, not delivery commitments. The repository name and Python
package remain unchanged so existing links, imports and commands keep working.
