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
| 04 — Docker verification | Open | Build and run the container; verify `/health` and `/docs`; record the commands and observed results |
| 05 — Results and limitations | Open | Consolidate demo evidence and held-out model results, including benchmark comparison and synthetic-data/sampling limitations |
| 06 — Repository polish | Open | Check setup, examples, links and status against the finished implementation; confirm final tests and lint |

Implemented milestones describe current code. The remaining milestones must not
be marked complete solely because instructions or a Dockerfile exist.

## Boundaries between workflows

```text
Fixed 1 Hz demo traces → schema validation → detector → comparison summary + traces

Synthetic population → simulated trip summaries → behavioural features
                     → policy exposure + claims → Poisson model + evaluation

FastAPI /health + /docs → separate API/container demonstration
```

The population pipeline generates trip counts and speeding summaries directly.
It does not call the detector. The API serves a health check only; an interactive
journey-analysis endpoint is not required for the current finish line.

The Poisson baseline uses a latest-month holdout. It does not establish generalisation
to unseen drivers, real claims or a different generator. Gamma severity, Tweedie
pure premium and grouped-driver validation are optional future extensions.

## Configuration ownership

| File | Current use |
|---|---|
| `configs/simulation.yaml` | Population, time range and random seed for the synthetic pipeline |
| `configs/features.yaml` | Minimum monthly distance for behavioural rates |
| `configs/claims.yaml` | Synthetic claim-frequency and severity assumptions |
| `configs/telemetry.yaml`: `telemetry` | Detector settings used by the demo and explicit configuration loading |
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
- Add journey-analysis or indicative-premium API endpoints.
- Add orchestration or cloud deployment if a concrete need arises.

These are possibilities, not delivery commitments. The repository name and Python
package remain unchanged so existing links, imports and commands keep working.
